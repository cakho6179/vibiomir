"""Unified CLI entrypoint for ViBioMIR R2AI2026 pipeline.

Usage:
    python -m r2ai2026 <command> [options]

Commands:
    clean        Clean raw crawled corpus (drop 404s, remove headers/footers)
    translate    Translate Vietnamese queries to English using MarianMT
    index        Build partitioned BM25 indexes (PubMed & Web)
    search       Run dual-stream cross-lingual BM25 + RRF + Extractive Chunking
    rerank       Run GPU-accelerated Hybrid BGE-M3 dense re-ranking
    validate     Validate a submission ZIP file against Codabench requirements
    reproduce    Execute full end-to-end pipeline to reproduce winning submission
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path


def cmd_clean(args: argparse.Namespace) -> int:
    from r2ai2026.cleaning import clean_document_text, is_valid_document
    from r2ai2026.pipeline import load_documents
    import time

    print(f"Cleaning {args.input} -> {args.output}...")
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                rec = json.loads(line)
            except Exception:
                dropped += 1
                continue
            is_pm = "pubmed.ncbi.nlm.nih.gov" in rec.get("url", "")
            cln = clean_document_text(rec.get("text", "") or "", is_pubmed=is_pm)
            if not is_valid_document(cln, min_chars=args.min_chars):
                dropped += 1
                continue
            rec["text"] = cln
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    print(f"Done in {time.perf_counter() - started:.2f}s: kept {kept:,}, dropped {dropped:,}.")
    return 0


def cmd_translate(args: argparse.Namespace) -> int:
    from r2ai2026.translate import translate_queries_vi_to_en
    import pandas as pd

    print(f"Translating queries from {args.input} -> {args.output}...")
    df = pd.read_parquet(args.input)
    queries = df["query"].tolist()
    translations = translate_queries_vi_to_en(queries, batch_size=args.batch_size)
    df["query_en"] = translations
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df):,} translated queries to {args.output}.")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from r2ai2026.bm25 import BM25Index, BM25Params
    from r2ai2026.pipeline import load_documents
    import time

    print(f"Building BM25S index from {args.docs} -> {args.index_dir}...")
    started = time.perf_counter()
    args.index_dir.mkdir(parents=True, exist_ok=True)

    def text_stream():
        for rec in load_documents(args.docs):
            if args.kind and rec.get("kind") != args.kind:
                continue
            if args.pubmed_only and "pubmed.ncbi.nlm.nih.gov" not in rec.get("url", ""):
                continue
            if args.web_only and "pubmed.ncbi.nlm.nih.gov" in rec.get("url", ""):
                continue
            yield (rec["id"], rec.get("text") or "")

    index = BM25Index.build_from_stream(
        text_stream(),
        params=BM25Params(method="bm25+", delta=0.7),
        int_dtype="int32",
        csc_backend="scipy",
    )
    index.save(args.index_dir, name="docs")
    print(f"Index built and saved to {args.index_dir} in {time.perf_counter() - started:.2f}s.")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from r2ai2026.bm25 import BM25Index
    from r2ai2026.chunking import spread_windows_chunk, query_biased_paragraph_chunk
    from r2ai2026.pipeline import load_documents, document_text
    from r2ai2026.fusion import reciprocal_rank_fusion
    from r2ai2026.submit import write_submission_zip, submission_stats, validate_submission
    import pandas as pd
    import time

    print(f"Running cross-lingual retrieval -> {args.submission}...")
    started = time.perf_counter()
    df_queries = pd.read_parquet(args.queries)
    if "query_en" not in df_queries.columns:
        raise ValueError(f"'{args.queries}' must contain 'query_en'. Run 'translate' first.")

    queries_vi = df_queries["query"].tolist()
    queries_en = df_queries["query_en"].tolist()
    query_ids = df_queries["id"].tolist()

    # Load indexes
    if args.index_pubmed and args.index_web:
        index_pm = BM25Index.load(args.index_pubmed, name="docs", mmap=True)
        index_web = BM25Index.load(args.index_web, name="docs", mmap=True)
        pool = args.candidate_pool

        ranked_pm_vi = index_pm.query_batch(queries_vi, top_k=pool)
        ranked_pm_en = index_pm.query_batch(queries_en, top_k=pool)
        ranked_web_vi = index_web.query_batch(queries_vi, top_k=pool)
        ranked_web_en = index_web.query_batch(queries_en, top_k=pool)

        fused_results: list[list[int]] = []
        needed_docs: set[int] = set()
        for i in range(len(query_ids)):
            fused = reciprocal_rank_fusion(
                [ranked_web_vi[i], ranked_pm_en[i], ranked_web_en[i], ranked_pm_vi[i]],
                weights=[1.0, 1.0, 0.5, 0.3],
                k=args.rrf_k,
                top_n=args.topk_docs,
            )
            fused_results.append(fused)
            needed_docs.update(fused)
    else:
        index = BM25Index.load(args.index_dir, name="docs", mmap=True)
        ranked_vi = index.query_batch(queries_vi, top_k=args.candidate_pool)
        ranked_en = index.query_batch(queries_en, top_k=args.candidate_pool)
        fused_results = []
        needed_docs = set()
        for i in range(len(query_ids)):
            fused = reciprocal_rank_fusion(
                [ranked_vi[i], ranked_en[i]],
                weights=[1.0, 1.0],
                k=args.rrf_k,
                top_n=args.topk_docs,
            )
            fused_results.append(fused)
            needed_docs.update(fused)

    print(f"Retrieved and fused top-{args.topk_docs} across {len(query_ids):,} queries in {time.perf_counter() - started:.2f}s")
    print(f"Loading {len(needed_docs):,} document texts from {args.docs}...")
    doc_texts: dict[int, str] = {}
    for rec in load_documents(args.docs):
        did = rec["id"]
        if did in needed_docs:
            doc_texts[did] = document_text(rec)

    records: list[dict] = []
    for q_idx, qid in enumerate(query_ids):
        top_docs = fused_results[q_idx]
        chunks = []
        for did in top_docs:
            dtext = doc_texts.get(did, "")
            if args.chunk_policy == "spread":
                chk = spread_windows_chunk(dtext, max_tokens=args.chunk_budget, windows=args.spread_windows)
            elif args.chunk_policy == "query_biased":
                chk = query_biased_paragraph_chunk(dtext, query=queries_vi[q_idx], max_tokens=args.chunk_budget)
            else:
                chk = dtext[:args.chunk_budget * 4]
            chunks.append(chk)

        records.append({
            "query_id": qid,
            "relevant_docs": top_docs,
            "relevant_chunks": chunks,
        })

    args.submission.parent.mkdir(parents=True, exist_ok=True)
    write_submission_zip(records, args.submission)
    stats = submission_stats(records)
    print(f"Successfully generated {args.submission} ({args.submission.stat().st_size / 1e6:.2f} MB)")
    print(f"Mean chunk chars: {stats['mean_chunk_chars']:.0f}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from r2ai2026.submit import validate_submission, submission_stats
    import pandas as pd

    print(f"Validating {args.submission}...")
    df = pd.read_parquet(args.queries)
    expected_ids = df["id"].tolist()

    with zipfile.ZipFile(args.submission, "r") as z:
        names = z.namelist()
        if "submission.json" not in names:
            print("ERROR: ZIP does not contain 'submission.json' at root!")
            return 1
        with z.open("submission.json") as f:
            records = json.load(f)

    errors = validate_submission(records, expected_ids=expected_ids)
    if errors:
        print(f"FAILED with {len(errors)} errors:")
        for e in errors[:10]:
            print(f"  - {e}")
        return 1

    stats = submission_stats(records)
    print("VALIDATION PASSED! 0 errors.")
    print(f"  Queries:          {int(stats['n_queries'])}")
    print(f"  Docs per query:   {stats['docs_per_query']:.1f}")
    print(f"  Chunks per query: {stats['chunks_per_query']:.1f}")
    print(f"  Mean chunk chars: {stats['mean_chunk_chars']:.0f}")
    print(f"  ZIP Size:         {args.submission.stat().st_size / 1e6:.2f} MB")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Clean
    p_clean = subparsers.add_parser("clean", help="Clean raw crawled corpus")
    p_clean.add_argument("--input", type=Path, default=Path("data/raw/crawl/docs.jsonl"))
    p_clean.add_argument("--output", type=Path, default=Path("data/docs_cleaned.jsonl"))
    p_clean.add_argument("--min-chars", type=int, default=150)
    p_clean.set_defaults(func=cmd_clean)

    # Translate
    p_trans = subparsers.add_parser("translate", help="Translate queries vi->en via MarianMT")
    p_trans.add_argument("--input", type=Path, default=Path("data/raw/btc/query.parquet"))
    p_trans.add_argument("--output", type=Path, default=Path("data/query_translated.parquet"))
    p_trans.add_argument("--batch-size", type=int, default=32)
    p_trans.set_defaults(func=cmd_translate)

    # Index
    p_idx = subparsers.add_parser("index", help="Build BM25S index")
    p_idx.add_argument("--docs", type=Path, default=Path("data/docs_cleaned.jsonl"))
    p_idx.add_argument("--index-dir", type=Path, required=True)
    p_idx.add_argument("--kind", type=str, default=None)
    p_idx.add_argument("--pubmed-only", action="store_true")
    p_idx.add_argument("--web-only", action="store_true")
    p_idx.set_defaults(func=cmd_index)

    # Search
    p_search = subparsers.add_parser("search", help="Run dual-stream retrieval & chunking")
    p_search.add_argument("--docs", type=Path, default=Path("data/docs_cleaned.jsonl"))
    p_search.add_argument("--queries", type=Path, default=Path("data/query_translated.parquet"))
    p_search.add_argument("--index-dir", type=Path, default=None)
    p_search.add_argument("--index-pubmed", type=Path, default=None)
    p_search.add_argument("--index-web", type=Path, default=None)
    p_search.add_argument("--submission", type=Path, required=True)
    p_search.add_argument("--topk-docs", type=int, default=35)
    p_search.add_argument("--candidate-pool", type=int, default=100)
    p_search.add_argument("--chunk-budget", type=int, default=1400)
    p_search.add_argument("--spread-windows", type=int, default=4)
    p_search.add_argument("--chunk-policy", type=str, choices=["spread", "query_biased", "head"], default="spread")
    p_search.add_argument("--rrf-k", type=int, default=60)
    p_search.set_defaults(func=cmd_search)

    # Validate
    p_val = subparsers.add_parser("validate", help="Validate submission zip")
    p_val.add_argument("--submission", type=Path, required=True)
    p_val.add_argument("--queries", type=Path, default=Path("data/raw/btc/query.parquet"))
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
