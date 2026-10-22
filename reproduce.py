"""1-Click End-to-End Reproduction Script for ViBioMIR R2AI2026 (Shared Task 3).

Reproduces the winning submission for Multilingual Medical Document Retrieval
and Extractive Chunking on Codabench.

Usage:
    python reproduce.py [--corpus data/docs_cleaned.jsonl]
                        [--output runs/submissions/submission_reproduced.zip]

Author: Team ca_kho (R2AI2026 ViBioMIR Shared Task 3)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

from r2ai2026.bm25 import BM25Index, BM25Params
from r2ai2026.chunking import spread_windows_chunk
from r2ai2026.cleaning import clean_document_text, is_valid_document
from r2ai2026.fusion import reciprocal_rank_fusion
from r2ai2026.pipeline import load_documents, document_text
from r2ai2026.submit import write_submission_zip, submission_stats, validate_submission
from r2ai2026.translate import translate_queries_vi_to_en


def run_pipeline(
    raw_docs: Path,
    cleaned_docs: Path,
    queries_path: Path,
    translated_queries_path: Path,
    index_pubmed_dir: Path,
    index_web_dir: Path,
    submission_path: Path,
    force_rebuild: bool = False,
) -> None:
    print("=" * 70)
    print("ViBioMIR R2AI2026 - 1-Click Reproduction Pipeline")
    print("=" * 70)
    start_total = time.perf_counter()

    # Step 1: Query Translation (Vietnamese -> English)
    if not translated_queries_path.exists() or force_rebuild:
        print(f"\n[Step 1/5] Translating queries {queries_path} -> {translated_queries_path}...")
        df_q = pd.read_parquet(queries_path)
        queries = df_q["query"].tolist()
        translations = translate_queries_vi_to_en(queries, batch_size=32)
        df_q["query_en"] = translations
        translated_queries_path.parent.mkdir(parents=True, exist_ok=True)
        df_q.to_parquet(translated_queries_path, index=False)
        print(f"-> Translated {len(df_q):,} queries.")
    else:
        print(f"\n[Step 1/5] Translated queries found at {translated_queries_path}.")

    # Step 2: Corpus Cleaning (Drop 404/junk + Strip Boilerplate)
    if not cleaned_docs.exists() or force_rebuild:
        print(f"\n[Step 2/5] Cleaning raw corpus {raw_docs} -> {cleaned_docs}...")
        t0 = time.perf_counter()
        cleaned_docs.parent.mkdir(parents=True, exist_ok=True)
        kept, dropped = 0, 0
        with open(raw_docs, "r", encoding="utf-8") as fin, open(cleaned_docs, "w", encoding="utf-8") as fout:
            import json
            for line in fin:
                try:
                    rec = json.loads(line)
                except Exception:
                    dropped += 1
                    continue
                is_pm = "pubmed.ncbi.nlm.nih.gov" in rec.get("url", "")
                cln = clean_document_text(rec.get("text", "") or "", is_pubmed=is_pm)
                if not is_valid_document(cln, min_chars=150):
                    dropped += 1
                    continue
                rec["text"] = cln
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
        print(f"-> Cleaned corpus ready in {time.perf_counter() - t0:.2f}s: kept {kept:,}, dropped {dropped:,}.")
    else:
        print(f"\n[Step 2/5] Cleaned corpus found at {cleaned_docs}.")

    # Step 3: BM25S Partitioned Indexing (PubMed + Web)
    docs_to_index = cleaned_docs if cleaned_docs.exists() else raw_docs
    if not (index_pubmed_dir / "docs_bm25.json").exists() or force_rebuild:
        print(f"\n[Step 3a/5] Building PubMed BM25S index -> {index_pubmed_dir}...")
        index_pubmed_dir.mkdir(parents=True, exist_ok=True)
        pm_stream = ((r["id"], r.get("text") or "") for r in load_documents(docs_to_index) if "pubmed.ncbi.nlm.nih.gov" in r.get("url", ""))
        idx_pm = BM25Index.build_from_stream(pm_stream, params=BM25Params(method="bm25+", delta=0.7), int_dtype="int32", csc_backend="scipy")
        idx_pm.save(index_pubmed_dir, name="docs")
        print("-> PubMed index built.")
    else:
        print(f"\n[Step 3a/5] PubMed index found at {index_pubmed_dir}.")

    if not (index_web_dir / "docs_bm25.json").exists() or force_rebuild:
        print(f"\n[Step 3b/5] Building Web BM25S index -> {index_web_dir}...")
        index_web_dir.mkdir(parents=True, exist_ok=True)
        web_stream = ((r["id"], r.get("text") or "") for r in load_documents(docs_to_index) if "pubmed.ncbi.nlm.nih.gov" not in r.get("url", ""))
        idx_web = BM25Index.build_from_stream(web_stream, params=BM25Params(method="bm25+", delta=0.7), int_dtype="int32", csc_backend="scipy")
        idx_web.save(index_web_dir, name="docs")
        print("-> Web index built.")
    else:
        print(f"\n[Step 3b/5] Web index found at {index_web_dir}.")

    # Step 4: Dual-Stream Retrieval + RRF + Extractive Chunking
    print(f"\n[Step 4/5] Executing Cross-Lingual Dual-Stream Retrieval & Chunking...")
    t0 = time.perf_counter()
    df_queries = pd.read_parquet(translated_queries_path)
    queries_vi = df_queries["query"].tolist()
    queries_en = df_queries["query_en"].tolist()
    query_ids = df_queries["id"].tolist()

    idx_pm = BM25Index.load(index_pubmed_dir, name="docs", mmap=True)
    idx_web = BM25Index.load(index_web_dir, name="docs", mmap=True)

    print("Retrieving candidates across partitions...")
    ranked_pm_vi = idx_pm.query_batch(queries_vi, top_k=100)
    ranked_pm_en = idx_pm.query_batch(queries_en, top_k=100)
    ranked_web_vi = idx_web.query_batch(queries_vi, top_k=100)
    ranked_web_en = idx_web.query_batch(queries_en, top_k=100)

    fused_results: list[list[int]] = []
    needed_docs: set[int] = set()
    for i in range(len(query_ids)):
        fused = reciprocal_rank_fusion(
            [ranked_web_vi[i], ranked_pm_en[i], ranked_web_en[i], ranked_pm_vi[i]],
            weights=[1.0, 1.0, 0.5, 0.3],
            k=60,
            top_n=35,
        )
        fused_results.append(fused)
        needed_docs.update(fused)

    print(f"Loading {len(needed_docs):,} document texts for chunking...")
    doc_texts: dict[int, str] = {}
    for rec in load_documents(docs_to_index):
        did = rec["id"]
        if did in needed_docs:
            doc_texts[did] = document_text(rec)

    print("Generating spread extractive chunks (1400 tokens / 4 windows)...")
    records: list[dict] = []
    for q_idx, qid in enumerate(query_ids):
        top_docs = fused_results[q_idx]
        chunks = [spread_windows_chunk(doc_texts.get(did, ""), max_tokens=1400, windows=4) for did in top_docs]
        records.append({
            "query_id": qid,
            "relevant_docs": top_docs,
            "relevant_chunks": chunks,
        })

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission_zip(records, submission_path)
    print(f"-> Generated {submission_path} ({submission_path.stat().st_size / 1e6:.2f} MB) in {time.perf_counter() - t0:.2f}s")

    # Step 5: Official Codabench Schema Validation
    print(f"\n[Step 5/5] Validating generated submission against Codabench schema...")
    errors = validate_submission(records, expected_ids=query_ids)
    if errors:
        print(f"VALIDATION FAILED with {len(errors)} errors!")
        for e in errors[:10]:
            print(f"  - {e}")
        sys.exit(1)

    stats = submission_stats(records)
    print("=" * 70)
    print("REPRODUCTION COMPLETE: VALIDATION PASSED (0 ERRORS)")
    print("=" * 70)
    print(f"  Output File:      {submission_path.resolve()}")
    print(f"  Total Queries:    {int(stats['n_queries'])}")
    print(f"  Docs per Query:   {stats['docs_per_query']:.1f}")
    print(f"  Chunks per Query: {stats['chunks_per_query']:.1f}")
    print(f"  Mean Chunk Size:  {stats['mean_chunk_chars']:.0f} characters")
    print(f"  Total Time:       {time.perf_counter() - start_total:.2f}s")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-docs", type=Path, default=Path("data/raw/crawl/docs.jsonl"))
    parser.add_argument("--cleaned-docs", type=Path, default=Path("data/docs_cleaned.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/raw/btc/query.parquet"))
    parser.add_argument("--translated-queries", type=Path, default=Path("data/query_translated.parquet"))
    parser.add_argument("--index-pubmed", type=Path, default=Path("runs/indexes/index_pubmed"))
    parser.add_argument("--index-web", type=Path, default=Path("runs/indexes/index_web_650k"))
    parser.add_argument("--output", type=Path, default=Path("runs/submissions/submission_reproduced.zip"))
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild all intermediate artifacts")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    run_pipeline(
        raw_docs=args.raw_docs,
        cleaned_docs=args.cleaned_docs,
        queries_path=args.queries,
        translated_queries_path=args.translated_queries,
        index_pubmed_dir=args.index_pubmed,
        index_web_dir=args.index_web,
        submission_path=args.output,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
