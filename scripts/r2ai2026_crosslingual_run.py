"""Cross-lingual BM25 retrieval with Reciprocal Rank Fusion (RRF).

Combines Vietnamese clinical queries and their English translations (generated via
MarianMT) to query both Vietnamese medical articles and English PubMed papers,
fusing the results via Reciprocal Rank Fusion (RRF) and extracting spread-window chunks.

    python scripts/r2ai2026_crosslingual_run.py \
        --docs data/docs.jsonl \
        --queries data/query_translated.parquet \
        --index-dir runs/indexes/index_v1 \
        --submission runs/submissions/submission_crosslingual_v1.zip \
        --topk-docs 35 --chunk-budget 1400 --spread-windows 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai2026.bm25 import BM25Index  # noqa: E402
from r2ai2026.chunking import spread_windows_chunk  # noqa: E402
from r2ai2026.pipeline import load_documents, document_text  # noqa: E402
from r2ai2026.submit import (  # noqa: E402
    submission_stats,
    validate_submission,
    write_submission_zip,
)


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    weights: list[float] | None = None,
    k: int = 60,
    top_n: int = 35,
) -> list[int]:
    """Fuse multiple ranked doc lists using Reciprocal Rank Fusion."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[int, float] = defaultdict(float)
    for ranked_docs, weight in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked_docs, start=1):
            scores[doc_id] += weight / (k + rank)

    sorted_docs = sorted(scores.keys(), key=lambda doc: scores[doc], reverse=True)
    return sorted_docs[:top_n]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True,
                        help="Parquet file containing 'id', 'query', and 'query_en'")
    parser.add_argument("--index-dir", type=Path, default=None,
                        help="Single BM25 index directory containing both VI and EN docs")
    parser.add_argument("--index-pubmed", type=Path, default=None,
                        help="Partitioned index directory for PubMed English abstracts")
    parser.add_argument("--index-web", type=Path, default=None,
                        help="Partitioned index directory for Vietnamese/Chinese web docs")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--topk-docs", type=int, default=35)
    parser.add_argument("--candidate-pool", type=int, default=100)
    parser.add_argument("--chunk-budget", type=int, default=1400)
    parser.add_argument("--spread-windows", type=int, default=4)
    parser.add_argument("--weight-vi", type=float, default=1.0)
    parser.add_argument("--weight-en", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--stats-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.index_dir and not (args.index_pubmed and args.index_web):
        parser.error("Must provide either --index-dir OR both (--index-pubmed and --index-web)")

    sys.stdout.reconfigure(encoding="utf-8")

    # 1. Load Queries
    started = time.perf_counter()
    df_queries = pd.read_parquet(args.queries)
    if "query_en" not in df_queries.columns:
        raise ValueError(f"'{args.queries}' must contain 'query_en'. Run r2ai2026/translate.py first.")
    print(f"Loaded {len(df_queries):,} queries (VI + EN) in {time.perf_counter() - started:.2f}s")

    queries_vi = df_queries["query"].tolist()
    queries_en = df_queries["query_en"].tolist()
    query_ids = df_queries["id"].tolist()

    # 2. Retrieve with VI queries and EN queries
    started = time.perf_counter()
    if args.index_pubmed and args.index_web:
        index_pm = BM25Index.load(args.index_pubmed, name="docs", mmap=True)
        index_web = BM25Index.load(args.index_web, name="docs", mmap=True)
        print(f"Loaded partitioned indexes: PubMed ({index_pm.size:,} docs) & Web ({index_web.size:,} docs)")
        print(f"Searching Web index with Vietnamese queries (k={args.candidate_pool})...")
        hits_vi = index_web.search_batch(queries_vi, k=args.candidate_pool)
        print(f"Searching PubMed index with English translated queries (k={args.candidate_pool})...")
        hits_en = index_pm.search_batch(queries_en, k=args.candidate_pool)
    else:
        index = BM25Index.load(args.index_dir, name="docs", mmap=True)
        print(f"Loaded BM25 index ({index.size:,} docs) from {args.index_dir}")
        print(f"Searching BM25 index with Vietnamese queries (k={args.candidate_pool})...")
        hits_vi = index.search_batch(queries_vi, k=args.candidate_pool)
        print(f"Searching BM25 index with English translated queries (k={args.candidate_pool})...")
        hits_en = index.search_batch(queries_en, k=args.candidate_pool)

    search_time = time.perf_counter() - started
    print(f"Dual-language search completed in {search_time:.2f}s ({len(df_queries)/search_time:.1f} q/s)")

    # 4. Fuse rankings with RRF
    started = time.perf_counter()
    fused_rankings: dict[int, list[int]] = {}
    weights = [args.weight_vi, args.weight_en]

    for idx, qid in enumerate(query_ids):
        docs_vi = [int(hit[0][0]) for hit in hits_vi[idx]]
        docs_en = [int(hit[0][0]) for hit in hits_en[idx]]
        fused = reciprocal_rank_fusion([docs_vi, docs_en], weights=weights, k=args.rrf_k, top_n=args.topk_docs)
        fused_rankings[qid] = fused
    print(f"Fused rankings with RRF in {time.perf_counter() - started:.2f}s")

    # 5. Load Corpus for Chunk Extraction
    # Notice: we only need documents that appear in the fused rankings!
    needed_doc_ids = {doc_id for docs in fused_rankings.values() for doc_id in docs}
    print(f"Unique documents required for top-{args.topk_docs} across all queries: {len(needed_doc_ids):,}")

    started = time.perf_counter()
    doc_texts: dict[int, str] = {}
    for record in load_documents(args.docs):
        if not record.get("ok", True):
            continue
        doc_id = int(record["id"])
        if doc_id in needed_doc_ids:
            doc_texts[doc_id] = document_text(record)
            if len(doc_texts) == len(needed_doc_ids):
                break
    print(f"Loaded {len(doc_texts):,} needed document texts in {time.perf_counter() - started:.2f}s")

    # 6. Extract chunks and build submission
    started = time.perf_counter()
    records: list[dict] = []
    missing_docs = 0

    for qid in query_ids:
        docs = fused_rankings[qid]
        chunks: list[dict] = []
        for doc_id in docs:
            text = doc_texts.get(doc_id)
            if not text:
                missing_docs += 1
                continue
            if args.chunk_budget <= 0:
                chunks.append({"doc_id": int(doc_id), "chunk_text": text})
            else:
                doc_chunk = spread_windows_chunk(text, max_tokens=args.chunk_budget, windows=args.spread_windows)
                if doc_chunk:
                    chunks.append({"doc_id": int(doc_id), "chunk_text": doc_chunk})

        records.append({"id": int(qid), "relevant_docs": docs, "relevant_chunks": chunks})

    if missing_docs > 0:
        print(f"Warning: {missing_docs} document references were missing from corpus file")

    stats = submission_stats(records)
    print("Submission statistics:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    errors = validate_submission(records)
    if errors:
        print(f"Validation FAILED with {len(errors)} error(s):")
        for err in errors[:10]:
            print(f"  - {err}")
        return 1

    print("Validation PASSED (0 errors).")
    zip_stats = write_submission_zip(records, args.submission)
    zip_size_mb = zip_stats["zip_mb"]
    print(f"Successfully written {args.submission} ({zip_size_mb:.2f} MB)")

    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        zip_stats["submission_file"] = str(args.submission)
        args.stats_out.write_text(json.dumps(zip_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
