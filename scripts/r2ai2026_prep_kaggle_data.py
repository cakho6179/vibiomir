"""Prepare lightweight candidate pool dataset for Kaggle GPU execution.

Extracts top-100 candidates per query from 650k corpus via partitioned BM25,
extracts verbatim lead and paragraphs for those candidates (~35k docs, ~45 MB),
and formats as a Kaggle Dataset bundle ready for upload.

Usage:
    python scripts/r2ai2026_prep_kaggle_data.py --out-dir kaggle/vibiomir_data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai2026.bm25 import BM25Index  # noqa: E402
from r2ai2026.pipeline import document_text, load_documents  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path("data/docs.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/query_translated.parquet"))
    parser.add_argument("--index-pubmed", type=Path, default=Path("runs/indexes/index_pubmed"))
    parser.add_argument("--index-web", type=Path, default=Path("runs/indexes/index_web"))
    parser.add_argument("--candidate-pool", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("kaggle/vibiomir_data"))
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading queries from {args.queries}...")
    df_queries = pd.read_parquet(args.queries)
    queries_vi = df_queries["query"].tolist()
    queries_en = df_queries["query_en"].tolist()
    query_ids = df_queries["id"].tolist()
    print(f"Loaded {len(df_queries):,} queries.")

    print("[2/4] Retrieving BM25 candidate pools from 650k corpus...")
    index_pm = BM25Index.load(args.index_pubmed, name="docs", mmap=True)
    index_web = BM25Index.load(args.index_web, name="docs", mmap=True)

    hits_vi = index_web.search_batch(queries_vi, k=args.candidate_pool)
    hits_en = index_pm.search_batch(queries_en, k=args.candidate_pool)

    # Build candidate mapping per query and gather all unique candidate document IDs
    query_candidates: dict[int, list[int]] = {}
    bm25_scores: dict[int, dict[int, float]] = {}
    all_needed_doc_ids: set[int] = set()

    for idx, qid in enumerate(query_ids):
        docs_vi = [int(hit[0][0]) for hit in hits_vi[idx]]
        docs_en = [int(hit[0][0]) for hit in hits_en[idx]]
        q_scores: dict[int, float] = defaultdict(float)
        for rank, d in enumerate(docs_vi, start=1):
            q_scores[d] += 1.0 / (60 + rank)
        for rank, d in enumerate(docs_en, start=1):
            q_scores[d] += 1.0 / (60 + rank)

        sorted_cands = sorted(q_scores.keys(), key=lambda d: q_scores[d], reverse=True)[:args.candidate_pool]
        query_candidates[qid] = sorted_cands
        bm25_scores[qid] = {d: q_scores[d] for d in sorted_cands}
        all_needed_doc_ids.update(sorted_cands)

    print(f"Found {len(all_needed_doc_ids):,} unique candidate documents for {len(query_ids):,} queries.")

    print(f"[3/4] Extracting candidate texts from {args.docs}...")
    started = time.perf_counter()
    doc_records: list[dict] = []
    found_count = 0

    for record in load_documents(args.docs):
        if not record.get("ok", True):
            continue
        did = int(record["id"])
        if did in all_needed_doc_ids:
            txt = document_text(record)
            doc_records.append({
                "doc_id": did,
                "url": str(record.get("url") or ""),
                "text": txt,
            })
            found_count += 1
            if found_count == len(all_needed_doc_ids):
                break

    print(f"Loaded {len(doc_records):,} candidate texts in {time.perf_counter() - started:.2f}s.")

    # Save candidates to parquet
    df_cands = pd.DataFrame(doc_records)
    cands_path = args.out_dir / "candidates.parquet"
    df_cands.to_parquet(cands_path, index=False, compression="snappy")
    print(f"Saved {cands_path} ({cands_path.stat().st_size / 1e6:.2f} MB)")

    # Save queries and candidate rankings mapping
    df_queries_out = df_queries.copy()
    df_queries_out["candidate_ids"] = [query_candidates[qid] for qid in query_ids]
    queries_path = args.out_dir / "queries.parquet"
    df_queries_out.to_parquet(queries_path, index=False)
    print(f"Saved {queries_path} ({queries_path.stat().st_size / 1e6:.2f} MB)")

    # Save dataset metadata for Kaggle
    dataset_metadata = {
        "title": "vibiomir-candidates-650k",
        "id": "snnguynvnk19hl/vibiomir-candidates-650k",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (args.out_dir / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2), encoding="utf-8"
    )
    print(f"[4/4] Kaggle dataset package ready at: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
