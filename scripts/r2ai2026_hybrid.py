"""Reciprocal-rank fusion of BM25 and dense candidates.

RRF is used instead of score blending because the two systems' scores are not
comparable (BM25 is unbounded, cosine sits in [-1, 1]) while ranks are. The
output parquet plugs straight into ``r2ai2026_run.py --rankings``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--dense", type=Path, default=None, help="parquet: query_id, doc_id, rank, score")
    parser.add_argument("--index-dir", type=Path, default=None, help="BM25 index directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-bm25", type=int, default=300)
    parser.add_argument("--top-dense", type=int, default=300)
    parser.add_argument("--top-out", type=int, default=300)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--limit-queries", type=int, default=None)
    args = parser.parse_args()

    import pandas as pd

    from r2ai2026.bm25 import BM25Index
    from r2ai2026.pipeline import load_queries

    queries = load_queries(args.queries)
    if args.limit_queries:
        queries = queries[: args.limit_queries]
    print(f"[hybrid] {len(queries):,} queries")

    fused: dict[int, dict[int, float]] = {}
    sources: dict[int, dict[int, dict[str, float]]] = {}

    def add(query_id: int, doc_id: int, weight: float, rank: float, source: str, score: float) -> None:
        contribution = weight / (args.rrf_k + rank)
        fused.setdefault(query_id, {})
        fused[query_id][doc_id] = fused[query_id].get(doc_id, 0.0) + contribution
        sources.setdefault(query_id, {}).setdefault(doc_id, {})[f"{source}_rank"] = float(rank)
        sources[query_id][doc_id][f"{source}_score"] = float(score)

    if args.index_dir is not None:
        started = time.perf_counter()
        index = BM25Index.load(args.index_dir, name="docs")
        hits = index.search_batch([query["text"] for query in queries], k=args.top_bm25)
        for query, per_query in zip(queries, hits):
            for rank, (descriptor, score) in enumerate(per_query, start=1):
                add(int(query["id"]), int(descriptor[0]), args.bm25_weight, rank, "bm25", float(score))
        print(f"[hybrid] bm25 done in {time.perf_counter() - started:.1f}s")

    if args.dense is not None:
        table = pd.read_parquet(args.dense)
        table = table[table["rank"] <= args.top_dense]
        for row in table.itertuples(index=False):
            add(int(row.query_id), int(row.doc_id), args.dense_weight, float(row.rank), "dense",
                float(getattr(row, "score", 0.0)))
        print(f"[hybrid] dense rows: {len(table):,}")

    rows = []
    for query_id, scores in fused.items():
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[: args.top_out]
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            detail = sources.get(query_id, {}).get(doc_id, {})
            rows.append({
                "query_id": query_id,
                "doc_id": doc_id,
                "score": score,
                "rank": rank,
                "bm25_rank": detail.get("bm25_rank", np.nan),
                "dense_rank": detail.get("dense_rank", np.nan),
                "bm25_score": detail.get("bm25_score", np.nan),
                "dense_score": detail.get("dense_score", np.nan),
            })
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    overlap = frame.dropna(subset=["bm25_rank", "dense_rank"])
    print(f"[hybrid] wrote {len(frame):,} rows for {frame['query_id'].nunique():,} queries -> {args.out}")
    print(f"[hybrid] candidates in both systems: {len(overlap):,} "
          f"({len(overlap) / max(1, len(frame)):.1%}); "
          f"bm25-only {frame['dense_rank'].isna().sum():,}; dense-only {frame['bm25_rank'].isna().sum():,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
