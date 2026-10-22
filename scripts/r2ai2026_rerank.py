"""Cross-encoder reranking of fused candidates with an open-weight multilingual model.

A bi-encoder compares query and document vectors that were computed independently;
a cross-encoder reads the pair together, which is what fixes the shallow ordering
of a hybrid list, especially across languages. Only candidate documents are loaded
into memory, so the corpus can stay on disk.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-n", type=int, default=100, help="candidates per query to rerank")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--text-chars", type=int, default=1600, help="document characters shown to the model")
    parser.add_argument("--limit-queries", type=int, default=None)
    args = parser.parse_args()

    import pandas as pd
    import torch
    from sentence_transformers import CrossEncoder

    from r2ai2026.pipeline import document_text, load_documents, load_queries

    queries = {int(query["id"]): str(query["text"]) for query in load_queries(args.queries)}
    if args.limit_queries:
        queries = dict(list(queries.items())[: args.limit_queries])

    candidates = pd.read_parquet(args.candidates)
    candidates = candidates[candidates["query_id"].isin(queries)]
    if "rank" in candidates.columns:
        candidates = candidates.sort_values(["query_id", "rank"], kind="stable")
    candidates = candidates.groupby("query_id", sort=False).head(args.top_n)
    wanted = set(int(value) for value in candidates["doc_id"].unique())
    print(f"[rerank] {len(candidates):,} pairs for {candidates['query_id'].nunique():,} queries, "
          f"{len(wanted):,} unique documents")

    started = time.perf_counter()
    texts: dict[int, str] = {}
    for record in load_documents(args.docs):
        if not record.get("ok"):
            continue
        doc_id = int(record["id"])
        if doc_id in wanted:
            text = document_text(record)
            if text:
                texts[doc_id] = text[: args.text_chars]
    print(f"[rerank] loaded {len(texts):,} candidate documents in {time.perf_counter() - started:.1f}s")

    model = CrossEncoder(args.model, device=args.device, max_length=args.max_length,
                         activation_fn=torch.nn.Sigmoid())
    pairs = []
    keys = []
    for row in candidates.itertuples(index=False):
        doc_id = int(row.doc_id)
        text = texts.get(doc_id)
        if not text:
            continue
        pairs.append((queries[int(row.query_id)], text))
        keys.append((int(row.query_id), doc_id))
    print(f"[rerank] scoring {len(pairs):,} pairs with {args.model}")

    started = time.perf_counter()
    scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True,
                           convert_to_numpy=True)
    elapsed = max(1e-9, time.perf_counter() - started)
    print(f"[rerank] {len(pairs) / elapsed:.1f} pairs/s")

    table = pd.DataFrame({
        "query_id": [key[0] for key in keys],
        "doc_id": [key[1] for key in keys],
        "ce_score": scores.astype("float32"),
    })
    table = table.sort_values(["query_id", "ce_score"], ascending=[True, False], kind="stable")
    table["rank"] = table.groupby("query_id", sort=False).cumcount() + 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(f"[rerank] wrote {len(table):,} rows -> {args.out}")
    print(table.groupby("query_id")["ce_score"].max().describe().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
