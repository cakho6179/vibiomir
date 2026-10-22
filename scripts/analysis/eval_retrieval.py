"""Offline retrieval A/B: does a ranking put a document's own title on top?

Health articles carry question-like titles, so using each title as a query and
asking whether the document itself is ranked highly gives a labelled set for free
("title-as-query"), broken down by language so a weak language cannot hide inside
an average. Absolute numbers are optimistic (the title is in the document) but
*relative* differences between rankings are informative.

Rankings come either from saved parquets (``--ranking name=path``) or from a live
BM25 index (``--bm25-index path``).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--bm25-index", type=Path, default=None)
    parser.add_argument("--ranking", action="append", default=[], metavar="name=path.parquet")
    parser.add_argument("--n", type=int, default=400, help="how many title queries to sample")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--per-kind", action="store_true", help="sample evenly per kind instead of randomly")
    args = parser.parse_args()

    from r2ai2026.pipeline import load_documents
    from r2ai2026.text import word_count

    started = time.perf_counter()
    documents: dict[int, dict] = {}
    for record in load_documents(args.docs):
        if not record.get("ok"):
            continue
        title = str(record.get("title") or "").strip()
        if word_count(title) < args.min_words:
            continue
        documents[int(record["id"])] = {"title": title, "kind": str(record.get("kind") or "?")}
    print(f"[eval] {len(documents):,} documents with usable titles ({time.perf_counter() - started:.1f}s)")

    rng = random.Random(args.seed)
    if args.per_kind:
        buckets: dict[str, list[int]] = defaultdict(list)
        for doc_id, record in documents.items():
            buckets[record["kind"]].append(doc_id)
        per_kind = max(1, args.n // max(1, len(buckets)))
        sample: list[int] = []
        for kind, ids in sorted(buckets.items()):
            sample.extend(rng.sample(ids, min(per_kind, len(ids))))
    else:
        sample = rng.sample(sorted(documents), min(args.n, len(documents)))
    print(f"[eval] sampling {len(sample):,} title queries")

    rankings: dict[str, dict[int, list[int]]] = {}

    if args.bm25_index is not None:
        from r2ai2026.bm25 import BM25Index

        index = BM25Index.load(args.bm25_index, name="docs")
        hits = index.search_batch([documents[doc_id]["title"] for doc_id in sample], k=args.depth)
        rankings["bm25"] = {
            doc_id: [int(descriptor[0]) for descriptor, _ in per_query]
            for doc_id, per_query in zip(sample, hits)
        }
        print(f"[eval] bm25 rankings ready ({time.perf_counter() - started:.1f}s)")

    for spec in args.ranking:
        if "=" not in spec:
            raise SystemExit(f"--ranking expects name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        import pandas as pd

        table = pd.read_parquet(path)
        table = table[table["query_id"].isin(sample)]
        if "rank" in table.columns:
            table = table.sort_values(["query_id", "rank"], kind="stable")
        table = table.groupby("query_id", sort=False).head(args.depth)
        rankings[name] = {
            int(query_id): [int(value) for value in group["doc_id"].tolist()]
            for query_id, group in table.groupby("query_id", sort=False)
        }
        print(f"[eval] {name}: {len(rankings[name]):,} queries from {path}")

    ks = (1, 5, 10, 20, 50, 100)
    header = f"{'ranking':12s} {'kind':6s} {'n':>6s} " + " ".join(f"R@{k:<4d}" for k in ks) + f"{'MRR':>7s}"
    print()
    print(header)
    print("-" * len(header))
    for name, per_query in rankings.items():
        by_kind: dict[str, list[int]] = defaultdict(list)
        for doc_id in sample:
            by_kind[documents[doc_id]["kind"]].append(doc_id)
        for kind in [*sorted(by_kind), "ALL"]:
            ids = sample if kind == "ALL" else by_kind[kind]
            if not ids:
                continue
            hits_at = {k: 0 for k in ks}
            reciprocal = 0.0
            for doc_id in ids:
                ranked = per_query.get(doc_id, [])
                for k in ks:
                    if doc_id in set(ranked[:k]):
                        hits_at[k] += 1
                for position, candidate in enumerate(ranked, start=1):
                    if candidate == doc_id:
                        reciprocal += 1.0 / position
                        break
            row = f"{name:12s} {kind:6s} {len(ids):6,d} " + " ".join(
                f"{hits_at[k] / len(ids):<6.3f}"[:6] for k in ks
            ) + f"{reciprocal / len(ids):7.3f}"
            print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
