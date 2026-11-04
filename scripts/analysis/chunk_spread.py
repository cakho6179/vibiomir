"""How are the submitted chunks spread over the submitted docs?

    python scripts/analysis/chunk_spread.py submissions/r8_m3rr197.zip [--qid 0]

conv (CHUNKS_RECALL / DOCS_RECALL) sits at 0.65: a third of the gold docs we do find produce no matched
chunk. One reason would be an uneven chunk budget -- if the global top-Nc chunk list piles several chunks
on a handful of docs, most docs get none, and a gold doc with no submitted chunk can never convert.
This measures the actual spread so the fix can be sized.
"""
from __future__ import annotations

import argparse
import collections
import statistics
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
import sys

sys.path.insert(0, str(ROOT))
from r2ai2026.core.submission import read_submission_zip  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zip", type=Path)
    ap.add_argument("--qid", type=int, default=-1)
    args = ap.parse_args()
    recs = read_submission_zip(args.zip)
    picks = recs if args.qid < 0 else [r for r in recs if int(r["id"]) == args.qid]
    per_doc_all: list[int] = []
    covered: list[float] = []
    zero_docs = 0
    for r in picks:
        docs = [d if isinstance(d, (int, str)) else d.get("doc_id") for d in r["relevant_docs"]]
        docs = [str(d) for d in docs]
        chunks = r["relevant_chunks"]
        c2d = collections.Counter(str(c.get("doc_id") if isinstance(c, dict) else c[0]) for c in chunks)
        per_doc_all += [c2d.get(d, 0) for d in docs]
        n_doc = len(docs)
        n_cov = sum(1 for d in docs if c2d.get(d, 0) > 0)
        covered.append(n_cov / n_doc if n_doc else 0.0)
        zero_docs += sum(1 for d in docs if c2d.get(d, 0) == 0)
    n = len(picks)
    print(f"{args.zip.name}: {n} queries, {len(per_doc_all)} doc slots, "
          f"chunks/doc mean {statistics.mean(per_doc_all):.2f}")
    print(f"  docs with >=1 chunk: {1 - zero_docs / len(per_doc_all):.1%}  (mean per query {statistics.mean(covered):.1%})")
    print(f"  docs with 0 chunks: {zero_docs:,}")
    hist = collections.Counter(per_doc_all)
    for k in sorted(hist)[:8]:
        print(f"  {k} chunk(s): {hist[k]:,} docs ({hist[k] / len(per_doc_all):.1%})")
    # how concentrated: share of chunks held by the docs that have any
    tot_chunks = sum(per_doc_all)
    print(f"  total chunk slots {tot_chunks:,} over {len(per_doc_all):,} docs "
          f"=> {tot_chunks / n:.1f} chunks/query, {tot_chunks / max(1, len(per_doc_all) - zero_docs):.2f} "
          f"per covered doc")


if __name__ == "__main__":
    main()
