"""Build a small doc list from multi-channel top-K consensus.

This is the C1 of the progressive pruning plan: a doc survives only when it is
retrieved near the top by at least `--min-support` independent channels.  The
chunk channel is left to the caller; the probe keeps r24 chunks so only the doc
channel changes.

    python select_consensus.py --out temp_submission/consensus46/lists.parquet \
        --top-k 40 --min-support 2
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
CHANNELS = {
    "w12": ROOT / "runs/submissions/w12_VI_ranking.parquet",
    "rerank": ROOT / "runs/submissions/rerank_r1_ranking.parquet",
    "m3only": ROOT / "runs/submissions/m3only_ranking.parquet",
    "e1v2m3": ROOT / "runs/rankings/e1v2_m3_w12_VIonly.parquet",
    "bm25": ROOT / "runs/bm25_vi_full/ranking.parquet",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--job", default="consensus")
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--channels", default="")
    args = ap.parse_args()

    names = args.channels.split(",") if args.channels else list(CHANNELS)
    counters: dict[int, Counter] = {}
    best: dict[int, dict[int, int]] = {}
    for name in names:
        d = pd.read_parquet(CHANNELS[name], columns=["qid", "doc_id", "rank"])
        d = d[d["rank"] < args.top_k]
        for qid, g in d.groupby("qid"):
            q = int(qid)
            counters.setdefault(q, Counter()).update(g["doc_id"].astype(int).tolist())
            b = best.setdefault(q, {})
            for doc, rank in zip(g["doc_id"].astype(int), g["rank"].astype(int)):
                b[doc] = min(b.get(doc, 10**9), int(rank))

    rows = []
    stats = []
    for qid in sorted(counters):
        keep = [(doc, c) for doc, c in counters[qid].items() if c >= args.min_support]
        keep.sort(key=lambda x: (-x[1], best[qid][x[0]], x[0]))
        docs = [doc for doc, _ in keep]
        rows.append(pd.DataFrame({"job": args.job, "qid": qid, "kind": "doc",
                                  "pos": np.arange(len(docs)), "doc_id": docs,
                                  "size": 0, "src": ""}))
        stats.append({"qid": qid, "n_docs": len(docs),
                      "n_sup2": sum(1 for _, c in keep if c == 2),
                      "n_sup3": sum(1 for _, c in keep if c >= 3)})
    out = pd.concat(rows, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    st = pd.DataFrame(stats)
    print(f"channels={names} top_k={args.top_k} min_support={args.min_support}")
    print(f"-> {args.out}: {len(out):,} rows, {st.shape[0]} queries")
    print(st[["n_docs", "n_sup2", "n_sup3"]].describe().to_string())
    print("queries with docs:", int((st.n_docs > 0).sum()))


if __name__ == "__main__":
    main()
