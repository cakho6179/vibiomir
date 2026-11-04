"""Promote LLM-selected tail docs into the submitted doc list.

    python scripts/analysis/promote_tail.py --scores tail/scores_0.parquet --ref head/scores_0.parquet \
        --promote 30 --nd 197 --nc 148 --out kaggle/datasets/mat_lists16

The doc side is the only side with room left (docsF2 0.3421 vs a 0.626 ceiling at N=197), and the missing
gold docs sit *below* rank 197: pB at 1200 docs/query reaches dR 0.81 while r8 at 197 reaches 0.57, so ~20
gold docs per query are in ranks 198-1200. Reordering inside the top-197 cannot change docsF2 (it is a set
metric), so the only way is to swap docs in from the tail.

Selection is per query and relative: a tail doc is promoted only if its `direct` logodds beats the median
of the same query's head sample (--ref), because absolute logodds are not comparable across queries. The
promoted docs go on top and the base list loses the same number of docs from its END -- the least likely
gold docs -- so the doc set changes only by that swap.

Emits lists.parquet in the materialize_lists format (job, qid, kind, pos, doc_id, size, src).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from decide_lists import load_pool, order  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, required=True, help="tail scores: qid, doc_id, score")
    ap.add_argument("--ref", type=Path, default=None, help="head scores used to set the per-query bar")
    ap.add_argument("--promote", type=int, default=30, help="max docs promoted per query")
    ap.add_argument("--quantile", type=float, default=0.5, help="bar = this quantile of the head sample")
    ap.add_argument("--nd", type=int, default=197)
    ap.add_argument("--nc", type=int, default=148)
    ap.add_argument("--base", default="m3rr")
    ap.add_argument("--pool", default="vi800,zh200,en200")
    ap.add_argument("--job", default="r24")
    ap.add_argument("--out", type=Path, default=ROOT / "kaggle/datasets/mat_lists16")
    args = ap.parse_args()

    cut = {s[:2]: int(s[2:]) for s in args.pool.split(",")}
    pool = load_pool(cut)
    ranked = order(pool, args.base).copy()
    ranked["rank"] = ranked.groupby("qid").cumcount()
    base = ranked[ranked["rank"] < args.nd].sort_values(["qid", "rank"])

    tail = pd.read_parquet(args.scores)[["qid", "doc_id", "score"]]
    tail["qid"] = tail["qid"].astype(int)
    tail["doc_id"] = tail["doc_id"].astype(int)
    ref = None
    if args.ref:
        ref = pd.read_parquet(args.ref)[["qid", "doc_id", "score"]]
        ref["qid"] = ref["qid"].astype(int)
    bar = (ref.groupby("qid")["score"].quantile(args.quantile) if ref is not None
           else pd.Series(dtype=float))

    base_docs = {int(q): g["doc_id"].astype(int).tolist() for q, g in base.groupby("qid")}
    rows = []
    n_prom = []
    for qid, g in tail.groupby("qid"):
        b = base_docs.get(int(qid), [])
        if not b:
            continue
        thr = float(bar.get(int(qid), 0.0))
        cand = g[(g["score"] > thr) & (~g["doc_id"].isin(b))].sort_values("score", ascending=False)
        take = cand["doc_id"].astype(int).tolist()[:args.promote]
        n_prom.append(len(take))
        keep = b[:args.nd - len(take)]  # the tail of the base list is what gets swapped out
        order_docs = take + keep
        rows += [(args.job, int(qid), "doc", i, d, 0, "") for i, d in enumerate(order_docs)]
        rows += [(args.job, int(qid), "chunk", i, d, 0, "") for i, d in enumerate(order_docs[:args.nc])]

    out = pd.DataFrame(rows, columns=["job", "qid", "kind", "pos", "doc_id", "size", "src"])
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out / "lists.parquet", index=False)
    print(f"{args.out / 'lists.parquet'}: {len(out):,} rows, {out['qid'].nunique()} queries, "
          f"promoted/query mean {sum(n_prom) / max(len(n_prom), 1):.1f} max {max(n_prom or [0])}")


if __name__ == "__main__":
    main()
