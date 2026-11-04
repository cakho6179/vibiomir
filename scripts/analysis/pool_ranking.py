"""Write the full m3rr ordering of the pool as a (qid, doc_id, rank) parquet, for tail sampling.

    python scripts/analysis/pool_ranking.py --method m3rr --out runs/reports/pool_m3rr.parquet

decide_lists.py only emits per-query lists (the materialize input). build_pairs.py needs the whole pool
with ranks so a tail band (e.g. 150-600) can be sampled -- the band where the ~20 gold docs we miss in the
top-197 must sit, since pB (1200 docs/query) reaches dR 0.81 while r8 (197) reaches 0.57.
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

CUT_DEFAULT = "vi800,zh200,en200"  # the 1,200-doc pool decide_lists uses by default


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="m3rr")
    ap.add_argument("--pool", default=CUT_DEFAULT)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/reports/pool_m3rr.parquet")
    args = ap.parse_args()
    cut = {s[:2]: int(s[2:]) for s in args.pool.split(",")}
    pool = load_pool(cut)
    ranked = order(pool, args.method).copy()
    ranked["rank"] = ranked.groupby("qid").cumcount()
    out = ranked[["qid", "doc_id", "rank"]].astype({"qid": int, "doc_id": int, "rank": int})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"{args.out}: {len(out):,} pairs, {out['qid'].nunique()} queries, "
          f"max rank {out['rank'].max()}, per-query {len(out) / out['qid'].nunique():.0f}")


if __name__ == "__main__":
    main()
