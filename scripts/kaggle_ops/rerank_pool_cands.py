"""Candidate pairs for reranking the 3-language pool, plus the query file, as one Kaggle dataset folder.

    python scripts/kaggle_ops/rerank_pool_cands.py --sets vi1500,zh500,en500 --out kaggle/datasets/rerank_pool

Reads the pool sets of kaggle/datasets/probe_inputs/additions.parquet (lang_pool_sets.py) and writes
cands.parquet (qid, doc_id, set, rank), deduplicated on (qid, doc_id) keeping the first listed set, and a copy of
query.parquet. rerank.py --cands cands.parquet reads both from the dataset mount.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", required=True, help="comma-separated set names of additions.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--additions", type=Path, default=ROOT / "kaggle/datasets/probe_inputs/additions.parquet",
                    help="pool sets written by lang_pool_sets.py")
    ap.add_argument("--query", type=Path, default=ROOT / "kaggle/datasets/query_only/query.parquet",
                    help="query.parquet copied next to cands.parquet")
    args = ap.parse_args()
    sets = args.sets.split(",")
    add = pd.read_parquet(args.additions)
    add = add[add["set"].isin(sets)]
    add["set"] = pd.Categorical(add["set"], categories=sets)
    cands = add.sort_values(["qid", "set", "rank"]).drop_duplicates(["qid", "doc_id"])
    cands["set"] = cands["set"].astype(str)
    args.out.mkdir(parents=True, exist_ok=True)
    cands[["qid", "doc_id", "set", "rank"]].to_parquet(args.out / "cands.parquet", index=False)
    shutil.copy(args.query, args.out / "query.parquet")
    print(f"{len(cands):,} pairs, {cands['qid'].nunique()} queries;",
          cands.groupby("set").size().to_dict())


if __name__ == "__main__":
    main()
