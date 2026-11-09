"""Build the candidate list for the LLM relevance judge (top-N of the submission #7230 order).

    python scripts/kaggle_ops/llm_judge_inputs.py [--top 300] [--out kaggle/datasets/judge_cands]

The base order is exactly the one that produced submission #7230 (FINAL 0.2983):
decide_lists.order(load_pool(vi800/zh200/en200), "m3rr") over the rr3 reranker parts. Rebuilding it
reproduces 197.0/197 documents per query of that zip (checked), so the judge is compared against the
real system and only the judge changes.

Writes judge_cands.parquet: qid, doc_id, bp, rank, rr, m3all
  bp    index of the passage bge-reranker-v2-m3 liked best (rr4 best_passage), so the judge reads the
        same evidence the cross-encoder already bets on - the only thing that changes is the judge
  rank  position in the #7230 order
Writes queries.parquet: qid, query (original Vietnamese query; the kernel has no data/ mount).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from decide_lists import load_pool, order  # noqa: E402

CUT = {"vi": 800, "zh": 200, "en": 200}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--out", type=Path, default=ROOT / "kaggle/datasets/judge_cands")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pool = order(load_pool(CUT), "m3rr")
    pool["rank"] = pool.groupby("qid").cumcount()
    pool = pool[pool["rank"] < args.top]

    bp = pd.concat([pd.read_parquet(p, columns=["qid", "doc_id", "best_passage"])
                    for p in sorted(glob.glob(str(ROOT / "kaggle/outputs/rr4-p*/rerank.parquet")))],
                   ignore_index=True)
    pool = pool.merge(bp, on=["qid", "doc_id"], how="left")
    out = pool[["qid", "doc_id", "best_passage", "rank", "score", "m3all"]].rename(
        columns={"best_passage": "bp", "score": "rr"})
    print(f"rr4 best_passage missing for {out['bp'].isna().mean():.2%} of pairs")
    out["bp"] = out["bp"].fillna(0).astype("int16")
    out["rank"] = out["rank"].astype("int16")
    out = out.sort_values(["qid", "rank"]).reset_index(drop=True)
    out.to_parquet(args.out / "judge_cands.parquet", index=False)

    qv = pd.read_parquet(ROOT / "data/query_variants_v3/query_variants.parquet")
    orig = qv[qv["variant"] == "orig"][["qid", "text"]].rename(columns={"text": "query"}).drop_duplicates("qid")
    orig.sort_values("qid").to_parquet(args.out / "queries.parquet", index=False)

    print(f"{len(out):,} pairs, {out['qid'].nunique()} queries, {out['doc_id'].nunique():,} docs, "
          f"{len(out) / out['qid'].nunique():.0f} per query")
    print(out.head(3).to_string())
    print(f"-> {args.out}/judge_cands.parquet + queries.parquet")


if __name__ == "__main__":
    main()
