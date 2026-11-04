"""Offline diagnosis of the doc2query channel against the reranker order (no submission spent).

    python scripts/analysis/dq_diag.py [--top 197]

Prints, per ordering method, the language mix and the overlap with the #7230 order (m3rr, FINAL 0.2983), plus how the
promoted docs look: reranker rank of the docs that dq lifts into the top-N and dq rank of the docs it drops out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decide_lists import load_pool, order  # noqa: E402

BASE = "m3rr"


def top(pool: pd.DataFrame, method: str, n: int) -> pd.DataFrame:
    o = order(pool, method)
    o["pos"] = o.groupby("qid").cumcount()
    return o[o["pos"] < n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=197)
    args = ap.parse_args()
    pool = load_pool({"vi": 800, "zh": 200, "en": 200})
    n = args.top
    ref = top(pool, BASE, n)[["qid", "doc_id", "rr_rank", "score", "lang"]]
    nq = pool["qid"].nunique()
    print(f"\nbaseline {BASE} top-{n}: mix {(ref.groupby('lang').size() / nq).round(1).to_dict()}")
    for m in ("dq", "dqrr", "dqrw2", "dqrw3", "d3", "d3w2", "d3w3", "d3w4"):
        try:
            cur = top(pool, m, n)[["qid", "doc_id", "rr_rank", "score", "lang", "dq_score", "dq_rank"]]
        except KeyError as e:
            print(f"{m}: skipped ({e})")
            continue
        share = cur.merge(ref, on=["qid", "doc_id"], how="inner").shape[0] / (n * nq)
        gained = cur.merge(ref, on=["qid", "doc_id"], how="left", indicator=True)
        gained = gained[gained["_merge"] == "left_only"]
        dropped = ref.merge(cur[["qid", "doc_id"]], on=["qid", "doc_id"], how="left", indicator=True)
        dropped = dropped[dropped["_merge"] == "left_only"]
        g_rr = gained["rr_rank_x"].median() if len(gained) else float("nan")
        print(f"{m:6s} top-{n}: mix {(cur.groupby('lang').size() / nq).round(1).to_dict()} "
              f"shared {share:.1%} | +{len(gained) / nq:.1f} new/q (rr_rank med {g_rr:.0f}) "
              f"| -{len(dropped) / nq:.1f} dropped/q (rr_rank med {dropped['rr_rank'].median():.0f})")
    print("\ndq coverage inside the reranker top-197 (documents with generated questions):")
    refd = ref.merge(pool[["qid", "doc_id", "dq_score", "dq_rank"]], on=["qid", "doc_id"], how="left")
    print(f"  covered {refd['dq_score'].gt(-1).mean():.1%}, mean dq_score {refd.loc[refd['dq_score'] > -1, 'dq_score'].mean():.3f}")
    print(f"  pool overall covered {pool['dq_score'].gt(-1).mean():.1%}")


if __name__ == "__main__":
    main()
