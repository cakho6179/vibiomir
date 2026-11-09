"""Calibrated probability for an AnyJev judge, from aggregate scores only (no per-pair labels).

    python scripts/analysis/llm_calib.py probe --scores "kaggle/outputs/decide35-acc*/scores_*.parquet" --job p9b_top
    python scripts/analysis/llm_calib.py calib --scores ... --job p9b_top --obs-p 0.123

The A/B sample (kaggle/datasets/tail_ab, 12,000 pool pairs at list ranks >= 197) was submitted as two lists split by
Qwen3-4B `direct`: ab_yes #7537 (docs P .0472) and ab_no #7548 (P .0192). Against the p(gold) model they hold
m_yes = obs / expected and m_no times the expected gold, so the gold of the whole sample is known.
probe: the top set of a new judge (logodds > 0, or the top --frac when that is smaller) as a doc list to submit.
calib: its observed P gives m_top; the rest of the sample holds the remainder, m_rest; the calibrated probability of a
pool doc is p_cal = p * m(bin), bin = the judge's answer at the same logodds cut. Likelihood ratio = m_top / m_rest.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AB = {"ab_yes": 0.0472, "ab_no": 0.0192}  # observed macro docs P of #7537 / #7548


def expected_p(lst: pd.DataFrame, cands: pd.DataFrame) -> float:
    """Expected macro docs P of a list under the p(gold) model (queries without docs are 0, as the scorer)."""
    x = lst.merge(cands, on=["qid", "doc_id"], how="left").fillna({"p": 0.0})
    q = x.groupby("qid")["p"].agg(["sum", "size"])
    return float((q["sum"] / q["size"]).sum() / 1200)


def load(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = [f for f in sorted(glob.glob(args.scores)) if "_h" not in Path(f).stem]
    s = pd.concat([pd.read_parquet(f, columns=["qid", "doc_id", "logodds"]) for f in files]).drop_duplicates(["qid", "doc_id"])
    ab = pd.read_parquet(ROOT / "kaggle/datasets/tail_ab/docs.parquet")[["job", "qid", "doc_id"]]
    cands = pd.read_parquet(args.cands, columns=["qid", "doc_id", "p"])
    s = s.merge(ab[["qid", "doc_id"]].drop_duplicates(), on=["qid", "doc_id"])
    print(f"judge scores on the A/B sample: {len(s):,} of {ab[['qid', 'doc_id']].drop_duplicates().shape[0]:,}", flush=True)
    return s, ab, cands


def top_set(s: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, float]:
    cut = 0.0 if (s["logodds"] > 0).mean() >= frac else float(s["logodds"].quantile(1 - frac))
    return s[s["logodds"] > cut], cut


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["probe", "calib"])
    ap.add_argument("--scores", required=True, help="glob of decide.py scores parquet")
    ap.add_argument("--cands", type=Path, default=ROOT / "runs/reports/pgold_feats/base4/cands_p.parquet")
    ap.add_argument("--job", default="p9b_top")
    ap.add_argument("--frac", type=float, default=0.10, help="minimum share of the sample in the top set")
    ap.add_argument("--obs-p", type=float, help="calib: observed docs P of the submitted top set")
    ap.add_argument("--out", type=Path, default=ROOT / "kaggle/datasets/p9b_docs")
    args = ap.parse_args()
    s, ab, cands = load(args)
    top, cut = top_set(s, args.frac)
    exp_top = expected_p(top[["qid", "doc_id"]], cands)
    print(f"top set: logodds > {cut:.3f}: {len(top):,} docs ({len(top) / len(s):.1%}), {top['qid'].nunique()} queries, "
          f"expected P under p(gold) {exp_top:.4f}", flush=True)
    if args.step == "probe":
        args.out.mkdir(parents=True, exist_ok=True)
        lst = top.sort_values(["qid", "logodds"], ascending=[True, False])
        lst = lst.assign(job=args.job, rank=lst.groupby("qid").cumcount())[["job", "qid", "doc_id", "rank"]]
        lst.to_parquet(args.out / "docs.parquet", index=False)
        print(f"wrote {args.out / 'docs.parquet'}", flush=True)
        return
    # gold of the whole sample from the 4B split, in units of expected p: sum over pairs of p * m
    c = cands.set_index(["qid", "doc_id"])["p"]
    gold = 0.0
    for job, obs in AB.items():
        part = ab[ab["job"] == job][["qid", "doc_id"]]
        m = obs / expected_p(part, cands)
        gold += m * c.reindex(pd.MultiIndex.from_frame(part)).fillna(0).sum()
        print(f"{job}: m = {m:.3f}", flush=True)
    m_top = args.obs_p / exp_top
    p_top = c.reindex(pd.MultiIndex.from_frame(top[["qid", "doc_id"]])).fillna(0).sum()
    rest = s.drop(top.index)
    p_rest = c.reindex(pd.MultiIndex.from_frame(rest[["qid", "doc_id"]])).fillna(0).sum()
    m_rest = (gold - m_top * p_top) / p_rest
    out = {"cut_logodds": cut, "m_top": m_top, "m_rest": m_rest, "likelihood_ratio": m_top / max(m_rest, 1e-9),
           "share_top": len(top) / len(s), "gold_sample_exp": gold}
    print(json.dumps(out, indent=1))
    (ROOT / f"runs/reports/llm_calib_{args.job}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
