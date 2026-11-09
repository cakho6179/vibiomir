"""How much does `direct` actually fall, and how much does the truth fall with it?

The saturation claim has to be a number, not an adjective. This compares the two quantities that matter on the
same axis -- m3rr rank:

    signal  : median p_yes and median logodds per rank band
    truth   : gold density per rank band, measured from the submissions themselves
              (dP x N at depth 88 / 197 / 1200 gives TP, so the density between two depths is arithmetic)

and then asks the only question that decides whether the score can be used at all: **can it order two
documents from different bands?** That is a pairwise AUC -- the probability that a document drawn from the
upper band scores above one drawn from the lower band. 0.5 means the score is blind to the band; 1.0 means it
separates perfectly. A monotone-looking median can still hide an AUC of 0.55.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())

# gold documents per query at each depth, from the scored submissions (dP x N).
# r8: dP 0.1792 x 197 = 35.3 ; w12T2_win6000 (88 docs): dP 0.2385 x 88 = 21.0 ; pB (1200 docs): 0.0463 x 1200 = 55.6
DEPTH = {0: 0.0, 88: 21.0, 197: 35.3, 1200: 55.6}
N_QUERIES = 1200


def density(a: int, b: int) -> float:
    """Gold docs per candidate in the open rank interval (a, b]."""
    ta, tb = DEPTH[a], DEPTH[b]
    return (tb - ta) / ((b - a) * 1.0)


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """P(x > y) + 0.5 P(x == y), by rank (Mann-Whitney)."""
    allv = np.concatenate([x, y])
    r = pd.Series(allv).rank().to_numpy()
    rx = r[: len(x)].sum()
    return (rx - len(x) * (len(x) + 1) / 2) / (len(x) * len(y))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head", type=Path, default=ROOT / "kaggle/outputs/vibiomir-decide4/direct/scores_0.parquet")
    ap.add_argument("--tail", type=Path, default=ROOT / "kaggle/outputs/vibiomir-decide5/scores_0.parquet")
    args = ap.parse_args()

    d = pd.concat([pd.read_parquet(args.head), pd.read_parquet(args.tail)], ignore_index=True)
    d = d.drop_duplicates(["qid", "doc_id"])
    bands = [(0, 20), (20, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300),
             (300, 400), (400, 500), (500, 600)]
    d["band"] = pd.cut(d["rank"], [b[0] for b in bands] + [600], right=False, labels=[f"{a}-{b}" for a, b in bands])

    rows = []
    for lab, g in d.groupby("band", observed=True):
        rows.append({"band": str(lab), "n": len(g),
                     "p_yes med": g["p_yes"].median(), "logodds med": g["logodds"].median(),
                     "% p>0.99": 100 * (g["p_yes"] > 0.99).mean(),
                     "% p>0.9": 100 * (g["p_yes"] > 0.9).mean(),
                     "% p>0.5": 100 * (g["p_yes"] > 0.5).mean()})
    t = pd.DataFrame(rows)
    print("== signal by m3rr rank band ==")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}" if v < 1 else f"{v:.2f}"))

    first, last = t.iloc[0], t.iloc[-1]
    print("\n== decay from band 0-20 to band 500-600 ==")
    for c in ["p_yes med", "logodds med", "% p>0.99", "% p>0.5"]:
        print(f"  {c:14s} {first[c]:10.4f} -> {last[c]:10.4f}   ({100 * (last[c] / first[c] - 1):+.1f} %)")

    print("\n== gold density by rank band (from the submissions, not the model) ==")
    cuts = [(0, 88), (88, 197), (197, 1200)]
    for a, b in cuts:
        print(f"  {a:>4}-{b:<4}: {100 * density(a, b):5.2f} % of candidates are gold")
    print(f"  density drop 0-88 -> 197-1200: {100 * (density(197, 1200) / density(0, 88) - 1):+.1f} %")

    print("\n== can the score order two documents from different bands? (pairwise AUC) ==")
    def sc(a, b):
        return (d.loc[(d["rank"] >= a) & (d["rank"] < b), "logodds"].to_numpy(),
                d.loc[(d["rank"] >= b) & (d["rank"] < min(b + (b - a), 600)), "logodds"].to_numpy())
    pairs = [(0, 20, 20, 50), (20, 50, 50, 100), (50, 100, 100, 150), (100, 150, 150, 200),
             (150, 200, 200, 250), (200, 250, 250, 300), (300, 400, 400, 500), (400, 500, 500, 600)]
    for a, b, c, e in pairs:
        x = d.loc[(d["rank"] >= a) & (d["rank"] < b), "logodds"].to_numpy()
        y = d.loc[(d["rank"] >= c) & (d["rank"] < e), "logodds"].to_numpy()
        if len(x) and len(y):
            print(f"  {a:>3}-{b:<3} vs {c:>3}-{e:<3}: AUC {auc(x, y):.3f}")
    x = d.loc[(d["rank"] >= 0) & (d["rank"] < 20), "logodds"].to_numpy()
    y = d.loc[(d["rank"] >= 500) & (d["rank"] < 600), "logodds"].to_numpy()
    print(f"    0- 20 vs 500-600: AUC {auc(x, y):.3f}   <- the whole range it has to work over")
    x = d.loc[(d["rank"] >= 150) & (d["rank"] < 200), "logodds"].to_numpy()
    y = d.loc[(d["rank"] >= 500) & (d["rank"] < 600), "logodds"].to_numpy()
    print(f"  150-200 vs 500-600: AUC {auc(x, y):.3f}   <- the range the tail promotion needs")

    print("\n== implied gold probability vs actual, if the head calibration held ==")
    # fitted on the head: P(gold) = sigmoid(a + b*logodds) through (logodds 10.41, p 0.30) and (1.94, p 0.13)
    b = (np.log(0.30 / 0.70) - np.log(0.13 / 0.87)) / (10.41 - 1.94)
    a = np.log(0.30 / 0.70) - b * 10.41
    for lo, hi, dens in [(0, 88, density(0, 88)), (88, 197, density(88, 197)), (150, 600, density(197, 1200))]:
        m = d.loc[(d["rank"] >= lo) & (d["rank"] < hi), "logodds"].median()
        pred = 1 / (1 + np.exp(-(a + b * m)))
        print(f"  rank {lo:>3}-{hi:<4}: median logodds {m:5.2f} -> model says {100 * pred:5.1f} % gold, "
              f"truth {100 * dens:5.2f} %  (x{pred / dens:.1f} over-estimate)")


if __name__ == "__main__":
    main()
