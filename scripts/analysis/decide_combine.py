"""Does combining the questions beat the single best question?

    python scripts/analysis/decide_combine.py --root kaggle/outputs/vibiomir-decide4

The idea under test ("ghep ca 2 hoac 3 cau hoi, dung any-yes") only pays if the questions carry
*complementary* signal. `helps` and `graded` are saturated (97.6 % / median 4.0), so an any-yes fusion is
dominated by the lenient question and throws the discriminative one away. This measures, on the same pairs,
how well each question and each fusion agrees with the m3rr rank (the only ordering we have), by Spearman.

Spearman against rank is a *proxy*, not accuracy: a question can agree with the current ranker and still be
better. But a fusion that agrees LESS than its best input cannot be an improvement, and that is what this
rules out.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())


def z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() or 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT / "kaggle/outputs/vibiomir-decide4")
    args = ap.parse_args()
    frames = {}
    for q in ("helps", "direct", "graded"):
        p = args.root / q / "scores_0.parquet"
        if p.exists():
            frames[q] = pd.read_parquet(p)
    if not frames:
        raise SystemExit(f"no scores under {args.root}")
    base = frames["direct"][["qid", "doc_id", "rank", "score"]].rename(columns={"score": "direct"})
    for q, d in frames.items():
        if q != "direct":
            base = base.merge(d[["qid", "doc_id", "score"]].rename(columns={"score": q}), on=["qid", "doc_id"])
    base = base.dropna(subset=["rank"])
    print(f"{len(base):,} pairs, {base['qid'].nunique()} queries\n")

    cands = {
        "helps": base["helps"],
        "direct": base["direct"],
        "graded": base["graded"],
        "direct+graded (mean z)": z(base["direct"]) + z(base["graded"]),
        "direct+helps (mean z)": z(base["direct"]) + z(base["helps"]),
        "all three (mean z)": z(base["direct"]) + z(base["helps"]) + z(base["graded"]),
        "any-yes (max p_yes)": base[["helps", "direct"]].max(axis=1),
        "direct x graded": z(base["direct"]) * z(base["graded"]),
    }
    print(f"{'score':26} {'spearman vs rank':>17} {'AUC (top-50 vs rest)':>21}")
    for name, s in cands.items():
        rho = spearmanr(s, base["rank"]).statistic
        y = (base["rank"] < 50).astype(int)
        # rank-based AUC without sklearn: mean over pairs, computed on a subsample of the negatives
        pos = s[y == 1]
        neg = s[y == 0]
        n = min(len(neg), 40000)
        neg = neg.sample(n, random_state=0) if len(neg) > n else neg
        auc = ((pos.values[:, None] > neg.values[None, :]).mean()
               + 0.5 * (pos.values[:, None] == neg.values[None, :]).mean())
        print(f"{name:26} {rho:>17.4f} {auc:>21.4f}")


if __name__ == "__main__":
    main()
