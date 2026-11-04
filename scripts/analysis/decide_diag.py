"""Diagnose the AnyJev questions already scored: does a question separate rank strata?

    python scripts/analysis/decide_diag.py [--root kaggle/outputs/vibiomir-decide-q3-4b-v2]

No qrels exist for this task, so "separates" is measured against the m3rr rank of the pair, which is the
only ordering we have: a question that is worth asking should fall monotonically from rank 0-5 to
rank 150-200, and should have a spread wide enough to reorder the list. A saturated question (p_yes ~ 1
everywhere) cannot reorder anything no matter how monotone it is.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
BANDS = [(0, 5), (5, 20), (20, 50), (50, 100), (100, 150), (150, 200)]


def load(root: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for p in sorted(root.glob("scores_0.parquet")) + sorted(root.glob("*/scores_0.parquet")):
        name = p.parent.name if p.parent != root else root.name.split("-")[-1]
        out[name] = pd.read_parquet(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT / "kaggle/outputs/vibiomir-decide-q3-4b-v2")
    ap.add_argument("--col", default="score")
    args = ap.parse_args()
    frames = load(args.root)
    if not frames:
        raise SystemExit(f"no scores under {args.root}")
    for name, d in frames.items():
        col = args.col if args.col in d.columns else "score"
        print(f"\n== {name}  ({len(d):,} pairs, column {col})")
        if "p_yes" in d.columns:
            print(f"   p_yes  mean {d['p_yes'].mean():.4f}  median {d['p_yes'].median():.4f}  "
                  f"share>0.99 {(d['p_yes'] > 0.99).mean():.3f}  share>0.5 {(d['p_yes'] > 0.5).mean():.3f}")
        rows = []
        for lo, hi in BANDS:
            s = d[(d["rank"] >= lo) & (d["rank"] < hi)][col]
            if len(s) == 0:
                continue
            rows.append({"rank": f"{lo}-{hi}", "n": len(s), "mean": s.mean(), "median": s.median(),
                         "p10": s.quantile(0.10), "p90": s.quantile(0.90)})
        t = pd.DataFrame(rows)
        print(t.round(4).to_string(index=False))
        top = d[d["rank"] < 20][col].mean()
        bot = d[d["rank"] >= 150][col].mean()
        spread = d[col].quantile(0.9) - d[col].quantile(0.1)
        print(f"   top20 mean {top:.4f} | tail mean {bot:.4f} | drop {(top - bot) / max(abs(top), 1e-9):.1%} | "
              f"p10-p90 spread {spread:.4f}")


if __name__ == "__main__":
    main()
