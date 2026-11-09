"""Does a locally rebuilt ordering reproduce a submitted doc list?

    python scripts/analysis/check_order.py submissions/r8_m3rr197.zip --ranking runs/reports/pool_m3rr.parquet

A submission's relevant_docs is a prefix of whatever ordering built it. Comparing set overlap per query
tells us whether a locally recomputed ranking is the same object, which has to be true before any tail
sample or promoted list can be trusted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT))
from r2ai2026.core.submission import read_submission_zip  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zip", type=Path)
    ap.add_argument("--ranking", type=Path, default=ROOT / "runs/reports/pool_m3rr.parquet")
    ap.add_argument("--n", type=int, default=0, help="depth to compare (default: whatever the zip has)")
    args = ap.parse_args()
    recs = read_submission_zip(args.zip)
    rk = pd.read_parquet(args.ranking)
    rk = rk.sort_values(["qid", "rank"])
    overlaps, sizes = [], []
    for r in recs:
        docs = [str(d if not isinstance(d, dict) else d.get("doc_id")) for d in r["relevant_docs"]]
        n = args.n or len(docs)
        mine = rk[rk["qid"] == int(r["id"])].head(n)["doc_id"].astype(str).tolist()
        overlaps.append(len(set(docs) & set(mine)) / max(len(set(mine)), 1))
        sizes.append(len(docs))
    print(f"{args.zip.name}: {len(recs)} queries, docs/query {sum(sizes) / len(sizes):.1f}, "
          f"mean overlap with the rebuilt top-{args.n or 'all'} {sum(overlaps) / len(overlaps):.3f}, "
          f"min {min(overlaps):.3f}")


if __name__ == "__main__":
    main()
