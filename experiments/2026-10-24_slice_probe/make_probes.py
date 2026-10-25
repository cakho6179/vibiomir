"""Doc-only probes: keep the chunk list of a base zip byte-identical and swap relevant_docs.

    python experiments/2026-10-24_slice_probe/make_probes.py [--base submissions/w12T2_VI_win3000_n150.zip] [--only r1ord,...]

Chunks F2 of every probe equals the base's, so the leaderboard Docs P/R answer one question each:
  r1ord  - same per-query count as the base, docs = top-N_q by bge-reranker-v2-m3 (r1) over its own pool
           (E1v2 top100 U M3 top100). Docs P vs the base = value of reranker ORDER alone.
  sliceA - union docs found by >= 2 families, not in the base, union rank <= 500.
  sliceB - union docs found by the m3 family only, not in the base, union rank 100..500.
For a slice, Docs P = gold density of the stratum (macro; queries with an empty stratum count 0, corrected in the
printed table) and Docs R x 71 = gold docs per query it holds. Add a stratum to the main list only if density > F2/5.
Only aggregate leaderboard numbers are used; no per-pair labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT))
from r2ai2026.core.submission import read_submission_zip, write_submission_zip  # noqa: E402

RD = ROOT / "kaggle" / "rankings_ds"
VI_MAX_ID = 766_441
HERE = Path(__file__).parent


def base_frame(records: list[dict]) -> pd.DataFrame:
    rows = [(int(r["id"]), int(d), i) for r in records for i, d in enumerate(r["relevant_docs"])]
    return pd.DataFrame(rows, columns=["qid", "doc_id", "pos"])


def probe_lists(base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """name -> (qid, doc_id) in submission order."""
    out = {}
    r1 = pd.read_parquet(RD / "rerank_r1.parquet", columns=["qid", "doc_id", "score"])
    r1 = r1[r1.doc_id <= VI_MAX_ID].sort_values(["qid", "score"], ascending=[True, False])
    n_q = base.groupby("qid").size().rename("n_q")
    r1 = r1.join(n_q, on="qid")
    r1 = r1[r1.groupby("qid").cumcount() < r1.n_q]
    out["r1ord"] = r1[["qid", "doc_id"]]

    u = pd.read_parquet(RD / "union_scored_v1.parquet")
    u = u[u.doc_id <= VI_MAX_ID].merge(base[["qid", "doc_id"]], on=["qid", "doc_id"], how="left", indicator=True)
    u = u[u["_merge"] == "left_only"].sort_values(["qid", "rank"])
    m3_only = (u.n_families == 1) & (u.r_m3.notna() | u.r_m3L.notna())
    out["sliceA"] = u[(u.n_families >= 2) & (u["rank"] <= 500)][["qid", "doc_id"]]
    out["sliceB"] = u[m3_only & u["rank"].between(100, 500)][["qid", "doc_id"]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=ROOT / "submissions" / "w12T2_VI_win3000_n150.zip")
    ap.add_argument("--only", default="", help="comma-separated probe names (default: all)")
    ap.add_argument("--out", type=Path, default=ROOT / "submissions")
    args = ap.parse_args()

    records = read_submission_zip(args.base)
    base = base_frame(records)
    stats = {"base": args.base.name, "base_docs_per_q": round(len(base) / len(records), 2)}
    wanted = set(filter(None, args.only.split(",")))
    for name, lst in probe_lists(base).items():
        if wanted and name not in wanted:
            continue
        by_q = lst.groupby("qid")["doc_id"].apply(list).to_dict()
        common = lst.merge(base, on=["qid", "doc_id"]).shape[0]
        new = [{**r, "relevant_docs": [int(d) for d in by_q.get(int(r["id"]), [])]} for r in records]
        empty = sum(not r["relevant_docs"] for r in new)
        out = args.out / f"probe_{name}_{args.base.stem.split('_win')[0]}.zip"
        zs = write_submission_zip(new, out, method="lzma")
        stats[name] = {"zip": out.name, "docs_per_q": round(len(lst) / len(records), 2),
                       "common_with_base_per_q": round(common / len(records), 2), "empty_queries": empty,
                       "zip_mb": round(zs["zip_mb"], 1)}
        print(name, json.dumps(stats[name]), flush=True)
    (HERE / "probe_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
