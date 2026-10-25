"""Union of every retrieval source that was part of a SCORED submission, merged per (qid, doc_id).

    python experiments/2026-10-24_union_probe/union_sources.py

Design (docs/DATA_PIPELINE.md, L1 brainstorm): one row per (qid, doc_id) with each source's rank, the number of
sources and families that found the doc; order = family-aware RRF (inside a family only the best rank votes, so
four BM25 variants count once). Sources never in a scored submission (b1, b2, b4, b5, c1, s1, s2, m3c, BM25 top-2000)
are left out on purpose.

Writes kaggle/rankings_ds/union_scored_v1.parquet (qid, doc_id, score, rank, n_sources, n_families, r_<source>...)
and prints per-query size / overlap statistics.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
RD = ROOT / "kaggle" / "rankings_ds"
OUT = ROOT / "kaggle" / "outputs"
RRF_K = 60

# source -> (family, files); every source appears in a submission with a leaderboard score (EXPERIMENTS.md table A)
SOURCES = {
    "e1v2": ("bm25", [RD / "bm25_e1v2.parquet"]),
    "e2v2": ("bm25", [OUT / "vibiomir-bm25-e2v2" / "ranking.parquet"]),
    "e3v2": ("bm25", [RD / "bm25_e3v2.parquet"]),
    "b3": ("bm25", [RD / "bm25_b3.parquet"]),
    "m3": ("m3", [RD / f"m3_{s}.parquet" for s in ("s01", "s23", "s45", "s67", "d1")]),
    "m3L": ("m3", [RD / f"m3L_{s}.parquet" for s in ("a", "b", "c")]),
    "q3e": ("qwen", [RD / f"q3e_{s}.parquet" for s in ("s01", "s23", "s45", "s67", "d1")]),
    "e1en": ("xl", [OUT / "vibiomir-bm25-e1en" / "ranking.parquet"]),
    "e1zh": ("xl", [OUT / "vibiomir-bm25-e1zh" / "ranking.parquet"]),
    "r1": ("rerank", [RD / "rerank_r1.parquet"]),
}
FAMILY_WEIGHT = {"bm25": 1.0, "m3": 2.0, "qwen": 1.0, "xl": 0.5, "rerank": 2.0}


def load(files: list[Path]) -> pd.DataFrame:
    """One ranking from one or more shard files: best score per (qid, doc_id), rank 1.. by score."""
    df = pd.concat([pd.read_parquet(f, columns=["qid", "doc_id", "score"]) for f in files], ignore_index=True)
    df = df.sort_values(["qid", "score"], ascending=[True, False]).drop_duplicates(["qid", "doc_id"])
    df["rank"] = df.groupby("qid").cumcount() + 1
    return df[["qid", "doc_id", "rank"]]


def main() -> None:
    merged = None
    for name, (_, files) in SOURCES.items():
        r = load(files).rename(columns={"rank": f"r_{name}"})
        merged = r if merged is None else merged.merge(r, on=["qid", "doc_id"], how="outer")
        print(f"{name:5s} {len(r) / r.qid.nunique():7.1f} docs/q", flush=True)
    rank_cols = [f"r_{n}" for n in SOURCES]
    retr = [f"r_{n}" for n, (fam, _) in SOURCES.items() if fam != "rerank"]
    merged["n_sources"] = merged[retr].notna().sum(axis=1)
    score = 0.0
    fam_found = 0
    for fam in dict.fromkeys(f for f, _ in SOURCES.values()):
        cols = [f"r_{n}" for n, (f, _) in SOURCES.items() if f == fam]
        best = merged[cols].min(axis=1)
        score = score + (FAMILY_WEIGHT[fam] / (RRF_K + best)).fillna(0.0)
        if fam != "rerank":
            fam_found = fam_found + best.notna().astype(int)
    merged["score"] = score
    merged["n_families"] = fam_found
    merged = merged.sort_values(["qid", "score"], ascending=[True, False])
    merged["rank"] = merged.groupby("qid").cumcount() + 1
    for c in rank_cols + ["n_sources", "n_families", "rank"]:
        merged[c] = merged[c].astype("Int32")
    merged.to_parquet(RD / "union_scored_v1.parquet", index=False)

    per_q = merged.groupby("qid").size()
    vi = merged[merged["doc_id"] <= 766_441].groupby("qid").size()
    stats = {"docs_per_query_median": int(per_q.median()), "docs_per_query_mean": round(per_q.mean(), 1),
             "docs_per_query_max": int(per_q.max()), "vi_docs_per_query_mean": round(vi.mean(), 1),
             "by_n_families": {str(k): round(v / per_q.size, 1) for k, v in merged.groupby("n_families").size().items()},
             "only_one_source": round((merged["n_sources"] == 1).sum() / per_q.size, 1)}
    print(json.dumps(stats, indent=1, default=str))
    (Path(__file__).parent / "union_stats.json").write_text(json.dumps(stats, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
