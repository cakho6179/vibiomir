"""Merge the doc2query shards and attach the queries, ready for stages/dq_embed.py.

    python scripts/analysis/dq_prepare.py [--out kaggle/outputs/dq-all] [--variant orig]

Reads every kaggle/outputs/dq-g*/questions.parquet (one per generation shard, doc_id split by --part i/4)
and data/query_variants_v3/query_variants.parquet, then writes into --out:
  questions.parquet   doc_id, qid, pid, idx, question   (deduplicated on doc_id/idx)
  query.parquet       id, query                          (the `orig` Vietnamese query of every qid)
  stats.json          docs covered, questions, questions/doc, parse rate, per-shard contribution

The query table is the plain original query on purpose: the doc2query channel asks "would a patient
phrase this the way the passage's own generated question is phrased", which is only meaningful against
the query as written, not against an LLM-rewritten variant.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "kaggle/outputs/dq-all")
    ap.add_argument("--variant", default="orig")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    parts = sorted(glob.glob(str(ROOT / "kaggle/outputs/dq-g*/questions.parquet")))
    if not parts:
        raise SystemExit("no shard outputs found (kaggle/outputs/dq-g*/questions.parquet)")
    frames, per_part = [], {}
    for p in parts:
        df = pd.read_parquet(p)
        per_part[Path(p).parent.name] = {"rows": int(len(df)), "docs": int(df["doc_id"].nunique())}
        frames.append(df)
        print(f"{Path(p).parent.name:16s} {len(df):7,d} questions, {df['doc_id'].nunique():6,d} docs")
    q = pd.concat(frames, ignore_index=True).drop_duplicates(["doc_id", "idx"])
    q["idx"] = q["idx"].astype("int16")
    q = q.sort_values(["doc_id", "idx"]).reset_index(drop=True)
    q.to_parquet(args.out / "questions.parquet", index=False)

    qv = pd.read_parquet(ROOT / "data/query_variants_v3/query_variants.parquet")
    orig = qv[qv["variant"] == args.variant][["qid", "text"]].rename(columns={"qid": "id", "text": "query"})
    orig = orig.drop_duplicates("id").sort_values("id").reset_index(drop=True)
    orig.to_parquet(args.out / "query.parquet", index=False)

    docs = q["doc_id"].nunique()
    head = pd.read_parquet(ROOT / "kaggle/datasets/dq_head/head_docs.parquet", columns=["doc_id"])
    want = head["doc_id"].nunique()
    stats = {"shards": per_part, "questions": int(len(q)), "docs": int(docs), "head_docs": int(want),
             "coverage": round(docs / max(want, 1), 4),
             "questions_per_doc": round(len(q) / max(docs, 1), 2),
             "median_chars": int(q["question"].str.len().median()) if len(q) else 0,
             "queries": int(len(orig))}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(f"-> {args.out}/questions.parquet + query.parquet")


if __name__ == "__main__":
    main()
