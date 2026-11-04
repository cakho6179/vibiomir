"""Build the X candidates: a clean lexical head in front of the m3rr list, chunks held constant.

Control design: every job keeps **197 docs** and **the exact 148 chunk docs of r8** (same size, same seed), so a
change in FINAL is attributable to the document set alone. `X*c` jobs additionally add one chunk for each head
document, which is free: the two channels are scored independently (proved by `chunkmix40`, conv 0.88).

    python scripts/analysis/build_lex_candidates.py --lex runs/reports/lex_head2.parquet --out temp_submission/X
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())


def r8_chunks(path: Path) -> dict[int, list[int]]:
    z = zipfile.ZipFile(path)
    sub = json.loads(z.read([n for n in z.namelist() if n.endswith(".json")][0]).decode("utf-8"))
    return {int(e["id"]): [int(c["doc_id"]) for c in e["relevant_chunks"]] for e in sub}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lex", type=Path, default=ROOT / "runs/reports/lex_head2.parquet")
    ap.add_argument("--pool", type=Path, default=ROOT / "runs/reports/pool_m3rr.parquet")
    ap.add_argument("--r8", type=Path, default=ROOT / "submissions/r8_m3rr197.zip")
    ap.add_argument("--out", type=Path, default=ROOT / "temp_submission/X")
    ap.add_argument("--n-docs", type=int, default=197)
    ap.add_argument("--jobs", default="X40:40:0,X80:80:0,X20c:40:20",
                    help="job:KHEAD:NEXTRA -- NEXTRA extra chunks for the first NEXTRA head docs")
    args = ap.parse_args()

    lex = pd.read_parquet(args.lex)
    pool = pd.read_parquet(args.pool).rename(columns={"rank": "m3rr"})
    chunk0 = r8_chunks(args.r8)
    print(f"lex {len(lex):,} rows | pool {len(pool):,} | r8 chunk docs "
          f"{sum(len(v) for v in chunk0.values()):,}")

    rows = []
    for spec in args.jobs.split(","):
        job, k, extra = spec.split(":")
        k, extra = int(k), int(extra)
        ndoc = nchunk = 0
        for qid, g in pool.groupby("qid"):
            qid = int(qid)
            head = list(lex[lex.qid == qid].sort_values("rank").head(k)["doc_id"].astype(int))
            tail = [int(d) for d in g.sort_values("m3rr")["doc_id"] if d not in set(head)]
            docs = (head + tail)[: args.n_docs]
            for pos, d in enumerate(docs):
                rows.append((job, qid, "doc", pos, d, 0, ""))
            ndoc += len(docs)
            chunks = list(chunk0[qid])
            if extra:
                chunks = chunks + [d for d in head if d not in set(chunks)]
            for pos, d in enumerate(chunks):
                rows.append((job, qid, "chunk", pos, int(d), 4500, ""))
            nchunk += len(chunks)
        print(f"{job}: {ndoc:,} doc rows ({ndoc/1200:.1f}/query) | {nchunk:,} chunk rows ({nchunk/1200:.1f}/query)")

    df = pd.DataFrame(rows, columns=["job", "qid", "kind", "pos", "doc_id", "size", "src"])
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out / "lists.parquet", index=False)
    print(f"wrote {args.out/'lists.parquet'} ({len(df):,} rows, jobs {sorted(df.job.unique())})")


if __name__ == "__main__":
    main()
