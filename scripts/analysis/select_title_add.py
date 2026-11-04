"""Add high-confidence title-only docs to an existing submission list (probe).

The production pool has high doc recall but its top-197 cut is fixed.  The title cross-encoder can find
relevant documents that the passage pool misses; this script keeps the existing doc/chunk list intact and
adds only title-only documents whose title score is positive, so the probe tests the expansion without
risking the current best list.  The number of additions varies per query; nothing is padded to a fixed N.

    python select_title_add.py --base-zip runs/mat_out/st1/r24.zip \
        --title kaggle/outputs/vibiomir-title-rerank/title_rerank.parquet \
        --out temp_submission/title_add --job title_add
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())


def load_base(path: Path) -> dict[int, tuple[list[int], list[int]]]:
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("submission.json"))
    return {int(x["id"]): ([int(d) for d in x["relevant_docs"]],
                           [int(c["doc_id"]) for c in x["relevant_chunks"]]) for x in data}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-zip", type=Path, required=True)
    ap.add_argument("--title", type=Path, required=True)
    ap.add_argument("--meta", type=Path, default=ROOT / "data/docstore/v2.1/docs_meta.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--job", default="title_add")
    ap.add_argument("--window", type=int, default=4500)
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="only title-only docs with a positive cross-encoder logit are added")
    ap.add_argument("--max-title-rank", type=int, default=30)
    ap.add_argument("--max-add-docs", type=int, default=20)
    ap.add_argument("--max-chunks", type=int, default=160)
    args = ap.parse_args()

    base = load_base(args.base_zip)
    title = pd.read_parquet(args.title)
    title = title[(title["title_rank"] < args.max_title_rank) & (title["title_score"] > args.min_score)]
    title = title.sort_values(["qid", "title_rank"])
    meta = pd.read_parquet(args.meta, columns=["id", "content_key"]).set_index("id")
    ckey = meta["content_key"].fillna("").to_dict()

    rows = []
    stats = []
    for qid, (base_docs, base_chunks) in base.items():
        seen = {ckey.get(d, str(d)) for d in base_docs}
        add = []
        for did in title.loc[title["qid"] == qid, "doc_id"].astype(int):
            k = ckey.get(did, str(did))
            if did in base_docs or k in seen:
                continue
            seen.add(k)
            add.append(did)
            if len(add) >= args.max_add_docs:
                break
        docs = base_docs + add
        chunks = list(base_chunks)
        for did in add:
            if did not in chunks and len(chunks) < args.max_chunks:
                chunks.append(did)
        rows.append(pd.DataFrame({"job": args.job, "qid": qid, "kind": "doc",
                                  "pos": np.arange(len(docs)), "doc_id": docs,
                                  "size": 0, "src": ""}))
        rows.append(pd.DataFrame({"job": args.job, "qid": qid, "kind": "chunk",
                                  "pos": np.arange(len(chunks)), "doc_id": chunks,
                                  "size": args.window, "src": ""}))
        stats.append({"qid": qid, "n_docs": len(docs), "n_chunks": len(chunks),
                      "n_add_docs": len(add), "n_add_chunks": len(chunks) - len(base_chunks)})

    out = pd.concat(rows, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out / "lists.parquet", index=False)
    st = pd.DataFrame(stats)
    print(f"-> {args.out / 'lists.parquet'}: {len(out):,} rows")
    print(st[["n_docs", "n_chunks", "n_add_docs", "n_add_chunks"]].describe().to_string())
    print(f"queries with additions: {(st['n_add_docs'] > 0).sum():,}/{len(st):,}")


if __name__ == "__main__":
    main()
