"""Swap the weakest base docs for high-confidence title-only docs (probe).

The X40 experiment proposed swapping title-head docs for the m3rr tail, but it used
lex_head2 ranks.  This version uses the BGE title cross-encoder and swaps only
title-only docs with a positive score, so the doc/chunk counts stay at the base
operating point while the added docs are evidence-backed.  It is a controlled
probe, not the final dynamic selector.

    python select_title_swap.py --base-zip runs/mat_out/st1/r24.zip \
        --title kaggle/outputs/vibiomir-title-rerank/title_rerank.parquet \
        --pool runs/reports/pool_m3rr.parquet --out temp_submission/title_swap_3 --job title_swap_3
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
    ap.add_argument("--pool", type=Path, default=ROOT / "runs/reports/pool_m3rr.parquet")
    ap.add_argument("--meta", type=Path, default=ROOT / "data/docstore/v2.1/docs_meta.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--job", default="title_swap")
    ap.add_argument("--window", type=int, default=4500)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-title-rank", type=int, default=30)
    ap.add_argument("--max-swap", type=int, default=3)
    args = ap.parse_args()

    base = load_base(args.base_zip)
    title = pd.read_parquet(args.title)
    title = title[(title["title_rank"] < args.max_title_rank) & (title["title_score"] > args.min_score)]
    title = title.sort_values(["qid", "title_rank"])
    pool = pd.read_parquet(args.pool)
    rank = {(int(q), int(d)): int(r) for q, d, r in pool[["qid", "doc_id", "rank"]].itertuples(index=False)}
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
            if len(add) >= args.max_swap:
                break
        # Remove the weakest base docs by m3rr rank, not by submission order.
        weakest = sorted(base_docs, key=lambda d: rank.get((qid, d), 10**9), reverse=True)[:len(add)]
        removed = set(weakest)
        docs = [d for d in base_docs if d not in removed] + add
        keep = [d for d in base_chunks if d not in removed]
        newc = [d for d in add if d not in keep]
        slots = max(0, len(base_chunks) - len(newc))
        chunks = keep[:slots] + newc
        if len(chunks) < len(base_chunks):
            chunks += [d for d in base_chunks if d not in chunks][:len(base_chunks) - len(chunks)]
        rows.append(pd.DataFrame({"job": args.job, "qid": qid, "kind": "doc",
                                  "pos": np.arange(len(docs)), "doc_id": docs,
                                  "size": 0, "src": ""}))
        rows.append(pd.DataFrame({"job": args.job, "qid": qid, "kind": "chunk",
                                  "pos": np.arange(len(chunks)), "doc_id": chunks,
                                  "size": args.window, "src": ""}))
        stats.append({"qid": qid, "n_docs": len(docs), "n_chunks": len(chunks),
                      "n_add_docs": len(add), "n_swap_docs": len(weakest)})

    out = pd.concat(rows, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out / "lists.parquet", index=False)
    st = pd.DataFrame(stats)
    print(f"-> {args.out / 'lists.parquet'}: {len(out):,} rows")
    print(st[["n_docs", "n_chunks", "n_add_docs", "n_swap_docs"]].describe().to_string())
    print(f"queries with swaps: {(st['n_add_docs'] > 0).sum():,}/{len(st):,}")
    print(f"mean swapped docs: {st['n_swap_docs'].mean():.2f}")


if __name__ == "__main__":
    main()
