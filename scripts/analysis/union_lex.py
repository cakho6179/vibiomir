"""Add a clean lexical-head candidate list to a decided doc list, without dropping base docs.

This is the recall-safe C1 used after the 29/10 correction: the base list is kept
100% intact, and lexical-head docs are only *added* when they are new, in the
docstore, and inside the requested top-K. Chunks are copied unchanged, so the
chunk channel remains an independent A/B.

    python scripts/analysis/union_lex.py --base lists.parquet --lex lex_head3.parquet \
        --docstore data/docstore/v2.1 --top-k 20 --out union/lists.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="decided lists.parquet with doc and chunk rows")
    ap.add_argument("--lex", type=Path, required=True, help="lexical ranking parquet: qid, doc_id, rank[, score]")
    ap.add_argument("--out", type=Path, required=True, help="output lists.parquet")
    ap.add_argument("--top-k", type=int, default=20, help="only add lexical docs with rank < top-k")
    ap.add_argument("--min-score", type=float, default=None, help="optional lexical score floor")
    ap.add_argument("--docstore", type=Path, default=None, help="v2.1 root; filters docs absent from docs_meta")
    ap.add_argument("--base-job", default="", help="job to read when the base file has several jobs")
    ap.add_argument("--job", default="", help="output job name; default <base>_lex")
    args = ap.parse_args()

    base = pd.read_parquet(args.base)
    required = {"job", "qid", "kind", "pos", "doc_id"}
    missing = required - set(base.columns)
    if missing:
        raise SystemExit(f"base is missing columns: {sorted(missing)}")
    base_jobs = sorted(base["job"].astype(str).unique())
    if args.base_job:
        if args.base_job not in base_jobs:
            raise SystemExit(f"base job {args.base_job!r} not in {base_jobs}")
        base = base[base["job"].astype(str) == args.base_job]
        base_jobs = [args.base_job]
    if len(base_jobs) != 1:
        raise SystemExit(f"expected exactly one base job, got {base_jobs}; pass --base-job")
    job = args.job or f"{base_jobs[0]}_lex"

    lex = pd.read_parquet(args.lex)
    if not {"qid", "doc_id", "rank"}.issubset(lex.columns):
        raise SystemExit(f"lex is missing qid/doc_id/rank: {lex.columns.tolist()}")
    lex = lex[lex["rank"].astype(int) < args.top_k].copy()
    if args.min_score is not None and "score" in lex.columns:
        lex = lex[lex["score"].astype(float) >= args.min_score]
    lex["qid"] = lex["qid"].astype(int)
    lex["doc_id"] = lex["doc_id"].astype(int)

    if args.docstore is not None:
        meta = pd.read_parquet(args.docstore / "docs_meta.parquet", columns=["id"])
        official = set(meta["id"].astype(int))
        before = len(lex)
        lex = lex[lex["doc_id"].isin(official)]
        print(f"docstore filter: {before:,} -> {len(lex):,} lexical rows")

    docs = base[base["kind"] == "doc"].copy()
    chunks = base[base["kind"] == "chunk"].copy()
    docs["qid"] = docs["qid"].astype(int)
    docs["doc_id"] = docs["doc_id"].astype(int)
    chunks["qid"] = chunks["qid"].astype(int)
    chunks["doc_id"] = chunks["doc_id"].astype(int)
    docs["pos"] = docs["pos"].astype(int)
    chunks["pos"] = chunks["pos"].astype(int)

    added = 0
    out_docs = []
    for qid, g in docs.groupby("qid", sort=True):
        g = g.sort_values("pos")
        seen = set(g["doc_id"].astype(int))
        order = list(g["doc_id"].astype(int))
        lg = lex[lex["qid"] == qid].sort_values(["rank", "doc_id"])
        for did in lg["doc_id"].astype(int):
            if did not in seen:
                seen.add(did)
                order.append(did)
                added += 1
        out_docs.append(pd.DataFrame({
            "job": job, "qid": qid, "kind": "doc",
            "pos": range(len(order)), "doc_id": order,
            "size": 0, "src": "",
        }))
    # Keep every base query even when it has no lexical hit.
    base_qids = set(docs["qid"].unique())
    lex_qids = set(lex["qid"].unique())
    missing_qids = sorted(lex_qids - base_qids)
    if missing_qids:
        raise SystemExit(f"lexical ranking has qids absent from base: {missing_qids[:5]}")

    out_chunks = chunks.copy()
    out_chunks["job"] = job
    out = pd.concat([*out_docs, out_chunks], ignore_index=True)
    out = out[["job", "qid", "kind", "pos", "doc_id", "size", "src"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    nq = docs["qid"].nunique()
    nd = out_docs and sum(len(x) for x in out_docs) / nq
    print(f"base job={base_jobs[0]} -> {job}")
    print(f"added {added:,} docs ({added / nq:.2f}/query), docs/query {nd:.2f}, chunks/query {len(chunks) / nq:.1f}")
    print(f"-> {args.out}: {len(out):,} rows")


if __name__ == "__main__":
    main()
