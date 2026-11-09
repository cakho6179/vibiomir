"""Dynamic doc/chunk selection from reranker + title/slug evidence (probe, not wired into v3 yet).

The fixed `NDOCS` job spec in `scripts/analysis/decide_lists.py` is a diagnostic operating point, not a target.
This script makes the count a per-query decision:

    combined = 1/(60 + rr_rank) + w_title/(60 + title_rank) + rare_bonus
    docs      = candidates with combined >= doc_frac * max(combined), at least min_docs, at most max_docs
    chunks    = one 4500-char window for the top candidates with combined >= chunk_frac * max(combined)

Candidates are the union of the production pool (reranker scores) and the title/slug head (`lex_head2` or
`title_rerank`).  `title_rerank` is the cross-encoder score of `(query, title_text)`; using its rank as RRF
keeps the high-recall title expansion while removing generic title matches.  The title-only docs are the pool
expansion; the dynamic cutoff is what keeps the list from being a fixed 197.  `--max-chunks` is a payload
guard, not a target: the actual zip is checked after materialization.

    python scripts/analysis/select_dynamic.py --out temp_submission/dyn1 --job dyn1
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())


def load_pool(pool_file: Path, rr_glob: str | None, cut: str) -> pd.DataFrame:
    cands = pd.read_parquet(pool_file)
    cands["lang"] = cands["set"].str[:2]
    want = {s[:2]: int(s[2:]) for s in cut.split(",")}
    cands = cands[cands["rank"] < cands["lang"].map(want)].copy()
    parts = sorted(glob.glob(str(rr_glob or ROOT / "kaggle/outputs/rr3-p*/rerank.parquet")))
    rr = pd.concat([pd.read_parquet(p, columns=["qid", "doc_id", "score"]) for p in parts], ignore_index=True)
    pool = cands.merge(rr, on=["qid", "doc_id"], how="left")
    pool["score"] = pool["score"].fillna(-99.0)
    pool["rr_rank"] = pool.groupby("qid")["score"].rank(ascending=False, method="first").astype(int) - 1
    return pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-file", type=Path, default=ROOT / "kaggle/datasets/rerank_pool/cands.parquet")
    ap.add_argument("--rr", default=None, help="glob of rr3 rerank.parquet parts")
    ap.add_argument("--title", type=Path, default=ROOT / "runs/reports/title_rerank.parquet")
    ap.add_argument("--meta", type=Path, default=ROOT / "data/docstore/v2.1/docs_meta.parquet")
    ap.add_argument("--pool", default="vi800,zh200,en200")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--job", default="dyn")
    ap.add_argument("--window", type=int, default=4500)
    ap.add_argument("--w-title", type=float, default=1.0)
    ap.add_argument("--fuse", choices=["sum", "max"], default="max",
                    help="max keeps the pool rank intact while adding strong title-only docs; sum dilutes it")
    ap.add_argument("--title-top", type=int, default=200,
                    help="only title candidates with title_rank below this are eligible")
    ap.add_argument("--rare-bonus", type=float, default=0.0)
    ap.add_argument("--doc-frac", type=float, default=0.22)
    ap.add_argument("--chunk-frac", type=float, default=0.22)
    ap.add_argument("--min-docs", type=int, default=40)
    ap.add_argument("--max-docs", type=int, default=300)
    ap.add_argument("--max-chunks", type=int, default=160)
    args = ap.parse_args()

    pool = load_pool(args.pool_file, args.rr, args.pool)
    title = pd.read_parquet(args.title)
    if "title_rank" not in title:
        title = title.rename(columns={"rank": "title_rank", "score": "title_score"})
    if "phrase" not in title:
        title["phrase"] = ""
    if "title_score" not in title:
        title["title_score"] = 0.0
    meta = pd.read_parquet(args.meta, columns=["id", "n_chars", "lang", "content_key"]).set_index("id")
    n_chars = meta["n_chars"].to_dict()
    lang = meta["lang"].to_dict()
    content_key = meta["content_key"].fillna("").to_dict()

    # The title head is a candidate source, not a filter: a title-only document gets a title rank but no rr rank.
    title = title[title["title_rank"] < args.title_top].copy()
    title["rare"] = title["phrase"].fillna("").astype(str).ne("")
    allc = pool.merge(title[["qid", "doc_id", "title_rank", "title_score", "phrase", "rare"]],
                      on=["qid", "doc_id"], how="left")
    # Add title docs that are not in the pool at all.
    extra = title.merge(pool[["qid", "doc_id", "lang", "rank", "score", "rr_rank"]],
                        on=["qid", "doc_id"], how="left")
    extra = extra[extra["rr_rank"].isna()].copy()
    extra["score"] = -99.0
    extra["rr_rank"] = np.nan
    extra["rank"] = np.nan
    extra["lang"] = extra["doc_id"].map(lang)
    cols = ["qid", "doc_id", "lang", "rank", "score", "rr_rank", "title_rank", "title_score", "phrase", "rare"]
    allc = pd.concat([allc[cols], extra[cols]], ignore_index=True)
    allc["title_rank"] = allc["title_rank"].fillna(1000).astype(int)
    allc["rr_rank"] = allc["rr_rank"].fillna(10000).astype(int)
    allc["rare"] = allc["rare"].fillna(False).astype(bool)
    allc["content_key"] = allc["doc_id"].map(content_key)
    allc["content_key"] = allc["content_key"].fillna(allc["doc_id"].astype(str))

    rr_rrf = np.where(allc["rr_rank"] < 10000, 1.0 / (60.0 + allc["rr_rank"]), 0.0)
    title_rrf = np.where(allc["title_rank"] < 1000, 1.0 / (60.0 + allc["title_rank"]), 0.0)
    title_evidence = args.w_title * title_rrf + args.rare_bonus * title_rrf * allc["rare"]
    if args.fuse == "max":
        allc["combined"] = np.maximum(rr_rrf, title_evidence)
    else:
        allc["combined"] = rr_rrf + title_evidence
    allc["combined"] = allc["combined"].fillna(0.0)
    # Duplicate documents share content_key.  Keep the highest-scoring copy per query so a duplicate
    # title hit cannot spend a second doc/chunk slot.
    allc = allc.sort_values(["qid", "combined", "rr_rank", "title_rank"],
                            ascending=[True, False, True, True]).drop_duplicates(["qid", "content_key"],
                                                                                 keep="first")

    rows = []
    stats = []
    for qid, g in allc.groupby("qid", sort=True):
        g = g.sort_values(["combined", "rr_rank", "title_rank"], ascending=[False, True, True])
        top = float(g["combined"].iloc[0]) if len(g) else 0.0
        docs = g[g["combined"] >= args.doc_frac * top].head(args.max_docs)
        if len(docs) < args.min_docs:
            docs = g.head(args.min_docs)
        chunks = g[g["combined"] >= args.chunk_frac * top].head(args.max_chunks)
        rows.append(docs.assign(job=args.job, kind="doc", pos=np.arange(len(docs)), size=0, src="")[
            ["job", "qid", "kind", "pos", "doc_id", "size", "src"]])
        rows.append(chunks.assign(job=args.job, kind="chunk", pos=np.arange(len(chunks)), size=args.window, src="")[
            ["job", "qid", "kind", "pos", "doc_id", "size", "src"]])
        stats.append({
            "qid": int(qid), "n_docs": len(docs), "n_chunks": len(chunks),
            "n_title": int((docs["title_rank"] < 1000).sum()),
            "n_title_only": int((docs["rr_rank"] >= 10000).sum()),
            "n_rare": int(docs["rare"].sum()),
            "est_chars": int(sum(min(args.window, int(n_chars.get(int(d), args.window))) for d in chunks["doc_id"])),
        })

    out = pd.concat(rows, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out / "lists.parquet", index=False)
    st = pd.DataFrame(stats)
    print(f"-> {args.out/'lists.parquet'}: {len(out):,} rows")
    print(st[["n_docs", "n_chunks", "n_title", "n_title_only", "n_rare", "est_chars"]].describe().round(1).to_string())
    print("language mix docs:", (out[out.kind == "doc"].merge(
        meta.reset_index()[["id", "lang"]], left_on="doc_id", right_on="id", how="left")
        .groupby("lang").size() / st.shape[0]).round(2).to_dict())
    print("est total chars MB:", round(st.est_chars.sum() / 1e6, 1),
          "| est payload MB:", round(st.est_chars.sum() / 1e6, 1))


if __name__ == "__main__":
    main()
