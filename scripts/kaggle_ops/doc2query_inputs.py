"""Build the doc2query input set: the head documents that can reach the top of a query list.

    python scripts/kaggle_ops/doc2query_inputs.py [--top 100] [--out kaggle/datasets/dq_head]

Why: gold looks like it was produced by matching a query against chunk-level `similar_queries` of the organizers'
KB (admin BM25 reaches P .88 that way, and gold has 1.05 chunks per doc). Rebuilding that channel needs patient-style
questions written *from the document*, then a query-to-question similarity score. This script only prepares the input:
one row per distinct head document, with the passage the reranker liked best, so the Kaggle kernel can slice the text
from the corpus and generate questions.

Head = union over queries of the top `--top` docs of the m3rr order (RRF k=60 of BGE-M3 dense/sparse/ColBERT and the
bge-reranker-v2-m3 MaxP score), i.e. the same order as the best submission #7230. Writes head_docs.parquet:
doc_id, pid, start, end, heading, qid, rr_score, n_head (how many queries have this doc in their head).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())

LINK_RE = re.compile(r"\[[^\]]{0,300}\]\([^)]{0,300}\)")
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"[ \t\r\f\v]+")


def load_bodies(wanted: set[int]) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    for part in sorted(glob.glob(str(ROOT / "data/docstore/v2.1/corpus/part-*.jsonl*"))):
        opener = gzip.open if part.endswith(".gz") else open
        with opener(part, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["id"] in wanted and d["id"] not in out:
                    out[d["id"]] = (d.get("t") or "", d.get("x") or "")
    return out


def clean_passage(body: str, start: int, end: int, furn: list[tuple[int, int]], limit: int) -> str:
    """Passage text with furniture spans blanked, markdown links and URLs dropped, whitespace collapsed."""
    seg = body[start:end]
    if furn:
        chars = list(seg)
        for a, b in furn:
            for i in range(max(a, start), min(b, end)):
                chars[i - start] = "\n"
        seg = "".join(chars)
    seg = LINK_RE.sub(" ", seg)
    seg = URL_RE.sub(" ", seg)
    lines = [WS_RE.sub(" ", ln).strip() for ln in seg.split("\n")]
    seg = "\n".join(ln for ln in lines if len(ln) > 20)
    return seg[:limit].strip()


def m3rr_order(cut: dict[str, int], top: int) -> pd.DataFrame:
    cands = pd.read_parquet(ROOT / "kaggle/datasets/rerank_pool/cands.parquet")
    cands["lang"] = cands["set"].str[:2]
    cands = cands[cands["rank"] < cands["lang"].map(cut)].copy()
    rr = pd.concat([pd.read_parquet(p, columns=["qid", "doc_id", "score", "best_passage"])
                    for p in sorted(glob.glob(str(ROOT / "kaggle/outputs/rr4-p*/rerank.parquet")))],
                   ignore_index=True)
    pool = cands.merge(rr, on=["qid", "doc_id"], how="left")
    m3 = pd.concat([pd.read_parquet(p, columns=["qid", "doc_id", "dense", "sparse", "colbert"])
                    for p in sorted(glob.glob(str(ROOT / "kaggle/outputs/m3m-p*/m3multi.parquet")))],
                   ignore_index=True)
    pool = pool.merge(m3, on=["qid", "doc_id"], how="left")
    pool["m3all"] = 0.4 * pool["dense"] + 0.2 * pool["sparse"] + 0.4 * pool["colbert"]
    print(f"pool {len(pool):,} pairs; rr unscored {pool['score'].isna().mean():.2%}; "
          f"m3 unscored {pool['m3all'].isna().mean():.2%}", flush=True)
    pool["rr_rank"] = pool.groupby("qid")["score"].rank(ascending=False, method="first") - 1
    m3_rank = pool.groupby("qid")["m3all"].rank(ascending=False, method="first") - 1
    pool["key"] = -(1 / (60 + m3_rank) + 1 / (60 + pool["rr_rank"]))
    pool = pool.sort_values(["qid", "key"])
    pool["pos"] = pool.groupby("qid").cumcount()
    return pool[pool["pos"] < top]


def base_order(cands_p: Path, top: int, exclude: Path | None) -> pd.DataFrame:
    """Top `top` docs per query by a p(gold) model (pgold_all cands_p.parquet), with the rr4 MaxP passage, minus the docs
    that already have questions; prio = the doc's highest p over queries (the kernel works through high prio first)."""
    c = pd.read_parquet(cands_p, columns=["qid", "doc_id", "p"]).sort_values(["qid", "p"], ascending=[True, False])
    c = c[c.groupby("qid").cumcount() < top]
    if exclude is not None:
        c = c[~c["doc_id"].isin(set(pd.read_parquet(exclude, columns=["doc_id"])["doc_id"]))]
    rr = pd.concat([pd.read_parquet(p, columns=["qid", "doc_id", "score", "best_passage"])
                    for p in sorted(glob.glob(str(ROOT / "kaggle/outputs/rr4-p*/rerank.parquet")))], ignore_index=True)
    c = c.merge(rr, on=["qid", "doc_id"], how="left")
    c["prio"] = c.groupby("doc_id")["p"].transform("max")
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--cands-p", type=Path, help="head = top --top docs per query by this p(gold) table, not m3rr")
    ap.add_argument("--exclude", type=Path, help="questions.parquet of docs already done (skipped)")
    ap.add_argument("--pool", default="vi800,zh200,en200")
    ap.add_argument("--text", action="store_true", help="add the cleaned passage text (kernel then needs no corpus)")
    ap.add_argument("--chars", type=int, default=1100)
    ap.add_argument("--out", type=Path, default=ROOT / "kaggle/datasets/dq_head")
    args = ap.parse_args()
    cut = {s[:2]: int(s[2:]) for s in args.pool.split(",")}
    head = base_order(args.cands_p, args.top, args.exclude) if args.cands_p else m3rr_order(cut, args.top)
    n_docs = head["doc_id"].nunique()
    print(f"head: {len(head):,} pairs, {n_docs:,} distinct docs, "
          f"{len(head) / head['qid'].nunique():.0f} per query", flush=True)

    pas = pd.read_parquet(ROOT / "data/docstore/v2.1/passages.parquet",
                          columns=["doc_id", "pid", "start", "end", "heading"])
    best = (head.dropna(subset=["best_passage"])
            .assign(pid=lambda d: d["best_passage"].astype(int))
            .sort_values("score", ascending=False)
            .drop_duplicates("doc_id")[["doc_id", "qid", "pid", "score"]])
    n_head = head.groupby("doc_id").size().rename("n_head")
    out = (best.merge(pas, on=["doc_id", "pid"], how="left")
           .merge(n_head, on="doc_id", how="left")
           .rename(columns={"score": "rr_score"}))
    if "prio" in head.columns:
        out = out.merge(head.groupby("doc_id")["prio"].max(), on="doc_id", how="left")
    miss = out["start"].isna().mean()
    print(f"best passage resolved for {1 - miss:.2%} of docs", flush=True)
    out = out.dropna(subset=["start"]).assign(
        pid=lambda d: d["pid"].astype(int), start=lambda d: d["start"].astype(int),
        end=lambda d: d["end"].astype(int), n_head=lambda d: d["n_head"].astype(int))
    out["chars"] = out["end"] - out["start"]
    if args.text:
        bodies = load_bodies(set(out["doc_id"].astype(int)))
        furn = pd.read_parquet(ROOT / "data/docstore/v2.1/furniture.parquet")
        furn = furn[furn["doc_id"].isin(set(out["doc_id"].astype(int)))]
        spans: dict[int, list[tuple[int, int]]] = {}
        for r in furn.itertuples():
            spans.setdefault(int(r.doc_id), []).append((int(r.start), int(r.end)))
        texts, titles, keep = [], [], []
        for r in out.itertuples():
            hit = bodies.get(int(r.doc_id))
            t = clean_passage(hit[1], int(r.start), int(r.end), spans.get(int(r.doc_id), []), args.chars) if hit else ""
            texts.append(t)
            titles.append((hit[0] if hit else "")[:200])
            keep.append(len(t) >= 200)
        out["text"] = texts
        out["title"] = titles
        print(f"text built for {sum(keep):,}/{len(out):,} docs (>=200 chars after cleaning)", flush=True)
        out = out[keep]
    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["doc_id", "qid", "pid", "start", "end", "heading", "rr_score", "n_head", "chars"] + (
        ["prio"] if "prio" in out.columns else []) + (["title", "text"] if args.text else [])
    out[cols].to_parquet(args.out / "head_docs.parquet", index=False)
    print(f"-> {args.out / 'head_docs.parquet'}: {len(out):,} docs, passage chars "
          f"median {out['chars'].median():.0f} / p90 {out['chars'].quantile(.9):.0f}", flush=True)
    print("docs by n_head:", out["n_head"].value_counts().head(5).to_dict(), flush=True)


if __name__ == "__main__":
    main()
