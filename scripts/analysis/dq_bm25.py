"""Score (query, doc) by BM25 over the doc2query questions: doc score = max BM25 of the query against its questions.

    python scripts/analysis/dq_bm25.py [--questions kaggle/outputs/dq-all] [--out runs/reports/dq_bm25] [--topk 1000]

The lexical twin of doc2query_score.py: if gold was labelled by matching queries against MedKB `similar_queries`
with BM25 (admin BM25 baseline: docs P .88), BM25 over generated questions is the closest copy of that channel.
Tokens are lower-cased syllables (Vietnamese is written syllable by syllable), diacritics kept.
Writes scores.parquet (qid, doc_id, dq_score, dq_rank), the same layout as doc2query_score.py.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
TOKEN = re.compile(r"\w+")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, default=ROOT / "kaggle/outputs/dq-all")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/reports/dq_bm25")
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    qs = pd.read_parquet(args.questions / "questions.parquet", columns=["doc_id", "question"])
    queries = pd.read_parquet(args.questions / "query.parquet")
    vocab: dict[str, int] = {}
    rows, cols, vals = [], [], []
    for i, text in enumerate(qs["question"]):
        tf: dict[int, int] = {}
        for t in tokens(text):
            j = vocab.setdefault(t, len(vocab))
            tf[j] = tf.get(j, 0) + 1
        rows += [i] * len(tf)
        cols += list(tf)
        vals += list(tf.values())
    tf = sparse.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(qs), len(vocab)))
    dl = np.asarray(tf.sum(axis=1)).ravel()
    df = np.bincount(tf.indices, minlength=len(vocab))
    idf = np.log1p((len(qs) - df + 0.5) / (df + 0.5)).astype(np.float32)
    norm = args.k1 * (1 - args.b + args.b * dl / dl.mean())
    w = tf.copy()
    w.data = w.data * (args.k1 + 1) / (w.data + np.repeat(norm, np.diff(w.indptr)))
    w = (w @ sparse.diags(idf)).T.tocsr()  # vocab x questions
    print(f"{len(qs):,} questions, {len(vocab):,} terms, mean length {dl.mean():.1f}", flush=True)

    doc_ids = qs["doc_id"].to_numpy()
    udocs, inv = np.unique(doc_ids, return_inverse=True)
    out = []
    for qid, text in zip(queries["id"].astype(int), queries["query"]):
        terms = sorted({vocab[t] for t in tokens(text) if t in vocab})
        s = np.asarray(w[terms].sum(axis=0)).ravel() if terms else np.zeros(len(qs), np.float32)
        best = np.full(len(udocs), 0.0, np.float32)
        np.maximum.at(best, inv, s)
        k = min(args.topk, len(udocs))
        top = np.argpartition(-best, k - 1)[:k]
        top = top[np.argsort(-best[top])]
        out.append(pd.DataFrame({"qid": qid, "doc_id": udocs[top], "dq_score": best[top], "dq_rank": np.arange(k)}))
    sc = pd.concat(out, ignore_index=True)
    sc.to_parquet(args.out / "scores.parquet", index=False)
    print(f"-> {args.out / 'scores.parquet'}: {len(sc):,} rows, {sc['qid'].nunique()} queries", flush=True)


if __name__ == "__main__":
    main()
