"""Score (query, doc) with the doc2query channel: max cosine between the query and the questions of that document.

    python scripts/analysis/doc2query_score.py --emb kaggle/outputs/dq-embed --out runs/reports/dq [--topk 500]

--emb holds emb.npy / meta.parquet / q_emb.npy / q_ids.npy from stages/dq_embed.py (Kaggle GPU).
The product is computed on the CPU in blocks with BLAS: questions are sorted by doc_id once, so the per-document max
is a single `np.maximum.reduceat` per block. Only the top `--topk` documents of each query are written.

Outputs: scores.parquet (qid, doc_id, dq_score, dq_rank), stats.json, and a short report on stdout: score
distribution, how many of the top-20 docs are shared with the m3rr order of #7230, and the language mix.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())


def m3rr_head(n: int = 20) -> pd.DataFrame:
    """The first n docs of the #7230 order, for the overlap report (needs the pool cands and the feature tables)."""
    cands = pd.read_parquet(ROOT / "kaggle/datasets/rerank_pool/cands.parquet")
    cands["lang"] = cands["set"].str[:2]
    cands = cands[cands["rank"] < cands["lang"].map({"vi": 800, "zh": 200, "en": 200})]
    rr = pd.concat([pd.read_parquet(p, columns=["qid", "doc_id", "score"])
                    for p in sorted(glob.glob(str(ROOT / "kaggle/outputs/rr4-p*/rerank.parquet")))], ignore_index=True)
    m3 = pd.concat([pd.read_parquet(p, columns=["qid", "doc_id", "dense", "sparse", "colbert"])
                    for p in sorted(glob.glob(str(ROOT / "kaggle/outputs/m3m-p*/m3multi.parquet")))], ignore_index=True)
    pool = cands.merge(rr, on=["qid", "doc_id"], how="left").merge(m3, on=["qid", "doc_id"], how="left")
    pool["m3all"] = 0.4 * pool["dense"] + 0.2 * pool["sparse"] + 0.4 * pool["colbert"]
    pool["rr_rank"] = pool.groupby("qid")["score"].rank(ascending=False, method="first") - 1
    m3_rank = pool.groupby("qid")["m3all"].rank(ascending=False, method="first") - 1
    pool["key"] = -(1 / (60 + m3_rank) + 1 / (60 + pool["rr_rank"]))
    pool = pool.sort_values(["qid", "key"])
    pool["pos"] = pool.groupby("qid").cumcount()
    return pool[pool["pos"] < n][["qid", "doc_id"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", type=Path, default=ROOT / "kaggle/outputs/dq-embed")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/reports/dq")
    ap.add_argument("--topk", type=int, default=500)
    ap.add_argument("--block", type=int, default=50000)
    ap.add_argument("--qchunk", type=int, default=200)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(args.emb / "meta.parquet")
    E = np.load(args.emb / "emb.npy", mmap_mode="r")
    Q = np.load(args.emb / "q_emb.npy").astype(np.float32)
    qids = np.load(args.emb / "q_ids.npy")
    print(f"{E.shape[0]:,} questions x {E.shape[1]} dim; {Q.shape[0]} queries", flush=True)

    order = np.argsort(meta["doc_id"].to_numpy(), kind="stable")
    docs_sorted = meta["doc_id"].to_numpy()[order]
    udocs = np.unique(docs_sorted)
    print(f"{len(udocs):,} distinct docs", flush=True)

    rows = []
    for qs in range(0, len(Q), args.qchunk):
        qc = Q[qs:qs + args.qchunk]
        best = np.full((len(qc), len(udocs)), -1.0, dtype=np.float32)
        for lo in range(0, len(order), args.block):
            idx = order[lo:lo + args.block]
            eb = np.asarray(E[idx], dtype=np.float32)
            s = qc @ eb.T
            local_docs, local_start = np.unique(docs_sorted[lo:lo + len(idx)], return_index=True)
            m = np.maximum.reduceat(s, local_start, axis=1)
            pos = np.searchsorted(udocs, local_docs)
            best[:, pos] = np.maximum(best[:, pos], m)
        k = min(args.topk, len(udocs) - 1)
        top = np.argpartition(-best, k, axis=1)[:, : args.topk]
        val = np.take_along_axis(best, top, axis=1)
        o = np.argsort(-val, axis=1)
        top, val = np.take_along_axis(top, o, axis=1), np.take_along_axis(val, o, axis=1)
        for i in range(len(qc)):
            rows.append(pd.DataFrame({"qid": int(qids[qs + i]), "doc_id": udocs[top[i]], "dq_score": val[i],
                                      "dq_rank": np.arange(len(top[i]))}))
        print(f"queries {qs + len(qc)}/{len(Q)}", flush=True)

    sc = pd.concat(rows, ignore_index=True)
    sc.to_parquet(args.out / "scores.parquet", index=False)

    head = m3rr_head(20)
    top20 = sc[sc["dq_rank"] < 20][["qid", "doc_id"]].assign(dq=1)
    j = head.merge(top20, on=["qid", "doc_id"], how="left")
    shared = j.groupby("qid")["dq"].sum().mean()
    s1 = sc[sc["dq_rank"] == 0]["dq_score"]
    stats = {"queries": int(sc["qid"].nunique()), "docs_per_query": int(len(sc) / sc["qid"].nunique()),
             "score_top1_mean": round(float(s1.mean()), 4), "score_top1_p10": round(float(s1.quantile(.1)), 4),
             "score_top1_p90": round(float(s1.quantile(.9)), 4),
             "shared_with_m3rr_top20": round(float(shared), 2),
             "vi_share_top20": round(float((sc[sc["dq_rank"] < 20]["doc_id"] <= 766441).mean()), 3)}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(f"-> {args.out / 'scores.parquet'}: {len(sc):,} rows")


if __name__ == "__main__":
    main()
