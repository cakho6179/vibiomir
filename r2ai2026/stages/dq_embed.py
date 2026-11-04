"""Embed the doc2query questions and the queries (Kaggle GPU), so the score can be computed locally.

    python dq_embed.py --data DIR --out DIR [--model BAAI/bge-m3] [--batch 64] [--max-len 128] [--max-questions N]

DIR holds questions.parquet (doc_id, qid, pid, idx, question from stages/doc2query.py) and query.parquet.
Writes to OUT:
  emb.npy        float16, L2-normalised question vectors, row order = meta.parquet
  meta.parquet   row, doc_id, qid, idx, question
  q_emb.npy      float16, L2-normalised query vectors
  q_ids.npy      int64 query ids, same order as q_emb.npy
  stats.json     questions, queries, dim, questions/s

The scoring itself (max cosine between a query and the questions of a document) runs on the CPU afterwards:
1200 x N dot products is a single BLAS call per block, no GPU needed.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def find(root: str, pattern: str) -> str:
    hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    if not hits:
        raise SystemExit(f"not found under {root}: {pattern}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--pairs", default="", help="e.g. head_docs.parquet: also score these qid/doc_id pairs")
    ap.add_argument("--no-save-emb", action="store_true", help="skip the big .npy outputs (scores only)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    q = pd.read_parquet(find(str(args.data), "questions.parquet")).reset_index(drop=True)
    if args.max_questions:
        q = q.head(args.max_questions)
    queries = pd.read_parquet(find(str(args.data), "query.parquet")).sort_values("id").reset_index(drop=True)
    log(f"{len(q):,} questions, {len(queries):,} queries")

    model = SentenceTransformer(args.model, device="cuda")
    model.max_seq_length = args.max_len
    model.half()

    emb = model.encode(q["question"].astype(str).tolist(), batch_size=args.batch, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True).astype(np.float16)
    log(f"questions embedded {emb.shape}, {len(q) / max(time.time() - T0, 1):.0f}/s")
    if not args.no_save_emb:
        np.save(args.out / "emb.npy", emb)
    q.assign(row=np.arange(len(q)))[["row", "doc_id", "qid", "pid", "idx", "question"]].to_parquet(
        args.out / "meta.parquet", index=False)

    qe = model.encode(queries["query"].astype(str).tolist(), batch_size=args.batch, normalize_embeddings=True,
                      convert_to_numpy=True).astype(np.float16)
    if not args.no_save_emb:
        np.save(args.out / "q_emb.npy", qe)
        np.save(args.out / "q_ids.npy", queries["id"].to_numpy().astype(np.int64))
    stats = {"questions": int(len(q)), "queries": int(len(queries)), "dim": int(emb.shape[1]),
             "model": args.model, "max_len": args.max_len, "seconds": round(time.time() - T0)}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    log(json.dumps(stats))

    if args.pairs:
        # Score on the GPU and ship only the small scores table: the 387 MB emb.npy download
        # breaks on a flaky link, a 2 MB parquet does not.
        import torch

        pairs = pd.read_parquet(find(str(args.data), args.pairs), columns=["qid", "doc_id"])
        pairs = pairs.drop_duplicates(["qid", "doc_id"]).reset_index(drop=True)
        docs, inv = np.unique(q["doc_id"].to_numpy(), return_inverse=True)
        qrow = {int(i): n for n, i in enumerate(queries["id"].to_numpy())}
        dcol = {int(d): n for n, d in enumerate(docs)}
        keep = pairs["qid"].isin(qrow) & pairs["doc_id"].isin(dcol)
        pairs = pairs[keep].reset_index(drop=True)
        log(f"scoring {len(pairs):,} pairs over {len(docs):,} docs x {len(queries):,} queries")

        E = torch.from_numpy(emb).cuda()
        Q = torch.from_numpy(qe).cuda()
        D = torch.full((len(Q), len(docs)), -1.0, dtype=torch.float32, device="cuda")
        inv_t = torch.from_numpy(inv.astype(np.int64)).cuda()
        B = 4096
        with torch.inference_mode():
            for lo in range(0, len(E), B):
                s = (Q.float() @ E[lo:lo + B].float().T)  # queries x block
                D.scatter_reduce_(1, inv_t[lo:lo + B].expand(len(Q), -1), s, reduce="amax")
                if lo % (B * 10) == 0:
                    log(f"scored {lo:,}/{len(E):,} question blocks")
        out = pairs.copy()
        qi = torch.from_numpy(pairs["qid"].map(qrow).to_numpy()).cuda()
        di = torch.from_numpy(pairs["doc_id"].map(dcol).to_numpy()).cuda()
        out["dq_score"] = D[qi, di].float().cpu().numpy()
        out["dq_rank"] = out.groupby("qid")["dq_score"].rank(ascending=False, method="first").astype("int32")
        out.to_parquet(args.out / "scores.parquet", index=False)
        log(f"scores.parquet: {len(out):,} rows, {out['qid'].nunique()} queries, "
            f"mean score {out['dq_score'].mean():.4f}")


if __name__ == "__main__":
    main()
