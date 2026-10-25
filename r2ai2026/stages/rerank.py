"""Cross-encoder reranking of candidate documents on Kaggle GPU(s), full text.

    python rerank.py --data DIR --cands candidates.parquet --out DIR [--model BAAI/bge-reranker-v2-m3]
                     [--part I/N] [--max-queries N]

candidates.parquet: (qid, doc_id) pairs, e.g. the union of BM25 and dense top-K. Every document is
split into ~1500-char passages (title on each); doc score = max passage logit. --part I/N keeps
queries with index % N == I so several accounts can share the work. One worker per GPU.
Output: rerank.parquet (qid, doc_id, score, best_passage)
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
PASSAGE_CHARS = 1500


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def split_passages(title: str, text: str) -> list[str]:
    pieces = []
    for para in text.split("\n"):
        while len(para) > PASSAGE_CHARS:
            pieces.append(para[:PASSAGE_CHARS])
            para = para[PASSAGE_CHARS:]
        if para.strip():
            pieces.append(para)
    out, buf = [], ""
    for p in pieces:
        if buf and len(buf) + len(p) > PASSAGE_CHARS:
            out.append(buf)
            buf = ""
        buf = f"{buf}\n{p}" if buf else p
    if buf or not out:
        out.append(buf)
    return [f"{title}\n{p}".strip() for p in out]


def load_inputs(args):
    if not args.cands.exists():  # bare file name: look under the Kaggle inputs
        args.cands = Path(glob.glob(f"/kaggle/input/**/{args.cands.name}", recursive=True)[0])
    cands = pd.read_parquet(args.cands)
    qids = sorted(cands["qid"].unique())
    i, n = (int(x) for x in args.part.split("/"))
    keep = set(qids[i::n][: args.max_queries] if args.max_queries else qids[i::n])
    cands = cands[cands["qid"].isin(keep)]
    queries = pd.read_parquet(glob.glob(f"{args.data}/**/query.parquet", recursive=True)[0])
    qtext = dict(zip(queries["id"].astype(int), queries["query"]))
    need = set(cands["doc_id"].astype(int))
    docs = {}
    for part in sorted(glob.glob(f"{args.data}/**/part-*.jsonl*", recursive=True)):
        opener = gzip.open if part.endswith(".gz") else open
        with opener(part, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["id"] in need:
                    docs[d["id"]] = split_passages(d["t"], d["x"])[: args.max_passages]
    return cands, qtext, docs


def worker(args) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    cands, qtext, docs = load_inputs(args)
    g, n = args.worker, args.ngpu
    qids = sorted(cands["qid"].unique())[g::n]
    cands = cands[cands["qid"].isin(set(qids))]
    pairs = [(int(q), int(d), k, qtext[int(q)], p)
             for q, d in zip(cands["qid"], cands["doc_id"]) if int(d) in docs
             for k, p in enumerate(docs[int(d)])]
    log(f"worker {g}: {len(qids)} queries, {len(pairs):,} pairs")
    device = f"cuda:{g}" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=torch.float16 if device != "cpu" else torch.float32).to(device).eval()
    order = np.argsort([-len(p[4]) for p in pairs], kind="stable")
    scores = np.empty(len(pairs), dtype=np.float32)
    for lo in range(0, len(order), args.batch):
        idx = order[lo:lo + args.batch]
        enc = tok([pairs[i][3] for i in idx], [pairs[i][4] for i in idx], padding=True, truncation="only_second",
                  max_length=args.max_len, return_tensors="pt").to(device)
        with torch.inference_mode():
            scores[idx] = model(**enc).logits[:, 0].float().cpu().numpy()
        if (lo // args.batch) % 200 == 0:
            log(f"worker {g}: {lo + len(idx):,}/{len(pairs):,}")
    df = pd.DataFrame({"qid": [p[0] for p in pairs], "doc_id": [p[1] for p in pairs],
                       "passage": [p[2] for p in pairs], "score": scores})
    best = df.loc[df.groupby(["qid", "doc_id"])["score"].idxmax()]
    best.rename(columns={"passage": "best_passage"}).to_parquet(args.out / f"rerank_w{g}.parquet", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--max-passages", type=int, default=None,
                    help="cap on passages scored per doc; default None = whole document (user: no character limits)")
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--ngpu", type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker >= 0:
        worker(args)
        return
    import torch
    n = max(1, torch.cuda.device_count())
    procs = [subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--worker", str(g), "--ngpu", str(n)])
             for g in range(n)]
    if any(p.wait() != 0 for p in procs):
        raise SystemExit("a worker failed")
    parts = [args.out / f"rerank_w{g}.parquet" for g in range(n)]
    pd.concat([pd.read_parquet(p) for p in parts]).to_parquet(args.out / "rerank.parquet", index=False)
    for p in parts:
        p.unlink()
    log("done")


if __name__ == "__main__":
    main()
