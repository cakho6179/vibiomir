"""BGE-M3 dense + sparse (lexical weights) + multi-vector (ColBERT) scores of candidate pairs, every passage (MaxP).

    python m3_multi.py --data DIR --cands cands.parquet --out DIR [--cut vi800,zh200,en200] [--part I/N]

Same inputs as rerank.py (docstore parts + query.parquet under --data, cands.parquet of (qid, doc_id, set, rank)).
Passages are those of rerank.split_passages (~1500 chars, title on each, whole document). Every passage of a doc
is encoded once and scored against every query whose pool holds the doc; a pair keeps the max over its passages
of each score and the passage that gave it. The heads follow the reference model (FlagEmbedding BGEM3FlagModel):
dense = CLS, L2-normalised, dot product; sparse = relu(sparse_linear(h)) per token, max per token id, special
tokens dropped, score = sum over shared ids of w_q * w_p; colbert = normalise(colbert_linear(h[1:])), score =
mean over query tokens of the max similarity over passage tokens. --part I/N keeps docs with doc_id % N == I
(each doc is encoded by one kernel only); one worker per GPU.
Output: m3multi.parquet (qid, doc_id, dense, sparse, colbert, p_dense, p_sparse, p_colbert, n_passages)
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from rerank import find_input, split_passages

T0 = time.time()
SPECIAL = {0, 1, 2, 3}  # XLM-R <s> <pad> </s> <unk>


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def sparse_dict(ids: np.ndarray, w: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = {}
    for t, x in zip(ids.tolist(), w.tolist()):
        if t not in SPECIAL and x > out.get(t, 0.0):
            out[t] = x
    return out


def load(args, g: int, n: int):
    """Cands of this kernel part and GPU worker, query texts, passages of the docs."""
    cands = pd.read_parquet(args.cands if args.cands.exists() else find_input(args.cands.name))
    if args.cut:
        cut = {s[:2]: int(s[2:]) for s in args.cut.split(",")}
        cands = cands[cands["rank"] < cands["set"].str[:2].map(cut)]
    i, k = (int(x) for x in args.part.split("/"))
    cands = cands[cands["doc_id"] % k == i]
    docs_all = np.sort(cands["doc_id"].unique())
    cands = cands[cands["doc_id"].isin(set(docs_all[g::n]))]
    queries = pd.read_parquet(glob.glob(f"{args.data}/**/query.parquet", recursive=True)[0])
    qtext = dict(zip(queries["id"].astype(int), queries["query"]))
    need = set(cands["doc_id"].astype(int))
    docs = {}
    for part in sorted(glob.glob(f"{args.data}/**/part-*.jsonl*", recursive=True)):
        opener = gzip.open if part.endswith(".gz") else open
        with opener(part, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["id"] in need and d["id"] not in docs:
                    docs[d["id"]] = split_passages(d["t"], d["x"])
    by_doc = defaultdict(list)
    for q, d in zip(cands["qid"].astype(int), cands["doc_id"].astype(int)):
        by_doc[d].append(q)
    return by_doc, qtext, docs


def worker(args) -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoTokenizer
    g, n = args.worker, args.ngpu
    by_doc, qtext, docs = load(args, g, n)
    qids = sorted({q for qs in by_doc.values() for q in qs})
    flat = [(d, k, p) for d in by_doc if d in docs for k, p in enumerate(docs[d])]
    log(f"worker {g}: {len(by_doc):,} docs, {len(flat):,} passages, {sum(map(len, by_doc.values())):,} pairs")
    device = f"cuda:{g}" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    colbert = torch.nn.Linear(1024, 1024)
    colbert.load_state_dict(torch.load(hf_hub_download(args.model, "colbert_linear.pt"), map_location="cpu"))
    sparse = torch.nn.Linear(1024, 1)
    sparse.load_state_dict(torch.load(hf_hub_download(args.model, "sparse_linear.pt"), map_location="cpu"))
    colbert, sparse = colbert.to(device, dtype).eval(), sparse.to(device, dtype).eval()

    def encode(texts: list[str], max_len: int):
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.inference_mode():
            h = model(**enc).last_hidden_state
            dense = torch.nn.functional.normalize(h[:, 0].float(), dim=-1)
            sw = torch.relu(sparse(h)).squeeze(-1).float()
            mask = enc["attention_mask"][:, 1:].float()
            cv = torch.nn.functional.normalize(colbert(h[:, 1:]).float(), dim=-1) * mask[..., None]
        return dense, sw.cpu().numpy(), cv, mask, enc["input_ids"].cpu().numpy()

    qenc = {}
    for lo in range(0, len(qids), 64):
        batch = qids[lo:lo + 64]
        dense, sw, cv, mask, ids = encode([qtext[q] for q in batch], 128)
        for i, q in enumerate(batch):
            qenc[q] = (dense[i], sparse_dict(ids[i], sw[i]), cv[i, :int(mask[i].sum())])
    best: dict[tuple[int, int], list[float]] = {}  # (q, d) -> [dense, sparse, colbert, p_dense, p_sparse, p_colbert]
    order = np.argsort([-len(p[2]) for p in flat], kind="stable")
    for lo in range(0, len(order), args.batch):
        idx = order[lo:lo + args.batch]
        dense, sw, cv, mask, ids = encode([flat[i][2] for i in idx], args.max_len)
        for j, i in enumerate(idx):
            d, k, _ = flat[i]
            ps = sparse_dict(ids[j], sw[j])
            pc = cv[j, :int(mask[j].sum())]
            for q in by_doc[d]:
                qd, qs, qc = qenc[q]
                s = (float(qd @ dense[j]), sum(w * ps[t] for t, w in qs.items() if t in ps),
                     float((qc @ pc.T).max(dim=-1).values.mean()))
                cur = best.setdefault((q, d), [-1e9, -1e9, -1e9, -1, -1, -1])
                for m in range(3):
                    if s[m] > cur[m]:
                        cur[m], cur[3 + m] = s[m], k
        if (lo // args.batch) % 200 == 0:
            log(f"worker {g}: {lo + len(idx):,}/{len(flat):,} passages")
    rows = [(q, d, *v, len(docs[d])) for (q, d), v in best.items()]
    df = pd.DataFrame(rows, columns=["qid", "doc_id", "dense", "sparse", "colbert", "p_dense", "p_sparse",
                                     "p_colbert", "n_passages"])
    df.to_parquet(args.out / f"m3multi_w{g}.parquet", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--cut", default="", help="e.g. vi800,zh200,en200: keep cands with rank < N in their set")
    ap.add_argument("--part", default="0/1", help="I/N: docs with doc_id %% N == I")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
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
    parts = [args.out / f"m3multi_w{g}.parquet" for g in range(n)]
    pd.concat([pd.read_parquet(p) for p in parts]).to_parquet(args.out / "m3multi.parquet", index=False)
    for p in parts:
        p.unlink()
    log("done")


if __name__ == "__main__":
    main()
