"""doc2query dense score over head + tail questions: per (query, doc) the max cosine of the query vs the doc's questions.

Inputs: the outputs of kernels vibiomir-dq-emb (head) and vibiomir-dq-emb-tail (tail), each emb.npy / meta.parquet /
q_emb.npy / q_ids.npy. Writes /kaggle/working/scores.parquet (qid, doc_id, dq_score, dq_rank), top 1000 docs per query,
the same layout as scripts/analysis/doc2query_score.py.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

dirs = sorted({str(Path(p).parent) for p in glob.glob("/kaggle/input/**/emb.npy", recursive=True)})
print(dirs, flush=True)
assert len(dirs) == 2, dirs
E, docs, qe, qids = [], [], None, None
for d in dirs:
    E.append(np.load(f"{d}/emb.npy"))
    docs.append(pd.read_parquet(f"{d}/meta.parquet", columns=["doc_id"])["doc_id"].to_numpy())
    q, i = np.load(f"{d}/q_emb.npy"), np.load(f"{d}/q_ids.npy")
    if qids is None:
        qe, qids = q, i
    else:
        assert (qids == i).all()
        print("query emb max abs diff", float(np.abs(q.astype(np.float32) - qe.astype(np.float32)).max()), flush=True)
E = np.concatenate(E)
docs = np.concatenate(docs)
udocs, inv = np.unique(docs, return_inverse=True)
print(f"{len(E):,} questions, {len(udocs):,} docs", flush=True)

dev = "cuda"
Et = torch.from_numpy(E).to(dev)
Qt = torch.from_numpy(qe).to(dev)
inv_t = torch.from_numpy(inv).to(dev)
TOPK = 1000
rows = []
for lo in range(0, len(Qt), 50):
    s = Qt[lo:lo + 50] @ Et.T  # fp16
    best = torch.full((s.shape[0], len(udocs)), -1.0, device=dev, dtype=torch.float32)
    best.scatter_reduce_(1, inv_t.expand(s.shape[0], -1), s.float(), reduce="amax", include_self=True)
    val, top = best.topk(TOPK, dim=1)
    val, top = val.cpu().numpy(), top.cpu().numpy()
    for k in range(len(top)):
        rows.append(pd.DataFrame({"qid": int(qids[lo + k]), "doc_id": udocs[top[k]], "dq_score": val[k],
                                  "dq_rank": np.arange(TOPK)}))
    print(lo, flush=True)
sc = pd.concat(rows, ignore_index=True)
sc.to_parquet("/kaggle/working/scores.parquet", index=False)
Path("/kaggle/working/stats.json").write_text(json.dumps({"questions": int(len(E)), "docs": int(len(udocs))}))
print("done", len(sc), flush=True)
