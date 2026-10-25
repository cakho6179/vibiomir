"""Dense search over saved passage embeddings (dense_embed.py --save-emb) for query variants.

    python dense_search.py --model BAAI/bge-m3 --emb "/kaggle/input/**/emb.npy" --variants query_variants.parquet \
                           --out DIR [--only orig,sub0,hyde,zh_q] [--topk 3000] [--query-prompt "..."]

Each emb.npy sits next to its ids.npy (owning doc per passage). Every embedding file is loaded to the GPU in turn
and scored against all selected variants; document score = best passage. Shards hold disjoint documents, so the
per-file top-k lists merge exactly. --only keeps those variant names (default: all rows). bge-m3 is multilingual,
so a Vietnamese variant may search ZH/EN embeddings as well. Outputs in OUT:
  ranking.parquet (qid, variant, rank, doc_id, score), q_emb.npy (variant order of variants.parquet), stats.json
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def top_docs(q, emb, ids, topk: int, device: str, batch: int = 64) -> list[list[tuple[int, float]]]:
    """Per query row: up to topk (doc_id, score) by best passage, from one embedding file."""
    import torch
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    docs_t = torch.from_numpy(emb).to(device=device, dtype=dtype)
    k = min(topk * 10, len(ids))  # passages; doc score = best passage
    out = []
    for lo in range(0, len(q), batch):
        qt = torch.from_numpy(q[lo:lo + batch]).to(device=device, dtype=dtype)
        scores, idx = torch.topk(qt @ docs_t.T, k, dim=1)
        scores, idx = scores.float().cpu().numpy(), idx.cpu().numpy()
        for j in range(len(qt)):
            best: dict[int, float] = {}
            for i, s in zip(idx[j], scores[j]):
                doc = int(ids[i])
                if doc not in best:
                    best[doc] = float(s)
                    if len(best) >= topk:
                        break
            out.append(list(best.items()))
    del docs_t
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def merge(variants: pd.DataFrame, per_file: list[list[list[tuple[int, float]]]], topk: int) -> pd.DataFrame:
    rows = []
    for r, (qid, name) in enumerate(zip(variants["qid"].astype(int), variants["variant"].astype(str))):
        hits = sorted((h for f in per_file for h in f[r]), key=lambda h: -h[1])[:topk]
        rows += [(qid, name, rank, doc, s) for rank, (doc, s) in enumerate(hits)]
    return pd.DataFrame(rows, columns=["qid", "variant", "rank", "doc_id", "score"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--emb", required=True, help="glob of emb.npy files (recursive **)")
    ap.add_argument("--variants", required=True, help="query_variants.parquet, or a glob finding it")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="", help="comma list of variant names to search")
    ap.add_argument("--topk", type=int, default=3000)
    ap.add_argument("--query-prompt", default="")
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    variants = pd.read_parquet(sorted(glob.glob(args.variants, recursive=True))[0])
    if args.only:
        variants = variants[variants["variant"].isin(args.only.split(","))].reset_index(drop=True)
    kwargs = {"torch_dtype": torch.float16} if device.startswith("cuda") else {}
    model = SentenceTransformer(args.model, device=device, model_kwargs=kwargs)
    model.max_seq_length = args.max_len
    q = model.encode(variants["text"].astype(str).tolist(), prompt=args.query_prompt or None, batch_size=64,
                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float16)
    np.save(args.out / "q_emb.npy", q)
    del model
    log(f"encoded {len(q)} variant rows")

    files = sorted(glob.glob(args.emb, recursive=True))
    if not files:
        raise SystemExit(f"no embeddings match {args.emb}")
    per_file, passages = [], 0
    for f in files:
        emb, ids = np.load(f), np.load(Path(f).with_name("ids.npy"))
        per_file.append(top_docs(q, emb, ids, args.topk, device))
        passages += len(ids)
        log(f"{f}: {emb.shape}")
        del emb, ids
    ranking = merge(variants, per_file, args.topk)
    ranking.to_parquet(args.out / "ranking.parquet", index=False)
    (args.out / "stats.json").write_text(json.dumps({"model": args.model, "files": files, "passages": passages,
                                                     "rows": len(variants), "seconds": round(time.time() - T0)}))
    log("done")


if __name__ == "__main__":
    main()
