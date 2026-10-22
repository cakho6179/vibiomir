"""Dense retrieval on Kaggle GPU(s): embed full documents as ~1500-char passages + queries;
exact cosine search, document score = best passage.

    python dense_embed.py --model BAAI/bge-m3 --data DIR --out DIR [--shards 0,1,2,3]
                          [--query-prompt "..."] [--max-len 512] [--max-docs N] [--max-queries N]

DIR holds corpus/part-XX.jsonl.gz ({"id","t","x"}) and query.parquet. One worker process per
GPU (sentence-transformers, fp16); the parent merges and scores. Outputs in OUT:
  q_emb.npy, ranking.parquet (qid, rank, doc_id, score), stats.json
  [--save-emb] emb.npy (float16 passage vectors, L2-normalised) + ids.npy (owning doc per passage)
The ranking covers only the shards given, so rankings from several kernels merge exactly.
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


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


PASSAGE_CHARS = 1500  # ~400-500 tokens: fits max_len 512 with the title prefix


def split_passages(title: str, text: str) -> list[str]:
    """Whole document -> passages of <= PASSAGE_CHARS (paragraph-aligned), each prefixed by the title."""
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


VI_MAX_ID = 766441  # official id blocks: Vietnamese ids come first


def overlapping_passages(title: str, text: str, size: int, overlap: int, with_title: bool) -> list[str]:
    """Paragraph-packed passages of <= size chars, each starting with the last `overlap` chars of
    the previous one (cut at a space), like LangChain's RecursiveCharacterTextSplitter."""
    pieces = []
    for para in text.split("\n"):
        while len(para) > size:
            cut = para.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            pieces.append(para[:cut])
            para = para[cut:].lstrip()
        if para.strip():
            pieces.append(para)
    out, buf = [], ""
    for p in pieces:
        if buf and len(buf) + len(p) + 1 > size:
            out.append(buf)
            tail = buf[-overlap:] if overlap else ""
            sp_ = tail.find(" ")
            buf = tail[sp_ + 1:] if 0 <= sp_ < len(tail) - 1 else tail
        buf = f"{buf}\n{p}" if buf else p
    if buf or not out:
        out.append(buf)
    return [f"{title}\n{p}".strip() if with_title else (p.strip() or title) for p in out]


CHUNKING: dict = {}  # set from the command line: size, overlap, with_title, vi_only


def passages_of(title: str, text: str) -> list[str]:
    if CHUNKING.get("size"):
        return overlapping_passages(title, text, CHUNKING["size"], CHUNKING["overlap"], CHUNKING["with_title"])
    return split_passages(title, text)


def load_docs(data: Path, shards: list[int], max_docs: int | None,
              parts_glob: str | None = None) -> tuple[np.ndarray, list[str]]:
    """Return (owning doc id per passage, passage texts) for the given shards, full text."""
    if parts_glob:  # explicit file set(s), comma-separated, e.g. "**/part-00.jsonl*,**/part-d1-*.jsonl*"
        files = sorted({Path(p) for g in parts_glob.split(",") for p in glob.glob(f"{data}/{g}", recursive=True)})
    else:
        corpus = Path(glob.glob(f"{data}/**/part-00.jsonl*", recursive=True)[0]).parent
        files = [next(corpus.glob(f"part-{s:02d}.jsonl*")) for s in shards]  # Kaggle stores .gz decompressed
    owner, texts, n = [], [], 0
    for part in files:
        opener = gzip.open if part.suffix == ".gz" else open
        with opener(part, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if CHUNKING.get("vi_only") and d["id"] > VI_MAX_ID:
                    continue
                ps = passages_of(d["t"], d["x"])
                owner.extend([d["id"]] * len(ps))
                texts.extend(ps)
                n += 1
                if max_docs and n >= max_docs:
                    return np.asarray(owner, dtype=np.int64), texts
    return np.asarray(owner, dtype=np.int64), texts


def load_model(name: str, device: str, max_len: int):
    import torch
    from sentence_transformers import SentenceTransformer
    kwargs = {"torch_dtype": torch.float16} if device.startswith("cuda") else {}
    model = SentenceTransformer(name, device=device, model_kwargs=kwargs, trust_remote_code=True)
    model.max_seq_length = max_len
    return model


def worker(args: argparse.Namespace) -> None:
    """Embed rows g::n of the doc list on GPU g."""
    ids, texts = load_docs(args.data, args.shard_list, args.max_docs, args.parts)
    g, n = args.worker, args.ngpu
    device = f"cuda:{g}" if args.ngpu_real else "cpu"
    model = load_model(args.model, device, args.max_len)
    part = texts[g::n]
    log(f"worker {g}: {len(part):,} docs on {device}")
    out =np.empty((len(part), model.get_sentence_embedding_dimension()), dtype=np.float16)
    step = 20_000
    for lo in range(0, len(part), step):
        out[lo:lo + step] = model.encode(part[lo:lo + step], batch_size=args.batch, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        log(f"worker {g}: {min(lo + step, len(part)):,}/{len(part):,}")
    np.save(args.out / f"emb_w{g}.npy", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shards", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--parts", default=None, help="glob under --data instead of --shards, e.g. **/part-d1-*.jsonl*")
    ap.add_argument("--query-prompt", default="")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--save-emb", action="store_true", help="also write passage embeddings (large)")
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--ngpu", type=int, default=0)
    ap.add_argument("--chunk-chars", type=int, default=0,
                    help="> 0: overlapping passages instead of 1500-char ones; 4500 chars needs --max-len 1536 "
                         "(measured: <= 1518 BGE-M3 tokens, 3.72 chars/token on Vietnamese)")
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--no-title", action="store_true", help="do not prefix passages with the title")
    ap.add_argument("--vi-only", action="store_true", help="only Vietnamese ids (<= 766441)")
    args = ap.parse_args()
    CHUNKING.update(size=args.chunk_chars, overlap=args.overlap, with_title=not args.no_title, vi_only=args.vi_only)
    args.shard_list = [int(s) for s in args.shards.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    args.ngpu_real = torch.cuda.device_count()
    if args.worker >= 0:
        worker(args)
        return

    n = max(1, args.ngpu_real)
    log(f"{args.model}: {n} worker(s), shards {args.shard_list}")
    base = [sys.executable, __file__] + [a for a in sys.argv[1:]]
    procs = [subprocess.Popen(base + ["--worker", str(g), "--ngpu", str(n)]) for g in range(n)]
    if any(p.wait() != 0 for p in procs):
        raise SystemExit("a worker failed")

    ids, _ = load_docs(args.data, args.shard_list, args.max_docs, args.parts)  # owning doc id per passage
    parts = [np.load(args.out / f"emb_w{g}.npy") for g in range(n)]
    emb = np.empty((len(ids), parts[0].shape[1]), dtype=np.float16)
    for g in range(n):
        emb[g::n] = parts[g]
        (args.out / f"emb_w{g}.npy").unlink()
    if args.save_emb:
        np.save(args.out / "emb.npy", emb)
        np.save(args.out / "ids.npy", ids)
    log(f"passage embeddings {emb.shape} from {len(np.unique(ids)):,} docs")

    queries = pd.read_parquet(glob.glob(f"{args.data}/**/query.parquet", recursive=True)[0])
    if args.max_queries:
        queries = queries.head(args.max_queries)
    device = "cuda:0" if args.ngpu_real else "cpu"
    model = load_model(args.model, device, args.max_len)
    q = model.encode(queries["query"].tolist(), prompt=args.query_prompt or None, batch_size=64,
                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float16)
    np.save(args.out / "q_emb.npy", q)
    del model

    dtype = torch.float16 if args.ngpu_real else torch.float32
    docs_t = torch.from_numpy(emb).to(device=device, dtype=dtype)
    k = min(args.topk * 10, len(ids))  # passages; doc score = best passage
    rows = []
    for lo in range(0, len(q), 64):
        qt = torch.from_numpy(q[lo:lo + 64]).to(device=device, dtype=dtype)
        scores, idx = torch.topk(qt @ docs_t.T, k, dim=1)
        scores, idx = scores.float().cpu().numpy(), idx.cpu().numpy()
        for j in range(len(qt)):
            qid = int(queries["id"].iloc[lo + j])
            seen: set[int] = set()
            for i, s in zip(idx[j], scores[j]):
                doc = int(ids[i])
                if doc not in seen:
                    rows.append((qid, len(seen), doc, float(s)))
                    seen.add(doc)
                    if len(seen) >= args.topk:
                        break
        log(f"scored {min(lo + 64, len(q))}/{len(q)} queries")
    pd.DataFrame(rows, columns=["qid", "rank", "doc_id", "score"]).to_parquet(args.out / "ranking.parquet", index=False)
    (args.out / "stats.json").write_text(json.dumps({"model": args.model, "docs": len(ids), "shards": args.shard_list,
                                                     "dim": int(emb.shape[1]), "seconds": round(time.time() - T0)}))
    log("done")


if __name__ == "__main__":
    main()
