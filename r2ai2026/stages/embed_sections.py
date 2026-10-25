"""Embed docstore sections once and keep the vectors for cheap query experiments.

    python embed_sections.py --data DIR --out DIR [--model BAAI/bge-m3] [--max-len 1536]

DIR is searched for sections.parquet and part-*.jsonl* (docstore v1). Each GPU worker
takes a round-robin subset of the corpus part files and writes
  emb_w<g>.npy   float16 (n_sections, dim), L2-normalised
  ids_w<g>.npy   int64 (n_sections, 2): doc_id, section id
  stats_w<g>.json
The parent only launches workers and writes manifest.json; there is no giant merge step.
Query time embeds 1,200 queries and multiplies against these shards (minutes, not hours).
Source is ASCII-only (Kaggle decodes kernel source badly otherwise).
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


def find_docstore(data: Path) -> Path:
    hits = sorted(glob.glob(f"{data}/**/sections.parquet", recursive=True))
    if not hits:
        raise SystemExit(f"sections.parquet not found under {data}")
    return Path(hits[0]).parent


def corpus_parts(root: Path, pattern: str) -> list[Path]:
    parts = sorted(Path(p) for p in glob.glob(f"{root}/{pattern}"))
    if not parts:
        parts = sorted(Path(p) for p in glob.glob(f"{root}/**/{pattern}", recursive=True))
    if not parts:
        raise SystemExit(f"no corpus parts matching {pattern} under {root}")
    return parts


def section_map(root: Path) -> dict[int, list[tuple[int, int, int, str]]]:
    sections = pd.read_parquet(root / "sections.parquet", columns=["doc_id", "sid", "start", "end", "heading"])
    out: dict[int, list[tuple[int, int, int, str]]] = {}
    for row in sections.itertuples(index=False):
        out.setdefault(int(row.doc_id), []).append(
            (int(row.sid), int(row.start), int(row.end), str(row.heading or ""))
        )
    for items in out.values():
        items.sort()
    return out


def iter_units(root: Path, parts: list[Path], max_chars: int, max_sections: int = 0):
    by_doc = section_map(root)
    n, docs = 0, 0
    for part in parts:
        opener = gzip.open if part.suffix == ".gz" else open
        with opener(part, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                did = int(d["id"])
                items = by_doc.pop(did, None)
                if not items:
                    continue
                title = str(d.get("t") or "")
                body = str(d.get("x") or "")
                for sid, start, end, heading in items:
                    text = "\n".join(x for x in (title, heading, body[start:end]) if x).strip()
                    yield did, sid, text[:max_chars]
                    n += 1
                    if max_sections and n >= max_sections:
                        log(f"worker limit: {n:,} sections")
                        return
                docs += 1
                if docs % 50_000 == 0:
                    log(f"read {docs:,} docs / {n:,} sections")


def load_model(name: str, device: str, max_len: int):
    import torch
    from sentence_transformers import SentenceTransformer
    kwargs = {"torch_dtype": torch.float16} if device.startswith("cuda") else {}
    model = SentenceTransformer(name, device=device, model_kwargs=kwargs, trust_remote_code=True)
    model.max_seq_length = max_len
    return model


def worker(args: argparse.Namespace) -> None:
    import torch
    root = find_docstore(args.data)
    parts = corpus_parts(root, args.parts)
    mine = parts[args.worker::args.ngpu]
    if args.dry_run:
        n, docs = 0, set()
        for did, _sid, _text in iter_units(root, mine, args.max_chars, args.max_sections):
            n += 1
            docs.add(did)
        stats = {"worker": args.worker, "parts": [p.name for p in mine], "sections": n,
                 "docs": len(docs), "dry_run": True, "seconds": round(time.time() - T0)}
        (args.out / f"dry_w{args.worker}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        log(f"worker {args.worker}: dry run {stats}")
        return
    device = f"cuda:{args.worker}" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device, args.max_len)
    dim = model.get_sentence_embedding_dimension()
    log(f"worker {args.worker}: {len(mine)} parts on {device}, dim {dim}")

    embs, keys, buf, buf_keys = [], [], [], []
    n = 0
    for did, sid, text in iter_units(root, mine, args.max_chars, args.max_sections):
        buf.append(text)
        buf_keys.append((did, sid))
        if len(buf) >= args.batch * 8:
            arr = model.encode(buf, batch_size=args.batch, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
            embs.append(arr.astype(np.float16))
            keys.append(np.asarray(buf_keys, dtype=np.int64))
            n += len(buf)
            if n % (args.batch * 32) == 0:
                log(f"worker {args.worker}: {n:,} sections")
            buf, buf_keys = [], []
    if buf:
        arr = model.encode(buf, batch_size=args.batch, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        embs.append(arr.astype(np.float16))
        keys.append(np.asarray(buf_keys, dtype=np.int64))
        n += len(buf)
    emb = np.concatenate(embs, axis=0) if embs else np.zeros((0, dim), dtype=np.float16)
    ids = np.concatenate(keys, axis=0) if keys else np.zeros((0, 2), dtype=np.int64)
    np.save(args.out / f"emb_w{args.worker}.npy", emb)
    np.save(args.out / f"ids_w{args.worker}.npy", ids)
    stats = {"worker": args.worker, "parts": [p.name for p in mine], "sections": int(emb.shape[0]),
             "dim": int(emb.shape[1]), "seconds": round(time.time() - T0)}
    (args.out / f"stats_w{args.worker}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    log(f"worker {args.worker}: saved {emb.shape}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--parts", default="part-*.jsonl*")
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--max-chars", type=int, default=5600)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-sections", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="count sections and check the path; do not load a model")
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--ngpu", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker >= 0:
        worker(args)
        return

    import torch
    n = max(1, torch.cuda.device_count())
    root = find_docstore(args.data)
    parts = corpus_parts(root, args.parts)
    log(f"{args.model}: {n} worker(s), {len(parts)} parts, {root}")
    base = [sys.executable, __file__] + [a for a in sys.argv[1:]]
    procs = [subprocess.Popen(base + ["--worker", str(g), "--ngpu", str(n)]) for g in range(n)]
    if any(p.wait() != 0 for p in procs):
        raise SystemExit("a worker failed")
    manifest = {"model": args.model, "max_len": args.max_len, "max_chars": args.max_chars,
                "workers": n, "parts": [p.name for p in parts], "seconds": round(time.time() - T0)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    for g in range(n):
        p = args.out / f"stats_w{g}.json"
        if p.exists():
            log(p.read_text(encoding="utf-8"))
    log("done")


if __name__ == "__main__":
    main()
