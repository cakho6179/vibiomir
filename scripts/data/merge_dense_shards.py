"""Merge disjoint ``dense_embed.py --save-emb`` shards into one passage index.

    python scripts/data/merge_dense_shards.py --lang vi --out INDEX/dense/vi SHARD_DIR [SHARD_DIR ...]

Each shard directory holds ``emb.npy`` (passages x dim) and ``ids.npy`` (owning
document per passage). Shards must cover disjoint documents; this is checked
before writing. The merged files are float16 (or the input dtype) and keep the
same order, so a later ``dense_search.py --emb .../emb.npy`` scores the whole
language without re-reading the corpus.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, help="language label stored in meta.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect-docs", type=int, default=None)
    ap.add_argument("--expect-passages", type=int, default=None)
    ap.add_argument("shards", nargs="+", type=Path)
    args = ap.parse_args()

    files = []
    for d in args.shards:
        emb, ids = d / "emb.npy", d / "ids.npy"
        if not emb.exists() or not ids.exists():
            raise SystemExit(f"{d}: need emb.npy and ids.npy")
        files.append((emb, ids))

    shapes, docs = [], []
    dim = None
    dtype = None
    for emb_path, ids_path in files:
        emb = np.load(emb_path, mmap_mode="r")
        ids = np.load(ids_path)
        if emb.ndim != 2 or ids.ndim != 1 or emb.shape[0] != ids.shape[0]:
            raise SystemExit(f"{emb_path}: bad shapes {emb.shape} / {ids.shape}")
        if dim is None:
            dim, dtype = emb.shape[1], emb.dtype
        elif emb.shape[1] != dim:
            raise SystemExit(f"{emb_path}: dim {emb.shape[1]} != {dim}")
        elif emb.dtype != dtype:
            raise SystemExit(f"{emb_path}: dtype {emb.dtype} != {dtype}")
        uniq = np.unique(ids)
        docs.append(set(int(x) for x in uniq))
        shapes.append(emb.shape[0])
        log(f"{emb_path.parent.name}: {emb.shape[0]:,} passages, {len(uniq):,} docs")

    for i, left in enumerate(docs):
        for right in docs[i + 1:]:
            overlap = left & right
            if overlap:
                raise SystemExit(f"shards overlap on {len(overlap):,} documents, e.g. {min(overlap)}")

    total = sum(shapes)
    all_docs = set().union(*docs)
    if args.expect_passages is not None and total != args.expect_passages:
        raise SystemExit(f"passages {total:,} != expected {args.expect_passages:,}")
    if args.expect_docs is not None and len(all_docs) != args.expect_docs:
        raise SystemExit(f"docs {len(all_docs):,} != expected {args.expect_docs:,}")

    args.out.mkdir(parents=True, exist_ok=True)
    out_emb = args.out / "emb.npy"
    merged = np.lib.format.open_memmap(out_emb, mode="w+", dtype=dtype, shape=(total, dim))
    offset = 0
    ids_parts = []
    for emb_path, ids_path in files:
        emb = np.load(emb_path, mmap_mode="r")
        merged[offset:offset + emb.shape[0]] = emb
        offset += emb.shape[0]
        ids_parts.append(np.load(ids_path))
        log(f"copied {emb_path.parent.name} -> {offset:,}/{total:,}")
    merged.flush()
    del merged
    np.save(args.out / "ids.npy", np.concatenate(ids_parts))

    meta = {
        "lang": args.lang,
        "dim": int(dim),
        "dtype": str(dtype),
        "passages": int(total),
        "docs": len(all_docs),
        "shards": [str(p) for p in args.shards],
        "seconds": round(time.time() - T0),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote {out_emb} ({total:,} passages, {len(all_docs):,} docs, dim {dim})")


if __name__ == "__main__":
    main()
