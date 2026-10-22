"""Embed corpus documents or queries with a multilingual open-weight encoder.

Writes fp16 embeddings next to an int64 id array, one shard per ``--shard-size``
documents, so a run over a corpus that is still being crawled can stop and resume:
every shard is written to temporary files and renamed only when complete, and
already-embedded ids are skipped on the next start.

The dense index is what lets a Vietnamese query reach the English and Chinese
half of the corpus, which a monolingual BM25 index cannot do.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402


def load_id_set(directory: Path) -> set[int]:
    ids: set[int] = set()
    for ids_path in sorted(directory.glob("shard_*/ids.npy")):
        array = np.load(ids_path, mmap_mode="r")
        ids.update(int(value) for value in array)
    return ids


def write_shard(directory: Path, index: int, ids: list[int], vectors: np.ndarray) -> None:
    """Write one completed shard; ``os.replace`` keeps half-written shards invisible."""
    shard = directory / f"shard_{index:05d}"
    shard.mkdir(parents=True, exist_ok=True)
    ids_tmp = shard / "ids.npy.tmp"
    emb_tmp = shard / "emb.npy.tmp"
    np.save(ids_tmp, np.asarray(ids, dtype=np.int64))
    np.save(emb_tmp, vectors.astype(np.float16))
    os.replace(ids_tmp, shard / "ids.npy")
    os.replace(emb_tmp, shard / "emb.npy")


def iter_documents(args: argparse.Namespace):
    from r2ai2026.pipeline import document_text, load_documents

    kinds = {value.strip() for value in args.kinds.split(",") if value.strip()} if args.kinds else None
    for record in load_documents(args.input):
        if not record.get("ok"):
            continue
        if kinds is not None and str(record.get("kind")) not in kinds:
            continue
        text = document_text(record).strip()
        if len(text) < args.min_chars:
            continue
        yield int(record["id"]), text


def iter_queries(args: argparse.Namespace):
    import pandas as pd

    frame = pd.read_parquet(args.input)
    text_column = next((column for column in frame.columns if column != "id"), frame.columns[-1])
    for row in frame.itertuples(index=False):
        text = str(getattr(row, text_column)).strip()
        if text:
            yield int(row.id), text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["docs", "queries"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory for ids.npy/emb.npy shards")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--min-chars", type=int, default=80, help="documents shorter than this are not indexed")
    parser.add_argument("--kinds", default=None, help="comma separated kind filter (docs mode only)")
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10, help="report every N batches")
    args = parser.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    args.out.mkdir(parents=True, exist_ok=True)
    done = load_id_set(args.out) if args.mode == "docs" else set()
    print(f"[embed] {args.mode}: {len(done):,} ids already stored in {args.out}")

    torch.set_grad_enabled(False)
    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_tokens
    if args.dtype == "float16":
        model.half()
    dimension = int(model.get_sentence_embedding_dimension())
    print(f"[embed] {args.model} on {args.device} ({args.dtype}, dim={dimension}, max_tokens={args.max_tokens})")

    source = iter_documents(args) if args.mode == "docs" else iter_queries(args)
    if args.mode == "queries":
        ids: list[int] = []
        texts: list[str] = []
        for doc_id, text in source:
            if args.limit is not None and len(ids) >= args.limit:
                break
            ids.append(doc_id)
            texts.append(text)
        vectors = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        np.save(args.out / "ids.npy", np.asarray(ids, dtype=np.int64))
        np.save(args.out / "emb.npy", vectors.astype(np.float16))
        print(f"[embed] wrote {len(ids)} query vectors to {args.out}")
        return 0

    shard_index = max([int(path.name.split("_")[1]) for path in args.out.glob("shard_*")] or [-1]) + 1
    pending_ids: list[int] = []
    pending_texts: list[str] = []
    written = 0
    started = time.perf_counter()
    scanned = 0
    batches = 0
    total_done = len(done)

    def flush() -> None:
        nonlocal written, pending_ids, pending_texts, shard_index, total_done, batches
        if not pending_ids:
            return
        vectors = model.encode(pending_texts, batch_size=args.batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        write_shard(args.out, shard_index, pending_ids, vectors)
        total_done += len(pending_ids)
        written += len(pending_ids)
        shard_index += 1
        pending_ids, pending_texts = [], []
        elapsed = time.perf_counter() - started
        rate = written / elapsed if elapsed else 0.0
        print(f"  [embed] shard {shard_index - 1} done: +{written:,} this run ({total_done:,} total) "
              f"{rate:.1f} doc/s", flush=True)

    for doc_id, text in source:
        scanned += 1
        if doc_id in done:
            continue
        pending_ids.append(doc_id)
        pending_texts.append(text)
        if len(pending_ids) >= args.shard_size:
            flush()
            batches = 0
        elif args.limit is not None and total_done + len(pending_ids) >= args.limit:
            break
        batches += 1
        if args.log_every and batches and batches % args.log_every == 0:
            elapsed = time.perf_counter() - started
            processed = written + len(pending_ids)
            rate = processed / elapsed if elapsed else 0.0
            print(f"  [embed] scanned {scanned:,} | queued {len(pending_ids):,} | {rate:.1f} doc/s | "
                  f"eta {(args.limit - total_done - len(pending_ids)) / rate / 60:.0f} min"
                  if rate and args.limit else
                  f"  [embed] scanned {scanned:,} | queued {len(pending_ids):,} | {rate:.1f} doc/s",
                  flush=True)
    flush()
    print(f"[embed] finished: {written:,} new vectors this run, {total_done:,} total, "
          f"{(time.perf_counter() - started) / 60:.1f} min")
    print(json.dumps({"new": written, "total": total_done, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
