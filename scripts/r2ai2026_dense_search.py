"""Exact dense (cosine) search over the fp16 shards written by r2ai2026_embed.py.

Embeddings are normalized, so the inner product is the cosine similarity. The
corpus is streamed through the GPU in chunks and only a running top-k is kept,
so memory stays flat as the corpus grows past a million documents.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def load_shards(directory: Path) -> tuple[np.ndarray, list[Path], list[tuple[int, int]]]:
    """Concatenate every shard's ids and return the shard directories with row ranges."""
    ids_parts: list[np.ndarray] = []
    shard_dirs: list[Path] = []
    ranges: list[tuple[int, int]] = []
    offset = 0
    for emb_path in sorted(directory.glob("shard_*/emb.npy")):
        shard_dir = emb_path.parent
        if not (shard_dir / "ids.npy").exists():
            continue
        ids = np.load(shard_dir / "ids.npy")
        ranges.append((offset, offset + int(ids.shape[0])))
        shard_dirs.append(shard_dir)
        ids_parts.append(ids)
        offset += int(ids.shape[0])
    if not ids_parts:
        raise SystemExit(f"no shards found under {directory}")
    return np.concatenate(ids_parts), shard_dirs, ranges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-emb", type=Path, required=True)
    parser.add_argument("--doc-emb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="parquet with query_id, doc_id, score, rank")
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-batch", type=int, default=2,
                        help="doc shards per GPU chunk (2 x 100k docs is ~800 MB in fp32)")
    args = parser.parse_args()

    import pandas as pd
    import torch

    query_ids = np.load(args.query_emb / "ids.npy")
    query_vectors = np.load(args.query_emb / "emb.npy")
    doc_ids, shard_dirs, ranges = load_shards(args.doc_emb)
    print(f"[dense] queries={len(query_ids):,} docs={len(doc_ids):,} dim={query_vectors.shape[1]}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    queries = torch.from_numpy(query_vectors.astype(np.float32)).to(device)
    queries = torch.nn.functional.normalize(queries, dim=1)

    topk = min(args.topk, len(doc_ids))
    best_scores = torch.full((len(query_ids), topk), -1.0, dtype=torch.float32, device=device)
    best_rows = torch.zeros((len(query_ids), topk), dtype=torch.int64, device=device)

    started = time.perf_counter()
    step = max(1, args.shard_batch)
    for start in range(0, len(shard_dirs), step):
        group = shard_dirs[start:start + step]
        block_start = ranges[start][0]
        vectors = [np.load(shard_dir / "emb.npy").astype(np.float32) for shard_dir in group]
        block = torch.from_numpy(np.concatenate(vectors, axis=0)).to(device)
        block = torch.nn.functional.normalize(block, dim=1)
        scores = queries @ block.T
        merged_scores = torch.cat([best_scores, scores], dim=1)
        merged_rows = torch.cat([
            best_rows,
            torch.arange(block_start, block_start + scores.shape[1], dtype=torch.int64, device=device)
            .unsqueeze(0).expand(scores.shape[0], -1),
        ], dim=1)
        best_scores, selection = torch.topk(merged_scores, topk, dim=1)
        best_rows = torch.gather(merged_rows, 1, selection)
        done = min(ranges[start + len(group) - 1][1], len(doc_ids))
        rate = done / max(1e-9, time.perf_counter() - started)
        print(f"  [dense] {done:,}/{len(doc_ids):,} docs scored ({rate:,.0f} doc/s)", flush=True)
        del block, scores, merged_scores, merged_rows

    rows = best_rows.cpu().numpy()
    scores = best_scores.cpu().numpy()
    frames = []
    for query_index, query_id in enumerate(query_ids):
        frames.append(pd.DataFrame({
            "query_id": np.full(topk, int(query_id), dtype=np.int64),
            "doc_id": doc_ids[rows[query_index]],
            "score": scores[query_index].astype(np.float32),
            "rank": np.arange(1, topk + 1, dtype=np.int32),
        }))
    table = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(f"[dense] wrote {len(table):,} rows -> {args.out} in {(time.perf_counter() - started) / 60:.1f} min")
    print(table.groupby("query_id")["score"].max().describe().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
