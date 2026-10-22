"""Pack data/docs_clean.jsonl into a slim, sharded Kaggle dataset folder.

Output: kaggle/corpus_dataset/corpus/part-XX.jsonl.gz  ({"id", "t": title, "x": text} per line)
        kaggle/corpus_dataset/query.parquet
    python scripts/r2ai2026_pack_corpus_dataset.py --shards 8
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path

import orjson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r2ai2026.paths import DATA, QUERIES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DATA / "docs_clean.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "kaggle" / "corpus_dataset")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--head-chars", type=int, default=0, help="keep only the first N chars of text (0 = all)")
    args = ap.parse_args()

    corpus = args.out / "corpus"
    if corpus.exists():
        shutil.rmtree(corpus)
    corpus.mkdir(parents=True)
    outs = [gzip.open(corpus / f"part-{i:02d}.jsonl.gz", "wb", compresslevel=6) for i in range(args.shards)]
    n = 0
    with open(args.src, "rb") as f:
        for line in f:
            d = orjson.loads(line)
            text = d.get("text") or ""
            rec = {"id": d["id"], "t": d.get("title") or "", "x": text[:args.head_chars] if args.head_chars else text}
            outs[d["id"] % args.shards].write(orjson.dumps(rec) + b"\n")
            n += 1
    for o in outs:
        o.close()
    shutil.copy2(QUERIES, args.out / "query.parquet")
    size = sum(p.stat().st_size for p in corpus.iterdir())
    print(f"packed {n:,} docs into {args.shards} shards, {size / 1e9:.2f} GB -> {args.out}")


if __name__ == "__main__":
    main()
