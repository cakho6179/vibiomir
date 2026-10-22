"""Pack the documents of data/docs_clean.jsonl that are not yet in the uploaded corpus dataset.

Base ids come from kaggle/corpus_dataset/corpus/part-*.jsonl.gz. Output folder holds
corpus_<tag>/part-<tag>-XX.jsonl.gz, so kernels that read every part-* under /kaggle/input
see base + delta as one corpus.
    python scripts/r2ai2026_pack_corpus_delta.py --tag d1
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import orjson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r2ai2026.paths import DATA  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", type=Path, default=ROOT / "kaggle" / "corpus_dataset" / "corpus")
    ap.add_argument("--shards", type=int, default=2)
    args = ap.parse_args()

    base = set()
    for part in sorted(args.base.glob("part-*.jsonl*")):
        with gzip.open(part, "rb") as f:
            base.update(orjson.loads(line)["id"] for line in f)
    out_dir = ROOT / "kaggle" / f"corpus_delta_{args.tag}" / f"corpus_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = [gzip.open(out_dir / f"part-{args.tag}-{i:02d}.jsonl.gz", "wb", compresslevel=6) for i in range(args.shards)]
    seen, n = set(), 0
    with open(DATA / "docs_clean.jsonl", "rb") as f:
        for line in f:
            d = orjson.loads(line)
            seen.add(d["id"])
            if d["id"] in base:
                continue
            outs[d["id"] % args.shards].write(orjson.dumps({"id": d["id"], "t": d.get("title") or "", "x": d.get("text") or ""}) + b"\n")
            n += 1
    for o in outs:
        o.close()
    print(f"base={len(base):,} new={n:,} base_ids_missing_from_clean={len(base - seen):,} -> {out_dir.parent}")


if __name__ == "__main__":
    main()
