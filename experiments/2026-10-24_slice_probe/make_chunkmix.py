"""Chunk-only probe: relevant_docs of the base byte-identical, base chunks followed by the first k chunks of another
zip whose doc is not among the base docs.

    python experiments/2026-10-24_slice_probe/make_chunkmix.py --add kaggle/outputs/vibiomir-build-slicea/sliceA_VI_win4500_d300c80.zip --k 40

Docs F2 equals the base's, so Chunks P/R give the marginal density of the added chunks (keep them if > chunks F2/5)
and show whether a second document's window can earn credit next to the base chunks.
Writes submissions/chunkmix{k}_{add stem}.zip.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_probes import HERE, ROOT, read_submission_zip, write_submission_zip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=ROOT / "submissions" / "w12T2_VI_win4500_n150.zip")
    ap.add_argument("--add", type=Path, required=True)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--out", type=Path, default=ROOT / "submissions")
    args = ap.parse_args()

    add = {int(r["id"]): r["relevant_chunks"] for r in read_submission_zip(args.add)}
    records = read_submission_zip(args.base)
    new = []
    for r in records:
        docs = set(r["relevant_docs"])
        extra = [c for c in add.get(int(r["id"]), []) if c["doc_id"] not in docs][:args.k]
        new.append({**r, "relevant_chunks": r["relevant_chunks"] + extra})
    out = args.out / f"chunkmix{args.k}_{args.add.stem}.zip"
    zs = write_submission_zip(new, out, method="lzma")
    stats = {"zip": out.name, "chunks_per_q": round(sum(len(r["relevant_chunks"]) for r in new) / len(new), 2),
             "added_per_q": round(sum(len(n["relevant_chunks"]) - len(r["relevant_chunks"])
                                      for n, r in zip(new, records)) / len(new), 2),
             "zip_mb": round(zs["zip_mb"], 1)}
    print(json.dumps(stats), flush=True)
    (HERE / f"{out.stem}_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
