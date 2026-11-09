"""Rewrite submission zips without chunks whose text is a binary file or a mis-decoded page.

The leaderboard scorer failed on every zip that sent doc 660649 (a 5M-character .docx read as text) as a chunk for
a query where it is gold (22/10: pool_VI_*, rd_VI_d150c90r, w12_VI_d150c28, w12_VI_n28). relevant_docs is unchanged.

    python scripts/submit/drop_binary_chunks.py submissions/pool_VI_d500c75r.zip [...]   -> <name>_nb.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())))
from r2ai2026.corpus.cleaning import is_binary_text  # noqa: E402
from r2ai2026.core.submission import read_submission_zip, write_submission_zip  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zips", nargs="+", type=Path)
    args = ap.parse_args()
    for path in args.zips:
        records = read_submission_zip(path)
        dropped = 0
        for rec in records:
            kept = [c for c in rec["relevant_chunks"] if not is_binary_text(c["chunk_text"])]
            dropped += len(rec["relevant_chunks"]) - len(kept)
            rec["relevant_chunks"] = kept
        out = path.with_name(f"{path.stem}_nb.zip")
        stats = write_submission_zip(records, out, method="lzma")
        print(f"{out.name}: dropped {dropped} binary chunks, {stats['zip_mb']:.1f} MiB", flush=True)
        del records


if __name__ == "__main__":
    main()
