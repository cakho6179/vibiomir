"""Check downloaded release files against release/MANIFEST.tsv (sha256), from the repo root.

    python scripts/release/verify_manifest.py MANIFEST.tsv [--groups A_pgold,B_materialize] [--big]

Files of the artifact groups are checked at their repo path (after unzipping at the repo root); the docstore / index
groups only with --big (12 GB to hash). Code and *_zip rows describe the downloaded archives themselves and are skipped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIG = {"E_docstore", "F_index"}
SKIP = {"code", "submitted"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--groups", default="")
    ap.add_argument("--big", action="store_true")
    args = ap.parse_args()
    want = set(filter(None, args.groups.split(",")))
    ok = bad = missing = 0
    with open(args.manifest, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            g = row["group"]
            if g in SKIP or g.endswith("_zip") or (want and g not in want) or (g in BIG and not args.big):
                continue
            p = ROOT / row["path"]
            if not p.exists():
                missing += 1
                print(f"MISSING {row['path']}")
            elif row["sha256"] != "-" and sha256(p) != row["sha256"]:
                bad += 1
                print(f"BAD     {row['path']}")
            else:
                ok += 1
    print(f"ok {ok}, bad {bad}, missing {missing}")
    raise SystemExit(1 if bad or missing else 0)


if __name__ == "__main__":
    main()
