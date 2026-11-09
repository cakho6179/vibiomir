"""Merge Kaggle re-crawl shard outputs into data/docs.jsonl and runs/reports/recrawl_state.tsv.

Each shard dir (from ``kaggle kernels output <user>/vibiomir-recrawl-<shard> -p <dir>``) holds
docs.jsonl + state.tsv. A dir is merged once (ledger: runs/reports/kaggle_merged.txt); records
that do not parse, are not ok, or whose id already has an ok record are skipped.

    python scripts/crawl/merge_kaggle_recrawl.py kaggle/recrawl/outputs/lc1 kaggle/recrawl/outputs/lc2
After merging: python scripts/data/build_clean_corpus.py --update-sqlite
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import orjson

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())))
from r2ai2026.paths import DOCS_RAW, REPORTS  # noqa: E402

STATE = REPORTS / "recrawl_state.tsv"
LEDGER = REPORTS / "kaggle_merged.txt"


def ok_ids_in_state() -> set[int]:
    ids: set[int] = set()
    if STATE.exists():
        for line in STATE.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 3 and parts[2] == "ok" and parts[0].isdigit():
                ids.add(int(parts[0]))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--dry", action="store_true", help="count only, write nothing")
    args = ap.parse_args()

    merged = set(LEDGER.read_text(encoding="utf-8").split()) if LEDGER.exists() else set()
    have_ok = ok_ids_in_state()
    for d in args.dirs:
        docs, state = d / "docs.jsonl", d / "state.tsv"
        if not docs.exists() or not state.exists():
            print(f"{d}: missing docs.jsonl/state.tsv, skipped")
            continue
        key = hashlib.md5(docs.read_bytes()).hexdigest()
        if key in merged:
            print(f"{d}: already merged, skipped")
            continue
        added = bad = dup = 0
        out_lines: list[bytes] = []
        with open(docs, "rb") as f:
            for line in f:
                try:
                    rec = orjson.loads(line)
                except orjson.JSONDecodeError:
                    bad += 1
                    continue
                if not rec.get("ok") or rec["id"] in have_ok:
                    dup += 1
                    continue
                have_ok.add(rec["id"])
                out_lines.append(orjson.dumps(rec) + b"\n")
                added += 1
        state_lines = [l for l in state.read_text(encoding="utf-8").splitlines() if l.count("\t") >= 2]
        print(f"{d}: +{added:,} docs  (bad_json={bad}, skipped={dup})  state_lines={len(state_lines):,}")
        if args.dry:
            continue
        with open(DOCS_RAW, "ab") as f:
            f.writelines(out_lines)
        with open(STATE, "a", encoding="utf-8") as f:
            f.write("".join(l + "\n" for l in state_lines))
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(key + "\n")
        merged.add(key)


if __name__ == "__main__":
    main()
