"""Merge crawled parquet partitions from Kaggle CPU into docs.jsonl and crawl.sqlite.

Usage:
    python scripts/r2ai2026_merge_crawled.py --input-dir kaggle/crawled_output
    # or single file:
    python scripts/r2ai2026_merge_crawled.py --input-file kaggle/crawled_output/vibiomir-crawl-p1-vinmec.parquet
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import orjson
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def merge_parquet(p_file: Path, docs_jsonl: Path, sqlite_path: Path) -> int:
    print(f"Reading {p_file}...")
    df = pd.read_parquet(p_file)
    if df.empty:
        print("  Empty parquet file.")
        return 0

    print(f"  Loaded {len(df):,} crawled records.")
    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()

    c.execute("SELECT id FROM docs WHERE ok = 1")
    existing_ok_ids = {r[0] for r in c.fetchall()}

    added = 0
    db_rows = []

    with open(docs_jsonl, "ab") as f_out:
        for row in df.itertuples(index=False):
            doc_id = int(row.id)
            if doc_id in existing_ok_ids:
                continue

            record = {
                "id": doc_id,
                "url": str(row.url),
                "title": str(getattr(row, "title", "") or ""),
                "text": str(getattr(row, "text", "") or ""),
                "ok": int(getattr(row, "ok", 1)),
            }
            line = orjson.dumps(record) + b"\n"
            f_out.write(line)

            db_rows.append((doc_id, str(row.url), 200, 1, len(record["text"]), None))
            existing_ok_ids.add(doc_id)
            added += 1

            if len(db_rows) >= 5000:
                c.executemany(
                    "INSERT OR REPLACE INTO docs (id, url, status, ok, n_chars, error) VALUES (?, ?, ?, ?, ?, ?)",
                    db_rows,
                )
                conn.commit()
                db_rows.clear()

    if db_rows:
        c.executemany(
            "INSERT OR REPLACE INTO docs (id, url, status, ok, n_chars, error) VALUES (?, ?, ?, ?, ?, ?)",
            db_rows,
        )
        conn.commit()

    conn.close()
    print(f"  Merged {added:,} new valid documents into {docs_jsonl.name} and {sqlite_path.name}.")
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("kaggle/crawled_output"))
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--docs", type=Path, default=Path("data/docs.jsonl"))
    parser.add_argument("--sqlite", type=Path, default=Path("data/crawl.sqlite"))
    args = parser.parse_args(argv)

    total_added = 0
    if args.input_file and args.input_file.is_file():
        total_added += merge_parquet(args.input_file, args.docs, args.sqlite)
    elif args.input_dir.is_dir():
        for p in sorted(args.input_dir.glob("*.parquet")):
            total_added += merge_parquet(p, args.docs, args.sqlite)
    else:
        print(f"No input files found at {args.input_dir} or {args.input_file}")
        return 1

    print(f"\nAll done! Total documents added: {total_added:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
