"""Build the canonical clean corpus from data/docs.jsonl (raw real crawl output).

Keeps only genuine page content (see ``r2ai2026.cleaning.junk_reason``), never edits text.
Writes:
  data/docs_clean.jsonl        one record per usable id
  runs/reports/excluded_docs.tsv   id, host, reason  (junk found in the raw crawl)
  runs/reports/recrawl_targets.tsv id, url, host, reason  (every id NOT in docs_clean)
and, with --update-sqlite, sets crawl.sqlite ok=1 only for ids in docs_clean.

    python scripts/r2ai2026_build_clean_corpus.py --update-sqlite
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import orjson
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2ai2026.cleaning import junk_reason  # noqa: E402
from r2ai2026.paths import CRAWL_DB, DATA, DOCS_RAW, LINKS, REPORTS  # noqa: E402

PUBMED = "pubmed.ncbi.nlm.nih.gov"


def host_of(url: str) -> str:
    return urlsplit(url).netloc.lower()


def iter_records(path: Path):
    """Parsed records; lines broken by concurrent appends are skipped and counted."""
    bad = 0
    with open(path, "rb") as f:
        for line in f:
            try:
                d = orjson.loads(line)
            except orjson.JSONDecodeError:
                bad += 1
                continue
            if d.get("ok") and d.get("url"):
                yield line, d
    if bad:
        print(f"skipped {bad} malformed lines in {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DOCS_RAW)
    ap.add_argument("--out", type=Path, default=DATA / "docs_clean.jsonl")
    ap.add_argument("--update-sqlite", action="store_true")
    args = ap.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)

    # last ok record per id wins (re-crawls are appended after older records)
    last: dict[int, int] = {}
    for n, (_, d) in enumerate(iter_records(args.src)):
        last[d["id"]] = n
    counts: Counter[bytes] = Counter()
    for n, (_, d) in enumerate(iter_records(args.src)):
        if last[d["id"]] == n and host_of(d["url"]) != PUBMED:
            counts[hashlib.md5((d.get("text") or "").encode("utf-8", "ignore")).digest()] += 1

    kept: set[int] = set()
    reasons: Counter[str] = Counter()
    excluded: list[tuple[int, str, str]] = []
    with open(args.out, "wb") as out:
        for n, (line, d) in enumerate(iter_records(args.src)):
            if last[d["id"]] != n:
                continue
            host = host_of(d["url"])
            text = d.get("text") or ""
            is_pm = host == PUBMED
            dup = 1 if is_pm else counts[hashlib.md5(text.encode("utf-8", "ignore")).digest()]
            why = junk_reason(text, is_pubmed=is_pm, dup_count=dup)
            if why:
                reasons[why] += 1
                excluded.append((d["id"], host, why))
            else:
                kept.add(d["id"])
                out.write(line)
    print(f"kept={len(kept):,}  excluded_junk={len(excluded):,}  {dict(reasons)}")

    with open(REPORTS / "excluded_docs.tsv", "w", encoding="utf-8") as f:
        for i, h, why in excluded:
            f.write(f"{i}\t{h}\t{why}\n")

    links = pd.read_parquet(LINKS)
    why_by_id = {i: why for i, _, why in excluded}
    miss = links[~links["id"].isin(kept)].copy()
    miss["host"] = miss["url"].map(host_of)
    miss["reason"] = miss["id"].map(why_by_id).fillna("no_real_record")
    miss.to_csv(REPORTS / "recrawl_targets.tsv", sep="\t", index=False, header=False,
                columns=["id", "url", "host", "reason"])
    print(f"recrawl targets={len(miss):,} of {len(links):,}  ({100 * len(miss) / len(links):.1f}%)")
    print(miss.groupby("host").size().sort_values(ascending=False).head(15).to_string())

    if args.update_sqlite:
        con = sqlite3.connect(CRAWL_DB)
        cur = con.cursor()
        cur.execute("UPDATE docs SET ok=0")
        cur.executemany("UPDATE docs SET ok=1 WHERE id=?", [(i,) for i in kept])
        con.commit()
        n_ok = cur.execute("SELECT count(*) FROM docs WHERE ok=1").fetchone()[0]
        con.close()
        print(f"sqlite ok=1 -> {n_ok:,}")


if __name__ == "__main__":
    main()
