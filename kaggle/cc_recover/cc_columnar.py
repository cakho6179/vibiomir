"""Recover missing corpus pages from the Common Crawl *columnar* index (Kaggle CPU kernel).

    python cc_columnar.py --missing missing.tsv --out DIR [--crawl-part I/N] [--since 2017]
                          [--skip-ids done.txt] [--exclude host,...] [--max-hours 11]

Unlike cc_recover.py (CDX server, overloaded: 503/504), everything is read from the static
data.commoncrawl.org bucket: per crawl the 300 parquet footers of cc-index/table (~40 KB each),
then, only in row groups whose url_surtkey range covers a wanted host, the url_host_name column
(~0.4 MB) and, if that host occurs, url + fetch_status + warc_* columns. Hits are fetched as single
WARC records by byte range and extracted with trafilatura (helpers shared with cc_recover.py).
Crawls are processed newest first; an id recovered once is not fetched again.
Outputs in DIR: docs.jsonl ({"id","url","t","x","crawl"}), hits.tsv, state.tsv, summary.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
import requests

from cc_recover import DATA, extract, fetch_record, get, key, log

COLS = ["url", "fetch_status", "warc_filename", "warc_record_offset", "warc_record_length"]
LOCK = threading.Lock()


def surt_host(host: str) -> str:
    return ",".join(reversed(host.split(".")))


def crawl_ids(session, since: int) -> list[str]:
    r = get(session, "https://index.commoncrawl.org/collinfo.json")
    ids = [c["id"] for c in r.json()] if r is not None else []
    return [c for c in ids if c.startswith("CC-MAIN-") and int(c[8:12]) >= since]


def footers(fs, crawl: str, session) -> list[tuple[str, int, str, str]]:
    """(path, row_group, min_surt, max_surt) for every row group of the warc subset of one crawl."""
    r = get(session, f"{DATA}crawl-data/{crawl}/cc-index-table.paths.gz")
    if r is None or r.status_code != 200:
        return []
    paths = [p for p in gzip.decompress(r.content).decode().split() if "/subset=warc/" in p]

    def one(path):
        for attempt in range(4):
            try:
                with fs.open(DATA + path, block_size=1 << 16) as f:
                    md = pq.ParquetFile(f).metadata
                col = [md.row_group(0).column(j).path_in_schema for j in range(md.num_columns)].index("url_surtkey")
                out = []
                for g in range(md.num_row_groups):
                    st = md.row_group(g).column(col).statistics
                    has = st is not None and st.has_min_max
                    out.append((path, g, st.min if has else "", st.max if has else "\uffff"))  # no stats: scan it
                return out
            except Exception as e:  # transient HTTP errors
                log(f"  footer retry {attempt}: {type(e).__name__} {path[-60:]}")
                time.sleep(5 * (attempt + 1))
        return []

    with ThreadPoolExecutor(4) as ex:  # data.commoncrawl.org answers 403 to bursts
        return [x for part in ex.map(one, paths) for x in part]


def scan_group(fs, path: str, g: int, hosts: set[str]) -> list[dict]:
    with fs.open(DATA + path, block_size=1 << 20) as f:
        pf = pq.ParquetFile(f)
        names = pf.read_row_group(g, columns=["url_host_name"]).column(0).to_pylist()
        rows = [i for i, h in enumerate(names) if h and h.lower().removeprefix("www.") in hosts]
        if not rows:
            return []
        t = pf.read_row_group(g, columns=COLS).take(rows).to_pylist()
    return [r for r in t if r["fetch_status"] == 200]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--missing", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crawl-part", default="0/1")
    ap.add_argument("--since", type=int, default=2017)
    ap.add_argument("--crawls", default="", help="explicit comma list of crawl ids (overrides collinfo)")
    ap.add_argument("--skip-ids", default="", help="file with ids (one per line) already recovered")
    ap.add_argument("--exclude", default="pubmed.ncbi.nlm.nih.gov")
    ap.add_argument("--hosts", default="", help="only these hosts (dry runs)")
    ap.add_argument("--fetch-workers", type=int, default=6)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-hours", type=float, default=11.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def find(name):
        return name if Path(name).exists() else glob.glob(f"/kaggle/input/**/{Path(name).name}", recursive=True)[0]

    skip = set()
    if args.skip_ids:
        skip = {int(x) for x in Path(find(args.skip_ids)).read_text().split()}
    excl = {h.removeprefix("www.") for h in args.exclude.split(",") if h}
    only = {h.removeprefix("www.") for h in args.hosts.split(",") if h}
    missing: dict[str, int] = {}
    with open(find(args.missing), encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            k = key(row["url"])
            host = k.split("/")[0]
            if int(row["id"]) not in skip and host not in excl and (not only or host in only):
                missing[k] = int(row["id"])
    hosts = {k.split("/")[0] for k in missing}
    ranges = sorted((surt_host(h) + ")", surt_host(h) + ")\uffff") for h in hosts)
    log(f"{len(missing):,} missing URLs on {len(hosts)} hosts (skipped {len(skip):,} ids)")

    session = requests.Session()
    fs = fsspec.filesystem("http")
    i, n = (int(x) for x in args.crawl_part.split("/"))
    all_crawls = args.crawls.split(",") if args.crawls else crawl_ids(session, args.since)
    mine = all_crawls[i::n]
    log(f"{len(all_crawls)} crawls; this part ({len(mine)}): {mine}")

    found: set[str] = set()
    docs_f = open(args.out / "docs.jsonl", "a", encoding="utf-8")
    hits_f = open(args.out / "hits.tsv", "a", encoding="utf-8")
    state_f = open(args.out / "state.tsv", "a", encoding="utf-8")
    pool = ThreadPoolExecutor(args.fetch_workers)
    t_end = time.time() + args.max_hours * 3600

    def handle(k: str, rec: dict, crawl: str) -> None:
        html = fetch_record(session, {"filename": rec["warc_filename"], "offset": rec["warc_record_offset"],
                                      "length": rec["warc_record_length"]})
        if not html:
            return
        title, text = extract(html)
        if len(text) < args.min_chars:
            return
        with LOCK:
            if k in found:
                return
            found.add(k)
            docs_f.write(json.dumps({"id": missing[k], "url": rec["url"], "t": title, "x": text, "crawl": crawl},
                                    ensure_ascii=False) + "\n")
            docs_f.flush()
            state_f.write(f"{missing[k]}\t{crawl}\n")
            state_f.flush()

    for crawl in mine:
        if time.time() > t_end:
            log("time budget reached")
            break
        groups = footers(fs, crawl, session)
        todo_hosts = {k.split("/")[0] for k in missing if k not in found}
        want = [(p, g) for p, g, lo, hi in groups
                if any(a <= hi and b >= lo for a, b in ranges if a[:-1] in {surt_host(h) for h in todo_hosts})]
        log(f"{crawl}: {len(groups)} row groups, {len(want)} cover wanted hosts")

        def scan(pg):
            for attempt in range(4):
                try:
                    return scan_group(fs, pg[0], pg[1], todo_hosts)
                except Exception as e:  # 403/503 throttling: back off and retry
                    log(f"  scan retry {attempt}: {type(e).__name__} {pg[0][-50:]}#{pg[1]}")
                    time.sleep(20 * 3 ** attempt)
            return []

        with ThreadPoolExecutor(3) as ex:
            recs = [r for part in ex.map(scan, want) for r in part]
        futs = []
        for rec in recs:
            k = key(rec["url"])
            if k in missing and k not in found:
                hits_f.write(f"{missing[k]}\t{crawl}\t{rec['warc_filename']}\t{rec['warc_record_offset']}\t"
                             f"{rec['warc_record_length']}\n")
                futs.append(pool.submit(handle, k, rec, crawl))
        hits_f.flush()
        for fu in futs:
            try:
                fu.result()
            except Exception as e:  # one bad record must not stop the crawl
                log(f"  fetch failed: {type(e).__name__}: {e}")
        log(f"{crawl}: {len(recs):,} host rows, {len(futs):,} matched, total recovered {len(found):,}/{len(missing):,}")

    pool.shutdown()
    for fh in (docs_f, hits_f, state_f):
        fh.close()
    per_host: dict[str, int] = {}
    for k in found:
        per_host[k.split("/")[0]] = per_host.get(k.split("/")[0], 0) + 1
    (args.out / "summary.json").write_text(json.dumps({"recovered": len(found), "missing": len(missing),
                                                      "per_host": per_host, "crawls": mine}, indent=2))
    log(f"done: {len(found):,} recovered")


if __name__ == "__main__":
    main()
