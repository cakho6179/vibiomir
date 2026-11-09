"""Recover missing corpus pages from Common Crawl (Kaggle CPU kernel). Output is kept separate.

    python cc_recover.py --missing missing.tsv --out DIR [--hosts h1,h2] [--crawl-part I/N]
                         [--since 2017] [--max-crawls N]

For each host and each crawl (newest first) the CDX index is listed by host prefix
(url=<host>/*, status 200, paginated), matched against the missing URLs, and each hit is
fetched as one WARC record by byte range from data.commoncrawl.org. Text via trafilatura.
Outputs in DIR: docs.jsonl ({"id","url","t","x","crawl","ts"}), state.tsv (id, crawl), cc.log
--crawl-part I/N keeps crawls with index % N == I so several kernels can share one host.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests

INDEX = "https://index.commoncrawl.org"
DATA = "https://data.commoncrawl.org/"
UA = {"User-Agent": "vibiomir-research/1.0 (biomedical IR dataset recovery)"}
T0 = time.time()
LOCK = threading.Lock()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def key(url: str) -> str:
    """Scheme-, www- and trailing-slash-insensitive URL key."""
    p = urlparse(url.strip())
    host = p.netloc.lower().removeprefix("www.").split(":")[0]
    path = p.path.rstrip("/") or "/"
    return f"{host}{path}" + (f"?{p.query}" if p.query else "")


def get(session: requests.Session, url: str, tries: int = 6, headers: dict | None = None, **kw):
    delay = 2.0
    for _ in range(tries):
        try:
            r = session.get(url, timeout=90, headers={**UA, **(headers or {})}, **kw)
            if r.status_code in (200, 206, 404):
                return r
            log(f"  http {r.status_code} {url[:120]}")
        except requests.RequestException as e:
            log(f"  {type(e).__name__} {url[:120]}")
        time.sleep(delay)
        delay = min(delay * 2, 120)
    return None


def crawls(session, since: int) -> list[str]:
    r = get(session, f"{INDEX}/collinfo.json")
    ids = [c["id"] for c in r.json()] if r is not None else []
    return [c for c in ids if re.match(r"CC-MAIN-\d{4}", c) and int(c[8:12]) >= since]


def list_host(session, crawl: str, prefix: str):
    """Yield CDX records (dict) with status 200 for every URL under prefix (host[/dir]) in one crawl."""
    base = f"{INDEX}/{crawl}-index"
    params = {"url": f"{prefix}/*", "output": "json", "filter": "status:200",
              "fl": "url,filename,offset,length,timestamp,mime"}
    r = get(session, base, params={**params, "showNumPages": "true"})
    if r is None or r.status_code == 404:
        return
    try:
        pages = int(json.loads(r.text.strip().splitlines()[0])["pages"])
    except (ValueError, KeyError, IndexError):
        return
    for page in range(pages):
        r = get(session, base, params={**params, "page": page})
        time.sleep(1.0)  # be polite to the shared index server
        if r is None or r.status_code == 404:
            continue
        for line in r.text.splitlines():
            try:
                yield json.loads(line)
            except ValueError:
                continue


def fetch_record(session, rec: dict) -> bytes | None:
    off, n = int(rec["offset"]), int(rec["length"])
    r = get(session, DATA + rec["filename"], headers={**UA, "Range": f"bytes={off}-{off + n - 1}"})
    if r is None or r.status_code not in (200, 206):
        return None
    from warcio.archiveiterator import ArchiveIterator
    for record in ArchiveIterator(io.BytesIO(r.content)):
        if record.rec_type == "response":
            return record.content_stream().read()
    return None


def decode(html: bytes) -> str:
    m = re.search(rb"charset=[\"']?([\w-]+)", html[:4000], re.I)
    enc = m.group(1).decode("ascii", "ignore").lower() if m else "utf-8"
    if enc in ("gb2312", "gbk"):
        enc = "gb18030"
    for e in (enc, "utf-8", "gb18030"):
        try:
            return html.decode(e)
        except (UnicodeDecodeError, LookupError):
            continue
    return html.decode("utf-8", "ignore")


def extract(html: bytes) -> tuple[str, str]:
    import trafilatura
    page = decode(html)
    text = trafilatura.extract(page, include_comments=False, include_tables=True) or ""
    if len(text) < 200:
        text = trafilatura.extract(page, include_comments=False, include_tables=True, favor_recall=True) or text
    meta = trafilatura.extract_metadata(page)
    title = (getattr(meta, "title", None) or "") if meta is not None else ""
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return title.strip(), "\n".join(ln for ln in lines if ln)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--missing", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hosts", default="", help="only these hosts (www. ignored); default all")
    ap.add_argument("--exclude", default="", help="skip these hosts")
    ap.add_argument("--crawl-part", default="0/1")
    ap.add_argument("--since", type=int, default=2017)
    ap.add_argument("--max-crawls", type=int, default=None)
    ap.add_argument("--fetch-workers", type=int, default=8)
    ap.add_argument("--min-chars", type=int, default=200)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    want_hosts = {h.removeprefix("www.") for h in args.hosts.split(",") if h}
    skip_hosts = {h.removeprefix("www.") for h in args.exclude.split(",") if h}
    missing: dict[str, int] = {}
    by_host: dict[str, int] = {}
    if not Path(args.missing).exists():  # bare file name: look under the Kaggle inputs
        import glob
        args.missing = glob.glob(f"/kaggle/input/**/{Path(args.missing).name}", recursive=True)[0]
    with open(args.missing, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            k = key(row["url"])
            host = k.split("/")[0]
            if (want_hosts and host not in want_hosts) or host in skip_hosts:
                continue
            missing[k] = int(row["id"])
            by_host[host] = by_host.get(host, 0) + 1
    log(f"{len(missing):,} missing URLs on {len(by_host)} hosts")

    session = requests.Session()
    i, n = (int(x) for x in args.crawl_part.split("/"))
    all_crawls = crawls(session, args.since)
    mine = all_crawls[i::n][: args.max_crawls]
    log(f"{len(all_crawls)} crawls since {args.since}; this part: {mine}")

    found: set[str] = set()
    docs_f = open(args.out / "docs.jsonl", "a", encoding="utf-8")
    state_f = open(args.out / "state.tsv", "a", encoding="utf-8")
    pool = ThreadPoolExecutor(args.fetch_workers)

    def handle(k: str, rec: dict, crawl: str) -> None:
        html = fetch_record(session, rec)
        if not html:
            return
        title, text = extract(html)
        if len(text) < args.min_chars:
            return
        with LOCK:
            if k in found:
                return
            found.add(k)
            docs_f.write(json.dumps({"id": missing[k], "url": rec["url"], "t": title, "x": text,
                                     "crawl": crawl, "ts": rec.get("timestamp")}, ensure_ascii=False) + "\n")
            docs_f.flush()
            state_f.write(f"{missing[k]}\t{crawl}\n")
            state_f.flush()

    hosts = sorted(by_host, key=lambda h: by_host[h])  # small hosts first
    for crawl in mine:
        for host in hosts:
            todo = {k for k in missing if k.startswith(host + "/") and k not in found}
            if not todo:
                continue
            # list only the shared first directory (e.g. 120ask.com/question) when there is one
            dirs = {k.split("/")[1] for k in todo if k.count("/") >= 2}
            prefix = f"{host}/{dirs.pop()}" if len(dirs) == 1 else host
            hits, futs = 0, []
            for rec in list_host(session, crawl, prefix):
                k = key(rec.get("url", ""))
                if k in todo:
                    todo.discard(k)
                    hits += 1
                    futs.append(pool.submit(handle, k, rec, crawl))
            for fu in futs:
                try:
                    fu.result()
                except Exception as e:  # one bad record must not stop the crawl
                    log(f"  fetch failed: {type(e).__name__}: {e}")
            log(f"{crawl} {host}: {hits} index hits, total recovered {len(found):,}/{len(missing):,}")
    pool.shutdown()
    docs_f.close()
    state_f.close()
    per_host: dict[str, int] = {}
    for k in found:
        per_host[k.split("/")[0]] = per_host.get(k.split("/")[0], 0) + 1
    (args.out / "summary.json").write_text(json.dumps({"recovered": len(found), "missing": len(missing),
                                                      "per_host": per_host, "crawls": mine}, indent=2))
    log(f"done: {len(found):,} recovered")


if __name__ == "__main__":
    main()
