"""Re-crawl the ids missing from the clean corpus (runs/reports/recrawl_targets.tsv).

Only genuine page text is ever written: every record must pass ``junk_reason`` and a host
is cut off as soon as it blocks us or serves one identical page for many URLs.

Methods
  direct   curl_cffi with a Chrome TLS fingerprint (httpx gets 403 on several hosts)
  wayback  newest archive.org snapshot, ``/web/9999id_/<url>``; adaptive throttle on 429

    # measure the yield on a random sample, write nothing
    python scripts/r2ai2026_recrawl.py --method direct --hosts vov.vn --sample 40 --dry
    # real run, resumable
    python scripts/r2ai2026_recrawl.py --method wayback --hosts nhathuoclongchau.com.vn --limit 5000
After a run: python scripts/r2ai2026_build_clean_corpus.py --update-sqlite
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import orjson
import pandas as pd
from curl_cffi.requests import AsyncSession

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2ai2026.cleaning import junk_reason  # noqa: E402
from r2ai2026.corpus import classify  # noqa: E402
from r2ai2026.fetch import clean_extracted_text, decode_html, extract_article  # noqa: E402
from r2ai2026.paths import DOCS_RAW, REPORTS  # noqa: E402
from r2ai2026.text import detect_lang  # noqa: E402

TARGETS = REPORTS / "recrawl_targets.tsv"
STATE = REPORTS / "recrawl_state.tsv"
WAYBACK = "https://web.archive.org/web/9999id_/{url}"
BLOCK_HINT = ("just a moment", "attention required", "access denied", "enable javascript and cookies")
MAX_BYTES = 4_000_000


def load_state() -> dict[int, str]:
    """id -> last outcome per ``method:outcome`` from earlier runs."""
    state: dict[int, str] = {}
    if STATE.exists():
        for line in STATE.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                state[int(parts[0])] = f"{parts[1]}:{parts[2]}"
    return state


class Run:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.counts: Counter[str] = Counter()
        self.by_host: dict[str, Counter[str]] = defaultdict(Counter)
        self.dead_hosts: dict[str, str] = {}
        self.consec_block: Counter[str] = Counter()
        self.text_seen: Counter[bytes] = Counter()
        self.host_last: dict[str, float] = {}
        self.delay = args.delay  # adaptive for wayback
        self.out = None if args.dry else open(DOCS_RAW, "ab")
        self.state = None if args.dry else open(STATE, "a", encoding="utf-8")

    def record(self, doc_id: int, host: str, outcome: str, detail: str = "") -> None:
        self.counts[outcome] += 1
        self.by_host[host][outcome] += 1
        if self.state:
            self.state.write(f"{doc_id}\t{self.args.method}\t{outcome}\t{detail}\n")
            self.state.flush()

    async def fetch_html(self, session: AsyncSession, url: str) -> tuple[int, str, str, str]:
        """Return (status, html, final_url, error)."""
        target = WAYBACK.format(url=url) if self.args.method == "wayback" else url
        try:
            r = await session.get(target, timeout=self.args.timeout, allow_redirects=True,
                                  headers={"Accept-Language": "vi,en;q=0.8,zh;q=0.6"})
        except Exception as exc:  # noqa: BLE001
            return 0, "", "", type(exc).__name__
        raw = r.content[:MAX_BYTES]
        html = decode_html(raw, r.headers.get("content-type", ""))
        return r.status_code, html, str(r.url), ""

    async def handle(self, session, sem_global, host_sems, doc_id: int, url: str, host: str) -> None:
        if host in self.dead_hosts:
            return
        async with sem_global, host_sems[host]:
            if host in self.dead_hosts:
                return
            gap = self.delay if self.args.method == "wayback" else self.args.delay
            wait = self.host_last.get(host, 0) + gap * (0.6 + random.random() * 0.8) - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            status, html, final_url, err = await self.fetch_html(session, url)
            self.host_last[host] = time.monotonic()

        low = html[:3000].lower()
        if self.args.method == "wayback" and status == 404:
            return self.record(doc_id, host, "no_snapshot")
        if status in (429, 503, 502) or err in ("Timeout", "ConnectionError"):
            if self.args.method == "wayback":
                self.delay = min(30.0, self.delay * 1.6)
            self.consec_block[host] += 1
            self._maybe_kill(host, f"http_{status or err}")
            return self.record(doc_id, host, "transient", f"{status}{err}")
        if status in (401, 403) or any(h in low for h in BLOCK_HINT) and len(html) < 30000:
            self.consec_block[host] += 1
            self._maybe_kill(host, f"blocked_{status}")
            return self.record(doc_id, host, "blocked", str(status))
        if status != 200 or not html:
            self.consec_block[host] += 1
            self._maybe_kill(host, f"http_{status or err}")
            return self.record(doc_id, host, "http_error", f"{status}{err}")

        self.consec_block[host] = 0
        if self.args.method == "wayback":
            self.delay = max(self.args.delay, self.delay * 0.95)
        title, text, strategy = extract_article(html, url)
        text = clean_extracted_text(text) if text else ""
        why = junk_reason(text, is_pubmed=False)
        if why:
            return self.record(doc_id, host, why)
        key = hashlib.md5(text.encode("utf-8", "ignore")).digest()
        self.text_seen[key] += 1
        if self.text_seen[key] >= 3:
            # one cluster of identical pages is not a reason to drop a host; a host
            # that mostly answers with the same page (redirect to home) is
            c = self.by_host[host]
            if c["duplicate_template"] >= 25 and c["duplicate_template"] > c["ok"]:
                self._kill(host, "duplicate_template")
            return self.record(doc_id, host, "duplicate_template")
        rec = {"id": doc_id, "url": url, "final_url": final_url, "host": host, "kind": classify(url),
               "status": 200, "ok": True, "error": "", "title": title, "text": text,
               "lang": detect_lang(text or title), "n_chars": len(text),
               "strategy": f"{self.args.method}:{strategy}"}
        if self.out:
            self.out.write(orjson.dumps(rec) + b"\n")
            self.out.flush()
        self.record(doc_id, host, "ok")

    def _maybe_kill(self, host: str, why: str) -> None:
        if self.consec_block[host] >= self.args.kill_after:
            self._kill(host, why)

    def _kill(self, host: str, why: str) -> None:
        if host not in self.dead_hosts:
            self.dead_hosts[host] = why
            print(f"  !! host cut off: {host} ({why})", flush=True)


async def main_async(args: argparse.Namespace) -> None:
    df = pd.read_csv(TARGETS, sep="\t", header=None, names=["id", "url", "host", "reason"])
    if args.hosts:
        df = df[df["host"].isin(args.hosts.split(","))]
    if args.exclude:
        df = df[~df["host"].isin(args.exclude.split(","))]
    state = load_state()
    if not args.retry_failed:
        prefix = f"{args.method}:"
        df = df[~df["id"].map(lambda i: state.get(int(i), "").startswith(prefix))]
    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=args.seed)
    elif args.limit:
        df = df.head(args.limit)
    print(f"{args.method}: {len(df):,} targets, hosts={df['host'].nunique()}  dry={args.dry}", flush=True)
    rows = list(df.itertuples(index=False))
    random.Random(args.seed).shuffle(rows)

    run = Run(args)
    sem = asyncio.Semaphore(args.concurrency)
    host_sems: dict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(args.per_host))
    done = 0
    t0 = time.monotonic()
    async with AsyncSession(impersonate="chrome124") as session:
        async def go(r):
            nonlocal done
            await run.handle(session, sem, host_sems, int(r.id), str(r.url), str(r.host))
            done += 1
            if done % args.progress == 0:
                rate = done / (time.monotonic() - t0)
                print(f"  {done}/{len(rows)}  {dict(run.counts)}  {rate:.1f}/s  wb_delay={run.delay:.2f}", flush=True)
        await asyncio.gather(*(go(r) for r in rows))
    print("\nRESULT", dict(run.counts))
    for h, c in sorted(run.by_host.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        print(f"  {h:36} n={n:<6} ok={c['ok']:<6} ({100 * c['ok'] / n:.0f}%)  {dict(c)}")
    if run.dead_hosts:
        print("cut off:", run.dead_hosts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["direct", "wayback"], required=True)
    ap.add_argument("--hosts", default=None, help="comma separated allow-list")
    ap.add_argument("--exclude", default=None, help="comma separated hosts to skip")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0, help="random sample size (measure yield)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="write nothing")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--per-host", type=int, default=2)
    ap.add_argument("--delay", type=float, default=0.8, help="min seconds between hits on one host")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--kill-after", type=int, default=8, help="cut a host after N consecutive failures")
    ap.add_argument("--progress", type=int, default=100)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
