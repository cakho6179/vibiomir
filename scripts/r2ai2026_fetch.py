"""Fetch ViBioMIR documents: PubMed via E-utilities, everything else via crawl.

Examples
--------
    # 500 PubMed abstracts (fast, deterministic, no HTML parsing)
    python scripts/r2ai2026_fetch.py pubmed --limit 500

    # 200 Vietnamese + Chinese pages spread over many hosts
    python scripts/r2ai2026_fetch.py web --limit 200 --kinds vi,zh --per-host-cap 5

    # one host only, small burst test
    python scripts/r2ai2026_fetch.py web --hosts suckhoedoisong.vn --limit 30

    # resume a stopped crawl (already-succeeded ids are skipped automatically)
    python scripts/r2ai2026_fetch.py web --limit 5000 --resume
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai2026.corpus import (  # noqa: E402
    classify,
    host_of,
    interleave_by_host,
    iter_links,
    load_links,
    pending_links,
    sample_links,
)
from r2ai2026.fetch import DEFAULT_HEADERS, DEFAULT_USER_AGENT, Crawler  # noqa: E402
from r2ai2026.pubmed import EFetchClient  # noqa: E402
from r2ai2026.store import CrawlSink  # noqa: E402
from r2ai2026.wayback import wayback_record  # noqa: E402

DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_LINKS = DEFAULT_DATA_DIR / "raw" / "btc" / "links_corpus_vi_en_zh.parquet"


def _paths(data_dir: Path) -> tuple[Path, Path]:
    return data_dir / "docs.jsonl", data_dir / "crawl.sqlite"


def _progress_printer(label: str, every: int = 25):
    def report(completed: int, total: int) -> None:
        if completed % every == 0 or completed == total:
            print(f"  [{label}] {completed}/{total}", flush=True)

    return report


async def run_pubmed(args: argparse.Namespace) -> int:
    links = load_links(args.links)
    subset = links[links["kind"] == "pubmed"].head(args.limit)
    if subset.empty:
        print("no pubmed links selected")
        return 1
    ids_by_pmid = {
        int(row.pmid): int(row.id)
        for row in subset.itertuples(index=False)
        if row.pmid is not None
    }
    jsonl_path, sqlite_path = _paths(args.data_dir)
    sink = CrawlSink(jsonl_path, sqlite_path)
    done = sink.succeeded_ids() if args.resume else set()
    pending = [pmid for pmid, doc_id in ids_by_pmid.items() if doc_id not in done]
    pending.sort()
    print(f"pubmed: {len(pending)} pmids pending (of {len(ids_by_pmid)} selected, {len(done)} already stored)")

    written = 0
    async with EFetchClient(api_key=args.ncbi_api_key, email=args.email) as client:
        for start in range(0, len(pending), args.batch):
            batch = pending[start : start + args.batch]
            records = await client.harvest(batch, ids_by_pmid=ids_by_pmid, batch_size=args.batch)
            for record in records:
                sink.append(record)
                written += 1
            print(f"  pubmed {min(start + args.batch, len(pending))}/{len(pending)} -> stored {written}", flush=True)
    sink.close()
    print(f"stored {written} records in {jsonl_path}")
    return 0


async def run_web(args: argparse.Namespace) -> int:
    links = load_links(args.links)
    kinds = [value.strip() for value in args.kinds.split(",") if value.strip()]
    hosts = [value.strip() for value in args.hosts.split(",")] if args.hosts else None
    if args.per_host_cap:
        subset = sample_links(links, args.limit, kinds=kinds or None, hosts=hosts, per_host_cap=args.per_host_cap)
    else:
        subset = links
        if kinds:
            subset = subset[subset["kind"].isin(kinds)]
        if hosts:
            subset = subset[subset["host"].isin(hosts)]
        subset = subset.head(args.limit).reset_index(drop=True)
    print(f"web: selected {len(subset)} urls across {subset['host'].nunique()} hosts")

    jsonl_path, sqlite_path = _paths(args.data_dir)
    sink = CrawlSink(jsonl_path, sqlite_path)
    done = sink.succeeded_ids() if args.resume else set()
    items = [
        (int(row.id), str(row.url), str(row.host), str(row.kind))
        for row in subset.itertuples(index=False)
        if int(row.id) not in done
    ]
    if not args.no_interleave:
        items = interleave_by_host(items)
    if not items:
        print("nothing to do (all selected ids already stored)")
        sink.close()
        return 0

    async with Crawler(
        concurrency=args.concurrency,
        per_host_concurrency=args.per_host_concurrency,
        timeout=args.timeout,
        respect_robots=not args.ignore_robots,
        min_host_delay=args.host_delay,
        max_forbidden_retries=args.forbidden_retries,
        host_failure_threshold=args.host_failure_threshold,
    ) as crawler:
        counters = await crawler.crawl(items, sink.append, progress=_progress_printer("web", args.progress_every))
        stats = crawler.stats
        blocked = crawler.blocked_hosts

    sink.close()
    print(f"crawl counters: {counters}")
    print(f"http stats: {stats}")
    if blocked:
        top = sorted(blocked.items(), key=lambda kv: kv[1])[:20]
        print(f"host circuit breaker opened for {len(blocked)} hosts, e.g. {top}")
    print(f"stored in {jsonl_path} (total rows {sink.counters['written']} this run)")
    return 0


async def run_wayback(args: argparse.Namespace) -> int:
    """Second pass: recover blocked/failed URLs from the Wayback Machine.

    Two selection modes: stored failures (``errors``/``hosts``) and
    ``--from-links`` for hosts that block *every* request profile -- those URLs
    have no store row at all, so ``failed_rows`` would find nothing.
    """
    jsonl_path, sqlite_path = _paths(args.data_dir)
    sink = CrawlSink(jsonl_path, sqlite_path)
    error_filter = [value.strip() for value in args.errors.split(",") if value.strip()]
    host_filter = [value.strip() for value in args.hosts.split(",") if value.strip()] if args.hosts else None

    if args.from_links:
        if not host_filter and not args.kinds:
            print("--from-links needs --hosts and/or --kinds, refusing to queue the whole corpus")
            sink.close()
            return 1
        links = load_links(args.links)
        kinds = [value.strip() for value in args.kinds.split(",") if value.strip()]
        subset = links
        if kinds:
            subset = subset[subset["kind"].isin(kinds)]
        if host_filter:
            subset = subset[subset["host"].isin(host_filter)]
        permanent = set() if args.include_no_snapshot else sink.ids_with_error_prefix("wayback_no_snapshot")
        done = sink.succeeded_ids()
        rows = pending_links(
            links,
            kinds=kinds,
            hosts=host_filter,
            exclude_ids=done | permanent,
            limit=args.limit,
        )
        print(
            f"wayback: {len(rows)} urls selected from {len(subset)} candidate links "
            f"(hosts={host_filter or 'any'}, kinds={kinds or 'any'}, "
            f"{len(done)} ok stored, {len(permanent)} skipped as never-archived)"
        )
    else:
        rows = sink.failed_rows(errors=error_filter, hosts=host_filter, limit=args.limit)
        if not rows:
            print("no failed rows matched")
            sink.close()
            return 0
        print(f"wayback: {len(rows)} urls to try (errors={error_filter or 'any'})")

    headers = {"User-Agent": DEFAULT_USER_AGENT, **DEFAULT_HEADERS}
    semaphore = asyncio.Semaphore(args.concurrency)
    counters = {"ok": 0, "failed": 0}
    # archive.org answers 429 as soon as the pass gets greedy, and a fixed delay
    # cannot see that coming: the pause grows on rate-limit/timeout answers and
    # decays again on successes.
    direct = not args.no_direct
    throttle = {"delay": max(0.05, args.delay)}

    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        async def handle(index: int, doc_id: int, url: str) -> None:
            async with semaphore:
                await asyncio.sleep(throttle["delay"] * (0.5 + random.random()))
                record = await wayback_record(
                    client, doc_id, url, host_of(url), classify(url), direct=direct,
                )
            sink.append(record)
            counters["ok" if record["ok"] else "failed"] += 1
            if record["ok"]:
                throttle["delay"] = max(args.delay, throttle["delay"] * 0.92)
            elif any(token in record["error"] for token in ("429", "503", "502", "transport", "timeout")):
                throttle["delay"] = min(30.0, throttle["delay"] * 1.6)
            done = counters["ok"] + counters["failed"]
            if done % 25 == 0 or done == len(rows):
                print(
                    f"  wayback {done}/{len(rows)} ok={counters['ok']} "
                    f"delay={throttle['delay']:.2f}s",
                    flush=True,
                )

        await asyncio.gather(*(handle(index, doc_id, url) for index, (doc_id, url) in enumerate(rows)))

    sink.close()
    print(f"wayback done: {counters}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pubmed = subparsers.add_parser("pubmed", help="harvest PubMed title+abstract via E-utilities")
    pubmed.add_argument("--limit", type=int, default=500)
    pubmed.add_argument("--batch", type=int, default=150)
    pubmed.add_argument("--resume", action="store_true", default=True)
    pubmed.add_argument("--ncbi-api-key", default=None)
    pubmed.add_argument("--email", default=None)

    web = subparsers.add_parser("web", help="crawl HTML pages from the published URL list")
    web.add_argument("--limit", type=int, default=200)
    web.add_argument("--kinds", default="vi,zh", help="comma separated: vi,zh,pubmed,other")
    web.add_argument("--hosts", default=None, help="comma separated host allow-list")
    web.add_argument("--per-host-cap", type=int, default=5)
    web.add_argument("--concurrency", type=int, default=32)
    web.add_argument("--per-host-concurrency", type=int, default=4)
    web.add_argument("--timeout", type=float, default=25.0)
    web.add_argument("--host-delay", type=float, default=0.0)
    web.add_argument("--progress-every", type=int, default=25)
    web.add_argument("--forbidden-retries", type=int, default=1,
                     help="extra attempts for HTTP 403 (URLs that keep failing go to the wayback pass)")
    web.add_argument("--no-interleave", action="store_true",
                     help="keep the published id order (one host at a time) instead of round-robining hosts")
    web.add_argument("--resume", action="store_true")
    web.add_argument("--ignore-robots", action="store_true")
    web.add_argument("--host-failure-threshold", type=int, default=8,
                     help="open a host circuit breaker after N consecutive blocks with no success (0 = never)")

    wayback = subparsers.add_parser("wayback", help="recover blocked/failed URLs from the Wayback Machine")
    wayback.add_argument("--limit", type=int, default=200)
    wayback.add_argument("--hosts", default=None, help="comma separated host allow-list")
    wayback.add_argument("--errors", default="http_403,empty_text", help="comma separated error codes to retry (stored-failure mode)")
    wayback.add_argument("--concurrency", type=int, default=6)
    wayback.add_argument("--delay", type=float, default=0.4,
                         help="mean seconds between requests per worker (archive.org rate-limits aggressively)")
    wayback.add_argument("--from-links", action="store_true",
                         help="select URLs from the corpus parquet instead of failed rows (needed for hosts that block every request profile)")
    wayback.add_argument("--kinds", default="vi,zh,other", help="kinds to consider with --from-links")
    wayback.add_argument("--include-no-snapshot", action="store_true",
                         help="also retry URLs already recorded as wayback_no_snapshot")
    wayback.add_argument("--no-direct", action="store_true",
                         help="skip the direct newest-snapshot endpoint and use the CDX API first")

    retry = subparsers.add_parser(
        "retry",
        help="re-crawl stored failures with the current extractor (e.g. after a decoding fix)",
    )
    retry.add_argument("--errors", default="empty_text", help="comma separated error codes to retry")
    retry.add_argument("--hosts", default=None, help="comma separated host allow-list")
    retry.add_argument("--limit", type=int, default=500)
    retry.add_argument("--concurrency", type=int, default=32)
    retry.add_argument("--per-host-concurrency", type=int, default=4)
    retry.add_argument("--timeout", type=float, default=25.0)
    retry.add_argument("--min-text-chars", type=int, default=200)
    retry.add_argument("--progress-every", type=int, default=25)
    retry.add_argument("--ignore-robots", action="store_true")
    return parser


async def run_retry(args: argparse.Namespace) -> int:
    """Re-crawl stored failures: a fixed extractor/decoder can turn them into documents."""
    jsonl_path, sqlite_path = _paths(args.data_dir)
    sink = CrawlSink(jsonl_path, sqlite_path)
    error_filter = [value.strip() for value in args.errors.split(",") if value.strip()]
    host_filter = [value.strip() for value in args.hosts.split(",") if value.strip()] if args.hosts else None
    rows = sink.failed_rows(errors=error_filter, hosts=host_filter, limit=args.limit)
    if not rows:
        print("no failed rows matched")
        sink.close()
        return 0
    done = sink.succeeded_ids()
    items = [
        (doc_id, url, host_of(url), classify(url))
        for doc_id, url in rows
        if doc_id not in done
    ]
    print(f"retry: {len(items)} urls (errors={error_filter}) hosts={args.hosts or 'any'}")

    async with Crawler(
        concurrency=args.concurrency,
        per_host_concurrency=args.per_host_concurrency,
        timeout=args.timeout,
        respect_robots=not args.ignore_robots,
        min_text_chars=args.min_text_chars,
    ) as crawler:
        counters = await crawler.crawl(items, sink.append, progress=_progress_printer("retry", args.progress_every))
        stats = crawler.stats

    sink.close()
    print(f"retry counters: {counters}")
    print(f"http stats: {stats}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "pubmed":
        return asyncio.run(run_pubmed(args))
    if args.command == "web":
        return asyncio.run(run_web(args))
    if args.command == "wayback":
        return asyncio.run(run_wayback(args))
    if args.command == "retry":
        return asyncio.run(run_retry(args))
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
