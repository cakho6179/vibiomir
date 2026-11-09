"""Live smoke test: the real stack against the real internet.

Unit tests run against a local server, so they cannot tell us what the public
websites actually do to us (TLS quirks, GBK encoding, WAF blocks, redirects,
robots rules, NCBI throttling). This script talks to the real hosts, prints
per-URL ground truth, and writes ``runs/reports/smoke_live_<ts>.json``.

    python scripts/crawl/smoke_live.py                 # 12 URLs/kind + PubMed + Wayback
    python scripts/crawl/smoke_live.py --per-kind 30   # bigger sample
    python scripts/crawl/smoke_live.py --strict        # exit 1 on weak results

Nothing here is simulated: every number is measured over the network.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import httpx  # noqa: E402

from r2ai2026.corpus.links import host_of, load_links, sample_links  # noqa: E402
from r2ai2026.crawl.fetch import Crawler  # noqa: E402
from r2ai2026.crawl.pubmed import EFetchClient  # noqa: E402
from r2ai2026.crawl.wayback import wayback_record  # noqa: E402

DEFAULT_LINKS = "data/raw/btc/links_corpus_vi_en_zh.parquet"
# hosts that returned 403 in the real crawl probe -> the Wayback path matters most
BLOCKED_HOSTS = ["nhathuoclongchau.com.vn", "www.39.net", "familydoctor.com.cn"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--links", default=DEFAULT_LINKS)
    parser.add_argument("--per-kind", type=int, default=12, help="URLs sampled for vi / zh / other")
    parser.add_argument("--per-host", type=int, default=1, help="max URLs taken from any single host")
    parser.add_argument("--pubmed", type=int, default=5, help="PMID abstracts to fetch via E-utilities")
    parser.add_argument("--wayback", type=int, default=3, help="blocked-host URLs to try in the archive")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--out", default="runs/reports")
    parser.add_argument("--skip-web", action="store_true")
    parser.add_argument("--skip-pubmed", action="store_true")
    parser.add_argument("--skip-wayback", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit 1 when the live results are weak")
    return parser.parse_args(argv)


async def crawl_sample(links, per_kind: int, per_host: int, concurrency: int, timeout: float, seed: int) -> list[dict]:
    """Crawl a real internet sample and return the raw records."""
    kinds = ["vi", "zh", "other"]
    subset = sample_links(links, per_kind * len(kinds), kinds=kinds, per_host_cap=per_host, seed=seed)
    items = [(row.id, row.url, host_of(row.url), row.kind) for row in subset.itertuples()]

    records: list[dict] = []
    started = time.perf_counter()
    async with Crawler(
        concurrency=concurrency,
        per_host_concurrency=2,
        timeout=timeout,
        respect_robots=True,
    ) as crawler:
        await crawler.crawl(items, records.append)
    elapsed = time.perf_counter() - started

    for record in records:
        record["wall_seconds"] = round(elapsed, 2)
    return records


async def pubmed_sample(links, count: int) -> list[dict]:
    pmids = [int(p) for p in links.loc[links["kind"] == "pubmed", "pmid"].dropna().head(count)]
    if not pmids:
        return []
    client = EFetchClient(email="vibiomir-baseline@example.org")
    async with client:
        articles = await client.fetch_abstracts(pmids)
    return [
        {
            "pmid": article.pmid,
            "title": article.title,
            "n_chars": len(article.text),
            "lang": article.languages[0] if article.languages else "",
            "ok": len(article.text) >= 200,
        }
        for article in articles
    ]


async def wayback_sample(links, count: int) -> list[dict]:
    rows: list[dict] = []
    subset = sample_links(links, count, hosts=BLOCKED_HOSTS, per_host_cap=1, seed=7)
    if subset.empty:
        return rows
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(45.0),
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/141.0.0.0"},
    ) as client:
        for row in subset.itertuples():
            record = await wayback_record(client, row.id, row.url, host_of(row.url), row.kind)
            rows.append(
                {
                    "id": row.id,
                    "url": row.url,
                    "host": row.host,
                    "ok": record.get("ok", False),
                    "status": record.get("status", 0),
                    "error": record.get("error", ""),
                    "snapshot": record.get("final_url", ""),
                    "n_chars": record.get("n_chars", 0),
                    "strategy": record.get("strategy", ""),
                }
            )
    return rows


def print_records(title: str, records: list[dict], fields: tuple[str, ...]) -> None:
    print(f"\n=== {title} ({len(records)}) ===")
    for record in records:
        print("  " + " | ".join(str(record.get(field, ""))[:70] for field in fields))


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("ok")]
    chars = sorted(r.get("n_chars", 0) for r in ok)
    median = chars[len(chars) // 2] if chars else 0
    return {
        "n": len(records),
        "ok": len(ok),
        "ok_rate": round(len(ok) / len(records), 3) if records else 0.0,
        "median_chars_ok": median,
        "strategies": dict(Counter(r.get("strategy", "") for r in ok)),
        "errors": dict(Counter(r.get("error", "") for r in records if not r.get("ok")).most_common(8)),
        "by_kind": {
            kind: {
                "n": len(group),
                "ok_rate": round(sum(1 for r in group if r.get("ok")) / len(group), 3),
                "median_chars_ok": sorted(r.get("n_chars", 0) for r in group if r.get("ok"))[
                    len([r for r in group if r.get("ok")]) // 2
                ]
                if any(r.get("ok") for r in group)
                else 0,
            }
            for kind, group in _group_by(records, "kind").items()
        },
        "by_host": {
            host: round(sum(1 for r in group if r.get("ok")) / len(group), 2)
            for host, group in sorted(_group_by(records, "host").items(), key=lambda kv: -len(kv[1]))[:15]
        },
    }


def _group_by(records: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key, ""))].append(record)
    return groups


async def main_async(args: argparse.Namespace) -> int:
    links = load_links(args.links)
    print(f"links: {len(links):,} rows from {args.links}")

    started = time.perf_counter()
    if args.skip_web:
        web = []
        crawl_seconds = 0.0
    else:
        web = await crawl_sample(links, args.per_kind, args.per_host, args.concurrency, args.timeout, args.seed)
        crawl_seconds = time.perf_counter() - started
    if not args.skip_web:
        print_records("live web crawl", web, ("kind", "status", "n_chars", "elapsed", "strategy", "error", "url"))

    pubmed = [] if args.skip_pubmed else await pubmed_sample(links, args.pubmed)
    if not args.skip_pubmed:
        print_records("pubmed efetch", pubmed, ("pmid", "lang", "n_chars", "title"))

    archive = [] if args.skip_wayback else await wayback_sample(links, args.wayback)
    if not args.skip_wayback:
        print_records("wayback (blocked hosts)", archive, ("host", "status", "n_chars", "strategy", "error", "url"))

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": vars(args),
        "crawl": summarize(web),
        "crawl_wall_seconds": round(crawl_seconds, 2),
        "crawl_urls_per_second": round(len(web) / crawl_seconds, 3) if crawl_seconds else 0.0,
        "pubmed": summarize(pubmed),
        "wayback": summarize(archive),
    }

    print("\n=== summary ===")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"smoke_live_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {path}")

    if args.strict:
        web_rate = report["crawl"]["ok_rate"]
        pubmed_rate = report["pubmed"]["ok_rate"]
        vi_rate = report["crawl"]["by_kind"].get("vi", {}).get("ok_rate", 0.0)
        zh_rate = report["crawl"]["by_kind"].get("zh", {}).get("ok_rate", 0.0)
        if web_rate < 0.5 or vi_rate < 0.5 or zh_rate < 0.3 or pubmed_rate < 0.8:
            print(f"\nSTRICT FAIL: web={web_rate} vi={vi_rate} zh={zh_rate} pubmed={pubmed_rate}")
            return 1
        print(f"\nSTRICT PASS: web={web_rate} vi={vi_rate} zh={zh_rate} pubmed={pubmed_rate}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
