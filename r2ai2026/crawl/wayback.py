"""Wayback Machine fallback for hosts that block direct crawling.

Some hosts in the corpus (``nhathuoclongchau.com.vn``, ``*.familydoctor.com.cn``)
return 403 for every request profile, so their text has to come from an archive.
The CDX API lists archived snapshots and ``/web/<timestamp>id_/<url>`` serves the
raw archived HTML (no Wayback toolbar), which keeps extraction clean.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

import httpx

from r2ai2026.crawl.fetch import clean_extracted_text, decode_html
from r2ai2026.core.text import detect_lang

CDX_URL = "https://web.archive.org/cdx/search/cdx"
WAYBACK_RAW_URL = "https://web.archive.org/web/{timestamp}id_/{url}"
# asking for the snapshot "closest to year 9999" returns the newest one, and the
# wayback endpoint keeps working while the CDX API rate-limits us with 429
WAYBACK_LATEST_STAMP = "9999"
DEFAULT_TIMEOUT = 45.0


@dataclass(slots=True)
class Snapshot:
    timestamp: str
    original: str
    status: str

    @property
    def raw_url(self) -> str:
        return WAYBACK_RAW_URL.format(timestamp=self.timestamp, url=self.original)


def parse_cdx_json(payload: object) -> list[Snapshot]:
    """Parse a CDX ``output=json`` payload (first row is the header)."""
    if not isinstance(payload, list) or len(payload) < 2:
        return []
    header = [str(value).lower() for value in payload[0]]
    try:
        timestamp_index = header.index("timestamp")
        original_index = header.index("original")
    except ValueError:
        return []
    status_index = header.index("statuscode") if "statuscode" in header else None
    snapshots: list[Snapshot] = []
    for row in payload[1:]:
        if not isinstance(row, list) or len(row) <= max(timestamp_index, original_index):
            continue
        snapshots.append(
            Snapshot(
                timestamp=str(row[timestamp_index]),
                original=str(row[original_index]),
                status=str(row[status_index]) if status_index is not None and len(row) > status_index else "200",
            )
        )
    return snapshots


async def cdx_lookup(
    client: httpx.AsyncClient,
    url: str,
    *,
    limit: int = 3,
    status_ok: bool = True,
    attempts: int = 4,
    cdx_url: str = CDX_URL,
) -> tuple[list[Snapshot], str]:
    """Return ``(snapshots, error)`` for one URL.

    Measured against the live API (2026-10-19): ``collapse=urlkey`` answers 503
    and domain/prefix queries either 503 or 504, while a plain exact query
    answers in 0.3-1.6 s. So: no collapse, few rows, and real retries -- a
    transient 503 must not be stored as "this URL was never archived".
    """
    params = {
        "url": url,
        "output": "json",
        "limit": str(limit),
        "fl": "timestamp,original,statuscode",
    }
    if status_ok:
        params["filter"] = "statuscode:200"

    last_error = ""
    for attempt in range(attempts):
        try:
            response = await client.get(cdx_url, params=params, timeout=DEFAULT_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            last_error = f"transport:{type(exc).__name__}"
        else:
            if response.status_code == 200:
                try:
                    return parse_cdx_json(response.json()), ""
                except Exception:  # noqa: BLE001
                    return [], "bad_json"
            last_error = f"http_{response.status_code}"
        if attempt + 1 < attempts:
            await asyncio.sleep(min(20.0, 2.0**attempt) * (1 + random.random() * 0.3))
    return [], last_error or "unknown"


async def fetch_snapshot(
    client: httpx.AsyncClient,
    snapshot: Snapshot,
    *,
    max_bytes: int = 4_000_000,
    wayback_url: str = WAYBACK_RAW_URL,
) -> tuple[str, str, str]:
    """Return ``(title, text, strategy)`` for an archived snapshot."""
    from r2ai2026.crawl.fetch import extract_article

    target = wayback_url.format(timestamp=snapshot.timestamp, url=snapshot.original)
    try:
        async with client.stream("GET", target, timeout=DEFAULT_TIMEOUT) as response:
            if response.status_code != 200:
                return "", "", f"wayback_http_{response.status_code}"
            raw = b""
            async for piece in response.aiter_bytes():
                raw += piece
                if len(raw) >= max_bytes:
                    break
            payload = decode_html(raw, response.headers.get("content-type", ""))
        title, text, strategy = extract_article(payload, snapshot.original)
        return title, text, f"wayback:{strategy}"
    except Exception as exc:  # noqa: BLE001
        return "", "", f"wayback_error:{type(exc).__name__}"


async def fetch_latest_snapshot(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_bytes: int = 4_000_000,
    wayback_url: str = WAYBACK_RAW_URL,
    stamp: str = WAYBACK_LATEST_STAMP,
) -> tuple[int, str, str, str]:
    """Fetch the newest snapshot in one request, skipping the CDX lookup.

    Returns ``(status, title, text, strategy)``: status 0 means a transport error
    and the strategy string carries the reason. Measured on archive.org
    (2026-10-19): a 6-worker CDX pass got throttled with ``http_429`` within
    minutes, while ``/web/9999id_/<url>`` kept serving pages -- so the rescue pass
    tries this first and only falls back to CDX when the newest snapshot has no
    usable text. A 404 is a definitive "this URL was never archived".
    """
    from r2ai2026.crawl.fetch import extract_article

    target = wayback_url.format(timestamp=stamp, url=url)
    try:
        async with client.stream("GET", target, timeout=DEFAULT_TIMEOUT) as response:
            status = response.status_code
            if status != 200:
                return status, "", "", f"wayback_http_{status}"
            raw = b""
            async for piece in response.aiter_bytes():
                raw += piece
                if len(raw) >= max_bytes:
                    break
            payload = decode_html(raw, response.headers.get("content-type", ""))
    except Exception as exc:  # noqa: BLE001
        return 0, "", "", f"wayback_error:{type(exc).__name__}"
    title, text, strategy = extract_article(payload, url)
    return status, title, text, f"wayback:{strategy}"


async def wayback_record(
    client: httpx.AsyncClient,
    doc_id: int,
    url: str,
    host: str,
    kind: str,
    *,
    min_chars: int = 200,
    cdx_url: str = CDX_URL,
    wayback_url: str = WAYBACK_RAW_URL,
    direct: bool = False,
) -> dict:
    """Build a corpus record from the newest usable Wayback snapshot."""
    direct_empty = False
    if direct:
        status, title, text, strategy = await fetch_latest_snapshot(client, url, wayback_url=wayback_url)
        text = clean_extracted_text(text) if text else ""
        if len(text) >= min_chars:
            return {
                "id": doc_id,
                "url": url,
                "final_url": wayback_url.format(timestamp=WAYBACK_LATEST_STAMP, url=url),
                "host": host,
                "kind": kind,
                "status": 200,
                "ok": True,
                "error": "",
                "title": title,
                "text": text,
                "lang": detect_lang(text or title),
                "n_chars": len(text),
                "strategy": f"{strategy}+latest",
                "wayback_timestamp": WAYBACK_LATEST_STAMP,
            }
        if status == 404:
            return _failed_record(doc_id, url, host, kind, "wayback_no_snapshot")
        direct_empty = status == 200

    snapshots, cdx_error = await cdx_lookup(client, url, cdx_url=cdx_url)
    if cdx_error:
        # the archive was unreachable -- keep the URL in the retry pool instead of
        # recording it as "never archived"
        return _failed_record(doc_id, url, host, kind, f"wayback_cdx_{cdx_error}")
    if not snapshots:
        return _failed_record(doc_id, url, host, kind, "wayback_empty_text" if direct_empty else "wayback_no_snapshot")

    for snapshot in sorted(snapshots, key=lambda item: item.timestamp, reverse=True):
        title, text, strategy = await fetch_snapshot(client, snapshot, wayback_url=wayback_url)
        text = clean_extracted_text(text) if text else ""
        if len(text) >= min_chars:
            return {
                "id": doc_id,
                "url": url,
                "final_url": snapshot.raw_url,
                "host": host,
                "kind": kind,
                "status": 200,
                "ok": True,
                "error": "",
                "title": title,
                "text": text,
                "lang": detect_lang(text or title),
                "n_chars": len(text),
                "strategy": strategy,
                "wayback_timestamp": snapshot.timestamp,
            }
    return _failed_record(doc_id, url, host, kind, "wayback_empty_text")


def _failed_record(doc_id: int, url: str, host: str, kind: str, error: str) -> dict:
    return {
        "id": doc_id, "url": url, "host": host, "kind": kind, "status": 0, "ok": False,
        "error": error, "title": "", "text": "", "lang": "", "n_chars": 0, "strategy": "wayback",
    }
