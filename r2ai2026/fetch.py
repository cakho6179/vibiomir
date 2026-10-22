"""Polite async fetching and article extraction.

Only public pages from the organizers' own URL list are fetched, at low
per-host concurrency with ``robots.txt`` respected. Extraction prefers
trafilatura (boilerplate removal) and falls back to a conservative lxml text
pass so a failure never silently drops a document.
"""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura
from lxml import html as lxml_html

from r2ai2026.text import detect_lang, normalize, split_paragraphs

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/141.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,zh-CN;q=0.8,en-US;q=0.7,en;q=0.6",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

_UNSUPPORTED_SUFFIXES = (
    ".pdf", ".zip", ".rar", ".7z", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp3", ".mp4", ".avi", ".mov", ".svg", ".ico",
)
_RETRY_STATUS = {403, 408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 524}
# status codes that mean "this host is blocked", not "this URL is broken"
_CIRCUIT_STATUS = {401, 403, 451, 520, 521, 522, 523}
_MIN_TEXT_CHARS = 200

_CHARSET_HEADER_RE = re.compile(r"charset\s*=\s*[\"']?([\w\-]+)", re.IGNORECASE)
_CHARSET_META_RE = re.compile(
    rb"""<meta[^>]{0,400}?charset\s*=\s*["']?\s*([\w\-]+)""",
    re.IGNORECASE,
)
_TEXTUAL_TOKENS = ("html", "text", "xml")


def charset_from_content_type(content_type: str) -> str | None:
    if not content_type:
        return None
    match = _CHARSET_HEADER_RE.search(content_type)
    return match.group(1) if match else None


def charset_from_meta(raw: bytes) -> str | None:
    """Sniff ``<meta charset>`` from the head of the document (first 8 KB)."""
    match = _CHARSET_META_RE.search(raw[:8192])
    return match.group(1).decode("ascii", errors="ignore") if match else None


def decode_html(raw: bytes, content_type: str = "") -> str:
    """Decode an HTML payload the way real Vietnamese/Chinese sites need.

    ``httpx`` only trusts the charset in the ``Content-Type`` header and silently
    falls back to UTF-8, which turns the 217k Chinese pages of this corpus into
    mojibake whenever a page is served as GBK/GB18030/Big5 with the charset only
    in a ``<meta>`` tag. So: header charset, then meta charset, then UTF-8, then
    statistical detection as a last resort.
    """
    for candidate in (charset_from_content_type(content_type), charset_from_meta(raw), "utf-8"):
        if not candidate:
            continue
        try:
            return raw.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(raw[:200_000]).best()
        if best is not None:
            return str(best)
    except Exception:
        pass
    return raw.decode("utf-8", errors="replace")


def is_textual_content_type(content_type: str) -> bool:
    if not content_type:
        return True  # some servers omit it; extraction + the length check decide
    lowered = content_type.lower()
    return any(token in lowered for token in _TEXTUAL_TOKENS)


_JS_COOKIE_ASSIGN_RE = re.compile(r"""document\.cookie\s*=\s*([^;]+)""")
_JS_STRING_RE = re.compile(r"""["']([^"']*)["']""")


def parse_js_cookie_challenge(payload: str) -> dict[str, str]:
    """Cookies from a JavaScript anti-bot wall, or ``{}`` when this is a real page.

    ``laodong.vn`` (31,282 corpus URLs) answers every request with 177 bytes of

        <script>document.cookie="D1N=3dd97c...";window.location.reload(true);</script>

    so the whole host looked like ``empty_text``. The wall is trivially solvable
    without a JS engine: take the literal, send it back as a cookie, and the next
    response is the article. Kept deliberately narrow -- small body, script only,
    no paragraph markup -- so real pages are never mistaken for a challenge.
    """
    if not payload or len(payload) > 4096:
        return {}
    lowered = payload.lower()
    if "document.cookie" not in lowered or "<script" not in lowered:
        return {}
    if "<p" in lowered or "<article" in lowered or "<div" in lowered:
        return {}
    cookies: dict[str, str] = {}
    for match in _JS_COOKIE_ASSIGN_RE.finditer(payload):
        # handles both "D1N=abc" and "D1N="+"abc" concatenation forms
        value = "".join(_JS_STRING_RE.findall(match.group(1)))
        value = value.split(";", 1)[0].strip().rstrip("+").strip()
        name, separator, cookie_value = value.partition("=")
        name = name.strip()
        if separator and name and len(name) < 64 and "\n" not in name:
            cookies[name] = cookie_value.strip()
    return cookies


@dataclass(slots=True)
class CrawlResult:
    url: str
    final_url: str
    status_code: int
    ok: bool
    error: str
    title: str
    text: str
    lang: str
    n_chars: int
    elapsed: float
    strategy: str = ""

    def to_record(self, doc_id: int, host: str, kind: str) -> dict:
        return {
            "id": doc_id,
            "url": self.url,
            "final_url": self.final_url,
            "host": host,
            "kind": kind,
            "status": self.status_code,
            "ok": self.ok,
            "error": self.error,
            "title": self.title,
            "text": self.text,
            "lang": self.lang,
            "n_chars": self.n_chars,
            "elapsed": round(self.elapsed, 3),
            "strategy": self.strategy,
        }


_CJK_RANGES = (
    (0x3400, 0x4DBF),  # CJK ext A
    (0x4E00, 0x9FFF),  # CJK unified
    (0x3040, 0x30FF),  # kana
    (0xAC00, 0xD7AF),  # hangul
    (0xF900, 0xFAFF),  # CJK compatibility
)


def has_cjk(text: str) -> bool:
    """True when the text contains CJK characters (no spaces between words)."""
    return any(any(low <= ord(ch) <= high for low, high in _CJK_RANGES) for ch in text)


def clean_extracted_text(text: str) -> str:
    """Drop menu junk and collapse whitespace while keeping paragraph breaks."""
    lines: list[str] = []
    for paragraph in split_paragraphs(text):
        collapsed = normalize(paragraph)
        if len(collapsed) < 3:
            continue
        if has_cjk(collapsed):
            # CJK writes without spaces, so the Latin "single token" rule would
            # delete every normal Chinese sentence -- use a length floor instead.
            if len(collapsed) < 6:
                continue
        elif collapsed.count(" ") < 1 and len(collapsed) < 12:
            continue  # single-token nav / breadcrumb crumbs
        lines.append(paragraph.strip())
    return "\n".join(lines).strip()


_META_TAG_RE = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_META_ATTR_RE = re.compile(r"""([\w:\-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""")
_META_KEYS = {
    "description", "og:description", "twitter:description",
    "og:title", "twitter:title", "keywords", "news_keywords",
}


class _TextCollector(HTMLParser):
    """Tag stripper that survives documents libxml2 refuses to parse.

    Some sites ship ``<html></html><head>..<body>..`` (head/body outside the
    root); libxml2 then keeps only a couple of characters. The stdlib parser is
    byte-oriented and structure-agnostic, so the prose survives.
    """

    _SKIP = {"script", "style", "noscript", "template", "svg", "iframe"}
    _BREAK = {"p", "div", "br", "li", "tr", "td", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BREAK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def strip_html_text(html_text: str) -> str:
    """Return the readable text of an HTML payload without a DOM parser."""
    collector = _TextCollector()
    try:
        collector.feed(html_text)
        collector.close()
    except Exception:  # noqa: BLE001 - stdlib parser is forgiving, but never crash a crawl
        return ""
    return "".join(collector.parts)


def meta_text(html_text: str, title: str = "", limit: int = 1500) -> str:
    """Title + meta description/keywords, for pages whose body is JS-rendered.

    A fifth of the corpus renders its article client-side; the chatty meta tags
    are then the only real content in the payload, and doc-level recall (half of
    FINAL_SCORE) still benefits from them.
    """
    pieces: list[str] = [title.strip()] if title.strip() else []
    for match in _META_TAG_RE.finditer(html_text):
        attrs = {
            attr.group(1).lower(): (attr.group(2) or attr.group(3) or attr.group(4) or "")
            for attr in _META_ATTR_RE.finditer(match.group(0))
        }
        key = (attrs.get("name") or attrs.get("property") or "").lower()
        content = unescape(attrs.get("content", "")).strip()
        if content and key in _META_KEYS:
            pieces.append(content)
    return " ".join(dict.fromkeys(pieces))[:limit]


def extract_article(html_text: str, url: str = "") -> tuple[str, str, str]:
    """Return ``(title, text, strategy)`` for an HTML payload."""
    if not html_text or not html_text.strip():
        return "", "", "empty"

    title = ""
    text = ""
    strategy = ""
    try:
        metadata = trafilatura.extract_metadata(html_text, default_url=url)
        title = (getattr(metadata, "title", "") or "").strip()
        extracted = trafilatura.extract(
            html_text,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=False,
            favor_recall=True,
        )
        if extracted:
            text = clean_extracted_text(extracted)
            strategy = "trafilatura"
    except Exception:  # trafilatura is defensive, but never let it kill a fetch
        text = ""

    if len(text) < _MIN_TEXT_CHARS:
        try:
            document = lxml_html.fromstring(html_text)
            for bad in document.xpath("//script|//style|//noscript|//nav|//footer|//header|//aside|//form"):
                bad.getparent().remove(bad)
            if not title:
                title_nodes = document.xpath("//title/text()")
                title = title_nodes[0].strip() if title_nodes else ""
            fallback = clean_extracted_text(document.text_content())
            if len(fallback) > len(text):
                text = fallback
                strategy = "lxml"
        except Exception:
            pass

    if not title:
        # libxml2 refuses documents with content after </html>; the title still exists
        match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.S | re.IGNORECASE)
        if match:
            title = unescape(match.group(1)).strip()

    if len(text) < _MIN_TEXT_CHARS:
        # last structural attempt: tag-strip without a DOM parser
        stripped = clean_extracted_text(strip_html_text(html_text))
        if len(stripped) > len(text):
            text = stripped
            strategy = "stdlib"

    if len(text) < 400:
        # JS-rendered page: the meta block is the only real description of the
        # article, so keep it instead of dropping the document entirely.
        meta = meta_text(html_text, title)
        if meta and len(meta) > len(text):
            text = f"{meta}\n{text}".strip() if text else meta
            strategy = "meta" if strategy in {"", "none", "empty"} else f"{strategy}+meta"

    return title, text, strategy or "none"


class RobotsCache:
    """Per-host robots.txt cache shared by every task in one crawl."""

    def __init__(self, client: httpx.AsyncClient, user_agent: str, enabled: bool = True) -> None:
        self._client = client
        self._user_agent = user_agent
        self._enabled = enabled
        self._cache: dict[str, RobotFileParser | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _parser(self, netloc: str, scheme: str) -> RobotFileParser | None:
        lock = self._locks.setdefault(netloc, asyncio.Lock())
        async with lock:
            if netloc in self._cache:
                return self._cache[netloc]
            parser: RobotFileParser | None = None
            try:
                response = await self._client.get(f"{scheme}://{netloc}/robots.txt", timeout=10.0)
                if response.status_code == 200 and response.text:
                    parser = RobotFileParser()
                    parser.parse(response.text.splitlines())
            except Exception:
                parser = None
            self._cache[netloc] = parser
            return parser

    async def allowed(self, url: str) -> bool:
        if not self._enabled:
            return True
        parts = urlsplit(url)
        # netloc keeps an explicit port ("host:8080") -- hostname would silently
        # send the robots request to port 80/443 on the wrong server.
        netloc = parts.netloc or parts.hostname or ""
        parser = await self._parser(netloc, parts.scheme or "https")
        if parser is None:
            return True
        try:
            return parser.can_fetch(self._user_agent, url)
        except Exception:
            return False


class Crawler:
    """Async crawler with global and per-host concurrency limits."""

    def __init__(
        self,
        *,
        concurrency: int = 32,
        per_host_concurrency: int = 4,
        timeout: float = 25.0,
        user_agent: str = DEFAULT_USER_AGENT,
        respect_robots: bool = True,
        max_retries: int = 2,
        max_forbidden_retries: int = 1,
        max_bytes: int = 4_000_000,
        min_text_chars: int = _MIN_TEXT_CHARS,
        min_host_delay: float = 0.0,
        host_failure_threshold: int = 8,
    ) -> None:
        self.concurrency = concurrency
        self.per_host_concurrency = per_host_concurrency
        self.timeout = timeout
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self.max_retries = max_retries
        self.max_forbidden_retries = max_forbidden_retries
        self.max_bytes = max_bytes
        self.min_text_chars = min_text_chars
        self.min_host_delay = min_host_delay
        self.host_failure_threshold = host_failure_threshold
        self._client: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._host_semaphores: dict[str, asyncio.Semaphore] = {}
        self._host_last_request: dict[str, float] = {}
        self._robots: RobotsCache | None = None
        self._stats = {
            "requests": 0, "ok": 0, "errors": 0, "skipped_robots": 0,
            "circuit_open": 0, "cookie_challenge": 0,
        }
        self._stats_lock = asyncio.Lock()
        # host-level circuit breaker: a whole host can be blocked (120ask.com and
        # iiyi.com answer 521 nginx for every URL). Without this, 98k URLs would
        # burn three attempts each before the wayback pass gets a chance.
        self._host_state: dict[str, list[int]] = {}  # host -> [consecutive_failures, successes]
        self._host_blocked: dict[str, str] = {}
        self._host_delay: dict[str, float] = {}

    @property
    def blocked_hosts(self) -> dict[str, str]:
        return dict(self._host_blocked)

    def _note_host(self, host: str, *, ok: bool, reason: str = "") -> None:
        state = self._host_state.setdefault(host, [0, 0])
        if ok:
            state[1] += 1
            state[0] = 0
            return
        state[0] += 1
        if (
            reason
            and self.host_failure_threshold > 0
            and state[1] == 0
            and state[0] >= self.host_failure_threshold
            and host not in self._host_blocked
        ):
            self._host_blocked[host] = reason
            self._stats["circuit_open"] += 1

    async def __aenter__(self) -> Crawler:
        limits = httpx.Limits(max_connections=self.concurrency * 2, max_keepalive_connections=self.concurrency)
        self._client = httpx.AsyncClient(
            http2=False,
            follow_redirects=True,
            timeout=httpx.Timeout(self.timeout, connect=min(10.0, self.timeout)),
            limits=limits,
            headers={"User-Agent": self.user_agent, **DEFAULT_HEADERS},
        )
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._robots = RobotsCache(self._client, self.user_agent, self.respect_robots)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    async def _throttle(self, host: str) -> None:
        delay = max(self.min_host_delay, self._host_delay.get(host, 0.0))
        if delay <= 0:
            return
        loop = asyncio.get_running_loop()
        last = self._host_last_request.get(host, 0.0)
        wait = delay - (loop.time() - last)
        if wait > 0:
            await asyncio.sleep(wait)
        self._host_last_request[host] = loop.time()

    async def fetch(self, url: str) -> CrawlResult:
        assert self._client is not None and self._semaphore is not None, "use as async context manager"
        host = urlsplit(url).hostname or ""
        if url.lower().endswith(_UNSUPPORTED_SUFFIXES):
            return CrawlResult(url, url, 0, False, "unsupported_content_type", "", "", "", 0, 0.0, "skipped")

        async with self._semaphore:
            host_semaphore = self._host_semaphores.setdefault(host, asyncio.Semaphore(self.per_host_concurrency))
            async with host_semaphore:
                if self._robots is not None and not await self._robots.allowed(url):
                    async with self._stats_lock:
                        self._stats["skipped_robots"] += 1
                    return CrawlResult(url, url, 0, False, "robots_disallowed", "", "", "", 0, 0.0, "robots")

                blocked_reason = self._host_blocked.get(host)
                if blocked_reason:
                    return CrawlResult(
                        url, url, 0, False, f"host_circuit_open:{blocked_reason}", "", "", "", 0, 0.0, "circuit",
                    )

                last_error = ""
                challenge_cookies: dict[str, str] = {}
                for attempt in range(self.max_retries + 1):
                    await self._throttle(host)
                    started = asyncio.get_running_loop().time()
                    async with self._stats_lock:
                        self._stats["requests"] += 1
                    try:
                        # streaming keeps memory bounded on 980k pages: a body is
                        # read up to max_bytes and the connection is then dropped.
                        async with self._client.stream("GET", url, cookies=challenge_cookies or None) as response:
                            elapsed = asyncio.get_running_loop().time() - started
                            status = response.status_code
                            final_url = str(response.url)
                            content_type = response.headers.get("content-type", "")

                            # 403 is usually a permanent block: one retry, then the
                            # wayback pass picks the URL up instead of hammering the host.
                            retries_allowed = (
                                min(self.max_retries, self.max_forbidden_retries)
                                if status == 403
                                else self.max_retries
                            )
                            if status in _RETRY_STATUS and attempt < retries_allowed:
                                if status == 429:
                                    # the host is alive but wants us slower
                                    self._host_delay[host] = min(5.0, max(0.5, self._host_delay.get(host, 0.0) * 2))
                                await asyncio.sleep(min(10.0, 1.0 * 2**attempt) * (1 + random.random() * 0.3))
                                continue

                            if status >= 400:
                                async with self._stats_lock:
                                    self._stats["errors"] += 1
                                self._note_host(
                                    host, ok=False,
                                    reason=f"http_{status}" if status in _CIRCUIT_STATUS else "",
                                )
                                return CrawlResult(
                                    url, final_url, status, False,
                                    f"http_{status}", "", "", "", 0, elapsed, "http_error",
                                )

                            if not is_textual_content_type(content_type):
                                async with self._stats_lock:
                                    self._stats["errors"] += 1
                                return CrawlResult(
                                    url, final_url, status, False,
                                    f"content_type:{content_type.split(';')[0]}", "", "", "", 0, elapsed, "skipped",
                                )

                            raw = b""
                            async for piece in response.aiter_bytes():
                                raw += piece
                                if len(raw) >= self.max_bytes:
                                    break

                            payload = decode_html(raw, content_type)
                            if not challenge_cookies:
                                solved = parse_js_cookie_challenge(payload)
                                if solved:
                                    challenge_cookies.update(solved)
                                    async with self._stats_lock:
                                        self._stats["cookie_challenge"] += 1
                                    await asyncio.sleep(0.2 + random.random() * 0.3)
                                    continue
                            title, text, strategy = extract_article(payload, final_url)
                            lang = detect_lang(text or title)
                            ready = len(text) >= self.min_text_chars
                            async with self._stats_lock:
                                if ready:
                                    self._stats["ok"] += 1
                                else:
                                    self._stats["errors"] += 1
                            self._note_host(host, ok=ready)
                            if ready and self._host_delay.get(host):
                                self._host_delay[host] = max(self.min_host_delay, self._host_delay[host] * 0.5)
                            return CrawlResult(
                                url, final_url, status, ready,
                                "" if ready else "empty_text", title, text, lang, len(text), elapsed, strategy,
                            )
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        if attempt < self.max_retries:
                            await asyncio.sleep(min(8.0, 0.6 * 2**attempt) * (1 + random.random() * 0.3))
                            continue
                        async with self._stats_lock:
                            self._stats["errors"] += 1
                        self._note_host(host, ok=False, reason=f"transport:{type(exc).__name__}")
                        return CrawlResult(url, url, 0, False, last_error, "", "", "", 0, 0.0, "error")

        raise RuntimeError("crawl loop exited without producing a result")

    async def crawl(
        self,
        items: Iterable[tuple[int, str, str, str]],
        on_result: Callable[[dict], None],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, int]:
        """Fetch ``(doc_id, url, host, kind)`` items, calling ``on_result`` per document.

        ``on_result`` runs on the event loop thread, so the sink must be cheap
        (append to a file) rather than CPU heavy.
        """
        items = list(items)
        total = len(items)
        completed = 0
        counters = {"ok": 0, "empty": 0, "failed": 0}
        queue: asyncio.Queue = asyncio.Queue()

        async def worker() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    return
                doc_id, url, host, kind = item
                result = await self.fetch(url)
                record = result.to_record(doc_id, host, kind)
                on_result(record)
                counters["ok" if result.ok else "failed"] += 1
                nonlocal completed
                completed += 1
                if progress is not None:
                    progress(completed, total)
                queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(self.concurrency)]
        for item in items:
            await queue.put(item)
        for _ in workers:
            await queue.put(None)
        await queue.join()
        for task in workers:
            task.cancel()
        return counters


async def iter_crawl(
    items: Iterable[tuple[int, str, str, str]],
    *,
    crawler: Crawler,
) -> AsyncIterator[dict]:
    """Stream crawl records (used by tests and small ad-hoc runs)."""
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    async def producer() -> None:
        for item in items:
            await queue.put(item)
        await queue.put(sentinel)

    producer_task = asyncio.create_task(producer())
    while True:
        item = await queue.get()
        if item is sentinel:
            break
        doc_id, url, host, kind = item
        result = await crawler.fetch(url)
        yield result.to_record(doc_id, host, kind)
    await producer_task


def summarize_records(records: Iterable[dict]) -> dict[str, float]:
    records = list(records)
    if not records:
        return {"n": 0}
    ok = [record for record in records if record.get("ok")]
    chars = [record.get("n_chars", 0) for record in ok]
    return {
        "n": float(len(records)),
        "n_ok": float(len(ok)),
        "ok_rate": len(ok) / len(records),
        "mean_chars": sum(chars) / len(chars) if chars else 0.0,
    }
