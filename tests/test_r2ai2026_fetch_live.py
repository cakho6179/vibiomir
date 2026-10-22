"""Fetch-stack tests against a real HTTP server on 127.0.0.1.

Mocked transports only prove the code agrees with its own assumptions, so these
tests start a real ``http.server`` in a thread and let the crawler talk to it
over real sockets: real status codes, real headers, real charset decoding, real
robots.txt fetching, real retry/backoff timings.

The URLs in this file are localhost-only by design; anything that touches the
public internet lives in ``scripts/crawl/smoke_live.py`` instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ARTICLE_HTML = (
    "<html><head><title>Tăng huyết áp</title><meta charset=\"utf-8\"></head><body>"
    "<nav>Trang chủ | Tin tức | Liên hệ | Đăng nhập | Quảng cáo</nav>"
    "<article><h1>Tăng huyết áp</h1>"
    "<p>Tăng huyết áp là bệnh lý mạn tính phổ biến, người bệnh cần theo dõi huyết áp thường xuyên "
    "và dùng thuốc theo chỉ định của bác sĩ để tránh biến chứng tim mạch, đột quỵ.</p>"
    "<p>Bệnh nhân nên giảm muối, tập thể dục đều đặn, hạn chế rượu bia và kiểm tra sức khỏe định kỳ "
    "tại cơ sở y tế để được tư vấn điều trị phù hợp.</p>"
    "</article><footer>Bản quyền thuộc về tòa soạn</footer></body></html>"
)


class _Response:
    __slots__ = ("status", "body", "content_type")

    def __init__(self, status: int, body: str | bytes, content_type: str = "text/html; charset=utf-8") -> None:
        self.status = status
        self.body = body
        self.content_type = content_type

    @property
    def payload(self) -> bytes:
        """Bytes exactly as a real server would send them (GBK pages stay GBK)."""
        return self.body if isinstance(self.body, bytes) else self.body.encode("utf-8")


class _State:
    """Per-test request log + response plan (the "server side" of the test)."""

    def __init__(self, respond) -> None:
        self.respond = respond
        self.counts: dict[str, int] = {}
        self.headers: dict[str, dict[str, str]] = {}
        self.history: dict[str, list[dict[str, str]]] = {}

    def hit(self, path: str, headers: dict[str, str]) -> _Response:
        self.counts[path] = self.counts.get(path, 0) + 1
        self.headers.setdefault(path, headers)
        self.history.setdefault(path, []).append(dict(headers))
        return self.respond(path, self.counts[path])


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: _State

    def log_message(self, *args: object) -> None:  # keep pytest output clean
        pass

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        planned = self.state.hit(self.path, dict(self.headers))
        payload = planned.payload
        self.send_response(planned.status)
        self.send_header("Content-Type", planned.content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextlib.contextmanager
def local_server(respond) -> Iterator[tuple[str, _State]]:
    """Run a real HTTP server on 127.0.0.1 with a caller-supplied response plan."""
    state = _State(respond)
    handler = type("_BoundHandler", (_Handler,), {"state": state})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def crawl(urls: list[tuple[int, str]], *, respect_robots: bool = False, **kwargs) -> list[dict]:
    """Crawl ``(doc_id, url)`` pairs with the production crawler, synchronously."""
    from r2ai2026.corpus import classify, host_of
    from r2ai2026.fetch import Crawler

    records: list[dict] = []

    async def run() -> None:
        items = [
            (doc_id, url, host_of(url), classify(url))
            for doc_id, url in urls
        ]
        async with Crawler(concurrency=4, per_host_concurrency=2, respect_robots=respect_robots, **kwargs) as crawler:
            await crawler.crawl(items, records.append)

    asyncio.run(run())
    return records


# ----------------------------------------------------------------- behaviour
def test_forbidden_is_retried_once_on_a_real_server():
    def respond(path: str, count: int) -> _Response:
        return _Response(403, "forbidden")

    with local_server(respond) as (base, state):
        records = crawl([(1, f"{base}/bai-viet/chan")])

    assert state.counts["/bai-viet/chan"] == 2, "403 must be retried exactly once"
    assert records[0]["ok"] is False
    assert records[0]["error"] == "http_403"


def test_server_error_backoff_recovers_and_extracts_real_html():
    def respond(path: str, count: int) -> _Response:
        if count <= 2:
            return _Response(503, "service unavailable")
        return _Response(200, ARTICLE_HTML)

    with local_server(respond) as (base, state):
        records = crawl([(7, f"{base}/tang-huyet-ap")], max_retries=3)

    assert state.counts["/tang-huyet-ap"] == 3
    record = records[0]
    assert record["ok"] is True
    assert record["status"] == 200
    assert record["n_chars"] > 200
    assert "mạn tính phổ biến" in record["text"]
    assert "Quảng cáo" not in record["text"]  # boilerplate dropped
    assert record["lang"] == "vi"
    assert record["strategy"] in {"trafilatura", "lxml"}


def test_robots_disallow_prevents_the_request_and_uses_the_right_port():
    html_path = "/private/no-crawl"
    robots = "User-agent: *\nDisallow: /private/\n"

    def respond(path: str, count: int) -> _Response:
        if path == "/robots.txt":
            return _Response(200, robots, "text/plain; charset=utf-8")
        return _Response(200, ARTICLE_HTML)

    with local_server(respond) as (base, state):
        records = crawl([(3, f"{base}{html_path}")], respect_robots=True)

    assert state.counts.get("/robots.txt") == 1, "robots.txt must be fetched from the same host:port"
    assert state.counts.get(html_path) is None, "a disallowed URL must never be requested"
    assert records[0]["error"] == "robots_disallowed"
    assert records[0]["strategy"] == "robots"


def test_robots_allow_still_crawls():
    def respond(path: str, count: int) -> _Response:
        if path == "/robots.txt":
            return _Response(200, "User-agent: *\nDisallow: /private/\n", "text/plain; charset=utf-8")
        return _Response(200, ARTICLE_HTML)

    with local_server(respond) as (base, _state):
        records = crawl([(4, f"{base}/public/bai-viet")], respect_robots=True)

    assert records[0]["ok"] is True
    assert records[0]["error"] == ""


def test_robots_fetch_failure_does_not_block_crawling():
    def respond(path: str, count: int) -> _Response:
        if path == "/robots.txt":
            return _Response(500, "boom", "text/plain")
        return _Response(200, ARTICLE_HTML)

    with local_server(respond) as (base, _state):
        records = crawl([(5, f"{base}/bai-viet")], respect_robots=True)

    assert records[0]["ok"] is True


def test_media_suffix_is_skipped_without_a_request():
    def respond(path: str, count: int) -> _Response:
        return _Response(200, "\x89PNG\r\n\x1a\n binary", "image/png")

    with local_server(respond) as (base, state):
        records = crawl([(6, f"{base}/hinh-anh.png")])

    assert records[0]["ok"] is False
    assert records[0]["error"] == "unsupported_content_type"
    assert state.counts == {}, "a media URL must never be requested"


def test_non_html_content_type_is_rejected_after_the_response():
    def respond(path: str, count: int) -> _Response:
        return _Response(200, "\x89PNG\r\n\x1a\n binary", "image/png")

    with local_server(respond) as (base, state):
        records = crawl([(6, f"{base}/hinh-anh?id=7")])  # no media suffix

    assert state.counts["/hinh-anh?id=7"] == 1
    assert records[0]["ok"] is False
    assert records[0]["error"] == "content_type:image/png"


def test_empty_html_is_reported_as_empty_text():
    def respond(path: str, count: int) -> _Response:
        return _Response(200, "<html><body></body></html>")

    with local_server(respond) as (base, _state):
        records = crawl([(8, f"{base}/trang-trong")])

    assert records[0]["ok"] is False
    assert records[0]["error"] == "empty_text"


CHINESE_ARTICLE = (
    "<html><head><title>高血压的日常管理</title></head><body><article><p>"
    + "高血压患者的常见症状包括头痛、头晕和心悸，需要长期服药治疗。"
    + "日常生活中应减少盐分摄入，保持规律作息，并定期监测血压变化。"
    + "如果出现胸痛、呼吸困难等症状，应立即就医接受专业治疗。"
    + "医生会根据患者的具体情况调整用药方案，切勿自行停药或更改剂量。"
    + "合理的饮食和适度的运动有助于控制血压，降低心脑血管事件的风险。"
    + "</p></article></body></html>"
)


def test_gbk_page_with_charset_only_in_meta_tag_is_decoded_correctly():
    """Chinese sites routinely serve GBK with the charset only in a <meta> tag."""

    body = CHINESE_ARTICLE.replace("<head>", '<head><meta charset="gbk">')
    encoded = body.encode("gbk")
    assert b"<meta charset" in encoded

    def respond(path: str, count: int) -> _Response:
        # httpx decodes these bytes as UTF-8 and produces mojibake; decode_html sniffs the meta.
        return _Response(200, encoded, "text/html")

    with local_server(respond) as (base, state):
        records = crawl([(10, f"{base}/gao-xue-ya")], min_text_chars=50)

    assert records[0]["ok"] is True
    assert "高血压" in records[0]["text"]
    assert "血压" in records[0]["text"]
    assert records[0]["lang"] == "zh"


def test_gbk_page_with_charset_in_the_header_is_decoded_correctly():
    encoded = CHINESE_ARTICLE.encode("gbk")

    def respond(path: str, count: int) -> _Response:
        return _Response(200, encoded, "text/html; charset=gbk")

    with local_server(respond) as (base, _state):
        records = crawl([(11, f"{base}/gao-xue-ya-2")], min_text_chars=50)

    assert records[0]["ok"] is True
    assert "高血压" in records[0]["text"]


MALFORMED_HTML = (
    '<!DOCTYPE html>\n<html lang="" class="desktop">\n\n</html>\n\n<head>\n<meta charset="utf-8">\n'
    "<title>Báo Tây Ninh</title>\n</head>\n<body>\n"
    "<nav>Trang chủ | Tin tức | Liên hệ</nav>\n"
    "<article><h1>Nghe có giúp giảm đau dạ dày?</h1>"
    "<p>Chuyên gia cho biết chế độ ăn uống đóng vai trò quan trọng trong điều trị bệnh dạ dày, "
    "người bệnh nên ăn đúng giờ và tránh các thực phẩm nhiều dầu mỡ, cay nóng để giảm triệu chứng "
    "đau và đầy bụng, đồng thời dùng thuốc theo chỉ định của bác sĩ chuyên khoa tiêu hóa.</p>"
    "<p>Nếu tình trạng đau kéo dài kèm theo nôn ra máu hoặc sụt cân, người bệnh cần đi khám ngay "
    "tại cơ sở y tế để nội soi và xác định nguyên nhân cụ thể trước khi tự ý điều trị tại nhà.</p>"
    "</article>\n</body>\n"
)


def test_head_after_broken_html_is_still_extracted():
    """Real sites (baotayninh.vn) ship <head>/<body> after </html>; libxml2 drops all of it."""

    def respond(path: str, count: int) -> _Response:
        return _Response(200, MALFORMED_HTML)

    with local_server(respond) as (base, _state):
        records = crawl([(20, f"{base}/nghe-giam-dau-da-day")])

    record = records[0]
    assert record["ok"] is True
    assert record["title"] == "Báo Tây Ninh"
    assert "đau dạ dày" in record["text"]
    assert record["strategy"] in {"lxml", "stdlib"}


JS_ONLY_HTML = (
    "<html><head><title>Bệnh tiểu đường - Bệnh viện Đa khoa</title>"
    '<meta name="description" content="Bệnh tiểu đường là bệnh rối loạn chuyển hóa mạn tính '
    'cần theo dõi đường huyết và dùng thuốc hằng ngày theo chỉ định.">'
    '<meta property="og:description" content="Triệu chứng tiểu đường gồm khát nước, tiểu nhiều, '
    'sụt cân và mờ mắt, cần xét nghiệm máu để chẩn đoán chính xác.">'
    "</head><body><div id=\"app\"></div></body></html>"
)


def test_js_rendered_page_falls_back_to_meta_description():
    """A client-side page carries no prose; the meta block keeps the doc retrievable."""

    def respond(path: str, count: int) -> _Response:
        return _Response(200, JS_ONLY_HTML)

    with local_server(respond) as (base, _state):
        records = crawl([(21, f"{base}/benh-tieu-duong")], min_text_chars=50)

    record = records[0]
    assert record["ok"] is True
    assert record["strategy"] in {"meta", "stdlib+meta", "lxml+meta"}
    assert "Bệnh tiểu đường" in record["text"]
    assert "khát nước" in record["text"]


def test_short_pages_are_kept_when_min_text_chars_allows_it():
    short = "<html><body><article><p>高血压需要长期治疗。</p></article></body></html>"

    def respond(path: str, count: int) -> _Response:
        return _Response(200, short)

    with local_server(respond) as (base, _state):
        strict = crawl([(12, f"{base}/duan")])
        lenient = crawl([(13, f"{base}/duan-2")], min_text_chars=5)

    assert strict[0]["ok"] is False and strict[0]["error"] == "empty_text"
    assert lenient[0]["ok"] is True and "高血压" in lenient[0]["text"]


def test_crawl_writes_records_through_the_sink(tmp_path: Path):
    """The real path from HTTP response to resumable JSONL + SQLite state."""

    from r2ai2026.store import CrawlSink

    def respond(path: str, count: int) -> _Response:
        return _Response(200, ARTICLE_HTML)

    with local_server(respond) as (base, _state):
        records = crawl([(11, f"{base}/a"), (12, f"{base}/b")])

    sink = CrawlSink(tmp_path / "docs.jsonl", tmp_path / "crawl.sqlite", flush_every=1)
    for record in records:
        sink.append(record)
    sink.close()

    reopened = CrawlSink(tmp_path / "docs.jsonl", tmp_path / "crawl.sqlite")
    assert reopened.succeeded_ids() == {11, 12}
    assert reopened.failed_rows() == []
    reopened.close()


def test_host_circuit_breaker_stops_hammering_a_dead_host():
    """120ask.com and iiyi.com answer 521 for every URL; 98k URLs must not be retried 3x each."""

    def respond(path: str, count: int) -> _Response:
        return _Response(521, "")

    urls = [(100 + index, f"__BASE__/cau-hoi/{index}.htm") for index in range(24)]
    with local_server(respond) as (base, state):
        records = crawl([(doc_id, url.replace("__BASE__", base)) for doc_id, url in urls],
                        max_retries=0, host_failure_threshold=3)

    requests_made = sum(state.counts.values())
    circuit_errors = [record for record in records if record["error"].startswith("host_circuit_open:")]
    assert len(records) == 24
    assert len(circuit_errors) >= 15, "after the threshold the rest must fail without a request"
    assert requests_made <= 8, f"only a handful of probes may reach a dead host, got {requests_made}"
    assert all(record["error"] in {"http_521", "host_circuit_open:http_521"} for record in records)


def test_rate_limit_slows_the_host_down_instead_of_blocking_it():
    """A 429 host is alive: back off adaptively and still collect the pages."""

    def respond(path: str, count: int) -> _Response:
        if count <= 2:
            return _Response(429, "slow down")
        return _Response(200, ARTICLE_HTML)

    with local_server(respond) as (base, state):
        records = crawl([(30, f"{base}/a")], max_retries=3)

    assert state.counts["/a"] == 3
    assert records[0]["ok"] is True


def test_circuit_breaker_does_not_open_on_empty_text_or_robots():
    """empty_text / robots_disallowed are per-URL outcomes, not host blocks."""

    def respond(path: str, count: int) -> _Response:
        return _Response(200, "<html><body></body></html>")

    with local_server(respond) as (base, _state):
        records = crawl([(40 + index, f"{base}/trang-trong-{index}") for index in range(10)],
                        host_failure_threshold=3)

    assert all(record["error"] == "empty_text" for record in records)


# ------------------------------------------------------------------- wayback
# ------------------------------------------------------- JS cookie challenge
# exact bytes served by laodong.vn for every request (177 bytes, 2026-10-19)
LIVE_CHALLENGE_HTML = (
    '<html><body><script>document.cookie="D1N=3dd97c956221517d5c899caa866f54be"+"; '
    'expires=Fri, 31 Dec 2099 23:59:59 GMT; path=/";window.location.reload(true);</script></body></html>'
)


def test_cookie_challenge_parser_matches_the_live_laodong_wall():
    from r2ai2026.fetch import parse_js_cookie_challenge

    assert parse_js_cookie_challenge(LIVE_CHALLENGE_HTML) == {"D1N": "3dd97c956221517d5c899caa866f54be"}
    # concatenated literal form ("a="+"b")
    assert parse_js_cookie_challenge('<script>document.cookie="ses="+"abc123";</script>') == {"ses": "abc123"}


def test_cookie_challenge_parser_never_fires_on_a_real_page():
    from r2ai2026.fetch import parse_js_cookie_challenge

    assert parse_js_cookie_challenge("") == {}
    # a real article with an analytics script that writes cookies
    page = "<html><body><article><p>Thuốc điều trị tăng huyết áp</p></article>" "<script>document.cookie='_ga=GA1.1'</script></body></html>"
    assert parse_js_cookie_challenge(page) == {}
    # a plain div-only landing page without any cookie script
    assert parse_js_cookie_challenge("<html><body><div>Loading the clinic list</div></body></html>") == {}
    # large documents are out of scope even if they mention document.cookie
    assert parse_js_cookie_challenge("document.cookie" + "x" * 5000) == {}


def test_crawler_solves_a_js_cookie_challenge_and_returns_the_article():
    def respond(path: str, count: int) -> _Response:
        if count == 1:
            return _Response(200, LIVE_CHALLENGE_HTML)
        return _Response(200, ARTICLE_HTML)

    with local_server(respond) as (base, state):
        records = crawl([(7, f"{base}/suc-khoe/bai-viet.ldo")])

    assert records[0]["ok"] is True, records[0]["error"]
    assert records[0]["n_chars"] > 200
    assert "huyết áp" in records[0]["text"]
    assert state.counts["/suc-khoe/bai-viet.ldo"] == 2
    # the retry must carry the cookie parsed from the wall, not a blind retry
    assert "D1N=3dd97c956221517d5c899caa866f54be" in state.history["/suc-khoe/bai-viet.ldo"][1].get("Cookie", "")
    assert state.history["/suc-khoe/bai-viet.ldo"][0].get("Cookie", "") == ""


def test_a_wall_that_never_lifts_is_recorded_as_empty_text_not_retried_forever():
    def respond(path: str, count: int) -> _Response:
        return _Response(200, LIVE_CHALLENGE_HTML)

    with local_server(respond) as (base, state):
        records = crawl([(8, f"{base}/wall")])

    assert records[0]["ok"] is False
    assert records[0]["error"] == "empty_text"
    assert state.counts["/wall"] == 2  # challenge + exactly one retry with the cookie


def _wayback_record(base: str, url: str, *, direct: bool = False, min_chars: int = 200) -> dict:
    import asyncio

    import httpx

    from r2ai2026.wayback import wayback_record

    async def run() -> dict:
        async with httpx.AsyncClient() as client:
            return await wayback_record(
                client, 1, url, "blocked.vn", "vi",
                cdx_url=f"{base}/cdx",
                wayback_url=f"{base}/web/{{timestamp}}id_/{{url}}",
                direct=direct,
                min_chars=min_chars,
            )

    return asyncio.run(run())


def hits(state: _State, prefix: str) -> int:
    """Requests whose path starts with ``prefix`` (paths carry the query string)."""
    return sum(count for path, count in state.counts.items() if path.startswith(prefix))


def test_wayback_transient_cdx_failure_is_not_recorded_as_never_archived():
    """A 503 from the archive must stay retryable, otherwise URLs are lost forever."""

    def respond(path: str, count: int) -> _Response:
        return _Response(503, "<html>Internet Archive: Temporarily Offline</html>")

    with local_server(respond) as (base, state):
        record = _wayback_record(base, "https://blocked.vn/bai-viet/x.html")

    assert record["ok"] is False
    assert record["error"].startswith("wayback_cdx_http_503")
    assert hits(state, "/cdx") >= 2, "transient failures must be retried"


def test_wayback_absent_snapshot_is_recorded_as_no_snapshot():
    """An empty CDX answer is a real answer: the URL is simply not archived."""

    def respond(path: str, count: int) -> _Response:
        return _Response(200, "[]", "application/json")

    with local_server(respond) as (base, state):
        record = _wayback_record(base, "https://blocked.vn/bai-viet/y.html")

    assert record["ok"] is False
    assert record["error"] == "wayback_no_snapshot"
    assert hits(state, "/cdx") == 1


def test_wayback_header_only_cdx_answer_is_no_snapshot():
    def respond(path: str, count: int) -> _Response:
        return _Response(200, '["urlkey","timestamp","original","mimetype","statuscode"]', "application/json")

    with local_server(respond) as (base, _state):
        record = _wayback_record(base, "https://blocked.vn/bai-viet/z.html")

    assert record["error"] == "wayback_no_snapshot"


def test_wayback_direct_endpoint_serves_the_newest_snapshot_without_cdx():
    """/web/9999id_/<url> is one request instead of CDX+fetch (the CDX API 429s)."""

    def respond(path: str, count: int) -> _Response:
        if path.startswith("/web/"):
            return _Response(200, ARTICLE_HTML)
        raise AssertionError(f"unexpected request to {path}")

    with local_server(respond) as (base, state):
        record = _wayback_record(base, "https://blocked.vn/bai-viet/direct.html", direct=True)

    assert record["ok"] is True, record["error"]
    assert record["n_chars"] > 200
    assert record["strategy"].endswith("+latest")
    assert hits(state, "/cdx") == 0, "the direct path must not touch the rate-limited CDX API"


def test_wayback_direct_404_is_a_definitive_never_archived():
    def respond(path: str, count: int) -> _Response:
        return _Response(404, "<html>not archived</html>")

    with local_server(respond) as (base, state):
        record = _wayback_record(base, "https://blocked.vn/bai-viet/missing.html", direct=True)

    assert record["error"] == "wayback_no_snapshot"
    assert hits(state, "/cdx") == 0


def test_wayback_direct_js_only_page_falls_back_to_the_cdx_snapshot_list():
    """A JS-only newest snapshot may be useless; older ones are listed by CDX."""

    def respond(path: str, count: int) -> _Response:
        if path.startswith("/web/9999"):
            return _Response(200, "<html><body><div id='app'></div></body></html>")
        if path.startswith("/web/"):
            return _Response(200, ARTICLE_HTML)
        if path.startswith("/cdx"):
            return _Response(
                200,
                '[["timestamp","original","statuscode"],["20200101000000","https://blocked.vn/b.html","200"]]',
                "application/json",
            )
        raise AssertionError(f"unexpected request to {path}")

    with local_server(respond) as (base, state):
        record = _wayback_record(base, "https://blocked.vn/bai-viet/js-only.html", direct=True)

    assert record["ok"] is True, record["error"]
    assert record["strategy"].startswith("wayback:")
    assert hits(state, "/cdx") == 1, "the fallback must go through the CDX list exactly once"


# ------------------------------------------------------------------- storage
def test_append_never_blocks_on_a_locked_database(tmp_path: Path):
    """The bug that stopped the crawl: a parallel process held the write lock and
    the event loop sat in a commit retry, so the crawl produced 0 docs/second."""

    import sqlite3
    import time as time_module

    from r2ai2026.store import CrawlSink

    jsonl_path = tmp_path / "docs.jsonl"
    sqlite_path = tmp_path / "crawl.sqlite"

    blocker = CrawlSink(jsonl_path, sqlite_path)
    blocker.close()

    holder = sqlite3.connect(sqlite_path, timeout=5.0)
    holder.execute("BEGIN EXCLUSIVE")
    holder.execute("UPDATE docs SET error = 'held' WHERE 0")  # keep the exclusive lock

    try:
        sink = CrawlSink(jsonl_path, sqlite_path)
        worst = 0.0
        for index in range(20):
            started = time_module.perf_counter()
            sink.append({"id": index, "url": f"https://host.vn/{index}", "ok": True, "n_chars": 100})
            worst = max(worst, time_module.perf_counter() - started)
        assert worst < 0.05, f"append blocked for {worst * 1000:.1f} ms while another process held the lock"
        # the JSONL payload must already be on disk even while SQLite waits
        assert len(jsonl_path.read_text(encoding="utf-8").strip().splitlines()) == 20
    finally:
        holder.rollback()
        holder.close()

    sink.close()
    con = sqlite3.connect(sqlite_path, timeout=30.0)
    assert con.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 20
    con.close()


def test_two_concurrent_sinks_append_intact_records(tmp_path: Path):
    """Two crawl processes share docs.jsonl; no line may be interleaved or cut."""

    import threading

    from r2ai2026.store import CrawlSink, iter_jsonl

    jsonl_path = tmp_path / "docs.jsonl"
    sqlite_path = tmp_path / "crawl.sqlite"
    big_text = "Bệnh nhân cần theo dõi huyết áp hằng ngày. " * 1200  # ~48 KB per record

    def writer(offset: int) -> None:
        sink = CrawlSink(jsonl_path, sqlite_path, flush_every=1)
        for index in range(40):
            sink.append(
                {
                    "id": offset + index,
                    "url": f"https://host.vn/bai-viet/{offset + index}",
                    "text": big_text,
                    "ok": True,
                    "n_chars": len(big_text),
                }
            )
        sink.close()

    threads = [threading.Thread(target=writer, args=(offset,)) for offset in (0, 1000, 2000)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    records = list(iter_jsonl(jsonl_path))
    assert len(records) == 120, "every appended record must survive intact"
    assert {record["id"] for record in records} == set(range(40)) | set(range(1000, 1040)) | set(range(2000, 2040))
    assert all(len(record["text"]) == len(big_text) for record in records)
