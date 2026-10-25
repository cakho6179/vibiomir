"""Real-file tests for the Wayback rescue selection (blocked hosts have no store rows).

``nhathuoclongchau.com.vn`` (79,598 URLs) and ``www.120ask.com`` (98,160) answer
403/521 for every request profile, so they never get a row in the crawl store:
a failed-row query finds nothing and the rescue pass has to select from the
published link list instead. These tests build a real parquet + real SQLite store
and check the selection.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai2026.corpus.links import load_links, pending_links  # noqa: E402
from r2ai2026.crawl.store import CrawlSink  # noqa: E402


@pytest.fixture()
def links_parquet(tmp_path: Path) -> Path:
    """Published-list stand-in: one blocked host, one reachable host, one zh host."""
    rows = [
        (101, "https://nhathuoclongchau.com.vn/thuoc/paracetamol.html"),
        (102, "https://nhathuoclongchau.com.vn/thuoc/aspirin.html"),
        (103, "https://nhathuoclongchau.com.vn/thuoc/ibuprofen.html"),
        (201, "https://suckhoedoisong.vn/benh-tieu-duong.htm"),
        (202, "https://suckhoedoisong.vn/benh-tim-mach.htm"),
        (301, "https://www.120ask.com/question/12345.html"),
    ]
    path = tmp_path / "links.parquet"
    pd.DataFrame(rows, columns=["id", "url"]).to_parquet(path, index=False)
    return path


def test_pending_links_selects_blocked_host_rows(links_parquet: Path):
    links = load_links(links_parquet)
    rows = pending_links(links, hosts=["nhathuoclongchau.com.vn"], exclude_ids=set(), limit=10)
    assert rows == [
        (101, "https://nhathuoclongchau.com.vn/thuoc/paracetamol.html"),
        (102, "https://nhathuoclongchau.com.vn/thuoc/aspirin.html"),
        (103, "https://nhathuoclongchau.com.vn/thuoc/ibuprofen.html"),
    ]


def test_pending_links_skips_ids_already_stored_and_respects_limit(links_parquet: Path):
    links = load_links(links_parquet)
    rows = pending_links(
        links,
        hosts=["nhathuoclongchau.com.vn", "suckhoedoisong.vn"],
        exclude_ids={101, 103, 201},
        limit=2,
    )
    assert rows == [(102, "https://nhathuoclongchau.com.vn/thuoc/aspirin.html"),
                    (202, "https://suckhoedoisong.vn/benh-tim-mach.htm")]


def test_pending_links_filters_by_kind(links_parquet: Path):
    links = load_links(links_parquet)
    assert [doc_id for doc_id, _ in pending_links(links, kinds=["zh"], exclude_ids=set())] == [301]
    # nhathuoclongchau.com.vn is a .vn host, so it classifies as vi, not "blocked"
    assert [doc_id for doc_id, _ in pending_links(links, kinds=["vi"], exclude_ids=set())] == [101, 102, 103, 201, 202]


def test_never_archived_urls_are_excluded_from_the_rescue_pool(tmp_path: Path, links_parquet: Path):
    """A URL whose CDX lookup returned no snapshot must not be re-queued forever."""
    jsonl_path = tmp_path / "docs.jsonl"
    sqlite_path = tmp_path / "crawl.sqlite"
    sink = CrawlSink(jsonl_path, sqlite_path)
    sink.append({"id": 101, "url": "https://nhathuoclongchau.com.vn/thuoc/paracetamol.html",
                 "ok": True, "n_chars": 900, "status": 200})
    sink.append({"id": 103, "url": "https://nhathuoclongchau.com.vn/thuoc/ibuprofen.html",
                 "ok": False, "n_chars": 0, "status": 0, "error": "wayback_no_snapshot"})
    sink.append({"id": 301, "url": "https://www.120ask.com/question/12345.html",
                 "ok": False, "n_chars": 0, "status": 0, "error": "wayback_cdx_http_503"})
    sink.close()

    sink = CrawlSink(jsonl_path, sqlite_path)
    done = sink.succeeded_ids()
    permanent = sink.ids_with_error_prefix("wayback_no_snapshot")
    transient = sink.ids_with_error_prefix("wayback_cdx")
    sink.close()

    assert done == {101}
    assert permanent == {103}
    assert transient == {301}  # a 503 from the archive stays retryable

    links = load_links(links_parquet)
    rescue = pending_links(
        links,
        hosts=["nhathuoclongchau.com.vn", "www.120ask.com"],
        exclude_ids=done | permanent,
        limit=10,
    )
    assert [doc_id for doc_id, _ in rescue] == [102, 301]


def test_store_state_survives_a_second_process_writing(tmp_path: Path, links_parquet: Path):
    """The rescue pass runs next to the main crawl: both must see each other's rows."""
    jsonl_path = tmp_path / "docs.jsonl"
    sqlite_path = tmp_path / "crawl.sqlite"
    first = CrawlSink(jsonl_path, sqlite_path)
    second = CrawlSink(jsonl_path, sqlite_path)
    try:
        first.append({"id": 101, "url": "https://nhathuoclongchau.com.vn/thuoc/paracetamol.html",
                      "ok": True, "n_chars": 900, "status": 200})
        second.append({"id": 102, "url": "https://nhathuoclongchau.com.vn/thuoc/aspirin.html",
                       "ok": True, "n_chars": 800, "status": 200})
        first.flush()
        second.flush()
        assert first.succeeded_ids() == {101, 102}
        assert second.succeeded_ids() == {101, 102}
    finally:
        first.close()
        second.close()

    assert len(jsonl_path.read_text(encoding="utf-8").strip().splitlines()) == 2
    con = sqlite3.connect(sqlite_path, timeout=30.0)
    assert con.execute("SELECT COUNT(*) FROM docs WHERE ok = 1").fetchone()[0] == 2
    con.close()


def test_interleave_by_host_keeps_every_item_and_mixes_hosts():
    """The published list is host-grouped; crawling it in order idles 88/96 slots."""
    from r2ai2026.corpus.links import interleave_by_host

    rows = [
        (1, "https://a.vn/1", "a.vn", "vi"),
        (2, "https://a.vn/2", "a.vn", "vi"),
        (3, "https://a.vn/3", "a.vn", "vi"),
        (4, "https://b.vn/1", "b.vn", "vi"),
        (5, "https://b.vn/2", "b.vn", "vi"),
        (6, "https://c.cn/1", "c.cn", "zh"),
    ]
    ordered = interleave_by_host(rows)

    assert sorted(ordered) == sorted(rows), "no item may be dropped or duplicated"
    assert [row[2] for row in ordered[:3]] == ["a.vn", "b.vn", "c.cn"], "hosts must rotate"
    # the three a.vn items arrive on the first three rounds, in their own order
    assert [row[0] for row in ordered if row[2] == "a.vn"] == [1, 2, 3]
    assert len(interleave_by_host([])) == 0
    assert interleave_by_host(rows[:1]) == rows[:1]
