"""Tests for core text/submission helpers and the crawl/corpus basics."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

VI_TEXT = (
    "Bệnh tim mạch là nguyên nhân gây tử vong hàng đầu trên thế giới. "
    "Triệu chứng thường gặp gồm đau ngực, khó thở và mệt mỏi khi gắng sức. "
    "Người bệnh cần đi khám sớm để được chẩn đoán và điều trị kịp thời."
)
ZH_TEXT = "高血压患者的常见症状包括头痛、头晕和心悸。治疗需要长期服药。"
EN_TEXT = "Myocardial infarction usually presents with chest pain radiating to the left arm."


# --------------------------------------------------------------------- text
def test_tokenize_strips_vietnamese_diacritics_by_default():
    from r2ai2026.core.text import strip_diacritics, tokenize

    tokens = tokenize("Đau ngực khi gắng sức")
    assert tokens == ["dau", "nguc", "khi", "gang", "suc"]
    assert tokenize("Đau ngực", keep_diacritics=True) == ["đau", "ngực"]
    assert strip_diacritics("Điều trị bệnh tim mạch") == "Dieu tri benh tim mach"


def test_tokenize_emits_cjk_chars_and_bigrams():
    from r2ai2026.core.text import tokenize

    tokens = tokenize("高血压")
    assert "高" in tokens and "血压" in tokens


def test_detect_lang_separates_vi_zh_en():
    from r2ai2026.core.text import detect_lang

    assert detect_lang(VI_TEXT) == "vi"
    assert detect_lang(ZH_TEXT) == "zh"
    assert detect_lang(EN_TEXT) == "en"


def test_sentence_spans_cover_text_without_gaps():
    from r2ai2026.core.text import sentence_spans

    spans = sentence_spans(VI_TEXT)
    assert len(spans) == 3
    reconstructed = "".join(VI_TEXT[start:end] for start, end in spans)
    assert reconstructed == VI_TEXT


# ------------------------------------------------------------------- corpus
def test_corpus_classification_and_pmid_extraction():
    from r2ai2026.corpus.links import classify, host_of, pmid_of

    assert classify("https://pubmed.ncbi.nlm.nih.gov/12345678/") == "pubmed"
    assert pmid_of("https://pubmed.ncbi.nlm.nih.gov/12345678/") == 12345678
    assert pmid_of("https://pubmed.ncbi.nlm.nih.gov/12345678/?format=pubmed") == 12345678
    assert classify("https://suckhoedoisong.vn/bai-viet.htm") == "vi"
    assert classify("https://www.120ask.com/question/1.html") == "zh"
    assert classify("https://example.org/page") == "other"
    assert host_of("https://WWW.Vinmec.com/a/b") == "www.vinmec.com"


# ----------------------------------------------------------------- chunking
def test_crawl_sink_is_resumable(tmp_path):
    from r2ai2026.crawl.store import CrawlSink

    jsonl = tmp_path / "docs.jsonl"
    sqlite = tmp_path / "state.sqlite"
    sink = CrawlSink(jsonl, sqlite, flush_every=1)
    sink.append({"id": 1, "url": "https://a.vn/x", "status": 200, "ok": True, "n_chars": 900})
    sink.append({"id": 2, "url": "https://a.vn/y", "status": 404, "ok": False, "n_chars": 0, "error": "http_404"})
    sink.append({"id": 3, "url": "https://b.39.net/z", "status": 403, "ok": False, "n_chars": 0, "error": "http_403"})
    sink.close()

    reopened = CrawlSink(jsonl, sqlite)
    assert reopened.done_ids() == {1, 2, 3}
    assert reopened.succeeded_ids() == {1}  # a resume must retry failures
    assert reopened.failed_rows(errors=["http_403"]) == [(3, "https://b.39.net/z")]
    assert reopened.failed_rows(hosts=["39.net"]) == [(3, "https://b.39.net/z")]
    assert reopened.failed_rows(hosts=["a.vn"]) == [(2, "https://a.vn/y")]
    reopened.close()
    assert len(jsonl.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_wayback_cdx_parsing_and_snapshot_url():
    from r2ai2026.crawl.wayback import parse_cdx_json

    payload = [
        ["timestamp", "original", "statuscode"],
        ["20240101120000", "https://nhathuoclongchau.com.vn/bai-viet/x.html", "200"],
        ["20230601120000", "https://nhathuoclongchau.com.vn/bai-viet/x.html", "200"],
    ]
    snapshots = parse_cdx_json(payload)
    assert len(snapshots) == 2
    assert snapshots[0].timestamp == "20240101120000"
    assert snapshots[0].raw_url.endswith("id_/https://nhathuoclongchau.com.vn/bai-viet/x.html")
    assert parse_cdx_json([]) == []
    assert parse_cdx_json([["timestamp"]]) == []


# ----------------------------------------------------------------- extraction
def test_extract_article_prefers_main_content():
    from r2ai2026.crawl.fetch import extract_article

    html = """
    <html><head><title>Tăng huyết áp</title></head><body>
    <nav>Trang chủ | Tin tức | Liên hệ</nav>
    <article><h1>Tăng huyết áp</h1>
    <p>Tăng huyết áp là bệnh lý mạn tính phổ biến, cần theo dõi huyết áp thường xuyên.</p>
    <p>Bệnh nhân nên giảm muối, tập thể dục và dùng thuốc theo chỉ định của bác sĩ.</p></article>
    <footer>Bản quyền</footer></body></html>
    """
    title, text, strategy = extract_article(html, "https://suckhoedoisong.vn/tang-huyet-ap.htm")
    assert "Tăng huyết áp" in title or "Tăng huyết áp" in text
    assert "mạn tính phổ biến" in text
    assert strategy in {"trafilatura", "lxml", "stdlib"}


# ------------------------------------------------------------------- pubmed
def test_pubmed_xml_parsing():
    from r2ai2026.crawl.pubmed import parse_efetch_xml

    xml = b"""<?xml version="1.0"?>
    <PubmedArticleSet><PubmedArticle><MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <Journal><Title>Journal of Cardiology</Title>
          <JournalIssue><PubDate><Year>2019</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>Hypertension and <i>stroke</i> risk</ArticleTitle>
        <Abstract><AbstractText Label="BACKGROUND">Stroke is common.</AbstractText>
        <AbstractText Label="METHODS">We studied 100 patients.</AbstractText></Abstract>
        <Language>eng</Language>
      </Article>
    </MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    articles = parse_efetch_xml(xml)
    assert len(articles) == 1
    article = articles[0]
    assert article.pmid == 12345678
    assert "stroke" in article.title
    assert "BACKGROUND: Stroke is common." in article.abstract
    assert "2019" in article.text


# -------------------------------------------------------------------- bm25
def test_build_and_validate_submission(tmp_path):
    from r2ai2026.core.submission import build_submission, read_submission_zip, submission_stats, validate_submission, write_submission_zip

    predictions = {
        2: {"relevant_docs": [7, 7, 9], "relevant_chunks": [{"doc_id": 7, "chunk_text": "đau ngực"}]},
        1: {"relevant_docs": [5], "relevant_chunks": []},
    }
    records = build_submission(predictions)
    assert [record["id"] for record in records] == [1, 2]
    assert records[1]["relevant_docs"] == [7, 9]  # duplicates removed, order kept
    assert validate_submission(records, expected_ids=[1, 2]) == []

    stats = submission_stats(records)
    assert stats["n_queries"] == 2 and stats["n_chunks"] == 1

    path = write_submission_zip(records, tmp_path / "submission.zip")
    assert path["zip_bytes"] > 0
    assert read_submission_zip(tmp_path / "submission.zip") == records


def test_validate_submission_reports_problems():
    from r2ai2026.core.submission import validate_submission

    problems = validate_submission(
        [
            {"id": 1, "relevant_docs": ["7"], "relevant_chunks": []},
            {"id": 1, "relevant_docs": [7], "relevant_chunks": [{"doc_id": 7, "chunk_text": " "}]},
            {"id": 99, "relevant_docs": [], "relevant_chunks": []},
        ],
        expected_ids=[1, 2],
    )
    joined = " | ".join(problems)
    assert "not an int" in joined
    assert "duplicate query id 1" in joined
    assert "outside the query set" in joined
    assert "non-empty string" in joined


# ----------------------------------------------------------------- devproxy


def test_stage_lang_blocks():
    from r2ai2026.stages.bm25_experiment import lang_of
    from r2ai2026.stages.dense_embed import lang_of as dense_lang_of

    for f in (lang_of, dense_lang_of):
        assert [f(i) for i in (1, 766441, 766442, 4392214, 4392215)] == ["vi", "vi", "zh", "zh", "en"]
