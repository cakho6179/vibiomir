"""Tests for the R2AI2026 baseline: text, corpus, chunking, index, submission, pipeline."""

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
    from r2ai2026.text import strip_diacritics, tokenize

    tokens = tokenize("Đau ngực khi gắng sức")
    assert tokens == ["dau", "nguc", "khi", "gang", "suc"]
    assert tokenize("Đau ngực", keep_diacritics=True) == ["đau", "ngực"]
    assert strip_diacritics("Điều trị bệnh tim mạch") == "Dieu tri benh tim mach"


def test_tokenize_emits_cjk_chars_and_bigrams():
    from r2ai2026.text import tokenize

    tokens = tokenize("高血压")
    assert "高" in tokens and "血压" in tokens


def test_detect_lang_separates_vi_zh_en():
    from r2ai2026.text import detect_lang

    assert detect_lang(VI_TEXT) == "vi"
    assert detect_lang(ZH_TEXT) == "zh"
    assert detect_lang(EN_TEXT) == "en"


def test_sentence_spans_cover_text_without_gaps():
    from r2ai2026.text import sentence_spans

    spans = sentence_spans(VI_TEXT)
    assert len(spans) == 3
    reconstructed = "".join(VI_TEXT[start:end] for start, end in spans)
    assert reconstructed == VI_TEXT


# ------------------------------------------------------------------- corpus
def test_corpus_classification_and_pmid_extraction():
    from r2ai2026.corpus import classify, host_of, pmid_of

    assert classify("https://pubmed.ncbi.nlm.nih.gov/12345678/") == "pubmed"
    assert pmid_of("https://pubmed.ncbi.nlm.nih.gov/12345678/") == 12345678
    assert pmid_of("https://pubmed.ncbi.nlm.nih.gov/12345678/?format=pubmed") == 12345678
    assert classify("https://suckhoedoisong.vn/bai-viet.htm") == "vi"
    assert classify("https://www.120ask.com/question/1.html") == "zh"
    assert classify("https://example.org/page") == "other"
    assert host_of("https://WWW.Vinmec.com/a/b") == "www.vinmec.com"


# ----------------------------------------------------------------- chunking
def test_chunk_document_covers_text_and_respects_target():
    from r2ai2026.chunking import chunk_document

    text = " ".join(f"câu số {index} nói về bệnh tim mạch." for index in range(1, 121))
    chunks = chunk_document(text, target_tokens=80, overlap_tokens=20)
    assert len(chunks) > 3
    assert all(chunk.text.strip() for chunk in chunks)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    assert max(chunk.n_tokens for chunk in chunks) < 200  # sentence rounding only


def test_merge_selected_keeps_document_order_and_budget():
    from r2ai2026.chunking import chunk_document, merge_selected

    text = " ".join(f"đoạn {index} mô tả triệu chứng đau ngực." for index in range(1, 81))
    chunks = chunk_document(text, target_tokens=40, overlap_tokens=0)
    merged_backwards = merge_selected(chunks, [2, 0])
    merged_forward = merge_selected(chunks, [0, 2])
    assert merged_backwards == merged_forward  # order is normalized to document order
    assert merged_forward.startswith(chunks[0].text)
    limited = merge_selected(chunks, [0, 1, 2], max_tokens=10)
    assert limited == chunks[0].text


def test_whole_document_chunk_truncates_only_at_sentence_boundary():
    from r2ai2026.chunking import whole_document_chunk
    from r2ai2026.text import word_count

    short = "Câu một. Câu hai."
    assert whole_document_chunk(short, max_tokens=100) == short
    long_text = " ".join(f"Câu {index} kể về bệnh nhân." for index in range(1, 200))
    trimmed = whole_document_chunk(long_text, max_tokens=50)
    assert word_count(trimmed) <= 60
    assert trimmed.endswith(".")
    assert long_text.startswith(trimmed)


# --------------------------------------------------------------------- store
def test_crawl_sink_is_resumable(tmp_path):
    from r2ai2026.store import CrawlSink

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
    from r2ai2026.wayback import parse_cdx_json

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
    from r2ai2026.fetch import extract_article

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
    from r2ai2026.pubmed import parse_efetch_xml

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
def test_bm25_ranks_matching_document_first():
    import numpy as np

    from r2ai2026.bm25 import BM25Index, rank_ids

    docs = {
        11: "bệnh tim mạch đau ngực khó thở khi gắng sức",
        22: "sốt cao co giật ở trẻ nhỏ cần xử trí ngay",
        33: "viêm dạ dày tá tràng do vi khuẩn helicobacter pylori",
    }
    index = BM25Index()
    index.build(np.array([[key] for key in docs], dtype=np.int64), list(docs.values()))
    hits = index.search("đau ngực gắng sức", k=3)
    assert rank_ids(hits)[0] == 11


# ------------------------------------------------------------------ submit
def test_build_and_validate_submission(tmp_path):
    from r2ai2026.submit import build_submission, read_submission_zip, submission_stats, validate_submission, write_submission_zip

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
    from r2ai2026.submit import validate_submission

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
def test_pseudo_gold_spans_are_recovered_by_whole_document_chunks():
    from r2ai2026.devproxy import build_pseudo_gold, score_span_recovery
    from r2ai2026.chunking import whole_document_chunk

    # index-unique sentences: no shared boilerplate for LCS to latch onto
    documents = {
        1: " ".join(f"s{i} t{i} u{i} v{i} w{i}." for i in range(1, 201)),
        2: " ".join(f"a{i} b{i} c{i} d{i} e{i}." for i in range(1, 201)),
        3: " ".join(f"p{i} q{i} r{i} yy{i} zz{i}." for i in range(1, 201)),
    }
    gold = build_pseudo_gold(documents, seed=1, spans=3, span_tokens=60)
    chunks = {doc_id: [(doc_id, whole_document_chunk(text, max_tokens=None))] for doc_id, text in documents.items()}
    score = score_span_recovery(chunks, gold, predicted_docs={doc_id: [doc_id] for doc_id in documents})
    assert score.chunks_precision == 1.0
    assert score.chunks_recall == 1.0
    assert score.chunks_f2 == 1.0

    truncated = {doc_id: [(doc_id, whole_document_chunk(text, max_tokens=40))] for doc_id, text in documents.items()}
    weak = score_span_recovery(truncated, gold, predicted_docs={doc_id: [doc_id] for doc_id in documents})
    assert weak.chunks_recall < score.chunks_recall


def test_chunk_policies_stay_within_budget_and_cover_document_ends():
    from r2ai2026.chunking import head_tail_chunk, spread_windows_chunk, whole_document_chunk
    from r2ai2026.text import word_count

    text = " ".join(f"s{i} t{i} u{i} v{i} w{i}." for i in range(1, 601))
    budget = 300

    head = whole_document_chunk(text, max_tokens=budget)
    assert text.startswith(head)
    assert word_count(head) <= budget * 1.25  # budget plus the sentence it lands in
    assert head != text

    mixed = head_tail_chunk(text, max_tokens=budget)
    assert mixed.startswith(text[:20])
    assert mixed.rstrip().endswith("w600.")  # the tail really reaches the document end
    assert word_count(mixed) <= budget * 1.5

    spread = spread_windows_chunk(text, max_tokens=budget, windows=6)
    pieces = [piece for piece in spread.split("\n") if piece.strip()]
    assert len(pieces) == 6
    positions = [text.index(piece) for piece in pieces]
    assert positions == sorted(positions)  # document order is preserved (LCS needs order)
    assert word_count(spread) <= budget * 1.5

    # a document that fits inside the budget is submitted verbatim by every policy
    short = "câu một. câu hai. câu ba."
    assert whole_document_chunk(short, max_tokens=budget) == short
    assert head_tail_chunk(short, max_tokens=budget) == short
    assert spread_windows_chunk(short, max_tokens=budget, windows=6) == short


def test_policies_reach_perfect_chunk_f2_when_the_budget_covers_the_document():
    from r2ai2026.chunking import head_tail_chunk, spread_windows_chunk, whole_document_chunk
    from r2ai2026.devproxy import build_pseudo_gold, score_span_recovery

    text = " ".join(f"s{i} t{i} u{i} v{i} w{i}." for i in range(1, 201))  # ~1000 tokens
    documents = {1: text, 2: text.replace("s", "z"), 3: text.replace("t", "y")}
    gold = build_pseudo_gold(documents, seed=5, spans=4, span_tokens=60)

    policies = (
        lambda body: whole_document_chunk(body, max_tokens=4000),
        lambda body: head_tail_chunk(body, max_tokens=4000),
        lambda body: spread_windows_chunk(body, max_tokens=4000, windows=6),
    )
    for policy in policies:
        chunks = {doc_id: [(doc_id, policy(body))] for doc_id, body in documents.items()}
        score = score_span_recovery(chunks, gold, predicted_docs={doc_id: [doc_id] for doc_id in documents})
        assert score.chunks_f2 == 1.0


def test_policy_comparison_reports_payload_and_f2():
    from r2ai2026.chunking import whole_document_chunk
    from r2ai2026.devproxy import policy_comparison

    documents = {doc_id: " ".join(f"w{i} x{i} y{i} z{i}." for i in range(1, 1201)) for doc_id in (1, 2, 3)}
    rows = policy_comparison(
        documents,
        {"head": lambda text, budget: whole_document_chunk(text, max_tokens=budget)},
        budgets=(200, 0),
        max_documents=None,
    )
    assert [int(row["budget_tokens"]) for row in rows] == [0, 200]
    assert all(0.0 <= row["chunk_f2"] <= 1.0 for row in rows)
    full_row = next(row for row in rows if int(row["budget_tokens"]) == 0)
    assert full_row["chunk_f2"] == 1.0  # the whole document covers every pseudo-gold span
    assert full_row["mean_chars"] > rows[1]["mean_chars"]


def test_retrieval_metrics_recall_and_mrr():
    from r2ai2026.devproxy import retrieval_metrics

    metrics = retrieval_metrics(
        {1: [4, 5, 6], 2: [7, 8]},
        {1: [5], 2: [9]},
        ks=(1, 3),
    )
    assert metrics.recall_at[1] == 0.0
    assert metrics.recall_at[3] == 0.5
    assert abs(metrics.mrr - 0.25) < 1e-9  # query 1 hits at rank 2, query 2 misses


# ----------------------------------------------------------------- pipeline
def _write_corpus(tmp_path):
    import pandas as pd

    documents = [
        {"id": 101, "url": "https://a.vn/1", "host": "a.vn", "kind": "vi", "ok": True, "title": "Đau ngực khi gắng sức",
         "text": "Đau ngực khi gắng sức là dấu hiệu của bệnh mạch vành. " * 12},
        {"id": 102, "url": "https://a.vn/2", "host": "a.vn", "kind": "vi", "ok": True, "title": "Sốt cao co giật",
         "text": "Sốt cao co giật thường gặp ở trẻ nhỏ dưới 5 tuổi. " * 12},
        {"id": 103, "url": "https://b.cn/3", "host": "b.cn", "kind": "zh", "ok": True, "title": "高血压",
         "text": "高血压患者的常见症状包括头痛头晕。 " * 12},
        {"id": 104, "url": "https://c.org/4", "host": "c.org", "kind": "other", "ok": False, "title": "", "text": ""},
    ]
    docs_path = tmp_path / "docs.jsonl"
    with open(docs_path, "w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")

    queries_path = tmp_path / "query.parquet"
    pd.DataFrame({"id": [1, 2], "question": ["đau ngực khi gắng sức", "sốt cao co giật"]}).to_parquet(queries_path)
    return docs_path, queries_path


def test_pipeline_end_to_end_writes_valid_submission(tmp_path):
    from r2ai2026.pipeline import BaselineConfig, BaselinePipeline
    from r2ai2026.submit import read_submission_zip, validate_submission

    docs_path, queries_path = _write_corpus(tmp_path)
    config = BaselineConfig(
        docs_path=docs_path,
        queries_path=queries_path,
        index_dir=tmp_path / "index",
        submission_path=tmp_path / "submission.zip",
        topk_docs=2,
        chunk_source="doc_head",
        chunk_budget_tokens=60,
        min_doc_chars=50,
    )
    pipeline = BaselinePipeline(config)
    assert pipeline.load_corpus() == 3  # the failed document is dropped
    pipeline.build_indexes()
    records = pipeline.run()

    problems = validate_submission(records, expected_ids=[1, 2])
    assert problems == [], problems
    assert all(len(record["relevant_docs"]) >= 1 for record in records)
    assert all(record["relevant_chunks"] for record in records)

    stats = pipeline.write_submission(records)
    assert stats["zip_mb"] > 0
    assert read_submission_zip(config.submission_path) == records


def test_pipeline_index_round_trip(tmp_path):
    from r2ai2026.pipeline import BaselineConfig, BaselinePipeline

    docs_path, queries_path = _write_corpus(tmp_path)
    config = BaselineConfig(
        docs_path=docs_path,
        queries_path=queries_path,
        index_dir=tmp_path / "index",
        topk_docs=2,
        chunk_source="bm25_merge",
        min_doc_chars=50,
    )
    pipeline = BaselinePipeline(config)
    pipeline.load_corpus()
    sizes = pipeline.build_indexes()
    assert sizes["n_docs"] == 3 and sizes["n_chunks"] > 0
    pipeline.save_indexes()
    records_before = pipeline.run()

    reloaded = BaselinePipeline(config)
    reloaded.load_corpus()
    reloaded.build_chunks()
    reloaded.load_indexes()
    assert reloaded.run() == records_before


def test_pipeline_doc_spread_chunk_source_covers_the_whole_document(tmp_path):
    from r2ai2026.pipeline import BaselineConfig, BaselinePipeline

    import pandas as pd

    long_body = " ".join(f"s{i} t{i} u{i} v{i} w{i}." for i in range(1, 401))
    docs_path = tmp_path / "docs.jsonl"
    with open(docs_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": 7, "ok": True, "title": "", "text": long_body}, ensure_ascii=False) + "\n")
    queries_path = tmp_path / "query.parquet"
    pd.DataFrame({"id": [1], "question": ["s5 t5 u5 v5 w5"]}).to_parquet(queries_path)

    config = BaselineConfig(
        docs_path=docs_path,
        queries_path=queries_path,
        topk_docs=1,
        chunk_source="doc_spread",
        chunk_budget_tokens=200,
        spread_windows=4,
        min_doc_chars=50,
    )
    pipeline = BaselinePipeline(config)
    pipeline.load_corpus()
    pipeline.build_indexes()
    records = pipeline.run()
    chunk_text = records[0]["relevant_chunks"][0]["chunk_text"]

    assert chunk_text != long_body[: len(chunk_text)]  # not a head-only slice
    pieces = [piece for piece in chunk_text.split("\n") if piece.strip()]
    assert len(pieces) == 4
    positions = [long_body.index(piece) for piece in pieces]
    assert positions == sorted(positions)
    assert long_body.rstrip().endswith(pieces[-1].rstrip()[-20:])  # the last slice is near the end


def test_document_text_merges_title_and_body():
    from r2ai2026.pipeline import document_text

    assert document_text({"title": "Tiêu đề", "text": "Nội dung"}) == "Tiêu đề\nNội dung"
    assert document_text({"title": "", "text": "Nội dung"}) == "Nội dung"
