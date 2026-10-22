"""Single source of truth for workspace paths (independent of the current directory)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"  # junction -> ..\data\r2ai2026-vibiomir (7 GB, not copied)
# Read-only source data (see data/raw/README.md); everything else under data/ is derived from it.
RAW = DATA / "raw"
QUERIES = RAW / "btc" / "query.parquet"
LINKS = RAW / "btc" / "links_corpus_vi_en_zh.parquet"
DOCS_RAW = RAW / "crawl" / "docs.jsonl"  # crawler output, frozen 2026-10-20
CRAWL_DB = RAW / "crawl" / "crawl.sqlite"
RECOVERED = RAW / "recovered"  # one append-only file per recovery source and date
# Derived
DOCS_CLEAN = DATA / "docs_clean.jsonl"  # corpus v2, 1,183,751 real docs, text as crawled
DOCS_PROCESSED = DATA / "processed_v3"  # output of scripts/r2ai2026_build_processed_corpus.py
# Qwen2.5-7B translations (the old data/query_translated.parquet was garbage, moved to _temp)
QUERIES_EN = Path(__file__).resolve().parents[1] / "kaggle" / "outputs" / "vibiomir-translate" / "query_en.parquet"
QUERIES_ZH = QUERIES_EN.with_name("query_zh.parquet")

RUNS = ROOT / "runs"
INDEXES = RUNS / "indexes"
SUBMISSIONS = RUNS / "submissions"
REPORTS = RUNS / "reports"
SUBMISSION_LOG = RUNS / "submissions_log.csv"

KNOWLEDGE = ROOT / "knowledge"
KAGGLE = ROOT / "kaggle"
