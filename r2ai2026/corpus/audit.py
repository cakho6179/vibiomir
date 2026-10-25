"""Corpus/query parquet auditing for retrieval competitions.

Generic readers for the standard `(query.parquet, corpus.parquet)` layout used by
information-retrieval shared tasks: both tables are expected to expose `id` and a
single text column (`query` / `url`).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq

# Host suffixes whose content language is unambiguous.
HOST_LANG_SUFFIXES: tuple[tuple[str, str], ...] = (
    (".vn", "vi"),
    (".com.vn", "vi"),
    (".com.cn", "zh"),
    (".cn", "zh"),
)

# Well-known publishers used to separate VI/ZH `.com` sites from noise.
KNOWN_EN_HOST_PARTS = (
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "nih.gov",
    "who.int",
    "mayo.edu",
    "mayoclinic.org",
    "webmd.com",
    "medscape.com",
    "cdc.gov",
    "nhs.uk",
)

KNOWN_ZH_HOST_PARTS = (
    "120ask.com",
    "39.net",
    "cnkang.com",
    "familydoctor.com.cn",
    "zysjonline.com",
    "zhongyibaodian.net",
    "haodf.com",
    "bohe.cn",
    "xywy.com",
    "39yst.com",
    "yiyaojd.com",
    "dxy.cn",
)


def infer_language(url: str) -> str:
    host = urlparse(url).netloc.lower()
    for suffix, lang in HOST_LANG_SUFFIXES:
        if host.endswith(suffix):
            return lang
    for part in KNOWN_EN_HOST_PARTS:
        if host.endswith(part):
            return "en"
    for part in KNOWN_ZH_HOST_PARTS:
        if host.endswith(part):
            return "zh"
    return "unknown"


def audit_query_table(path: Path) -> dict:
    table = pq.read_table(path)
    ids = table.column("id").to_pylist()
    queries = table.column("query").to_pylist()
    word_counts = sorted(len(text.split()) for text in queries)
    n = len(word_counts)
    return {
        "path": str(path),
        "rows": table.num_rows,
        "id_min": min(ids),
        "id_max": max(ids),
        "id_contiguous": sorted(ids) == list(range(min(ids), max(ids) + 1)),
        "words_min": word_counts[0],
        "words_median": word_counts[n // 2],
        "words_p95": word_counts[int(n * 0.95)],
        "words_max": word_counts[-1],
        "words_mean": round(sum(word_counts) / n, 1),
        "samples": [{"id": i, "query": q} for i, q in zip(ids[:5], queries[:5])],
    }


def audit_corpus_table(path: Path, top_hosts: int = 30) -> dict:
    table = pq.read_table(path)
    ids = table.column("id").to_pylist()
    urls = table.column("url").to_pylist()

    hosts: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    schemes: Counter[str] = Counter()
    for url in urls:
        parsed = urlparse(url)
        hosts[parsed.netloc.lower()] += 1
        languages[infer_language(url)] += 1
        schemes[parsed.scheme] += 1

    pubmed = sum(1 for url in urls if "pubmed.ncbi.nlm.nih.gov" in url)
    id_set = set(ids)
    return {
        "path": str(path),
        "rows": table.num_rows,
        "unique_urls": len(set(urls)),
        "unique_ids": len(id_set),
        "id_min": min(ids),
        "id_max": max(ids),
        "id_contiguous": sorted(id_set) == list(range(min(ids), max(ids) + 1)),
        "languages_heuristic": dict(languages),
        "schemes": dict(schemes),
        "pubmed_urls": pubmed,
        "top_hosts": hosts.most_common(top_hosts),
        "sample_urls": urls[:5],
    }


def audit_retrieval_data(query_path: Path, corpus_path: Path) -> dict:
    return {
        "query": audit_query_table(query_path),
        "corpus": audit_corpus_table(corpus_path),
    }


def render_audit_markdown(audit: dict, title: str = "Corpus audit") -> str:
    query = audit["query"]
    corpus = audit["corpus"]
    lines = [
        f"# {title}",
        "",
        "## Queries",
        "",
        f"- Rows: **{query['rows']:,}** (id {query['id_min']}..{query['id_max']}, contiguous={query['id_contiguous']})",
        f"- Words: min={query['words_min']} median={query['words_median']} p95={query['words_p95']} max={query['words_max']} mean={query['words_mean']}",
        "",
        "Sample queries:",
        "",
    ]
    lines += [f"- `{s['id']}` — {s['query'][:160]}" for s in query["samples"]]
    lines += [
        "",
        "## Corpus",
        "",
        f"- Rows: **{corpus['rows']:,}** | unique urls: {corpus['unique_urls']:,} | id range {corpus['id_min']}..{corpus['id_max']} (contiguous={corpus['id_contiguous']})",
        f"- Language heuristic: {corpus['languages_heuristic']}",
        f"- Schemes: {corpus['schemes']}",
        f"- PubMed URLs: {corpus['pubmed_urls']:,}",
        "",
        "Top hosts:",
        "",
        "| URLs | Host |",
        "|---:|---|",
    ]
    lines += [f"| {count:,} | {host} |" for host, count in corpus["top_hosts"]]
    return "\n".join(lines) + "\n"
