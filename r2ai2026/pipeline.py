"""Baseline pipeline: published URLs -> documents -> BM25 -> submission.

The pipeline is deliberately split so each stage is independently runnable and
inspectable:

    fetch (crawl / PubMed)  ->  documents.jsonl
    build_indexes           ->  index_dir/ (doc level, chunk level)
    run                     ->  submission.json + .zip (+ validation)

Three chunk policies are implemented because the official chunk rule (LCS
overlap >= 0.40 against a gold span of the same document) makes chunk *length*
the dominant factor:

* ``doc_head``  – one chunk per retrieved document: the head of the document up
  to a token budget, cut at a sentence boundary.
* ``doc_full``  – one chunk per retrieved document: the whole document (best
  coverage, largest payload, slowest to score).
* ``bm25_merge`` – chunk-level BM25 picks the best windows for the query, and the
  windows are merged back in document order into a single chunk per document.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from r2ai2026.bm25 import BM25Index, BM25Params, rank_ids, rank_rows
from r2ai2026.chunking import Chunk, chunk_document, merge_selected, spread_windows_chunk, whole_document_chunk
from r2ai2026.submit import build_submission, submission_stats, validate_submission, write_submission_zip


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    if path.suffix == ".zst":
        import zstandard

        return zstandard.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_documents(path: str | Path, *, on_corrupt: Callable[[int, str], None] | None = None) -> Iterator[dict]:
    """Yield document records from ``.jsonl`` / ``.jsonl.gz`` / ``.jsonl.zst``.

    A line can be truncated when a crawl is killed mid-append, and two crawl
    processes used to interleave halves of a 50 KB article. One bad line must not
    abort an index build over a million documents, so it is reported to
    ``on_corrupt`` and skipped -- same policy as ``store.iter_jsonl``.
    """
    path = Path(path)
    with _open_text(path) as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if on_corrupt is not None:
                    on_corrupt(number, line[:200])
                continue


def unslug_url(url: str) -> str:
    import re
    from urllib.parse import unquote, urlsplit

    try:
        path = unquote(urlsplit(url).path)
        path = re.sub(r"\.(html?|php|aspx?|jsp)$", "", path, flags=re.IGNORECASE)
        parts = re.split(r"[/_-]+", path)
        words = [p for p in parts if len(p) > 1 and not p.isdigit()]
        return " ".join(words)
    except Exception:
        return ""


def document_text(record: Mapping[str, object]) -> str:
    title = str(record.get("title") or "").strip()
    body = str(record.get("text") or "").strip()
    if not title:
        slug = unslug_url(str(record.get("url") or ""))
        if slug:
            title = slug
    if title and not body.startswith(title):
        return f"{title}\n{body}" if body else title
    return body or title


def load_queries(path: str | Path) -> list[dict]:
    import pandas as pd

    frame = pd.read_parquet(path)
    text_column = next((column for column in frame.columns if column != "id"), frame.columns[-1])
    return [
        {"id": int(row.id), "text": str(getattr(row, text_column))}
        for row in frame.itertuples(index=False)
    ]


@dataclass(slots=True)
class BaselineConfig:
    docs_path: Path
    queries_path: Path | None = None
    index_dir: Path | None = None
    submission_path: Path | None = None
    topk_docs: int = 20
    chunk_source: str = "doc_head"  # doc_head | doc_spread | doc_full | bm25_merge
    chunk_budget_tokens: int = 1400
    merge_chunks_per_doc: int = 3
    spread_windows: int = 4
    chunk_target_tokens: int = 320
    chunk_overlap_tokens: int = 64
    k1: float = 1.5
    b: float = 0.75
    keep_diacritics: bool = False
    limit_queries: int | None = None
    min_doc_chars: int = 200


@dataclass(slots=True)
class BaselineState:
    documents: dict[int, dict] = field(default_factory=dict)
    doc_chunks: dict[int, list[Chunk]] = field(default_factory=dict)
    rankings: dict[int, list[int]] = field(default_factory=dict)


class BaselinePipeline:
    def __init__(self, config: BaselineConfig) -> None:
        self.config = config
        self.state = BaselineState()
        self.doc_index: BM25Index | None = None
        self.chunk_index: BM25Index | None = None
        self.chunk_rows: np.ndarray = np.zeros((0, 2), dtype=np.int64)

    # ------------------------------------------------------------------ corpus
    def load_corpus(self, *, show_progress: bool = False) -> int:
        documents: dict[int, dict] = {}
        iterator: Iterable[dict] = load_documents(self.config.docs_path)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="documents")
        for record in iterator:
            if not record.get("ok", True):
                continue
            text = document_text(record)
            if len(text) < self.config.min_doc_chars:
                continue
            record_id = int(record["id"])
            record["_text"] = text
            documents[record_id] = record
        self.state.documents = documents
        return len(documents)

    def build_chunks(self, *, show_progress: bool = False) -> int:
        iterator: Iterable[int] = sorted(self.state.documents)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(list(iterator), desc="chunks")
        total = 0
        for doc_id in iterator:
            chunks = chunk_document(
                self.state.documents[doc_id]["_text"],
                target_tokens=self.config.chunk_target_tokens,
                overlap_tokens=self.config.chunk_overlap_tokens,
            )
            self.state.doc_chunks[doc_id] = chunks
            total += len(chunks)
        return total

    # ----------------------------------------------------------------- indexes
    def build_indexes(self, *, show_progress: bool = False) -> dict[str, int]:
        params = BM25Params(k1=self.config.k1, b=self.config.b, keep_diacritics=self.config.keep_diacritics)
        doc_ids = np.array(sorted(self.state.documents), dtype=np.int64).reshape(-1, 1)
        doc_texts: list[str] = []
        for doc_id in doc_ids[:, 0]:
            rec = self.state.documents[int(doc_id)]
            text = rec["_text"]
            slug = unslug_url(str(rec.get("url") or ""))
            if slug and slug not in text.lower():
                text = f"{slug}\n{text}"
            doc_texts.append(text)
        doc_index = BM25Index(params, name="docs")
        doc_index.build(doc_ids, doc_texts, show_progress=show_progress)
        self.doc_index = doc_index

        chunk_index = None
        if self.config.chunk_source == "bm25_merge":
            if not self.state.doc_chunks:
                self.build_chunks(show_progress=show_progress)
            rows: list[tuple[int, int]] = []
            texts: list[str] = []
            for doc_id in sorted(self.state.doc_chunks):
                for chunk in self.state.doc_chunks[doc_id]:
                    rows.append((doc_id, chunk.index))
                    texts.append(chunk.text)
            chunk_index = BM25Index(params, name="chunks")
            chunk_index.build(np.array(rows, dtype=np.int64), texts, show_progress=show_progress)
            self.chunk_index = chunk_index
            self.chunk_rows = np.array(rows, dtype=np.int64)

        return {
            "n_docs": doc_index.size,
            "n_chunks": chunk_index.size if chunk_index is not None else 0,
        }

    def save_indexes(self, directory: str | Path | None = None) -> Path:
        directory = Path(directory or self.config.index_dir or "index")
        assert self.doc_index is not None, "build or load the indexes first"
        self.doc_index.save(directory)
        if self.chunk_index is not None:
            self.chunk_index.save(directory)
        return directory

    def load_indexes(self, directory: str | Path | None = None) -> None:
        directory = Path(directory or self.config.index_dir or "index")
        self.doc_index = BM25Index.load(directory, name="docs")
        if self.config.chunk_source == "bm25_merge":
            self.chunk_index = BM25Index.load(directory, name="chunks")
            self.chunk_rows = np.asarray(self.chunk_index.rows, dtype=np.int64)

    def load_rankings(self, path: str | Path) -> int:
        """Load an externally produced ranking (hybrid, dense, reranked).

        The file is a parquet with ``query_id``, ``doc_id`` and optionally
        ``rank``/``score``. Retrieval then follows this order instead of BM25,
        which is how the dense, RRF and cross-encoder stages are combined
        without touching the submission writer.
        """
        import pandas as pd

        frame = pd.read_parquet(path)
        missing = {"query_id", "doc_id"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if "rank" in frame.columns:
            frame = frame.sort_values(["query_id", "rank"], kind="stable")
        elif "score" in frame.columns:
            frame = frame.sort_values(["query_id", "score"], ascending=[True, False], kind="stable")
        rankings: dict[int, list[int]] = {}
        for query_id, group in frame.groupby("query_id", sort=False):
            rankings[int(query_id)] = [int(value) for value in group["doc_id"].tolist()]
        self.state.rankings = rankings
        return len(rankings)

    # --------------------------------------------------------------- retrieval
    def _chunk_text_for_document(self, doc_id: int, query: str, merge_pool: Sequence[tuple[int, ...]]) -> str:
        text = self.state.documents[doc_id]["_text"]
        if self.config.chunk_source == "doc_full":
            return text
        if self.config.chunk_source == "doc_head":
            return whole_document_chunk(text, max_tokens=self.config.chunk_budget_tokens)
        if self.config.chunk_source == "doc_spread":
            return spread_windows_chunk(
                text,
                max_tokens=self.config.chunk_budget_tokens,
                windows=self.config.spread_windows,
            )

        chunks = self.state.doc_chunks.get(doc_id) or []
        if not chunks:
            return whole_document_chunk(text, max_tokens=self.config.chunk_budget_tokens)
        wanted = [row[1] for row in merge_pool if int(row[0]) == doc_id]
        selected = wanted[: self.config.merge_chunks_per_doc]
        if not selected:
            fallback = chunk_document(
                text,
                target_tokens=self.config.chunk_target_tokens,
                overlap_tokens=self.config.chunk_overlap_tokens,
            )
            selected = [0] if fallback else []
            chunks = fallback or chunks
        return merge_selected(chunks, selected, max_tokens=self.config.chunk_budget_tokens)

    def answer_query(self, query_id: int, query_text: str) -> dict:
        cfg = self.config
        merge_pool: Sequence[tuple[int, ...]] = ()

        if self.state.rankings:
            ranked = self.state.rankings.get(query_id, [])
            doc_ids = [doc_id for doc_id in ranked if doc_id in self.state.documents][: cfg.topk_docs]
        else:
            assert self.doc_index is not None, "build or load the indexes first"
            doc_hits = self.doc_index.search(query_text, k=cfg.topk_docs)
            doc_ids = [int(value) for value in rank_ids(doc_hits) if int(value) in self.state.documents]

        if cfg.chunk_source == "bm25_merge" and self.chunk_index is not None:
            assert self.doc_index is not None
            chunk_hits = self.chunk_index.search(query_text, k=cfg.topk_docs * cfg.merge_chunks_per_doc)
            merge_pool = rank_rows(chunk_hits)

        chunks: list[dict] = []
        for doc_id in doc_ids:
            chunk_text = self._chunk_text_for_document(doc_id, query_text, merge_pool)
            if chunk_text.strip():
                chunks.append({"doc_id": doc_id, "chunk_text": chunk_text})
        return {"relevant_docs": doc_ids, "relevant_chunks": chunks}

    def run(self, queries: Sequence[dict] | None = None) -> list[dict]:
        queries = list(queries if queries is not None else load_queries(self.config.queries_path))
        if self.config.limit_queries is not None:
            queries = queries[: self.config.limit_queries]
        predictions: dict[int, dict] = {}
        for query in queries:
            predictions[int(query["id"])] = self.answer_query(int(query["id"]), query["text"])
        return build_submission(predictions)

    # -------------------------------------------------------------- submission
    def write_submission(self, records: Sequence[dict], path: str | Path | None = None) -> dict[str, float]:
        path = Path(path or self.config.submission_path or "submission.zip")
        stats = write_submission_zip(records, path)
        return stats


def validate(records: Sequence[dict], *, expected_ids: Iterable[int] | None = None, corpus_ids: set[int] | None = None) -> list[str]:
    return validate_submission(records, expected_ids=expected_ids, corpus_ids=corpus_ids)


def describe(records: Sequence[dict]) -> dict[str, float]:
    return submission_stats(records)
