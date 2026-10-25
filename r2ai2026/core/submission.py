"""Submission packing and validation.

The organizers want a ZIP that contains exactly one ``submission.json``:
``[{"id": <query id>, "relevant_docs": [<doc id>, ...],
   "relevant_chunks": [{"doc_id": <doc id>, "chunk_text": "<verbatim>"}, ...]}]``

Queries that are missing from the file are simply not evaluated (they do not
burn a submission attempt), so the packer always emits exactly the queries we
actually solved.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

SUBMISSION_NAME = "submission.json"


def build_submission(predictions: Mapping[int, Mapping[str, object]]) -> list[dict]:
    """Normalize a ``{query_id: {...}}`` mapping into the official array shape."""
    records: list[dict] = []
    for query_id in sorted(predictions):
        prediction = predictions[query_id]
        docs: list[int] = []
        for doc_id in prediction.get("relevant_docs", []) or []:
            value = int(doc_id)
            if value not in docs:
                docs.append(value)
        chunks: list[dict] = []
        for chunk in prediction.get("relevant_chunks", []) or []:
            text = str(chunk.get("chunk_text", ""))
            if not text.strip():
                continue
            chunks.append({"doc_id": int(chunk["doc_id"]), "chunk_text": text})
        records.append({"id": int(query_id), "relevant_docs": docs, "relevant_chunks": chunks})
    return records


def submission_stats(records: Sequence[dict]) -> dict[str, float]:
    n_queries = len(records)
    n_docs = sum(len(record.get("relevant_docs", [])) for record in records)
    chunks = [chunk for record in records for chunk in record.get("relevant_chunks", [])]
    chars = sum(len(chunk.get("chunk_text", "")) for chunk in chunks)
    payload_bytes = len(json.dumps(records, ensure_ascii=False).encode("utf-8"))
    return {
        "n_queries": float(n_queries),
        "n_docs": float(n_docs),
        "docs_per_query": n_docs / n_queries if n_queries else 0.0,
        "n_chunks": float(len(chunks)),
        "chunks_per_query": len(chunks) / n_queries if n_queries else 0.0,
        "chunk_chars": float(chars),
        "payload_mb": payload_bytes / 1_048_576,
        "mean_chunk_chars": chars / len(chunks) if chunks else 0.0,
    }


def validate_submission(
    records: Sequence[dict],
    *,
    expected_ids: Iterable[int] | None = None,
    max_payload_mb: float = 1024.0,
    corpus_ids: set[int] | None = None,
) -> list[str]:
    """Return a list of problems; empty means the file is well formed.

    ``max_payload_mb`` defaults to the Codabench deployment limit (1 GB for the
    uploaded file); the public phase publishes no smaller limit of its own.
    """
    problems: list[str] = []
    if not isinstance(records, list):
        return ["submission root must be a list"]
    if not records:
        problems.append("submission is empty")

    expected = set(int(value) for value in expected_ids) if expected_ids is not None else None
    seen_ids: set[int] = set()
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            problems.append(f"[{position}] record is not an object")
            continue
        if "id" not in record:
            problems.append(f"[{position}] missing 'id'")
            continue
        query_id = record["id"]
        if not isinstance(query_id, int) or isinstance(query_id, bool):
            problems.append(f"[{position}] 'id' must be an int, got {type(query_id).__name__}")
            continue
        if query_id in seen_ids:
            problems.append(f"[{position}] duplicate query id {query_id}")
        seen_ids.add(query_id)
        if expected is not None and query_id not in expected:
            problems.append(f"[{position}] query id {query_id} is outside the query set")

        docs = record.get("relevant_docs")
        if not isinstance(docs, list):
            problems.append(f"[{position}] 'relevant_docs' must be a list")
            docs = []
        for doc_id in docs:
            if not isinstance(doc_id, int) or isinstance(doc_id, bool):
                problems.append(f"[{position}] doc id {doc_id!r} is not an int")
                break
            if corpus_ids is not None and doc_id not in corpus_ids:
                problems.append(f"[{position}] doc id {doc_id} is not in the published corpus")
                break

        chunks = record.get("relevant_chunks")
        if not isinstance(chunks, list):
            problems.append(f"[{position}] 'relevant_chunks' must be a list")
            continue
        for chunk_index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                problems.append(f"[{position}].chunks[{chunk_index}] is not an object")
                break
            if not isinstance(chunk.get("doc_id"), int):
                problems.append(f"[{position}].chunks[{chunk_index}] 'doc_id' must be an int")
                break
            text = chunk.get("chunk_text")
            if not isinstance(text, str) or not text.strip():
                problems.append(f"[{position}].chunks[{chunk_index}] 'chunk_text' must be a non-empty string")
                break

    stats = submission_stats(records)
    if stats["payload_mb"] > max_payload_mb:
        problems.append(f"payload too large: {stats['payload_mb']:.1f} MB > {max_payload_mb} MB")
    return problems


LZMA_DICT = 1 << 28  # 256 MiB window: the same document repeated across queries compresses to a reference


class _BigDictLZMACompressor(zipfile.LZMACompressor):
    """zipfile's LZMA (method 14) writer with a large dictionary; any zipfile reader decodes it."""

    def _init(self):
        import lzma
        import struct
        filt = {"id": lzma.FILTER_LZMA1, "preset": 9, "dict_size": LZMA_DICT}
        props = lzma._encode_filter_properties(filt)
        self._comp = lzma.LZMACompressor(lzma.FORMAT_RAW, filters=[filt])
        return struct.pack("<BBH", 9, 4, len(props)) + props


def write_submission_zip(records: Sequence[dict], path: str | Path, *, compresslevel: int = 6,
                         method: str = "deflate") -> dict[str, float]:
    """Write ``submission.json`` alone inside a ZIP and return size statistics.

    method="lzma" stores it with LZMA and a 256 MiB dictionary (~2.5x smaller than deflate when
    documents repeat across queries); the leaderboard rejects uploads above ~100 MB (HTTP 413).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(records, ensure_ascii=False).encode("utf-8")
    if method == "lzma":
        saved = zipfile.LZMACompressor
        zipfile.LZMACompressor = _BigDictLZMACompressor
        try:
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_LZMA) as archive:
                archive.writestr(SUBMISSION_NAME, payload)
        finally:
            zipfile.LZMACompressor = saved
        with zipfile.ZipFile(path) as archive:  # round-trip check with the stock reader
            if archive.read(SUBMISSION_NAME) != payload:
                raise RuntimeError(f"LZMA round-trip mismatch for {path}")
    else:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as archive:
            archive.writestr(SUBMISSION_NAME, payload)
    stats = submission_stats(records)
    stats["json_bytes"] = float(len(payload))
    stats["zip_bytes"] = float(path.stat().st_size)
    stats["zip_mb"] = path.stat().st_size / 1_048_576
    return stats


def read_submission_zip(path: str | Path) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != [SUBMISSION_NAME] and SUBMISSION_NAME not in names:
            raise ValueError(f"zip must contain {SUBMISSION_NAME}, found {names}")
        return json.loads(archive.read(SUBMISSION_NAME).decode("utf-8"))
