"""Docstore v1: one canonical document store shared by every retrieval stage (docs/DATA_PIPELINE.md §3).

Text is never rewritten beyond ``processing.prepare_body`` (entities, NFC, whitespace, empty/repeated lines,
leading title line). Page furniture is *marked* as character spans, not removed, and passages are spans that tile
the body exactly, so a submitted chunk can always be cut verbatim from the body.

Offsets (``start``/``end``) are character positions in the body ``x``; the title ``t`` is kept separately.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaggle" / "common"))
from docorg import furniture_kinds, is_heading  # noqa: E402

PUBMED_HOST = "pubmed.ncbi.nlm.nih.gov"
THIN_CHARS = 500

# recovered/<prefix>_<date>.jsonl -> source label; lower rank wins when an id is recovered by several sources
RECOVERED_SOURCES = [("direct", "direct"), ("wayback", "wayback"), ("ccc", "cc"), ("cc_", "cc"), ("pubmed", "pubmed")]
SOURCE_RANK = {"crawl": 0, "direct": 1, "wayback": 2, "cc": 3, "pubmed": 4}


def source_of(path: Path) -> str:
    name = path.name.lower()
    for prefix, label in RECOVERED_SOURCES:
        if name.startswith(prefix):
            return label
    raise ValueError(f"unknown recovered source file: {path.name}")


def host_of(url: str) -> str:
    return urlsplit(url or "").netloc.lower()


def text_hash(text: str) -> bytes:
    """Same hash as scripts/data/build_clean_corpus.py uses for duplicate-template counts."""
    return hashlib.md5((text or "").encode("utf-8", "ignore")).digest()


def as_doc(rec: dict, source: str) -> dict | None:
    """Uniform {id, url, host, title, text, source} from a crawler / recrawler / Common Crawl / PubMed record."""
    if rec.get("ok") is False or rec.get("id") is None:
        return None
    url = rec.get("url") or ""
    return {"id": int(rec["id"]), "url": url, "host": rec.get("host") or host_of(url),
            "title": rec.get("title") or rec.get("t") or "", "text": rec.get("text") or rec.get("x") or "",
            "source": source}


def pick_recovered(candidates: list[dict]) -> dict:
    """Best record for one id: highest-priority source, then the longest text."""
    return min(candidates, key=lambda d: (SOURCE_RANK[d["source"]], -len(d["text"])))


def line_offsets(lines: list[str]) -> list[int]:
    out, pos = [], 0
    for ln in lines:
        out.append(pos)
        pos += len(ln) + 1
    return out


def furniture_spans(lines: list[str], template: list[bool]) -> list[tuple[int, int, str]]:
    """Merged (start, end, kind) spans of template lines and clean_v4 furniture over the body ``"\\n".join(lines)``."""
    kinds = furniture_kinds(lines)
    offs = line_offsets(lines)
    spans: list[tuple[int, int, str]] = []
    for i, ln in enumerate(lines):
        kind = "template" if template[i] else kinds[i]
        if kind is None:
            continue
        start, end = offs[i], offs[i] + len(ln)
        if spans and spans[-1][2] == kind and spans[-1][1] + 1 == start:
            spans[-1] = (spans[-1][0], end, kind)
        else:
            spans.append((start, end, kind))
    return spans


def _units(lines: list[str], offs: list[int], a: int, b: int, size: int) -> list[tuple[int, int]]:
    """(start, end) char ranges of lines a..b-1; a line longer than ``size`` is cut at spaces."""
    out = []
    for i in range(a, b):
        start, ln = offs[i], lines[i]
        while len(ln) > size:
            cut = ln.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            out.append((start, start + cut))
            start, ln = start + cut, ln[cut:]
        out.append((start, start + len(ln)))
    return out


def passage_spans(lines: list[str], max_chars: int = 1500, min_chars: int = 400) -> list[tuple[int, int, str]]:
    """Heading-aware passages (start, end, heading) that tile ``"\\n".join(lines)`` exactly.

    Sections start at heuristic headings (docorg.is_heading); a section shorter than ``min_chars`` absorbs the next
    one; a section is packed into pieces of <= ``max_chars`` along line boundaries (giant lines cut at spaces). Each
    passage runs to the start of the next one, so separators belong to the passage before them.
    """
    if not lines:
        return [(0, 0, "")]
    offs = line_offsets(lines)
    starts = [0] + [i for i in range(1, len(lines))
                    if is_heading(lines[i], lines[i + 1] if i + 1 < len(lines) else None)]
    sections: list[list[int]] = []  # [first line, end line, heading line or -1]
    for k, a in enumerate(starts):
        b = starts[k + 1] if k + 1 < len(starts) else len(lines)
        head = a if a > 0 or is_heading(lines[0], lines[1] if len(lines) > 1 else None) else -1
        if sections and offs[a] - offs[sections[-1][0]] < min_chars:
            sections[-1][1] = b
        else:
            sections.append([a, b, head])
    pieces: list[tuple[int, str]] = []
    for a, b, head in sections:
        heading = lines[head] if head >= 0 else ""
        buf_start = None
        for s, e in _units(lines, offs, a, b, max_chars):
            if buf_start is not None and e - buf_start > max_chars:
                pieces.append((buf_start, heading))
                buf_start = None
            if buf_start is None:
                buf_start = s
        if buf_start is not None:
            pieces.append((buf_start, heading))
    total = offs[-1] + len(lines[-1])
    return [(s, pieces[k + 1][0] if k + 1 < len(pieces) else total, h) for k, (s, h) in enumerate(pieces)]


def pack_sections(passages: list[tuple[int, int, str]], max_chars: int = 4800,
                  min_chars: int = 2000) -> list[tuple[int, int, int, int]]:
    """Group consecutive passages of one doc into sections (pid_first, pid_last, start, end) of <= ``max_chars``.

    Sections are the chunk unit (gold chunks look like KB sections: median ~3.3k, max ~4.8k chars). A new section
    starts when adding the next passage would exceed ``max_chars``, or at a new heading once the current section
    holds ``min_chars``. A single passage longer than ``max_chars`` stays one section. Sections tile the passages.
    """
    out: list[tuple[int, int, int, int]] = []
    first = 0
    for k in range(1, len(passages) + 1):
        if k < len(passages):
            s0, e, head = passages[first][0], passages[k][1], passages[k][2]
            new_head = head and head != passages[k - 1][2]
            if e - s0 <= max_chars and not (new_head and passages[k - 1][1] - s0 >= min_chars):
                continue
        out.append((first, k - 1, passages[first][0], passages[k - 1][1]))
        first = k
    return out


def content_chars(start: int, end: int, furniture: list[tuple[int, int, str]]) -> int:
    """Characters of [start, end) outside furniture spans."""
    covered = sum(max(0, min(end, fe) - max(start, fs)) for fs, fe, _ in furniture)
    return end - start - covered
