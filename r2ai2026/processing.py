"""Corpus processing v3: normalise text, strip per-host page templates, tag language, flag duplicates.

docs_clean.jsonl keeps every page exactly as extracted by the crawler. This module turns it into
retrieval-ready text without inventing anything:

* ``normalize_text``  HTML entities, NFC, invisible characters, whitespace; empty and immediately
  repeated lines dropped. Nothing else is rewritten.
* ``TemplateModel``   learns, per host, the lines that recur in a large share of that host's pages
  (menus, "các tin khác", font-size widgets, booking banners) and removes them. A document is
  never cut below ``keep_min_share`` of its characters by template removal.
* ``lang_of``         VI / ZH / EN from the official id blocks (ids are grouped by source).
* ``content_key``     exact-duplicate key over the processed text (same article under two URLs).
* ``split_passages``  the paragraph-aligned ~1500-character passages used by the Kaggle kernels.
"""

from __future__ import annotations

import hashlib
import html
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# Official id blocks (links_corpus_vi_en_zh.parquet sorted by id): 740,598 VI, 240,000 ZH, 240,000 EN.
VI_MAX_ID = 766_441
ZH_MAX_ID = 4_392_214

_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_SPACES = re.compile(r"[ \t  -   　]+")
_WS = re.compile(r"\s+")


def lang_of(doc_id: int) -> str:
    if doc_id <= VI_MAX_ID:
        return "vi"
    return "zh" if doc_id <= ZH_MAX_ID else "en"


def normalize_text(text: str) -> str:
    """Entity-decode, NFC, drop invisible chars, collapse spaces; drop empty and repeated lines."""
    text = unicodedata.normalize("NFC", html.unescape(text or "")).translate(_INVISIBLE)
    out: list[str] = []
    for line in text.replace("\r", "\n").split("\n"):
        line = _SPACES.sub(" ", line).strip()
        if line and (not out or line != out[-1]):
            out.append(line)
    return "\n".join(out)


def line_key(line: str) -> str:
    return hashlib.blake2b(_WS.sub(" ", line).strip().lower().encode("utf-8"), digest_size=8).hexdigest()


def content_key(text: str) -> str:
    """Exact-duplicate key: lowercase text without whitespace."""
    return hashlib.blake2b(_WS.sub("", text).lower().encode("utf-8"), digest_size=12).hexdigest()


def strip_leading_title(title: str, text: str) -> str:
    """Drop the first line when it only repeats the title (PubMed and many news pages do this)."""
    first, _, rest = text.partition("\n")
    if title and _WS.sub(" ", first).strip().lower() == _WS.sub(" ", title).strip().lower():
        return rest
    return text


@dataclass
class TemplateModel:
    """Per-host template lines learned from a reservoir sample of that host's documents."""

    min_share: float = 0.2  # line must appear in >= 20% of the sampled pages of the host
    min_docs: int = 20  # ... and in at least this many of them
    sample_size: int = 3000
    keep_min_share: float = 0.2  # never remove more than 80% of a document's characters
    seed: int = 0
    lines: dict[str, set[str]] = field(default_factory=dict)  # host -> line keys
    examples: dict[str, list[tuple[float, str]]] = field(default_factory=dict)  # host -> (share, line)
    _seen: Counter = field(default_factory=Counter, repr=False)
    _sample: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list), repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def observe(self, host: str, text: str) -> None:
        """Feed one document body, as returned by ``prepare_body`` (reservoir sampling per host)."""
        self._seen[host] += 1
        bucket = self._sample[host]
        if len(bucket) < self.sample_size:
            bucket.append(text)
        else:
            j = self._rng.randrange(self._seen[host])
            if j < self.sample_size:
                bucket[j] = text

    def fit(self) -> TemplateModel:
        for host, texts in self._sample.items():
            df: Counter = Counter()
            example: dict[str, str] = {}
            for t in texts:
                keys = set()
                for ln in t.split("\n"):
                    k = line_key(ln)
                    keys.add(k)
                    example.setdefault(k, ln[:200])
                df.update(keys)
            need = max(self.min_docs, self.min_share * len(texts))
            keys = {k for k, c in df.items() if c >= need}
            if keys:
                self.lines[host] = keys
                self.examples[host] = sorted(((df[k] / len(texts), example[k]) for k in keys), reverse=True)
        self._sample.clear()
        return self

    def template_mask(self, host: str, lines: list[str]) -> list[bool]:
        """True for each template line of ``lines``; all False when removal would cut below keep_min_share."""
        keys = self.lines.get(host)
        if not keys:
            return [False] * len(lines)
        mask = [line_key(ln) in keys for ln in lines]
        kept = sum(len(ln) for ln, m in zip(lines, mask) if not m) + max(sum(not m for m in mask) - 1, 0)
        return mask if kept >= self.keep_min_share * len("\n".join(lines)) else [False] * len(lines)

    def apply(self, host: str, text: str) -> str:
        lines = text.split("\n")
        return "\n".join(ln for ln, m in zip(lines, self.template_mask(host, lines)) if not m)


def split_passages(title: str, text: str, max_chars: int = 1500) -> list[str]:
    """Paragraph-aligned passages of <= max_chars, each prefixed by the title (as on Kaggle)."""
    pieces = []
    for para in text.split("\n"):
        while len(para) > max_chars:
            pieces.append(para[:max_chars])
            para = para[max_chars:]
        if para.strip():
            pieces.append(para)
    out, buf = [], ""
    for p in pieces:
        if buf and len(buf) + len(p) > max_chars:
            out.append(buf)
            buf = ""
        buf = f"{buf}\n{p}" if buf else p
    if buf or not out:
        out.append(buf)
    return [f"{title}\n{p}".strip() for p in out]


def prepare_body(title: str, text: str) -> tuple[str, str]:
    """Normalised (title, body) before template removal; the body no longer repeats the title."""
    title = normalize_text(title).replace("\n", " ")
    return title, strip_leading_title(title, normalize_text(text))


def process_document(doc_id: int, host: str, title: str, text: str, model: TemplateModel) -> dict:
    """One processed record from a raw docs_clean record."""
    title, body = prepare_body(title, text)
    cleaned = model.apply(host, body)
    return {"id": doc_id, "host": host, "lang": lang_of(doc_id), "title": title, "text": cleaned,
            "n_chars": len(cleaned), "removed_chars": len(body) - len(cleaned), "content_key": content_key(cleaned)}
