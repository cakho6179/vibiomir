"""Text normalization, tokenization and language detection for the ViBioMIR corpus.

The official scorer normalizes text with NFKC + HTML-entity decoding +
lowercasing + whitespace collapsing before its token-LCS comparison. Indexing
and retrieval normalize the same way so that what the scorer sees is what the
ranker ranked.

Vietnamese is handled with optional diacritic stripping: queries typed without
tone marks still match documents that carry them, which matters because the
metric (F2) rewards recall much more than precision. Chinese is handled as
per-character tokens plus character bigrams, the standard cheap approach for
BM25 on CJK text.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RUN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SENTENCE_BREAK_RE = re.compile(r"[.!?。！？；;…]+[\s\"'”’)\]]*|\n{2,}")

_CJK_RANGES = (
    (0x2E80, 0x2EFF),
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2FA1F),
)

_VI_MARKERS = (
    " và ", " của ", " là ", " không ", " người ", " bệnh ", " khi ", " có ", " được ",
)
_ZH_MARKERS = ("的", "了", "是", "什么", "怎么", "治疗", "症状")


def normalize(text: str) -> str:
    """NFKC, HTML entities decoded, lowercased, whitespace collapsed and trimmed."""
    text = unicodedata.normalize("NFKC", text or "")
    text = html.unescape(text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


_DIACRITIC_FOLD = str.maketrans({"đ": "d", "Đ": "D"})


def strip_diacritics(text: str) -> str:
    """Remove combining marks and fold đ/Đ so diacritic-free queries still match.

    ``đ`` is a precomposed letter with no canonical decomposition, so it needs an
    explicit fold on top of the NFD/combining-mark pass.
    """
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return unicodedata.normalize("NFC", without_marks).translate(_DIACRITIC_FOLD)


def has_cjk(text: str) -> bool:
    return any(
        start <= ord(char) <= end for char in text for start, end in _CJK_RANGES
    )


def _cjk_chars(text: str) -> list[str]:
    return [
        char
        for char in text
        if any(start <= ord(char) <= end for start, end in _CJK_RANGES)
    ]


def tokenize(text: str, keep_diacritics: bool = False) -> list[str]:
    """Word tokens (plus CJK chars and bigrams) after normalization.

    ``keep_diacritics=False`` (default) indexes the diacritic-free form so that
    Vietnamese queries and documents meet on the same token alphabet.
    """
    normalized = normalize(text)
    tokens: list[str] = []
    for run in _WORD_RUN_RE.findall(normalized):
        if has_cjk(run):
            chars = _cjk_chars(run)
            tokens.extend(chars)
            tokens.extend(chars[index] + chars[index + 1] for index in range(len(chars) - 1))
            continue
        token = run if keep_diacritics else strip_diacritics(run)
        if token:
            tokens.append(token)
    return tokens


def word_count(text: str) -> int:
    return len(_WORD_RUN_RE.findall(text or ""))


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences, preserving order and full coverage."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_BREAK_RE.finditer(text):
        end = match.end()
        if end > cursor:
            spans.append((cursor, end))
            cursor = end
    if cursor < len(text):
        spans.append((cursor, len(text)))
    return [(start, end) for start, end in spans if text[start:end].strip()]


def detect_lang(text: str) -> str:
    """Cheap language tag: ``vi`` | ``zh`` | ``en`` | ``other``."""
    sample = (text or "")[:2000]
    if not sample.strip():
        return "other"
    cjk_chars = len(_cjk_chars(sample))
    letters = sum(1 for char in sample if char.isalpha())
    if letters and cjk_chars / max(letters, 1) > 0.25:
        return "zh"
    lowered = f" {sample.lower()} "
    if any(marker in lowered for marker in _ZH_MARKERS):
        return "zh"
    if any(char in sample for char in "ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"):
        return "vi"
    if any(marker in lowered for marker in _VI_MARKERS):
        return "vi"
    ascii_letters = sum(1 for char in sample if char.isascii() and char.isalpha())
    if letters and ascii_letters / letters > 0.9:
        return "en"
    return "other"


def split_paragraphs(text: str, min_chars: int = 2) -> list[str]:
    """Paragraph-ish lines, used to drop menu junk short lines."""
    paragraphs: list[str] = []
    for block in re.split(r"\n{1,}", text or ""):
        block = block.strip()
        if len(block) >= min_chars:
            paragraphs.append(block)
    return paragraphs


def dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
