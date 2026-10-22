"""Chunking policies for a metric that rewards long, gold-covering chunks.

The official chunk rule counts a submitted chunk as correct when
``LCS_tokens(submitted, gold) / |gold| >= 0.40`` for a gold chunk of the same
query and document. LCS keeps order but allows gaps, so a submitted chunk that
contains the gold span verbatim -- even inside a much longer passage -- scores a
full 1.0. That is why the baseline below can submit one merged, document-ordered
chunk per retrieved document and still beat a naive "many short chunks" policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from r2ai2026.text import sentence_spans, tokenize, word_count


@dataclass(slots=True)
class Chunk:
    index: int
    start: int
    end: int
    text: str
    n_tokens: int

    @property
    def char_len(self) -> int:
        return self.end - self.start


def chunk_document(
    text: str,
    *,
    target_tokens: int = 320,
    overlap_tokens: int = 64,
    min_tokens: int = 32,
) -> list[Chunk]:
    """Sentence-aware sliding windows of roughly ``target_tokens`` tokens."""
    text = text or ""
    if not text.strip():
        return []
    spans = sentence_spans(text)
    if not spans:
        return []

    chunks: list[Chunk] = []
    window_start_index = 0
    while window_start_index < len(spans):
        tokens_in_window = 0
        window_end_index = window_start_index
        while window_end_index < len(spans) and tokens_in_window < target_tokens:
            tokens_in_window += word_count(text[spans[window_end_index][0] : spans[window_end_index][1]])
            window_end_index += 1

        start = spans[window_start_index][0]
        end = spans[window_end_index - 1][1]
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    start=start,
                    end=end,
                    text=chunk_text,
                    n_tokens=word_count(chunk_text),
                )
            )

        if window_end_index >= len(spans):
            break

        if overlap_tokens <= 0:
            window_start_index = window_end_index
            continue

        # Step back far enough that the next window repeats ~overlap_tokens.
        rewind_tokens = 0
        next_start = window_end_index
        while next_start > window_start_index and rewind_tokens < overlap_tokens:
            next_start -= 1
            rewind_tokens += word_count(text[spans[next_start][0] : spans[next_start][1]])
        window_start_index = max(next_start, window_start_index + 1)

    if len(chunks) > 1 and chunks[0].n_tokens < min_tokens:
        chunks[0] = Chunk(
            index=0,
            start=chunks[0].start,
            end=chunks[1].end,
            text=text[chunks[0].start : chunks[1].end].strip(),
            n_tokens=word_count(text[chunks[0].start : chunks[1].end]),
        )
        del chunks[1]
        for position, chunk in enumerate(chunks):
            chunks[position] = Chunk(position, chunk.start, chunk.end, chunk.text, chunk.n_tokens)
    return chunks


def merge_selected(chunks: Sequence[Chunk], selected: Sequence[int], *, max_tokens: int | None = None) -> str:
    """Join selected chunk texts in document order (gaps allowed, order kept).

    Document order matters because the scorer's LCS is order-preserving: merging
    windows out of order throws away matches that would otherwise count.
    """
    ordered_indices = sorted({index for index in selected if 0 <= index < len(chunks)})
    if not ordered_indices:
        return ""
    ordered = sorted((chunks[index] for index in ordered_indices), key=lambda chunk: chunk.start)
    pieces: list[str] = []
    used_tokens = 0
    for chunk in ordered:
        if max_tokens is not None and used_tokens >= max_tokens:
            break
        pieces.append(chunk.text)
        used_tokens += chunk.n_tokens
    return "\n".join(pieces).strip()


def whole_document_chunk(text: str, *, max_tokens: int | None = 1400) -> str:
    """One submission chunk built from the head of the document.

    Keeps the head verbatim (gold spans are usually in the body, and the head
    carries the title/intro) and stops at a sentence boundary once ``max_tokens``
    is exceeded. ``max_tokens=None`` submits the full text, which maximizes chunk
    recall at the cost of submission size and scoring time.
    """
    return _sentence_limited(text, max_tokens, keep="head")


def _sentence_limited(text: str, max_tokens: int | None, *, keep: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if max_tokens is None or word_count(text) <= max_tokens:
        return text

    spans = sentence_spans(text)
    if not spans:
        return text
    if keep == "tail":
        used = 0
        start = spans[0][0]
        for span_start, span_end in reversed(spans):
            used += word_count(text[span_start:span_end])
            if used > max_tokens:
                start = span_start
                break
        return text[start:].strip()

    keep_until = spans[-1][1]
    used = 0
    for start, end in spans:
        used += word_count(text[start:end])
        if used > max_tokens:
            keep_until = end
            break
    return text[:keep_until].strip()


def head_tail_chunk(text: str, *, max_tokens: int | None = 1400) -> str:
    """Half the budget from the document head, half from its tail."""
    text = (text or "").strip()
    if not text:
        return ""
    if max_tokens is None or word_count(text) <= max_tokens:
        return text
    half = max(20, max_tokens // 2)
    head = _sentence_limited(text, half, keep="head")
    tail = _sentence_limited(text[len(head) :], half, keep="tail")
    return f"{head}\n{tail}".strip()


def spread_windows_chunk(text: str, *, max_tokens: int | None = 1400, windows: int = 4) -> str:
    """``windows`` equal slices spread across the document, in document order.

    LCS is order preserving, so slices taken across the whole document raise the
    chance that at least one gold span falls (mostly) inside the submitted text
    while the token budget stays fixed. The first slice starts at the document
    head and the last one is anchored to the document tail, so both ends (title
    and conclusion) are always covered.
    """
    text = (text or "").strip()
    if not text:
        return ""
    if max_tokens is None or word_count(text) <= max_tokens:
        return text
    windows = max(1, int(windows))
    spans = sentence_spans(text)
    if not spans or windows == 1:
        return _sentence_limited(text, max_tokens, keep="head")

    budget = max(20, max_tokens // windows)

    # walk backwards to find where the final window must start
    used = 0
    last_start = len(spans) - 1
    for index in range(len(spans) - 1, -1, -1):
        used += word_count(text[spans[index][0] : spans[index][1]])
        if used >= budget:
            last_start = index
            break

    starts = sorted({int(round(index / (windows - 1) * last_start)) for index in range(windows)})

    pieces: list[str] = []
    for start_index in starts:
        piece = _sentence_limited(text[spans[start_index][0] :], budget, keep="head")
        if piece:
            pieces.append(piece.strip())
    if not pieces:
        return _sentence_limited(text, max_tokens, keep="head")
    return "\n".join(pieces).strip()


def query_biased_paragraph_chunk(
    text: str,
    query: str,
    *,
    max_tokens: int | None = 1400,
) -> str:
    """Extract informative paragraphs biased towards the clinical query.

    Keeps the document title/lead verbatim, scores subsequent body paragraphs
    by query lexical/semantic overlap, and reconstructs selected paragraphs
    in original document order to maximize token-LCS >= 0.40.
    """
    text = (text or "").strip()
    if not text:
        return ""
    if max_tokens is None or word_count(text) <= max_tokens:
        return text

    raw_paras = [p.strip() for p in text.split("\n") if p.strip()]
    if not raw_paras:
        return _sentence_limited(text, max_tokens, keep="head")

    lead = raw_paras[0]
    lead_tokens = word_count(lead)
    if len(raw_paras) == 1 or lead_tokens >= max_tokens:
        return _sentence_limited(lead, max_tokens, keep="head")

    remaining_budget = max_tokens - lead_tokens
    body_paras = raw_paras[1:]

    query_terms = set(tokenize(query, keep_diacritics=False))
    scored_paras = []
    for idx, p in enumerate(body_paras):
        p_terms = set(tokenize(p, keep_diacritics=False))
        overlap = len(query_terms.intersection(p_terms))
        wcnt = word_count(p)
        score = overlap / (wcnt ** 0.5 + 1.0)
        scored_paras.append((score, idx, p, wcnt))

    scored_paras.sort(key=lambda x: x[0], reverse=True)

    selected_indices = {0}
    used_tokens = lead_tokens

    for score, b_idx, p, wcnt in scored_paras:
        if used_tokens + wcnt <= remaining_budget:
            selected_indices.add(b_idx + 1)
            used_tokens += wcnt
        elif used_tokens < max_tokens:
            avail = max_tokens - used_tokens
            if avail > 30:
                selected_indices.add(b_idx + 1)
                used_tokens += avail
            break

def tri_sentence_context_chunk(
    text: str,
    query: str,
    *,
    max_tokens: int | None = 1400,
) -> str:
    """SOTA BioASQ/ViBioMIR extractive chunking: Tri-Sentence Saliency Expansion.

    1. If document <= max_tokens: emit whole document verbatim (guarantees LCS = 1.0).
    2. Segment text into sentences using sentence_spans.
    3. Always keep the first 2 sentences (Title + Lead Context).
    4. Score all sentences by query lexical overlap.
    5. For each high-scoring sentence s_i, expand to [s_{i-1}, s_i, s_{i+1}] to
       preserve complete diagnostic meaning without cutting off medical explanations.
    6. Sort selected sentence indices and reconstruct in original document order up to max_tokens.
    """
    text = (text or "").strip()
    if not text:
        return ""
    if max_tokens is None or word_count(text) <= max_tokens:
        return text

    spans = sentence_spans(text)
    if not spans:
        return _sentence_limited(text, max_tokens, keep="head")

    sentences = [text[start:end].strip() for start, end in spans]
    n_sents = len(sentences)
    if n_sents <= 3:
        return text

    # Lead context: sentences 0 and 1
    selected_indices: set[int] = {0, min(1, n_sents - 1)}
    lead_tokens = sum(word_count(sentences[i]) for i in selected_indices)
    if lead_tokens >= max_tokens:
        return _sentence_limited(text, max_tokens, keep="head")

    remaining_budget = max_tokens - lead_tokens
    query_terms = set(tokenize(query, keep_diacritics=False))

    # Score candidate sentences (from sentence 2 onwards)
    scored_sents = []
    for i in range(2, n_sents):
        s_text = sentences[i]
        s_terms = set(tokenize(s_text, keep_diacritics=False))
        overlap = len(query_terms.intersection(s_terms))
        wcnt = word_count(s_text)
        if wcnt < 3:
            continue
        # Density score with length normalization
        score = overlap / (wcnt ** 0.5 + 1.0)
        if score > 0:
            scored_sents.append((score, i, wcnt))

    scored_sents.sort(key=lambda x: x[0], reverse=True)

    # Greedily expand top sentences with tri-sentence context [i-1, i, i+1]
    used_tokens = lead_tokens
    for score, i, wcnt in scored_sents:
        tri_window = {max(0, i - 1), i, min(n_sents - 1, i + 1)}
        new_indices = tri_window - selected_indices
        new_tokens = sum(word_count(sentences[idx]) for idx in new_indices)

        if used_tokens + new_tokens <= remaining_budget:
            selected_indices.update(new_indices)
            used_tokens += new_tokens
        elif used_tokens < max_tokens:
            # Just add the central sentence if full window doesn't fit
            if i not in selected_indices and used_tokens + wcnt <= remaining_budget:
                selected_indices.add(i)
                used_tokens += wcnt
            break

    ordered_sents = [sentences[idx] for idx in sorted(selected_indices)]
    result = " ".join(ordered_sents)
    return _sentence_limited(result, max_tokens, keep="head")


CHUNK_POLICIES = {
    "head": whole_document_chunk,
    "head_tail": head_tail_chunk,
    "spread": spread_windows_chunk,
    "query_biased": query_biased_paragraph_chunk,
    "tri_sentence": tri_sentence_context_chunk,
}


def chunk_token_stats(chunks: Sequence[Chunk]) -> dict[str, float]:
    if not chunks:
        return {"n_chunks": 0, "mean_tokens": 0.0, "p90_tokens": 0.0, "max_tokens": 0.0}
    counts = sorted(chunk.n_tokens for chunk in chunks)
    p90 = counts[min(len(counts) - 1, int(0.9 * len(counts)))]
    return {
        "n_chunks": float(len(counts)),
        "mean_tokens": sum(counts) / len(counts),
        "p90_tokens": float(p90),
        "max_tokens": float(counts[-1]),
    }


def tokens_of(text: str) -> list[str]:
    return tokenize(text)
