"""Stair-aware chunking: pick the chunk on the document's own hierarchy, not on a length budget.

STAIR (arXiv 2609.03874) argues that a retrieval unit should be a *semantic boundary* -- a section whose
title names its topic -- rather than a fixed-length window, because a window can straddle two topics or cut
one in half. Its own setting (a textbook with a deep table of contents) does not exist in this corpus: 1.19 M
web documents with a median of **1.4 sections** and a heading on only 34.5 % of them. So the hierarchy is
used where it does exist, and the flat case falls back to the current behaviour:

    document <= whole_max                  -> the whole document (61 % of the pool; nothing can beat this)
    section holding the best passage fits  -> that section, verbatim (semantic boundary, the STAIR unit)
    section too long for ``size``          -> window of ``size`` inside the section, grown from the best passage
    no sections / no heading               -> window grown from the best passage

Two details that the geometry of the metric forces:

* a section shorter than ``min_frac`` x ``size`` is merged with its neighbours (never past ``size``), because
  a chunk shorter than ~0.28 L cannot reach ``LCS(c, g) / |g| >= 0.4`` against a gold chunk of ~0.7 L;
* the seed passage is the reranker's argmax over the document's passages when a profile is given, and the
  lexical best otherwise. ``window_chunk`` always seeds lexically, which is the wrong signal for a question
  phrased in different words than the document.
"""
from __future__ import annotations


def _window(spans: list[tuple[int, int]], best: int, size: int, lo: int = 0, hi: int | None = None) -> tuple[int, int]:
    """Contiguous span around passage ``best``, grown next-then-previous, kept inside [lo, hi]."""
    n = len(spans)
    hi = n if hi is None else min(hi, n)
    i = j = min(max(best, lo), hi - 1)
    grew = True
    while grew:
        grew = False
        for ni, nj in ((i, j + 1), (i - 1, j)):
            if lo <= ni and nj < hi and spans[nj][1] - spans[ni][0] <= size:
                i, j, grew = ni, nj, True
    return spans[i][0], spans[j][1]


def stair_chunk(title: str, text: str, spans: list[tuple[int, int]], sections: list[tuple[int, int]] | None,
                pscore: list[float] | None, size: int, whole_max: int, min_frac: float = 0.30,
                lexical: int | None = None) -> str | None:
    """Chunk for (query, document) cut on the document's section boundary when that boundary fits."""
    full = f"{title}\n{text}".strip()
    if len(full) <= whole_max:
        return full
    n = len(spans)
    if n == 0:
        return None
    best = lexical if lexical is not None else 0
    if pscore:
        m = min(len(pscore), n)
        if m:
            best = max(range(m), key=lambda k: (pscore[k], -k))
    pos = spans[min(best, n - 1)][0]

    # passage index range of the section holding the best passage (sections are char spans over the body)
    if sections:
        k = None
        for idx, (a, b) in enumerate(sections):
            if a <= pos < b:
                k = idx
                break
        if k is not None:
            lo, hi = k, k + 1
            need = min_frac * size
            # merge neighbours while the section is too short to reach the LCS floor, never past `size`
            while sections[hi - 1][1] - sections[lo][0] < need or sections[hi - 1][1] - sections[lo][0] < min_frac * len(text):
                grew = False
                for nl, nh in ((lo, hi + 1), (lo - 1, hi)):
                    if 0 <= nl and nh <= len(sections) and sections[nh - 1][1] - sections[nl][0] <= size:
                        lo, hi, grew = nl, nh, True
                        break
                if not grew:
                    break
            a, b = sections[lo][0], sections[hi - 1][1]
            if b - a <= size:
                if a == 0 and len(title) + 1 + b <= size:
                    return f"{title}\n{text[:b]}".strip()
                return text[a:b].strip() or None
            # section longer than the budget: window inside it, seeded at the best passage
            in_section = [i for i in range(n) if a <= spans[i][0] < b]
            if in_section:
                seed = min(in_section, key=lambda i: abs(i - best))
                s, e = _window(spans, seed, size, lo=in_section[0], hi=in_section[-1] + 1)
                if e - s > 0:
                    return text[s:e].strip() or None

    s, e = _window(spans, best, size)
    if s == 0 and len(title) + 1 + e <= size:
        return f"{title}\n{text[:e]}".strip()
    return text[s:e].strip() or None
