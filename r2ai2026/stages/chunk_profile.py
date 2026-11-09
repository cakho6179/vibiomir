"""Cut a chunk where the query's relevance ends, from the per-passage reranker scores of the document.

Shipped standalone (prep_kaggle_script.py --modules) because the Kaggle code bundle inside the docstore dataset
predates this function; materialize_lists.py imports it from here when the bundle lacks it.
"""
from __future__ import annotations


def profile_chunk(title: str, text: str, spans: list[tuple[int, int]], pscore: list[float] | None,
                  size: int, whole_max: int, alpha: float = 0.5, min_frac: float = 0.30) -> str | None:
    """Chunk cut where the query's relevance ENDS, using the per-passage reranker scores of the document.

    window_chunk always fills ``size`` chars, so on a long document it has to guess where the answer lives.
    Here the growth is driven by the relevance profile instead, in two phases:

      1. reach ``min_frac`` of the document (a chunk shorter than that cannot overlap a gold chunk of
         ~0.7 x L enough on the LCS(c, g)/|g| side), always taking the better neighbour;
      2. keep growing while the next neighbour still scores >= ``alpha`` x the best passage -- i.e. until the
         document moves on to another topic -- and stop there even if ``size`` is not exhausted.

    Returns None when the document has no profile, so the caller can fall back to window_chunk.
    """
    full = f"{title}\n{text}".strip()
    if len(full) <= whole_max:
        return full
    n = min(len(spans), len(pscore or ()))
    if n == 0:
        return None
    best = max(range(n), key=lambda k: (pscore[k], -k))
    floor = alpha * pscore[best]
    i = j = best
    need = min_frac * len(text)
    while True:
        opts = []
        if j + 1 < n and spans[j + 1][1] - spans[i][0] <= size:
            opts.append((pscore[j + 1], 1, i, j + 1))
        if i - 1 >= 0 and spans[j][1] - spans[i - 1][0] <= size:
            opts.append((pscore[i - 1], 0, i - 1, j))
        if not opts:
            break
        short = spans[j][1] - spans[i][0] < need
        opts = [o for o in opts if short or o[0] >= floor]
        if not opts:
            break
        _, _, i, j = max(opts)
    start, end = spans[i][0], spans[j][1]
    return f"{title}\n{text[:end]}".strip() if start == 0 else text[start:end].strip()
