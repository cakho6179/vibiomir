"""R2AI2026 Shared Task 3 (ViBioMIR) — competition tooling.

Modules
-------
``scoring``   local replica of the official F2 metric (doc + chunk levels)
``text``      normalization, tokenization, language detection
``corpus``    the published URL list: classification, host stats, sampling
``fetch``     polite async crawler + article extraction
``pubmed``    E-utilities harvesting for the 240k PubMed URLs
``chunking``  chunk policies built around the "LCS >= 0.40" chunk rule
``bm25``      BM25 indexes (document level and chunk level)
``pipeline``  end-to-end baseline from documents to submission ZIP
``submit``    submission packing + validation
``devproxy``  label-free offline proxies (title queries, pseudo-gold spans)

Heavy dependencies (httpx, trafilatura, bm25s) stay inside their modules, so
importing this package only pulls numpy/pandas.
"""

from r2ai2026.chunking import Chunk, chunk_document, merge_selected, whole_document_chunk
from r2ai2026.scoring import (
    OVERLAP_THRESHOLD,
    QueryScore,
    SubmissionScore,
    chunk_overlap,
    f_beta,
    lcs_length,
    normalize_text,
    score_query,
    score_submission,
    tokenize as scorer_tokenize,
)
from r2ai2026.text import detect_lang, normalize, strip_diacritics, tokenize

__all__ = [
    "Chunk",
    "OVERLAP_THRESHOLD",
    "QueryScore",
    "SubmissionScore",
    "chunk_document",
    "chunk_overlap",
    "detect_lang",
    "f_beta",
    "lcs_length",
    "merge_selected",
    "normalize",
    "normalize_text",
    "score_query",
    "score_submission",
    "scorer_tokenize",
    "strip_diacritics",
    "tokenize",
    "whole_document_chunk",
]
