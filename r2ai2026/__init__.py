"""R2AI2026 Shared Task 3 (ViBioMIR): Vietnamese medical document retrieval system.

Subpackages
-----------
``core``    text normalisation/tokenisation, a replica of the official F2 metric, submission files
``corpus``  BTC link list, cleaning, processing, the versioned docstore (sections), audits
``crawl``   crawler, resumable crawl store, Wayback and PubMed fetching
``system``  end-to-end runner: ``python -m r2ai2026 run --config configs/system_v1.yaml``

The retrieval/ranking stages themselves are path-argument scripts run by the runner (see configs/).
Heavy dependencies stay inside their modules, so importing this package only pulls numpy/pandas.
"""

from r2ai2026.core.scoring import (
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
from r2ai2026.core.text import detect_lang, normalize, strip_diacritics, tokenize

__all__ = [
    "OVERLAP_THRESHOLD",
    "QueryScore",
    "SubmissionScore",
    "chunk_overlap",
    "detect_lang",
    "f_beta",
    "lcs_length",
    "normalize",
    "normalize_text",
    "score_query",
    "score_submission",
    "scorer_tokenize",
    "strip_diacritics",
    "tokenize",
]
