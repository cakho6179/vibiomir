"""R2AI2026 Shared Task 3 — local replica of the official scoring rule.

The organizers score a submission with:

    FINAL_SCORE = mean(DOCS_F2MACRO, CHUNKS_F2MACRO)
    F2          = 5 * P * R / (4 * P + R)          (macro-averaged per query)

Document level:
    P = |D ∩ G| / |D|      R = |D ∩ G| / |G|

Chunk level:
    a submitted chunk c counts as correct when, after text normalization,
    overlap(c, g) = LCS_tokens(c, g) / |g| >= 0.40 for at least one reference
    chunk g of the same query and the same doc_id.
    Normalization: NFKC, HTML entities decoded, lowercased, whitespace collapsed.
    chunk precision = correct submitted chunks / submitted chunks
    chunk recall    = reference chunks matched by >= 1 submitted chunk / reference chunks

This module implements that rule so tuning never has to spend submission quota.
It is a replica built from the published description, not a copy of the hidden
scoring program: treat small deviations (tokenizer choice, empty-gold handling)
as risks to be monitored, not as facts.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

OVERLAP_THRESHOLD = 0.40

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------------
# normalization / tokenization
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """NFKC, HTML entities decoded, lowercase, whitespace collapsed and trimmed."""
    text = unicodedata.normalize("NFKC", text or "")
    text = html.unescape(text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Word-ish tokens (``\\w+``) after normalization.

    The official scorer pipes text through a HF `tokenizers` pipeline, so its
    tokens are subwords rather than words. Word tokens are used here because the
    overlap ratio LCS(c, g) / |g| is far less sensitive to the token alphabet
    than to whether the gold span is present in the submitted text at all.
    """
    return _WORD_RE.findall(normalize_text(text))


# ---------------------------------------------------------------------------
# LCS
# ---------------------------------------------------------------------------
def _lcs_dp(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence (order kept, gaps allowed)."""
    try:  # pragma: no cover - exercised implicitly when rapidfuzz is present
        from rapidfuzz.distance import LCSseq

        return int(LCSseq.similarity(a, b))
    except Exception:
        return _lcs_dp(a, b)


def chunk_overlap(chunk_tokens: Sequence[str], gold_tokens: Sequence[str]) -> float:
    """LCS_tokens(c, g) / |g|, the quantity the official rule thresholds at 0.40."""
    if not gold_tokens:
        return 0.0
    return lcs_length(chunk_tokens, gold_tokens) / len(gold_tokens)


# ---------------------------------------------------------------------------
# per-query scoring
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class QueryScore:
    docs_precision: float = 0.0
    docs_recall: float = 0.0
    docs_f2: float = 0.0
    chunks_precision: float = 0.0
    chunks_recall: float = 0.0
    chunks_f2: float = 0.0

    @property
    def final(self) -> float:
        return (self.docs_f2 + self.chunks_f2) / 2


def f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator <= 0.0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denominator


def score_query(
    predicted_docs: Iterable[int | str],
    gold_docs: Iterable[int | str],
    predicted_chunks: Sequence[tuple[int | str, str]] = (),
    gold_chunks: Sequence[tuple[int | str, str]] = (),
    *,
    threshold: float = OVERLAP_THRESHOLD,
    empty_gold_policy: str = "skip",
) -> QueryScore:
    """Score one query.

    ``predicted_chunks`` / ``gold_chunks`` are ``(doc_id, text)`` pairs.
    ``empty_gold_policy``:
        ``"skip"``  — queries with no gold at either level score nothing (the
                      metric is only defined where references exist),
        ``"zero"``  — keep the query and score 0 (macro over all queries).
    """
    predicted_docs = list(dict.fromkeys(predicted_docs))
    gold_docs = list(dict.fromkeys(gold_docs))
    score = QueryScore()

    if gold_docs:
        hits = len(set(predicted_docs) & set(gold_docs))
        score.docs_precision = hits / len(predicted_docs) if predicted_docs else 0.0
        score.docs_recall = hits / len(gold_docs)
        score.docs_f2 = f_beta(score.docs_precision, score.docs_recall)

    gold_tokens: list[tuple[int | str, list[str]]] = [
        (doc_id, tokenize(text)) for doc_id, text in gold_chunks
    ]
    gold_tokens = [(doc_id, tokens) for doc_id, tokens in gold_tokens if tokens]

    if gold_tokens:
        submitted: list[tuple[int | str, list[str]]] = [
            (doc_id, tokenize(text)) for doc_id, text in predicted_chunks
        ]
        submitted = [(doc_id, tokens) for doc_id, tokens in submitted if tokens]

        found_gold: set[int] = set()
        correct_submitted = 0
        for doc_id, tokens in submitted:
            matched = False
            for gold_index, (gold_doc_id, gold) in enumerate(gold_tokens):
                if gold_doc_id != doc_id:
                    continue
                if chunk_overlap(tokens, gold) >= threshold:
                    matched = True
                    found_gold.add(gold_index)
            correct_submitted += int(matched)

        score.chunks_precision = correct_submitted / len(submitted) if submitted else 0.0
        score.chunks_recall = len(found_gold) / len(gold_tokens)
        score.chunks_f2 = f_beta(score.chunks_precision, score.chunks_recall)

    if empty_gold_policy == "skip" and not gold_docs and not gold_tokens:
        return QueryScore()
    return score


# ---------------------------------------------------------------------------
# submission-level scoring
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SubmissionScore:
    docs_precision: float
    docs_recall: float
    docs_f2: float
    chunks_precision: float
    chunks_recall: float
    chunks_f2: float
    final_score: float
    queries_scored: int
    queries_total: int
    per_query: dict[str, QueryScore] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "FINAL_SCORE": self.final_score,
            "DOCS_F2MACRO": self.docs_f2,
            "CHUNKS_F2MACRO": self.chunks_f2,
            "DOCS_PRECISION": self.docs_precision,
            "DOCS_RECALL": self.docs_recall,
            "CHUNKS_PRECISION": self.chunks_precision,
            "CHUNKS_RECALL": self.chunks_recall,
            "queries_scored": self.queries_scored,
            "queries_total": self.queries_total,
        }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_submission(
    predictions: Mapping[int | str, dict],
    references: Mapping[int | str, dict],
    *,
    threshold: float = OVERLAP_THRESHOLD,
    empty_gold_policy: str = "skip",
) -> SubmissionScore:
    """Score a whole submission.

    ``predictions[qid] = {"relevant_docs": [...], "relevant_chunks": [{"doc_id": .., "chunk_text": ..}]}``
    ``references[qid]``  = same shape, holding gold labels.
    """
    per_query: dict[str, QueryScore] = {}
    for query_id, reference in references.items():
        prediction = predictions.get(query_id) or {}
        gold_chunks = [
            (chunk.get("doc_id"), chunk.get("chunk_text", ""))
            for chunk in reference.get("relevant_chunks", [])
        ]
        pred_chunks = [
            (chunk.get("doc_id"), chunk.get("chunk_text", ""))
            for chunk in (prediction.get("relevant_chunks") or [])
        ]
        per_query[str(query_id)] = score_query(
            prediction.get("relevant_docs") or [],
            reference.get("relevant_docs") or [],
            pred_chunks,
            gold_chunks,
            threshold=threshold,
            empty_gold_policy=empty_gold_policy,
        )

    scored = [
        score
        for score in per_query.values()
        if not (score.docs_precision == score.docs_recall == 0.0 and score.chunks_recall == 0.0
                and score.chunks_precision == 0.0)
    ]
    docs_f2 = _mean([score.docs_f2 for score in scored])
    chunks_f2 = _mean([score.chunks_f2 for score in scored])
    return SubmissionScore(
        docs_precision=_mean([score.docs_precision for score in scored]),
        docs_recall=_mean([score.docs_recall for score in scored]),
        docs_f2=docs_f2,
        chunks_precision=_mean([score.chunks_precision for score in scored]),
        chunks_recall=_mean([score.chunks_recall for score in scored]),
        chunks_f2=chunks_f2,
        final_score=(docs_f2 + chunks_f2) / 2,
        queries_scored=len(scored),
        queries_total=len(references),
        per_query=per_query,
    )


# ---------------------------------------------------------------------------
# depth tuning helpers (F2-aware submission policy)
# ---------------------------------------------------------------------------
def depth_curve_docs(
    ranked: Mapping[int | str, Sequence[int | str]],
    references: Mapping[int | str, dict],
    depths: Sequence[int],
) -> list[tuple[int, float]]:
    """Macro docs F2 for each fixed submitted depth (offline proxy tuning)."""
    curve: list[tuple[int, float]] = []
    for depth in depths:
        predictions = {
            query_id: {"relevant_docs": list(docs)[:depth], "relevant_chunks": []}
            for query_id, docs in ranked.items()
        }
        curve.append((depth, score_submission(predictions, references).docs_f2))
    return curve


def depth_curve_chunks(
    ranked: Mapping[int | str, Sequence[tuple[int | str, str]]],
    references: Mapping[int | str, dict],
    depths: Sequence[int],
    *,
    predicted_docs: Mapping[int | str, Sequence[int | str]] | None = None,
) -> list[tuple[int, float]]:
    """Macro chunks F2 for each fixed submitted chunk depth."""
    curve: list[tuple[int, float]] = []
    for depth in depths:
        predictions: dict[int | str, dict] = {}
        for query_id, chunks in ranked.items():
            predictions[query_id] = {
                "relevant_docs": list((predicted_docs or {}).get(query_id, [])),
                "relevant_chunks": [
                    {"doc_id": doc_id, "chunk_text": text} for doc_id, text in list(chunks)[:depth]
                ],
            }
        curve.append((depth, score_submission(predictions, references).chunks_f2))
    return curve
