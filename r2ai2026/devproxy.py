"""Offline proxies for a competition with secret test labels.

Two independent dev signals, neither of which needs a leaderboard submission:

1. **Title-as-query** — health articles carry question-like titles, so ranking
   documents for their own titles measures document retrieval quality
   (recall@k, MRR) with zero labelling cost.
2. **Pseudo-gold spans** — a random sentence-aligned span of a document stands in
   for a gold chunk. Because the official chunk rule is
   ``LCS(submitted, gold) / |gold| >= 0.40``, this proxy directly measures how
   well a submission policy covers unknown gold spans in a document it already
   retrieved.

Both proxies are biased (real queries are short and messy; gold chunks were
chosen by the organizers), so they are used for *relative* decisions only.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from r2ai2026.scoring import SubmissionScore, score_submission
from r2ai2026.text import sentence_spans, word_count


def sentential_spans(text: str, *, target_tokens: int, rng: random.Random, jitter: float = 0.25) -> list[str]:
    """Random sentence-aligned spans of roughly ``target_tokens`` tokens."""
    spans = sentence_spans(text)
    if not spans:
        return []
    wanted = max(20, int(target_tokens * (1 + rng.uniform(-jitter, jitter))))
    chunks: list[str] = []
    start_index = 0
    while start_index < len(spans):
        tokens = 0
        end_index = start_index
        while end_index < len(spans) and tokens < wanted:
            tokens += word_count(text[spans[end_index][0] : spans[end_index][1]])
            end_index += 1
        chunks.append(text[spans[start_index][0] : spans[end_index - 1][1]].strip())
        start_index = end_index
    return [chunk for chunk in chunks if chunk]


def pseudo_gold_for_document(
    doc_id: int, text: str, *, rng: random.Random, spans: int = 2, span_tokens: int = 180
) -> dict:
    """A fake gold record: the document plus 1..spans pseudo-gold chunks from it."""
    candidates = sentential_spans(text, target_tokens=span_tokens, rng=rng)
    if not candidates:
        return {"relevant_docs": [], "relevant_chunks": []}
    rng.shuffle(candidates)
    picked = candidates[: max(1, min(spans, len(candidates)))]
    return {
        "relevant_docs": [doc_id],
        "relevant_chunks": [{"doc_id": doc_id, "chunk_text": chunk} for chunk in picked],
    }


def build_pseudo_gold(
    documents: Mapping[int, str],
    *,
    seed: int = 20261018,
    spans: int = 2,
    span_tokens: int = 180,
) -> dict[int, dict]:
    rng = random.Random(seed)
    gold: dict[int, dict] = {}
    for doc_id, text in documents.items():
        record = pseudo_gold_for_document(doc_id, text, rng=rng, spans=spans, span_tokens=span_tokens)
        if record["relevant_docs"]:
            gold[doc_id] = record
    return gold


@dataclass(slots=True)
class RetrievalMetrics:
    recall_at: dict[int, float]
    mrr: float
    n_queries: int

    def as_dict(self) -> dict[str, float]:
        output = {f"recall@{k}": value for k, value in sorted(self.recall_at.items())}
        output["mrr"] = self.mrr
        output["n_queries"] = float(self.n_queries)
        return output


def retrieval_metrics(
    rankings: Mapping[int, Sequence[int]],
    gold_docs: Mapping[int, Iterable[int]],
    *,
    ks: Sequence[int] = (1, 5, 10, 20, 50),
) -> RetrievalMetrics:
    recall_at = {k: 0.0 for k in ks}
    reciprocal_ranks: list[float] = []
    counted = 0
    for query_id, ranked in rankings.items():
        gold = set(int(value) for value in gold_docs.get(query_id, []))
        if not gold:
            continue
        counted += 1
        for k in ks:
            if gold & set(int(value) for value in list(ranked)[:k]):
                recall_at[k] += 1.0
        reciprocal = 0.0
        for position, doc_id in enumerate(ranked, start=1):
            if int(doc_id) in gold:
                reciprocal = 1.0 / position
                break
        reciprocal_ranks.append(reciprocal)
    if counted:
        recall_at = {k: value / counted for k, value in recall_at.items()}
    return RetrievalMetrics(
        recall_at=recall_at,
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        n_queries=counted,
    )


def title_queries(documents: Mapping[int, str], titles: Mapping[int, str], *, min_words: int = 5) -> dict[int, str]:
    """``{doc_id: title}`` pairs usable as stand-in queries."""
    return {
        doc_id: title.strip()
        for doc_id, title in titles.items()
        if doc_id in documents and word_count(title or "") >= min_words
    }


def score_span_recovery(
    predicted_chunks: Mapping[int, Sequence[tuple[int, str]]],
    pseudo_gold: Mapping[int, dict],
    *,
    predicted_docs: Mapping[int, Sequence[int]] | None = None,
) -> SubmissionScore:
    """Score a chunk policy against pseudo-gold spans with the official rule."""
    predictions: dict[int, dict] = {}
    for query_id, chunks in predicted_chunks.items():
        predictions[query_id] = {
            "relevant_docs": list((predicted_docs or {}).get(query_id, [])),
            "relevant_chunks": [{"doc_id": doc_id, "chunk_text": text} for doc_id, text in chunks],
        }
    references = {query_id: record for query_id, record in pseudo_gold.items()}
    return score_submission(predictions, references)


def span_recovery_curve(
    documents: Mapping[int, str],
    build_chunk: object,
    *,
    budgets: Sequence[int] = (200, 400, 800, 1200, 2000, 4000, 0),
    seed: int = 20261018,
    spans: int = 2,
    span_tokens: int = 180,
    max_documents: int | None = 400,
) -> list[dict[str, float]]:
    """Macro chunk F2 for a "one merged chunk per document" policy at several budgets.

    ``build_chunk(text, budget) -> str`` is the policy under test; a budget of 0
    means "whole document".
    """
    rng = random.Random(seed)
    doc_ids = sorted(documents)
    if max_documents is not None:
        doc_ids = doc_ids[:max_documents]
    picked = {doc_id: documents[doc_id] for doc_id in doc_ids}
    gold = build_pseudo_gold(picked, seed=seed, spans=spans, span_tokens=span_tokens)

    curve: list[dict[str, float]] = []
    for budget in budgets:
        chunks = {
            doc_id: [(doc_id, build_chunk(text, None if budget == 0 else budget))]  # type: ignore[operator]
            for doc_id, text in picked.items()
        }
        score = score_span_recovery(chunks, gold, predicted_docs={doc_id: [doc_id] for doc_id in picked})
        curve.append(
            {
                "budget_tokens": float(budget),
                "chunk_precision": score.chunks_precision,
                "chunk_recall": score.chunks_recall,
                "chunk_f2": score.chunks_f2,
                "docs_f2": score.docs_f2,
                "final": score.final_score,
            }
        )
    return curve


def policy_comparison(
    documents: Mapping[int, str],
    policies: Mapping[str, object],
    *,
    budgets: Sequence[int] = (400, 800, 1400, 3000, 0),
    seed: int = 20261018,
    spans: int = 2,
    span_tokens: int = 180,
    max_documents: int | None = 400,
) -> list[dict[str, float | str]]:
    """Compare chunk policies at equal token budgets on the same pseudo-gold spans.

    Every row reports macro chunk F2 *and* the average submitted characters, so a
    policy is only preferred when it wins at a comparable payload size.
    """
    doc_ids = sorted(documents)
    if max_documents is not None:
        doc_ids = doc_ids[:max_documents]
    picked = {doc_id: documents[doc_id] for doc_id in doc_ids}
    gold = build_pseudo_gold(picked, seed=seed, spans=spans, span_tokens=span_tokens)

    rows: list[dict[str, float | str]] = []
    for name, build_chunk in policies.items():
        for budget in budgets:
            chunks = {
                doc_id: [(doc_id, build_chunk(text, None if budget == 0 else budget))]  # type: ignore[operator]
                for doc_id, text in picked.items()
            }
            score = score_span_recovery(chunks, gold, predicted_docs={doc_id: [doc_id] for doc_id in picked})
            chars = sum(len(text) for pair in chunks.values() for _, text in pair)
            rows.append(
                {
                    "policy": name,
                    "budget_tokens": float(budget),
                    "chunk_precision": score.chunks_precision,
                    "chunk_recall": score.chunks_recall,
                    "chunk_f2": score.chunks_f2,
                    "mean_chars": chars / max(1, len(chunks)),
                    "total_mb_per_1200q": chars * 1200 / max(1, len(chunks)) / 1_048_576,
                }
            )
    rows.sort(key=lambda row: (str(row["policy"]), int(row["budget_tokens"])))
    return rows
