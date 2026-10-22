"""Decision Model / Verification Filter for R2AI2026.

Prunes low-confidence candidates after Cross-Encoder / BM25 reranking to stop
Precision Dilution and maximize F2 (where P = Hits / N).

Techniques:
1. Relative Score Dropoff Thresholding: Drop candidates where score < alpha * top_score.
2. Softmax Confidence Mass: Keep top candidates covering >= target_mass probability.
3. Dynamic Bounding: Clip output length to [min_docs, max_docs] (default: [1, 5]).
"""

from __future__ import annotations

import math
from typing import Sequence


def relative_dropoff_filter(
    doc_ids: Sequence[int],
    scores: Sequence[float],
    *,
    alpha: float = 0.50,
    min_docs: int = 1,
    max_docs: int = 5,
) -> list[int]:
    """Retain candidates whose score is within alpha of the top score.

    Parameters
    ----------
    doc_ids : Sequence[int]
        Ranked list of document IDs (highest score first).
    scores : Sequence[float]
        Corresponding relevance scores (monotonically decreasing).
    alpha : float, default 0.50
        Score retention ratio relative to the top candidate.
    min_docs : int, default 1
        Lower bound on returned documents (never return 0).
    max_docs : int, default 5
        Upper bound on returned documents (prevents precision dilution).
    """
    if not doc_ids:
        return []
    if len(doc_ids) == 1:
        return [doc_ids[0]]

    top_score = scores[0]
    # If top score is non-positive or very small, take min_docs
    if top_score <= 1e-6:
        return list(doc_ids[:min_docs])

    cutoff = top_score * alpha
    selected: list[int] = []

    for did, sc in zip(doc_ids, scores):
        if sc >= cutoff:
            selected.append(did)
        else:
            break

    # Apply bounding
    if len(selected) < min_docs:
        selected = list(doc_ids[:min_docs])
    elif len(selected) > max_docs:
        selected = selected[:max_docs]

    return selected


def softmax_mass_filter(
    doc_ids: Sequence[int],
    scores: Sequence[float],
    *,
    target_mass: float = 0.80,
    temperature: float = 2.0,
    min_docs: int = 1,
    max_docs: int = 5,
) -> list[int]:
    """Retain smallest set of candidates whose softmax probability reaches target_mass."""
    if not doc_ids:
        return []
    if len(doc_ids) <= min_docs:
        return list(doc_ids)

    # Consider top-10 for probability distribution
    pool_size = min(10, len(doc_ids))
    sub_scores = scores[:pool_size]
    max_s = max(sub_scores)

    exp_scores = [math.exp((s - max_s) / temperature) for s in sub_scores]
    sum_exp = sum(exp_scores)
    probs = [e / sum_exp for e in exp_scores]

    cum_prob = 0.0
    selected_count = 0
    for p in probs:
        cum_prob += p
        selected_count += 1
        if cum_prob >= target_mass:
            break

    k = max(min_docs, min(selected_count, max_docs))
    return list(doc_ids[:k])


def build_verification_judge_prompt(query: str, doc_title: str, doc_snippet: str) -> str:
    """Build a pointwise LLM-as-a-Judge binary verification prompt for border candidates (Rank 2 to 5).

    Instructs a fast sub-15B model (Qwen2.5-7B) to verify whether the candidate document
    actually contains the clinical information needed to answer the patient inquiry.
    """
    snippet_clean = doc_snippet.strip().replace("\n", " ")[:400]
    return (
        "Bạn là giám định viên y khoa. Nhiệm vụ của bạn là đánh giá tính liên quan trực tiếp của tài liệu đối với câu hỏi.\n\n"
        f"Câu hỏi của bệnh nhân: {query}\n\n"
        f"Tiêu đề tài liệu: {doc_title}\n"
        f"Đoạn trích nội dung: {snippet_clean}...\n\n"
        "Câu hỏi: Bài viết trên có chứa thông tin trả lời trực tiếp hoặc chẩn đoán liên quan đến câu hỏi của bệnh nhân không?\n"
        "Chỉ trả lời chính xác một từ duy nhất: YES hoặc NO."
    )


# -------------------------------------------------------------------------
# Joint Expected Value (JEV) Decision Model
# -------------------------------------------------------------------------

def joint_expected_value_filter(
    doc_ids: Sequence[int],
    scores: Sequence[float],
    *,
    k_gold: float = 2.0,
    beta: float = 2.0,
    p_top: float = 0.90,
    gamma: float = 2.0,
    min_docs: int = 1,
    max_docs: int = 4,
) -> list[int]:
    """Joint Expected Value (JEV) Decision Model for optimal F_beta retrieval cutoff.

    Directly optimizes the Joint Expected Value E[F_beta(S)] over candidate set S:
        E[F_beta(S)] = (1 + beta^2) * E[Hits] / (beta^2 * |S| + k_gold)
    
    The marginal gain in Joint Expected Value when adding candidate d_{N+1} with probability p_{N+1}:
        Delta JEV(d_{N+1}) = E[F_beta(S_{N+1})] - E[F_beta(S_N)]
    Candidate is admitted if and only if Delta JEV > 0, which reduces to the exact stopping condition:
        p_{N+1} > (beta^2 * H_N) / (beta^2 * N + k_gold)
    where H_N = sum_{j=1}^N p_j is the cumulative expected relevant hits.

    Parameters
    ----------
    doc_ids : Sequence[int]
        Ranked list of document IDs (highest score first).
    scores : Sequence[float]
        Corresponding relevance scores (monotonically decreasing).
    k_gold : float, default 2.0
        Expected number of gold relevant documents per telehealth query (~2.0).
    beta : float, default 2.0
        F-beta parameter (beta=2 weights recall higher than precision).
    p_top : float, default 0.90
        Prior relevance probability of the top-ranked candidate.
    gamma : float, default 2.0
        Power scaling exponent for probability calibration from monotonic scores.
    min_docs : int, default 1
        Lower bound on returned documents.
    max_docs : int, default 4
        Upper bound on returned documents.
    """
    if not doc_ids:
        return []
    if len(doc_ids) <= min_docs:
        return list(doc_ids[:min_docs])

    top_s = scores[0]
    if top_s <= 1e-6:
        return list(doc_ids[:min_docs])

    # Convert monotonic scores to calibrated probabilities via power scaling
    probs = [min(1.0, max(0.01, p_top * ((s / top_s) ** gamma))) for s in scores]

    beta_sq = beta * beta  # 4.0 for F2
    selected: list[int] = [doc_ids[0]]
    cum_hits: float = probs[0]

    for n in range(1, len(doc_ids)):
        if len(selected) >= max_docs:
            break

        p_next = probs[n]
        # JEV Marginal Threshold condition:
        # Delta JEV > 0  <===>  p_{N+1} > (beta^2 * H_N) / (beta^2 * N + k_gold)
        jev_threshold = (beta_sq * cum_hits) / (beta_sq * n + k_gold)

        if p_next >= jev_threshold:
            selected.append(doc_ids[n])
            cum_hits += p_next
        else:
            # JEV stopping boundary reached: adding this document would degrade expected F2
            break

    if len(selected) < min_docs:
        selected = list(doc_ids[:min_docs])

    return selected


