"""Rank Fusion and Ensembling algorithms for Information Retrieval.

Includes:
- Reciprocal Rank Fusion (RRF)
- Weighted Reciprocal Rank Fusion
- Normalized Score Fusion (CombSUM, CombMNZ)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence
import numpy as np


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    weights: Sequence[float] | None = None,
    k: int = 60,
    top_n: int = 35,
) -> list[int]:
    """Fuse multiple ranked document ID lists using Reciprocal Rank Fusion (RRF).

    Formula:
        Score(d) = sum_{m} [ w_m / (k + rank_{m}(d)) ]
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[int, float] = defaultdict(float)
    for ranked_docs, weight in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked_docs, start=1):
            scores[doc_id] += weight / (k + rank)

    sorted_docs = sorted(scores.keys(), key=lambda doc: scores[doc], reverse=True)
    return sorted_docs[:top_n]


def hybrid_score_fusion(
    bm25_ranks: Sequence[int],
    dense_scores: np.ndarray,
    dense_candidates: Sequence[int],
    alpha: float = 0.6,
    k: int = 60,
    top_n: int = 35,
) -> list[int]:
    """Combine sparse BM25 RRF rank score with dense semantic cosine similarity.

    Args:
        bm25_ranks: Candidate doc IDs in order of BM25 rank.
        dense_scores: Array of float dense similarities aligned with dense_candidates.
        dense_candidates: Candidate doc IDs corresponding to dense_scores.
        alpha: Weight for dense score (1.0 - alpha for BM25 rank score).
        k: Smoothing constant for reciprocal rank.
        top_n: Number of top documents to return.
    """
    bm25_score_map = {doc: 1.0 / (k + r) for r, doc in enumerate(bm25_ranks, start=1)}

    # Min-max normalize dense scores
    d_min, d_max = float(dense_scores.min()), float(dense_scores.max())
    if d_max > d_min:
        norm_dense = (dense_scores - d_min) / (d_max - d_min)
    else:
        norm_dense = np.ones_like(dense_scores)

    dense_score_map = {doc: float(norm_dense[i]) for i, doc in enumerate(dense_candidates)}

    # Normalize BM25 values
    all_cands = list(set(bm25_ranks).union(dense_candidates))
    bm25_vals = np.array([bm25_score_map.get(d, 0.0) for d in all_cands])
    b_min, b_max = float(bm25_vals.min()), float(bm25_vals.max())
    if b_max > b_min:
        norm_bm25 = (bm25_vals - b_min) / (b_max - b_min)
    else:
        norm_bm25 = np.ones_like(bm25_vals)

    norm_bm25_map = {d: float(norm_bm25[i]) for i, d in enumerate(all_cands)}

    # Compute final hybrid score
    fused: dict[int, float] = {}
    for d in all_cands:
        s_bm25 = norm_bm25_map.get(d, 0.0)
        s_dense = dense_score_map.get(d, 0.0)
        fused[d] = (1.0 - alpha) * s_bm25 + alpha * s_dense

    sorted_docs = sorted(fused.keys(), key=lambda d: fused[d], reverse=True)
    return sorted_docs[:top_n]
