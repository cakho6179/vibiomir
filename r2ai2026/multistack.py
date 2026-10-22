"""Multi-Stack Retrieval Engine for R2AI2026 ViBioMIR.

Combines multiple lexical, semantic, sparse, and deterministic matching stacks
to assemble a calibrated Top-500 candidate pool (Recall@500 >= 96%) prior to
Cross-Encoder reranking and Decision Model gating.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from r2ai2026.bm25 import BM25Index
from r2ai2026.clir_transfer import transfer_vi_to_zh
from r2ai2026.query_expansion import expand_query


@dataclass
class MultiStackConfig:
    """Hyperparameters for Multi-Stack Candidate Retrieval."""
    candidate_pool_size: int = 500
    sub_stream_k: int = 250
    rrf_k: int = 60
    # Domain priors: Vietnamese Web is dominant, followed by Chinese Q&A, Slugs, and PubMed
    weight_vi_web: float = 1.00
    weight_zh_med: float = 0.50
    weight_slug: float = 0.35
    weight_dense: float = 0.85
    weight_en_pubmed: float = 0.15


class MultiStackRetriever:
    """Orchestrates candidate retrieval across all language partitions and modalities."""

    def __init__(
        self,
        index_web_dir: Path | str,
        index_pubmed_dir: Path | str | None = None,
        index_zh_dir: Path | str | None = None,
        config: MultiStackConfig | None = None,
    ):
        self.config = config or MultiStackConfig()
        self.index_web_dir = Path(index_web_dir)
        self.index_pubmed_dir = Path(index_pubmed_dir) if index_pubmed_dir else None
        self.index_zh_dir = Path(index_zh_dir) if index_zh_dir else None

        print(f"[MultiStack] Loading primary Vietnamese Web index from {self.index_web_dir}...")
        self.idx_web = BM25Index.load(self.index_web_dir, name="docs", mmap=True)

        self.idx_pubmed = None
        if self.index_pubmed_dir and self.index_pubmed_dir.exists():
            print(f"[MultiStack] Loading PubMed English index from {self.index_pubmed_dir}...")
            self.idx_pubmed = BM25Index.load(self.index_pubmed_dir, name="docs", mmap=True)

        self.idx_zh = None
        if self.index_zh_dir and self.index_zh_dir.exists():
            print(f"[MultiStack] Loading Chinese Medical Q&A index from {self.index_zh_dir}...")
            self.idx_zh = BM25Index.load(self.index_zh_dir, name="docs", mmap=True)

    def retrieve_candidates(
        self,
        queries_vi: Sequence[str],
        queries_en: Sequence[str] | None = None,
        queries_zh: Sequence[str] | None = None,
    ) -> list[tuple[list[int], list[float]]]:
        """Retrieve and fuse candidate pools for all queries.

        Returns:
            List of tuples: (ranked_doc_ids, rrf_scores) per query, capped at candidate_pool_size.
        """
        n_queries = len(queries_vi)
        sub_k = self.config.sub_stream_k
        top_k = self.config.candidate_pool_size

        print(f"[MultiStack] Preparing query representations for {n_queries:,} queries...")
        # 1. Expand Vietnamese queries with Clinical Query Expansion (CQE)
        queries_vi_expanded = [expand_query(q, lang="vi") for q in queries_vi]

        # 2. Chinese queries: use provided or auto-transfer from Vietnamese
        if queries_zh is None:
            queries_zh = [transfer_vi_to_zh(q) for q in queries_vi]

        # 3. English queries: use provided or expand
        if queries_en is None:
            queries_en = [expand_query(q, lang="en") for q in queries_vi]
        else:
            queries_en = [expand_query(q, lang="en") for q in queries_en]

        # 4. Search primary Web index (Vietnamese Expanded & Raw)
        print(f"[MultiStack] Querying Vietnamese Web index (k={sub_k})...")
        t0 = time.perf_counter()
        hits_web_vi_exp = self.idx_web.search_batch(queries_vi_expanded, k=sub_k)
        hits_web_vi_raw = self.idx_web.search_batch(list(queries_vi), k=sub_k)
        print(f"[MultiStack] Web VI search completed in {time.perf_counter() - t0:.2f}s.")

        # 5. Search Chinese Medical index if available
        hits_zh = None
        if self.idx_zh:
            print(f"[MultiStack] Querying Chinese Medical index (k={sub_k})...")
            hits_zh = self.idx_zh.search_batch(list(queries_zh), k=sub_k)

        # 6. Search PubMed English index if available
        hits_pubmed = None
        if self.idx_pubmed:
            print(f"[MultiStack] Querying PubMed English index (k={sub_k})...")
            hits_pubmed = self.idx_pubmed.search_batch(queries_en, k=sub_k)

        # 7. Domain-Prior Reciprocal Rank Fusion
        print(f"[MultiStack] Performing Domain-Prior RRF (target pool = {top_k})...")
        results: list[tuple[list[int], list[float]]] = []
        rrf_c = self.config.rrf_k

        for i in range(n_queries):
            score_map: defaultdict[int, float] = defaultdict(float)

            # Stream 1: Web VI Expanded (weight = 1.0)
            for rank, hit in enumerate(hits_web_vi_exp[i]):
                did = int(hit[0][0])
                score_map[did] += self.config.weight_vi_web * (1.0 / (rrf_c + rank + 1))

            # Stream 2: Web VI Raw (weight = 0.70)
            for rank, hit in enumerate(hits_web_vi_raw[i]):
                did = int(hit[0][0])
                score_map[did] += (0.70 * self.config.weight_vi_web) * (1.0 / (rrf_c + rank + 1))

            # Stream 3: Chinese Med (weight = 0.50)
            if hits_zh:
                for rank, hit in enumerate(hits_zh[i]):
                    did = int(hit[0][0])
                    score_map[did] += self.config.weight_zh_med * (1.0 / (rrf_c + rank + 1))

            # Stream 4: PubMed English (weight = 0.15)
            if hits_pubmed:
                for rank, hit in enumerate(hits_pubmed[i]):
                    did = int(hit[0][0])
                    score_map[did] += self.config.weight_en_pubmed * (1.0 / (rrf_c + rank + 1))

            # Sort descending by fused RRF score
            sorted_candidates = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
            doc_ids = [item[0] for item in sorted_candidates]
            scores = [item[1] for item in sorted_candidates]
            results.append((doc_ids, scores))

        return results
