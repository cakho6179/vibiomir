"""Dense & Hybrid Re-ranking using BAAI/bge-m3 on RTX 4060 GPU for ViBioMIR.

Combines high-recall BM25 candidate retrieval with multilingual semantic scoring
from BAAI/bge-m3 (dense embeddings), re-ranks top candidates, and extracts
verbatim chunks formatted for Codabench.

Usage:
    python scripts/r2ai2026_dense_rerank.py \
        --docs data/docs.jsonl \
        --queries data/query_translated.parquet \
        --index-pubmed runs/indexes/index_pubmed \
        --index-web runs/indexes/index_web \
        --submission runs/submissions/submission_hybrid_bgem3_v1.zip \
        --candidate-pool 70 \
        --topk-docs 35 \
        --chunk-budget 1400 \
        --alpha 0.6
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai2026.bm25 import BM25Index  # noqa: E402
from r2ai2026.chunking import spread_windows_chunk  # noqa: E402
from r2ai2026.pipeline import document_text, load_documents  # noqa: E402
from r2ai2026.submit import validate_submission, write_submission_zip  # noqa: E402


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    weights: list[float] | None = None,
    k: int = 60,
    top_n: int = 50,
) -> list[int]:
    """Fuse multiple ranked doc lists using Reciprocal Rank Fusion."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[int, float] = defaultdict(float)
    for ranked_docs, weight in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked_docs, start=1):
            scores[doc_id] += weight / (k + rank)

    sorted_docs = sorted(scores.keys(), key=lambda doc: scores[doc], reverse=True)
    return sorted_docs[:top_n]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, default=Path("data/docs.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/query_translated.parquet"))
    parser.add_argument("--index-pubmed", type=Path, default=Path("runs/indexes/index_pubmed"))
    parser.add_argument("--index-web", type=Path, default=Path("runs/indexes/index_web"))
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=int, default=70,
                        help="Number of BM25 candidates to retrieve per language partition")
    parser.add_argument("--topk-docs", type=int, default=35,
                        help="Final number of documents to select per query")
    parser.add_argument("--chunk-budget", type=int, default=1400)
    parser.add_argument("--spread-windows", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Weight for dense semantic score (1 - alpha for BM25 RRF score)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for dense encoding on GPU")
    parser.add_argument("--stats-out", type=Path, default=None)
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device for dense encoding: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Queries
    started = time.perf_counter()
    df_queries = pd.read_parquet(args.queries)
    print(f"Loaded {len(df_queries):,} queries (VI + EN) in {time.perf_counter() - started:.2f}s")
    queries_vi = df_queries["query"].tolist()
    queries_en = df_queries["query_en"].tolist()
    query_ids = df_queries["id"].tolist()

    # 2. Candidate retrieval via Dual-Partition BM25
    started = time.perf_counter()
    index_pm = BM25Index.load(args.index_pubmed, name="docs", mmap=True)
    index_web = BM25Index.load(args.index_web, name="docs", mmap=True)
    print(f"Loaded partitioned indexes: PubMed ({index_pm.size:,} docs) & Web ({index_web.size:,} docs)")

    print(f"Searching Web index with VI queries (k={args.candidate_pool})...")
    hits_vi = index_web.search_batch(queries_vi, k=args.candidate_pool)
    print(f"Searching PubMed index with EN queries (k={args.candidate_pool})...")
    hits_en = index_pm.search_batch(queries_en, k=args.candidate_pool)
    print(f"BM25 search completed in {time.perf_counter() - started:.2f}s")

    # 3. Form initial fused candidate pool per query
    candidates_per_query: dict[int, list[int]] = {}
    bm25_rrf_scores: dict[int, dict[int, float]] = {}
    all_needed_doc_ids: set[int] = set()

    for idx, qid in enumerate(query_ids):
        docs_vi = [int(hit[0][0]) for hit in hits_vi[idx]]
        docs_en = [int(hit[0][0]) for hit in hits_en[idx]]
        # compute RRF scores for top candidates
        q_scores: dict[int, float] = defaultdict(float)
        for rank, d in enumerate(docs_vi, start=1):
            q_scores[d] += 1.0 / (60 + rank)
        for rank, d in enumerate(docs_en, start=1):
            q_scores[d] += 1.0 / (60 + rank)

        sorted_cand = sorted(q_scores.keys(), key=lambda d: q_scores[d], reverse=True)[:args.candidate_pool]
        candidates_per_query[qid] = sorted_cand
        bm25_rrf_scores[qid] = {d: q_scores[d] for d in sorted_cand}
        all_needed_doc_ids.update(sorted_cand)

    print(f"Total unique candidate documents across all queries: {len(all_needed_doc_ids):,}")

    # 4. Load text for candidate documents
    started = time.perf_counter()
    doc_texts: dict[int, str] = {}
    doc_leads: dict[int, str] = {}
    for record in load_documents(args.docs):
        if not record.get("ok", True):
            continue
        did = int(record["id"])
        if did in all_needed_doc_ids:
            full_txt = document_text(record)
            doc_texts[did] = full_txt
            # For dense embedding, take the title + first 400 words (the most information-dense lead)
            words = full_txt.split()
            doc_leads[did] = " ".join(words[:400]) if len(words) > 400 else full_txt
            if len(doc_texts) == len(all_needed_doc_ids):
                break
    print(f"Loaded {len(doc_texts):,} candidate texts in {time.perf_counter() - started:.2f}s")

    # Free BM25 index handles
    del index_pm, index_web
    gc.collect()

    # 5. Semantic Scoring with BAAI/bge-m3
    started = time.perf_counter()
    print("Loading BAAI/bge-m3 for dense re-ranking...")
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    if device == "cuda":
        model = model.half()  # fp16 for 2x speed and half VRAM

    print("Encoding queries...")
    # Encode both Vietnamese and English queries for multilingual matching
    q_emb_vi = model.encode(queries_vi, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=False)
    q_emb_en = model.encode(queries_en, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=False)

    print(f"Pre-encoding {len(all_needed_doc_ids):,} unique candidate documents on GPU...")
    unique_cands = sorted(all_needed_doc_ids)
    doc_to_idx = {did: i for i, did in enumerate(unique_cands)}
    unique_texts = [doc_leads.get(did, "") for did in unique_cands]
    all_d_embs = model.encode(unique_texts, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True)

    print("Scoring queries against candidate embeddings...")
    final_rankings: dict[int, list[int]] = {}

    # Score queries
    for idx, qid in enumerate(query_ids):
        cands = candidates_per_query[qid]
        if not cands:
            final_rankings[qid] = []
            continue

        cand_indices = [doc_to_idx[d] for d in cands if d in doc_to_idx]
        if not cand_indices:
            final_rankings[qid] = []
            continue

        d_embs = all_d_embs[cand_indices]

        # Dense similarity: max of VI query and EN query similarity against candidate
        sim_vi = np.dot(d_embs, q_emb_vi[idx])
        sim_en = np.dot(d_embs, q_emb_en[idx])
        dense_sim = np.maximum(sim_vi, sim_en)

        # Normalize dense_sim to [0, 1] range within candidate set
        dense_min, dense_max = float(dense_sim.min()), float(dense_sim.max())
        if dense_max > dense_min:
            norm_dense = (dense_sim - dense_min) / (dense_max - dense_min)
        else:
            norm_dense = np.ones_like(dense_sim)

        # Normalize BM25 RRF scores to [0, 1]
        bm25_vals = np.array([bm25_rrf_scores[qid].get(d, 0.0) for d in cands])
        b_min, b_max = float(bm25_vals.min()), float(bm25_vals.max())
        if b_max > b_min:
            norm_bm25 = (bm25_vals - b_min) / (b_max - b_min)
        else:
            norm_bm25 = np.ones_like(bm25_vals)

        # Hybrid Score: (1 - alpha) * BM25 + alpha * Dense
        hybrid_scores = (1.0 - args.alpha) * norm_bm25 + args.alpha * norm_dense

        # Sort candidates
        sorted_indices = np.argsort(-hybrid_scores)
        top_docs = [cands[i] for i in sorted_indices[:args.topk_docs]]
        final_rankings[qid] = top_docs

        if (idx + 1) % 200 == 0 or (idx + 1) == len(query_ids):
            print(f"  Processed {idx + 1}/{len(query_ids)} queries...")

    print(f"Dense re-ranking completed in {time.perf_counter() - started:.2f}s")

    # 6. Extract Chunks & Write Submission
    started = time.perf_counter()
    records: list[dict] = []
    missing_docs = 0

    for qid in query_ids:
        docs = final_rankings[qid]
        chunks: list[dict] = []
        for did in docs:
            txt = doc_texts.get(did, "")
            if not txt:
                missing_docs += 1
            chunk_txt = spread_windows_chunk(txt, max_tokens=args.chunk_budget, windows=args.spread_windows)
            chunks.append({"doc_id": int(did), "chunk_text": chunk_txt})
        records.append({
            "id": int(qid),
            "relevant_docs": [int(d) for d in docs],
            "relevant_chunks": chunks,
        })

    args.submission.parent.mkdir(parents=True, exist_ok=True)
    res = write_submission_zip(records, args.submission)
    val_errors = validate_submission(records, expected_ids=query_ids)

    elapsed = time.perf_counter() - started
    print(f"\n[DONE] Built submission in {elapsed:.2f}s -> {args.submission}")
    print(f"Validation errors: {len(val_errors)}")
    if val_errors:
        for err in val_errors[:5]:
            print(f"  ERROR: {err}")
    if missing_docs:
        print(f"  WARNING: {missing_docs} documents had empty or missing text in docs.jsonl")

    stats = {
        "submission_zip": str(args.submission),
        "zip_size_mb": args.submission.stat().st_size / 1e6,
        "queries": len(query_ids),
        "docs_per_query": args.topk_docs,
        "candidate_pool": args.candidate_pool,
        "alpha_dense": args.alpha,
        "chunk_budget": args.chunk_budget,
        "spread_windows": args.spread_windows,
        "val_errors": len(val_errors),
        "missing_docs": missing_docs,
    }
    if args.stats_out:
        args.stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Zip size: {stats['zip_size_mb']:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
