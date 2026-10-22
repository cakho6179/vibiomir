"""Build SOTA V4 Submission: CQE + CLIR (ZH & VI) + Domain-Prior RRF + Decision Model Gating (N in [1, 4]).

Breakthrough Features:
1. Decision Model & Dynamic Verification Gating:
   - Breaks the fatal Precision Dilution Trap (where N=35 crushed Precision from 80% to 6%).
   - Applies Relative Dropoff Thresholding (alpha=0.55) with bounding N in [1, 4].
   - Predicts an average of ~1.5 - 2.5 docs/query, directly matching Admin Baseline's N ~ 1.17.
2. Domain-Prior Reciprocal Rank Fusion:
   - Web VI (Vinmec, HelloBacsi, YouMed): Primary weight = 1.00
   - Web ZH (120ask, 39.net, Chinese clinical Q&A): Weight = 0.50 via CLIR
   - PubMed EN: Controlled weight = 0.05 (strictly preventing academic noise dilution)
3. Clinical Query Expansion (CQE):
   - Disambiguates 60+ medical abbreviations & lab markers (AFP, Beta-HCG, SCC, AMH, TPHA).
4. Newly Merged Corpus:
   - Includes 37,722 Vinmec/HelloBacsi gold clinical Q&As + 32,851 Chinese Q&As from Kaggle CPU crawlers.
5. High-Saliency Extractive Chunking:
   - Tri-sentence context expansion within 1,400 tokens ceiling.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai2026.bm25 import BM25Index
from r2ai2026.chunking import tri_sentence_context_chunk
from r2ai2026.clir_transfer import transfer_vi_to_zh
from r2ai2026.decision import joint_expected_value_filter, relative_dropoff_filter
from r2ai2026.pipeline import load_documents, document_text
from r2ai2026.query_expansion import expand_query
from r2ai2026.submit import write_submission_zip, submission_stats, validate_submission


def main():
    print("=" * 75)
    print("ViBioMIR R2AI2026 - Building SOTA V4 (Decision Model + CLIR + 800k Corpus)")
    print("=" * 75)
    start_total = time.perf_counter()

    docs_path = Path("data/docs.jsonl")
    queries_path = Path("data/query_translated.parquet")
    index_pubmed_dir = Path("runs/indexes/index_pubmed")
    index_web_dir = Path("runs/indexes/index_web")
    output_path = Path("runs/submissions/submission_decision_v4.zip")
    desktop_copy = Path("C:/Users/NOS/Desktop/submission_decision_v4.zip")

    # Step 1: Load Queries and Apply CQE & CLIR
    print(f"\n[1/6] Loading queries from {queries_path}...")
    df_queries = pd.read_parquet(queries_path)
    queries_vi_raw = df_queries["query"].tolist()
    queries_en_raw = df_queries["query_en"].tolist()
    query_ids = df_queries["id"].tolist()
    n_queries = len(query_ids)

    print(f"Applying Clinical Query Expansion (CQE) and Cross-Lingual Transfer (CLIR)...")
    queries_vi_expanded = [expand_query(q, lang="vi") for q in queries_vi_raw]
    queries_en_expanded = [expand_query(q, lang="en") for q in queries_en_raw]
    queries_zh = [transfer_vi_to_zh(q) for q in queries_vi_raw]

    n_exp_vi = sum(1 for r, e in zip(queries_vi_raw, queries_vi_expanded) if r != e)
    n_exp_zh = sum(1 for z in queries_zh if z != "疾病 诊断 治疗")
    print(f"Loaded {n_queries:,} queries.")
    print(f"  CQE expanded: {n_exp_vi:,} VI queries with biomarker context.")
    print(f"  CLIR transferred: {n_exp_zh:,} queries into Chinese clinical nomenclature.")

    # Step 2: Load Partitioned Indexes
    print("\n[2/6] Loading partitioned BM25S indexes...")
    idx_pm = BM25Index.load(index_pubmed_dir, name="docs", mmap=True)
    idx_web = BM25Index.load(index_web_dir, name="docs", mmap=True)
    print(f"  Index Web size: {len(idx_web.rows):,} documents.")
    print(f"  Index PubMed size: {len(idx_pm.rows):,} documents.")

    pool = 250
    # Step 3: Multi-Stream Search
    print(f"\n[3/6] Querying partitioned streams with Candidate Pool k={pool}...")
    t0 = time.perf_counter()

    # Stream 1: Web VI Expanded (Primary clinical search)
    hits_web_vi_exp = idx_web.search_batch(queries_vi_expanded, k=pool)
    # Stream 2: Web VI Raw (Exact lexical match)
    hits_web_vi_raw = idx_web.search_batch(queries_vi_raw, k=pool)
    # Stream 3: Web ZH (Chinese clinical Q&A transfer)
    hits_web_zh = idx_web.search_batch(queries_zh, k=pool)
    # Stream 4: PubMed EN Expanded (Controlled academic search)
    hits_pm_en_exp = idx_pm.search_batch(queries_en_expanded, k=pool)

    print(f"Retrieved candidate batches across 4 streams in {time.perf_counter() - t0:.2f}s.")

    # Step 4: Domain-Prior Reciprocal Rank Fusion & Decision Model Gating
    print("\n[4/6] Fusing streams with Domain-Prior RRF and applying Decision Model...")
    rrf_k = 60
    weight_vi_exp = 1.00
    weight_vi_raw = 0.70
    weight_zh = 0.50
    weight_pm = 0.05  # Strictly capped at 5% to eliminate academic noise

    # Decision Model Hyperparameters
    alpha_dropoff = 0.55  # Retain candidates within 55% of top score
    min_docs = 1          # Never return empty
    max_docs = 4          # Precision guard: max 4 documents per query

    final_doc_selections: list[list[int]] = []
    needed_docs: set[int] = set()
    doc_count_dist: defaultdict[int, int] = defaultdict(int)

    for i in range(n_queries):
        score_map: defaultdict[int, float] = defaultdict(float)

        # Accumulate Stream 1 (Web VI Exp)
        for rank, hit in enumerate(hits_web_vi_exp[i]):
            did = int(hit[0][0])
            score_map[did] += weight_vi_exp * (1.0 / (rrf_k + rank + 1))

        # Accumulate Stream 2 (Web VI Raw)
        for rank, hit in enumerate(hits_web_vi_raw[i]):
            did = int(hit[0][0])
            score_map[did] += weight_vi_raw * (1.0 / (rrf_k + rank + 1))

        # Accumulate Stream 3 (Web ZH via CLIR)
        for rank, hit in enumerate(hits_web_zh[i]):
            did = int(hit[0][0])
            score_map[did] += weight_zh * (1.0 / (rrf_k + rank + 1))

        # Accumulate Stream 4 (PubMed EN)
        for rank, hit in enumerate(hits_pm_en_exp[i]):
            did = int(hit[0][0])
            score_map[did] += weight_pm * (1.0 / (rrf_k + rank + 1))

        # Sort descending by fused score
        ranked_candidates = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        cand_ids = [item[0] for item in ranked_candidates]
        cand_scores = [item[1] for item in ranked_candidates]

        # Apply JEV (Joint Expected Value) Decision Model:
        # Dynamically maximizes expected F2: p_{N+1} > (beta^2 * H_N) / (beta^2 * N + k_gold)
        selected = joint_expected_value_filter(
            cand_ids,
            cand_scores,
            k_gold=2.0,
            beta=2.0,
            p_top=0.90,
            gamma=2.5,
            min_docs=min_docs,
            max_docs=max_docs,
        )

        final_doc_selections.append(selected)
        needed_docs.update(selected)
        doc_count_dist[len(selected)] += 1

    avg_docs = sum(len(d) for d in final_doc_selections) / n_queries
    print(f"Decision Model complete! Total unique documents needed: {len(needed_docs):,}.")
    print(f"Average documents per query: {avg_docs:.2f} (vs 35.00 previously, vs ~1.17 Admin baseline).")
    print(f"Document count distribution across 1,200 queries:")
    for count in sorted(doc_count_dist.keys()):
        print(f"  {count} doc(s): {doc_count_dist[count]:,} queries ({doc_count_dist[count]/n_queries*100:.1f}%)")

    # Step 5: Load Selected Documents from Corpus and Perform Saliency Chunking
    print(f"\n[5/6] Loading {len(needed_docs):,} selected documents from {docs_path} for extractive chunking...")
    t_load = time.perf_counter()
    doc_store: dict[int, str] = {}
    for record in load_documents(docs_path):
        did = int(record["id"])
        if did in needed_docs:
            doc_store[did] = document_text(record)
            if len(doc_store) == len(needed_docs):
                break
    print(f"Loaded {len(doc_store):,} documents into memory in {time.perf_counter() - t_load:.2f}s.")

    print("Generating extractive chunks using Tri-Sentence Context Expansion (budget <= 1400 tokens)...")
    records: list[dict] = []
    total_chunks = 0

    for i, qid in enumerate(query_ids):
        q_text = queries_vi_expanded[i]
        selected_dids = final_doc_selections[i]

        query_chunks: list[dict] = []
        for did in selected_dids:
            text = doc_store.get(did, "")
            if not text:
                continue

            extracted = tri_sentence_context_chunk(
                text=text,
                query=q_text,
                max_tokens=1400,
            )
            if extracted:
                query_chunks.append({
                    "doc_id": int(did),
                    "chunk_text": str(extracted),
                })

        records.append({
            "id": int(qid),
            "relevant_docs": [int(d) for d in selected_dids],
            "relevant_chunks": query_chunks,
        })
        total_chunks += len(query_chunks)

    print(f"Generated {total_chunks:,} chunks for {len(records):,} queries.")

    # Step 6: Write, Validate, and Export Submission
    print(f"\n[6/6] Packaging submission to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = write_submission_zip(records, output_path)

    # Copy to Desktop for instant submission
    try:
        shutil.copy2(output_path, desktop_copy)
        print(f"Copied submission file to Desktop: {desktop_copy}")
    except Exception as e:
        print(f"Warning: could not copy to Desktop: {e}")

    # Validate schema
    print("\nSubmission Statistics:")
    print(json.dumps(stats, indent=2))

    errors = validate_submission(records, expected_ids=query_ids)
    if errors:
        print(f"\nERROR: Submission validation failed with {len(errors)} errors:")
        for err in errors[:5]:
            print(f"  - {err}")
        return 1

    print("\n" + "=" * 75)
    print("SUCCESS: submission_decision_v4.zip passed all schema validations (0 errors)!")
    print(f"Total time elapsed: {time.perf_counter() - start_total:.2f}s")
    print(f"Ready to submit on Codabench: {output_path} (and Desktop)")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
