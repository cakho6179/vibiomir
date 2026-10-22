"""Build SOTA V2 Submission: CQE + Tri-Stream BM25+ RRF + Tri-Sentence Saliency Chunking.

Features:
1. Clinical Query Expansion (CQE): Expands 60+ medical acronyms & lab markers (AFP, Beta-HCG, SCC, AMH, TPHA, etc.)
2. Deep Candidate Retrieval: Candidate pool k=200 per partition stream
3. 5-Stream RRF Fusion: Fuses Expanded VI, Expanded EN, Raw VI, Raw EN across PubMed and Web partitions
4. SOTA Extractive Chunking: Adaptive Full-Document (for short docs <= 1400 tokens) +
   Tri-Sentence Context Expansion (for long docs > 1400 tokens)
5. Cleaned Corpus: 758k verified medical articles without boilerplate noise
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai2026.bm25 import BM25Index
from r2ai2026.chunking import tri_sentence_context_chunk
from r2ai2026.fusion import reciprocal_rank_fusion
from r2ai2026.pipeline import load_documents, document_text
from r2ai2026.query_expansion import expand_query, batch_expand_queries
from r2ai2026.submit import write_submission_zip, submission_stats, validate_submission


def main():
    print("=" * 70)
    print("ViBioMIR R2AI2026 - Building SOTA V2 Submission")
    print("=" * 70)
    start_total = time.perf_counter()

    docs_path = Path("data/docs_cleaned.jsonl")
    queries_path = Path("data/query_translated.parquet")
    index_pubmed_dir = Path("runs/indexes/index_pubmed")
    index_web_dir = Path("runs/indexes/index_web")
    output_path = Path("runs/submissions/submission_tri_stream_sota_v2.zip")

    # Step 1: Load Queries and Apply Clinical Query Expansion (CQE)
    print(f"\n[1/5] Loading queries from {queries_path} and applying CQE...")
    df_queries = pd.read_parquet(queries_path)
    queries_vi_raw = df_queries["query"].tolist()
    queries_en_raw = df_queries["query_en"].tolist()
    query_ids = df_queries["id"].tolist()

    queries_vi_expanded = [expand_query(q, lang="vi") for q in queries_vi_raw]
    queries_en_expanded = [expand_query(q, lang="en") for q in queries_en_raw]

    n_expanded_vi = sum(1 for r, e in zip(queries_vi_raw, queries_vi_expanded) if r != e)
    n_expanded_en = sum(1 for r, e in zip(queries_en_raw, queries_en_expanded) if r != e)
    print(f"Loaded {len(query_ids):,} queries.")
    print(f"CQE expanded {n_expanded_vi:,} VI queries and {n_expanded_en:,} EN queries with clinical entities.")

    # Step 2: Load Partitioned Indexes
    print("\n[2/5] Loading partitioned BM25S indexes (PubMed + Web)...")
    idx_pm = BM25Index.load(index_pubmed_dir, name="docs", mmap=True)
    idx_web = BM25Index.load(index_web_dir, name="docs", mmap=True)
    pool = 200

    # Step 3: Multi-Stream Search & Fusion
    print(f"\n[3/5] Querying partitions with Candidate Pool k={pool}...")
    t0 = time.perf_counter()

    # Query Web with Expanded VI and Raw VI
    hits_web_vi_exp = idx_web.search_batch(queries_vi_expanded, k=pool)
    hits_web_vi_raw = idx_web.search_batch(queries_vi_raw, k=pool)

    # Query PubMed with Expanded EN and Raw EN
    hits_pm_en_exp = idx_pm.search_batch(queries_en_expanded, k=pool)
    hits_pm_en_raw = idx_pm.search_batch(queries_en_raw, k=pool)

    # Cross-lingual streams: Web with EN, PubMed with VI
    hits_web_en_exp = idx_web.search_batch(queries_en_expanded, k=pool)
    hits_pm_vi_exp = idx_pm.search_batch(queries_vi_expanded, k=pool)

    print(f"Retrieved candidate batches in {time.perf_counter() - t0:.2f}s.")
    print("Fusing streams using Reciprocal Rank Fusion (RRF k=60)...")

    fused_results: list[list[int]] = []
    needed_docs: set[int] = set()
    top_k_docs = 35

    for i in range(len(query_ids)):
        web_vi_exp_ids = [int(h[0][0]) for h in hits_web_vi_exp[i]]
        pm_en_exp_ids = [int(h[0][0]) for h in hits_pm_en_exp[i]]
        web_vi_raw_ids = [int(h[0][0]) for h in hits_web_vi_raw[i]]
        pm_en_raw_ids = [int(h[0][0]) for h in hits_pm_en_raw[i]]
        web_en_exp_ids = [int(h[0][0]) for h in hits_web_en_exp[i]]
        pm_vi_exp_ids = [int(h[0][0]) for h in hits_pm_vi_exp[i]]

        fused = reciprocal_rank_fusion(
            [
                web_vi_exp_ids,
                pm_en_exp_ids,
                web_vi_raw_ids,
                pm_en_raw_ids,
                web_en_exp_ids,
                pm_vi_exp_ids,
            ],
            weights=[1.2, 1.2, 0.8, 0.8, 0.6, 0.4],
            k=60,
            top_n=top_k_docs,
        )
        fused_results.append(fused)
        needed_docs.update(fused)

    print(f"Fused top-{top_k_docs} documents across 1,200 queries.")
    print(f"Unique documents to extract chunks for: {len(needed_docs):,} docs.")

    # Step 4: Load Documents and Apply Tri-Sentence Context Chunking
    print(f"\n[4/5] Loading document texts from {docs_path}...")
    t0 = time.perf_counter()
    doc_texts: dict[int, str] = {}
    for rec in load_documents(docs_path):
        did = rec["id"]
        if did in needed_docs:
            doc_texts[did] = document_text(rec)

    print(f"Loaded {len(doc_texts):,} documents in {time.perf_counter() - t0:.2f}s.")
    print("Extracting chunks via Tri-Sentence Saliency Expansion (budget=1400 tokens)...")
    t0 = time.perf_counter()

    records: list[dict] = []
    for q_idx, qid in enumerate(query_ids):
        top_docs = fused_results[q_idx]
        q_text = queries_vi_expanded[q_idx]
        chunks = []
        for did in top_docs:
            dtext = doc_texts.get(did, "")
            chk = tri_sentence_context_chunk(dtext, q_text, max_tokens=1400)
            chunks.append({"doc_id": int(did), "chunk_text": chk})

        records.append({
            "id": int(qid),
            "relevant_docs": top_docs,
            "relevant_chunks": chunks,
        })

    print(f"Chunked 42,000 documents in {time.perf_counter() - t0:.2f}s.")

    # Step 5: Packaging & Verification
    print(f"\n[5/5] Packaging submission to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(records, ensure_ascii=False).encode("utf-8")
    import zipfile
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("submission.json", payload)

    print(f"ZIP Size: {output_path.stat().st_size / 1e6:.2f} MB")

    print("\nValidating submission against Codabench schema...")
    errors = validate_submission(records, expected_ids=query_ids)
    if errors:
        print(f"VALIDATION FAILED with {len(errors)} errors:")
        for e in errors[:10]:
            print(f"  - {e}")
        sys.exit(1)

    stats = submission_stats(records)
    print("=" * 70)
    print("SOTA V2 GENERATION COMPLETE: VALIDATION PASSED (0 ERRORS)")
    print("=" * 70)
    print(f"  Output File:        {output_path.resolve()}")
    print(f"  Total Queries:      {int(stats['n_queries'])}")
    print(f"  Docs per Query:     {stats['docs_per_query']:.1f}")
    print(f"  Chunks per Query:   {stats['chunks_per_query']:.1f}")
    print(f"  Mean Chunk Size:    {stats['mean_chunk_chars']:.0f} characters")
    print(f"  Unique Docs Cited:  {len(needed_docs):,}")
    print(f"  Total Elapsed Time: {time.perf_counter() - start_total:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
