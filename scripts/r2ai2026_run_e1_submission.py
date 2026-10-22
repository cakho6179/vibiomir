"""Run E1 Benchmark Submission: BM25 on Clean Web Corpus with Kept Diacritics.

Strategy Specifications:
- Model: Pure BM25 (reproducing Admin BM25 baseline)
- Tokenizer: Syllable tokenizer preserving 100% Vietnamese tone marks & diacritics
- Scope: 1,200 Vietnamese clinical patient questions on clean Web documents
- N: Exactly 28 documents per query (Admin BM25 baseline parameter)
- Chunks: 1 chunk = whole document verbatim (up to 5,000 characters)
- Evaluation Goal: Verify if Admin BM25 gold candidate pool hypothesis is confirmed (P >= 0.60)

Usage:
    python scripts/r2ai2026_run_e1_submission.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bm25s
from r2ai2026.pipeline import document_text, load_documents
from scripts.r2ai2026_build_web_index_e1 import tokenize_keep_diacritics
from r2ai2026.submit import build_submission, submission_stats, validate_submission, write_submission_zip


def run_e1(
    index_dir: Path,
    docs_clean_path: Path,
    query_path: Path,
    top_k: int = 28,
    max_chunk_chars: int = 5000,
) -> Path:
    started = time.perf_counter()
    print("=" * 70)
    print("RUNNING E1 SUBMISSION PIPELINE (BM25 + Kept Diacritics + N=28 + Fulltext)")
    print("=" * 70)

    # 1. Load Queries
    print(f"\n[1/5] Loading queries from {query_path}...")
    df_q = pd.read_parquet(query_path)
    text_col = "query" if "query" in df_q.columns else df_q.columns[-1]
    queries = [{"id": int(r.id), "text": str(getattr(r, text_col))} for r in df_q.itertuples(index=False)]
    print(f"Loaded {len(queries):,} queries.")

    # 2. Load BM25 Retriever
    print(f"\n[2/5] Loading Web BM25 index from {index_dir}...")
    retriever = bm25s.BM25.load(str(index_dir / "docs.bm25"), load_corpus=False)
    doc_ids_arr = np.load(index_dir / "docs.rows.npy").flatten()
    print(f"Index loaded successfully: {len(doc_ids_arr):,} Web documents indexed.")

    # 3. Tokenize queries & retrieve top K
    print(f"\n[3/5] Tokenizing queries with keep_diacritics=True and retrieving top {top_k}...")
    tokenized_queries: list[list[str]] = []
    for q in queries:
        toks = tokenize_keep_diacritics(q["text"])
        if not toks:
            toks = ["__empty__"]
        tokenized_queries.append(toks)

    # Batch retrieval with bm25s
    retrieve_start = time.perf_counter()
    results, scores = retriever.retrieve(tokenized_queries, k=top_k, show_progress=True)
    print(f"Retrieval finished in {time.perf_counter() - retrieve_start:.2f}s.")

    # Collect retrieved doc IDs
    query_predictions: dict[int, list[int]] = {}
    needed_doc_ids: set[int] = set()

    for idx, q in enumerate(queries):
        qid = q["id"]
        # results[idx] contains the row indices into doc_ids_arr
        retrieved_rows = results[idx]
        retrieved_dids = [int(doc_ids_arr[row_idx]) for row_idx in retrieved_rows]
        query_predictions[qid] = retrieved_dids
        needed_doc_ids.update(retrieved_dids)

    print(f"Total unique documents needed for chunks: {len(needed_doc_ids):,}")

    # 4. Stream docs_clean.jsonl to extract whole-document texts
    print(f"\n[4/5] Extracting whole-document texts for {len(needed_doc_ids):,} retrieved documents from {docs_clean_path}...")
    doc_texts: dict[int, str] = {}
    found_count = 0

    for rec in load_documents(docs_clean_path):
        did = int(rec["id"])
        if did in needed_doc_ids:
            txt = document_text(rec)
            # Cap at max_chunk_chars to prevent ballooning zip size while preserving verbatim LCS >= 0.40
            if len(txt) > max_chunk_chars:
                txt = txt[:max_chunk_chars]
            doc_texts[did] = txt
            found_count += 1
            if found_count == len(needed_doc_ids):
                break

    print(f"Found texts for {len(doc_texts):,} / {len(needed_doc_ids):,} required documents.")

    # 5. Build official submission structure
    print("\n[5/5] Building submission package...")
    records: list[dict] = []
    missing_texts = 0

    for q in queries:
        qid = q["id"]
        dids = query_predictions[qid]
        chunks: list[dict] = []
        valid_dids: list[int] = []

        for did in dids:
            valid_dids.append(did)
            text = doc_texts.get(did, "")
            if text:
                chunks.append({"doc_id": did, "chunk_text": text})
            else:
                missing_texts += 1

        records.append({
            "id": qid,
            "relevant_docs": valid_dids,
            "relevant_chunks": chunks,
        })

    if missing_texts > 0:
        print(f"Warning: {missing_texts} chunks had empty text!")

    # Validate
    expected_ids = {q["id"] for q in queries}
    problems = validate_submission(records, expected_ids=expected_ids)
    if problems:
        print("ERROR: Submission validation failed:")
        for p in problems[:10]:
            print(" ", p)
        raise RuntimeError("Submission validation failed!")
    print("Validation passed 100%! All 1,200 queries are present with correct schemas.")

    stats = submission_stats(records)
    print("\nSubmission Statistics:")
    print(f"  Queries:          {int(stats['n_queries'])}")
    print(f"  Total Docs:       {int(stats['n_docs']):,} ({stats['docs_per_query']:.2f} docs/query)")
    print(f"  Total Chunks:     {int(stats['n_chunks']):,} ({stats['chunks_per_query']:.2f} chunks/query)")
    print(f"  Mean Chunk Chars: {stats['mean_chunk_chars']:.1f} characters")
    print(f"  Payload Size:     {stats['payload_mb']:.2f} MB")

    # Save to runs/submissions and Desktop
    out_dir = ROOT / "runs" / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "submission_E1_bm25_fulltext.zip"
    write_submission_zip(records, zip_path)
    print(f"\nSaved submission to: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")

    # Copy to Desktop
    desktop = Path("C:/Users/NOS/Desktop")
    if desktop.exists():
        desktop_zip = desktop / "submission_E1_bm25_fulltext.zip"
        shutil.copyfile(zip_path, desktop_zip)
        print(f"COPIED TO DESKTOP: {desktop_zip}")

    print(f"Pipeline completed in {time.perf_counter() - started:.1f}s.")
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "runs" / "indexes" / "index_web_diacritics")
    parser.add_argument("--docs", type=Path, default=ROOT / "data" / "docs_clean.jsonl")
    parser.add_argument("--queries", type=Path, default=ROOT / "data" / "raw" / "btc" / "query.parquet")
    parser.add_argument("--top-k", type=int, default=28)
    parser.add_argument("--max-chunk-chars", type=int, default=5000)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    run_e1(
        index_dir=args.index_dir,
        docs_clean_path=args.docs,
        query_path=args.queries,
        top_k=args.top_k,
        max_chunk_chars=args.max_chunk_chars,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
