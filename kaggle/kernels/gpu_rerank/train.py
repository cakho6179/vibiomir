"""Kaggle GPU Re-ranking & Query-Biased Chunking Pipeline for ViBioMIR.

Runs on Kaggle GPU (Tesla T4 16GB / 2x T4).
Re-ranks top candidate documents with BAAI/bge-m3 or BAAI/bge-reranker-v2-m3,
applies Query-Biased Paragraph Chunking to maximize token-LCS overlap >= 0.40,
and outputs valid submission.zip.
"""

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# Ensure sentence-transformers is available
try:
    import sentence_transformers
except ImportError:
    print("Installing sentence-transformers...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=True)

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def word_count(text: str) -> int:
    return len(text.split())


def extract_query_biased_chunk(
    doc_text: str,
    query_text: str,
    max_tokens: int = 1400,
) -> str:
    """Query-biased extractive chunking.

    Preserves document title and opening context, then selects the most
    information-dense paragraphs relevant to the query to maximize LCS overlap.
    """
    doc_text = (doc_text or "").strip()
    if not doc_text:
        return ""
    if word_count(doc_text) <= max_tokens:
        return doc_text

    # Split into paragraphs (separated by double newlines or single newlines)
    raw_paras = [p.strip() for p in doc_text.split("\n") if p.strip()]
    if not raw_paras:
        return doc_text[:max_tokens * 6]

    # Lead paragraph (title + intro) is always kept
    lead = raw_paras[0]
    lead_tokens = word_count(lead)

    if len(raw_paras) == 1 or lead_tokens >= max_tokens:
        return " ".join(lead.split()[:max_tokens])

    remaining_budget = max_tokens - lead_tokens
    body_paras = raw_paras[1:]

    # Compute lexical term overlap with query for body paragraphs
    query_terms = set(query_text.lower().split())
    scored_paras = []
    for idx, p in enumerate(body_paras):
        p_terms = set(p.lower().split())
        overlap = len(query_terms.intersection(p_terms))
        wcnt = word_count(p)
        score = overlap / (wcnt ** 0.5 + 1.0)
        scored_paras.append((score, idx, p, wcnt))

    # Sort body paragraphs by relevance to query
    scored_paras.sort(key=lambda x: x[0], reverse=True)

    selected_indices = {0}  # lead is index 0 in raw_paras
    used_tokens = lead_tokens

    for score, b_idx, p, wcnt in scored_paras:
        if used_tokens + wcnt <= remaining_budget:
            selected_indices.add(b_idx + 1)
            used_tokens += wcnt
        elif used_tokens < max_tokens:
            # Add truncated piece
            avail = max_tokens - used_tokens
            if avail > 30:
                selected_indices.add(b_idx + 1)
                used_tokens += avail
            break

    # Reconstruct in original document order
    ordered_pieces = [raw_paras[i] for i in sorted(selected_indices)]
    result = "\n".join(ordered_pieces)
    words = result.split()
    if len(words) > max_tokens:
        result = " ".join(words[:max_tokens])
    return result


def main():
    print("=" * 60)
    print("Kaggle GPU ViBioMIR Re-ranking & Extractive Chunking")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")

    # Find dataset directory
    input_base = Path("/kaggle/input")
    cand_path = None
    queries_path = None

    for p in input_base.rglob("candidates.parquet"):
        cand_path = p
        break
    for p in input_base.rglob("queries.parquet"):
        queries_path = p
        break

    if not cand_path or not queries_path:
        raise FileNotFoundError(f"Could not find candidates.parquet or queries.parquet in {input_base}")

    print(f"Loading queries from {queries_path}...")
    df_queries = pd.read_parquet(queries_path)
    print(f"Loaded {len(df_queries):,} queries.")

    print(f"Loading candidate documents from {cand_path}...")
    df_cands = pd.read_parquet(cand_path)
    print(f"Loaded {len(df_cands):,} candidate documents.")

    doc_texts = dict(zip(df_cands["doc_id"].astype(int), df_cands["text"]))
    # Lead text (first 300 words) for dense embedding
    doc_leads = {
        did: " ".join(txt.split()[:300]) if len(txt.split()) > 300 else txt
        for did, txt in doc_texts.items()
    }
    del df_cands

    # Load BGE-M3 model on GPU
    print("Loading BAAI/bge-m3 model...")
    started = time.perf_counter()
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    if device == "cuda":
        model = model.half()
    print(f"Model loaded in {time.perf_counter() - started:.2f}s")

    # Encode queries
    print("Encoding queries...")
    queries_vi = df_queries["query"].tolist()
    queries_en = df_queries["query_en"].tolist()
    query_ids = df_queries["id"].astype(int).tolist()
    candidate_id_lists = df_queries["candidate_ids"].tolist()

    q_emb_vi = model.encode(queries_vi, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    q_emb_en = model.encode(queries_en, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    # Encode unique candidate documents
    all_needed_doc_ids = sorted(doc_leads.keys())
    doc_to_idx = {did: i for i, did in enumerate(all_needed_doc_ids)}
    unique_texts = [doc_leads[did] for did in all_needed_doc_ids]

    print(f"Encoding {len(unique_texts):,} candidate documents on GPU...")
    started = time.perf_counter()
    d_embs = model.encode(unique_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    print(f"Encoded candidates in {time.perf_counter() - started:.2f}s")

    # Re-rank candidates per query
    print("Scoring and re-ranking...")
    records = []
    top_k_docs = 35
    chunk_budget = 1400

    for idx, qid in enumerate(query_ids):
        cands = candidate_id_lists[idx]
        valid_cands = [d for d in cands if d in doc_to_idx]
        if not valid_cands:
            records.append({"id": qid, "relevant_docs": [], "relevant_chunks": []})
            continue

        cand_indices = [doc_to_idx[d] for d in valid_cands]
        sub_d_embs = d_embs[cand_indices]

        # Semantic similarity against VI and EN query
        sim_vi = np.dot(sub_d_embs, q_emb_vi[idx])
        sim_en = np.dot(sub_d_embs, q_emb_en[idx])
        dense_sim = np.maximum(sim_vi, sim_en)

        # BM25 rank score
        bm25_score = np.array([1.0 / (60 + r) for r in range(1, len(valid_cands) + 1)])

        # Normalize
        d_min, d_max = float(dense_sim.min()), float(dense_sim.max())
        norm_dense = (dense_sim - d_min) / (d_max - d_min + 1e-8)

        b_min, b_max = float(bm25_score.min()), float(bm25_score.max())
        norm_bm25 = (bm25_score - b_min) / (b_max - b_min + 1e-8)

        # Hybrid score (alpha=0.6)
        hybrid_scores = 0.4 * norm_bm25 + 0.6 * norm_dense
        sorted_order = np.argsort(-hybrid_scores)
        best_doc_ids = [valid_cands[i] for i in sorted_order[:top_k_docs]]

        # Query-biased chunk extraction
        chunks = []
        q_text = queries_vi[idx]
        for did in best_doc_ids:
            full_txt = doc_texts.get(did, "")
            chunk_txt = extract_query_biased_chunk(full_txt, q_text, max_tokens=chunk_budget)
            chunks.append({"doc_id": int(did), "chunk_text": chunk_txt})

        records.append({
            "id": int(qid),
            "relevant_docs": [int(d) for d in best_doc_ids],
            "relevant_chunks": chunks,
        })

    # Output to zip
    out_dir = Path("/kaggle/working")
    out_zip = out_dir / "submission.zip"
    print(f"Writing {out_zip}...")
    payload = json.dumps(records, ensure_ascii=False).encode("utf-8")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("submission.json", payload)

    zip_size_mb = out_zip.stat().st_size / 1e6
    print(f"Successfully generated {out_zip} ({zip_size_mb:.2f} MB)")
    print(f"Total queries: {len(records)}, Docs/query: {top_k_docs}")
    print("DONE!")


if __name__ == "__main__":
    main()
