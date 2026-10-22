"""Runtime sanity check for the scoring replica at competition scale.

Builds a synthetic submission (1,200 queries) with realistic chunk sizes and
measures the time to score it, plus the marginal cost of very long chunks
(the "whole document as one chunk" strategy).
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai2026.scoring import score_submission  # noqa: E402

rng = random.Random(7)
VOCAB = [f"t{i}" for i in range(30000)]


def make_text(tokens: int) -> str:
    return " ".join(rng.choices(VOCAB, k=tokens))


def build(queries: int, docs_per_query: int, chunks_per_doc: int, chunk_tokens: int, gold_tokens: int):
    references, predictions = {}, {}
    for qid in range(1, queries + 1):
        gold_chunks, pred_chunks, gold_docs = [], [], []
        for doc_index in range(docs_per_query):
            doc_id = qid * 100 + doc_index
            gold_docs.append(doc_id)
            gold_chunks.append({"doc_id": doc_id, "chunk_text": make_text(gold_tokens)})
            for _ in range(chunks_per_doc):
                pred_chunks.append({"doc_id": doc_id, "chunk_text": make_text(chunk_tokens)})
        references[qid] = {"relevant_docs": gold_docs, "relevant_chunks": gold_chunks}
        predictions[qid] = {"relevant_docs": gold_docs, "relevant_chunks": pred_chunks}
    return predictions, references


def run(label: str, **kwargs) -> None:
    predictions, references = build(**kwargs)
    chunks = sum(len(p["relevant_chunks"]) for p in predictions.values())
    started = time.perf_counter()
    score = score_submission(predictions, references)
    elapsed = time.perf_counter() - started
    print(
        f"{label:<46} chunks={chunks:>6}  {elapsed:6.1f}s  "
        f"chunks_F2={score.chunks_f2:.3f} docs_F2={score.docs_f2:.3f}"
    )


if __name__ == "__main__":
    run("1200q x 5doc x 3chunk @400t vs gold 300t", queries=1200, docs_per_query=5, chunks_per_doc=3, chunk_tokens=400, gold_tokens=300)
    run("1200q x 5doc x 1chunk @20000t (whole doc)", queries=1200, docs_per_query=5, chunks_per_doc=1, chunk_tokens=20000, gold_tokens=300)
