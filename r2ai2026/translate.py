"""Cross-lingual query translation and expansion for ViBioMIR.

Translates Vietnamese clinical queries to English using the local MarianMT model
(Helsinki-NLP/opus-mt-vi-en) so BM25 and dense retrievers can effectively search
the 240,000 English PubMed abstracts in the corpus.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch


def translate_queries_vi_to_en(
    queries: Sequence[str],
    *,
    batch_size: int = 32,
    device: str | None = None,
    max_length: int = 128,
    show_progress: bool = True,
) -> list[str]:
    """Translate a list of Vietnamese queries to English using MarianMT."""
    from transformers import MarianMTModel, MarianTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "Helsinki-NLP/opus-mt-vi-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name, local_files_only=True)
    model = MarianMTModel.from_pretrained(model_name, local_files_only=True)
    if device == "cuda":
        model = model.half()
    model = model.to(device)
    model.eval()

    results: list[str] = []
    total = len(queries)
    iterator = range(0, total, batch_size)
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="translating vi->en", total=(total + batch_size - 1) // batch_size)

    with torch.inference_mode():
        for start_idx in iterator:
            batch = list(queries[start_idx : start_idx + batch_size])
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            generated = model.generate(**inputs, max_length=max_length)
            translations = [
                tokenizer.decode(t, skip_special_tokens=True).strip()
                for t in generated
            ]
            results.extend(translations)

    return results


def translate_query_file(
    input_path: Path | str,
    output_path: Path | str,
    *,
    batch_size: int = 32,
    device: str | None = None,
) -> pd.DataFrame:
    """Read query.parquet, translate all queries to English, and save with query_en."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} queries from {input_path}")

    queries_vi = df["query"].tolist()
    queries_en = translate_queries_vi_to_en(queries_vi, batch_size=batch_size, device=device)

    df["query_en"] = queries_en
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved translated queries to {output_path}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate queries VI -> EN")
    parser.add_argument("--queries", type=Path, default=Path("data/raw/btc/query.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/query_translated.parquet"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    translate_query_file(args.queries, args.out, batch_size=args.batch_size)
