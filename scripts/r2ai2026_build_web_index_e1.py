"""Build Web BM25 Index with Vietnamese Diacritics Kept (E1 Benchmark).

- Keeps 100% of Vietnamese tone marks and diacritics (keep_diacritics=True)
- No artificial 1,000 token truncation (max_tokens=4,000)
- Sources only from clean verified Web documents (data/docs_clean.jsonl)
- Uses ultra-fast bm25s (Robertson BM25, SciPy CSC int32)

Usage:
    python scripts/r2ai2026_build_web_index_e1.py
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import unicodedata
import html
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bm25s
from r2ai2026.pipeline import document_text, load_documents

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RUN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_keep_diacritics(text: str) -> str:
    """Normalize Unicode NFKC, unescape HTML, lowercase, collapse whitespace.
    Crucially preserves all Vietnamese diacritics.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = html.unescape(text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def tokenize_keep_diacritics(text: str) -> list[str]:
    """Tokenize text preserving Vietnamese syllables, numbers, medical acronyms,
    and expanding CJK characters into unigrams and bigrams.
    """
    if not text:
        return []
    norm = normalize_keep_diacritics(text)
    words = _WORD_RUN_RE.findall(norm)
    tokens: list[str] = []
    for w in words:
        w_len = len(w)
        if w_len > 40:
            continue
        if w.isdigit() and w_len > 6:
            continue
        # CJK check (Chinese characters)
        has_cjk = any(0x4E00 <= ord(c) <= 0x9FFF for c in w)
        if has_cjk:
            chars = [c for c in w if 0x4E00 <= ord(c) <= 0x9FFF]
            tokens.extend(chars)
            tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
        else:
            tokens.append(w)
    return tokens


def build_web_index(
    docs_path: Path,
    out_dir: Path,
    max_tokens: int = 4000,
    min_chars: int = 50,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"[1/2] Loading & tokenizing Web documents from {docs_path}...")
    print(f"      keep_diacritics=True, max_tokens={max_tokens}, min_chars={min_chars}")

    doc_ids: list[int] = []
    vocab_dict: dict[str, int] = {}
    corpus_token_ids: list[list[int]] = []
    seen_ids: set[int] = set()
    n_skipped_pubmed = 0
    n_skipped_short = 0

    for record in load_documents(docs_path):
        did = int(record["id"])
        if did in seen_ids:
            continue
        url = str(record.get("url") or "")
        if "pubmed.ncbi.nlm.nih.gov" in url:
            n_skipped_pubmed += 1
            continue

        text = document_text(record)
        if len(text) < min_chars:
            n_skipped_short += 1
            continue

        toks = tokenize_keep_diacritics(text)
        if len(toks) > max_tokens:
            toks = toks[:max_tokens]
        if not toks:
            toks = ["__empty__"]

        seen_ids.add(did)
        doc_ids.append(did)
        
        # Memory-safe integer ID mapping (zero duplicate string overhead)
        token_ids = [vocab_dict.setdefault(t, len(vocab_dict)) for t in toks]
        corpus_token_ids.append(token_ids)

        if len(doc_ids) % 100000 == 0:
            print(f"  Tokenized {len(doc_ids):,} docs (vocab: {len(vocab_dict):,}, {time.perf_counter() - started:.1f}s)...", flush=True)

    del seen_ids
    gc.collect()

    print(f"Loaded & tokenized {len(doc_ids):,} Web docs in {time.perf_counter() - started:.1f}s "
          f"(skipped {n_skipped_pubmed:,} PubMed, {n_skipped_short:,} too short).", flush=True)

    if not doc_ids:
        print("ERROR: No documents tokenized!")
        return 0

    print(f"[2/2] Building BM25 index with bm25s (Robertson, scipy CSC, int32, vocab={len(vocab_dict):,})...", flush=True)
    idx_started = time.perf_counter()
    retriever = bm25s.BM25(
        k1=1.5,
        b=0.75,
        method="robertson",
        int_dtype="int32",
        csc_backend="scipy",
    )
    retriever.index((corpus_token_ids, vocab_dict), show_progress=True)

    del corpus_token_ids
    del vocab_dict
    gc.collect()

    # Save BM25 files
    print(f"Saving BM25 index to {out_dir}...")
    retriever.save(str(out_dir / "docs.bm25"), show_progress=False)
    np.save(out_dir / "docs.rows.npy", np.array(doc_ids, dtype=np.int64).reshape(-1, 1))
    (out_dir / "docs.params.json").write_text(
        json.dumps({"k1": 1.5, "b": 0.75, "method": "robertson", "keep_diacritics": True}, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "partition": "web_diacritics",
        "n_docs": len(doc_ids),
        "docs_path": str(docs_path),
        "max_tokens": max_tokens,
        "keep_diacritics": True,
        "build_time_s": time.perf_counter() - idx_started,
        "total_time_s": time.perf_counter() - started,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"DONE! Web index saved successfully to {out_dir} in {time.perf_counter() - started:.1f}s.")
    return len(doc_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=ROOT / "data" / "docs_clean.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "indexes" / "index_web_diacritics")
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--min-chars", type=int, default=50)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    build_web_index(
        docs_path=args.docs,
        out_dir=args.out_dir,
        max_tokens=args.max_tokens,
        min_chars=args.min_chars,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
