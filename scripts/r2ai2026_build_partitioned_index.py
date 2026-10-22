"""Build partitioned BM25 indexes (PubMed English vs. Web VI/ZH) for ViBioMIR.

Splitting into two clean partitions avoids memory exhaustion on 500k+ docs,
allows specialized tokenization, and matches our dual-language query strategy:
- index_pubmed: ~240k PubMed English abstracts (searched with English translated queries)
- index_web: ~271k Vietnamese/Chinese web portal articles (searched with Vietnamese queries)

Usage:
    python scripts/r2ai2026_build_partitioned_index.py --docs data/docs.jsonl \
        --partition pubmed --out-dir runs/indexes/index_pubmed

    python scripts/r2ai2026_build_partitioned_index.py --docs data/docs.jsonl \
        --partition web --out-dir runs/indexes/index_web
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bm25s
from r2ai2026.bm25 import BM25Params
from r2ai2026.pipeline import document_text, load_documents, unslug_url
from r2ai2026.text import normalize, strip_diacritics, _CJK_RANGES

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def fast_tokenize(text: str, is_pubmed: bool = False) -> list[str]:
    """Memory-safe, high-speed tokenization with token length filtering."""
    if not text:
        return []
    if is_pubmed:
        # PubMed is pure English: lowercase and fast regex
        tokens = [t for t in _WORD_RE.findall(text.lower()) if 2 <= len(t) <= 40 and not (t.isdigit() and len(t) > 6)]
        return tokens

    # Web (Vietnamese / Chinese):
    norm = normalize(text)
    # Strip diacritics on entire text once
    stripped = strip_diacritics(norm)
    words = _WORD_RE.findall(stripped)
    tokens: list[str] = []
    for w in words:
        if len(w) < 2 or len(w) > 40:
            continue
        if w.isdigit() and len(w) > 6:
            continue
        # Check CJK
        has_cjk = any(0x4E00 <= ord(c) <= 0x9FFF for c in w)
        if has_cjk:
            chars = [c for c in w if 0x4E00 <= ord(c) <= 0x9FFF]
            tokens.extend(chars)
            tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
        else:
            tokens.append(w)
    return tokens


def build_partition(
    docs_path: Path,
    out_dir: Path,
    partition: str,
    min_chars: int = 30,
    max_tokens: int = 500,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    is_pubmed_target = (partition == "pubmed")

    print(f"\n[1/2] Loading and streaming tokenization for partition: '{partition}' (is_pubmed={is_pubmed_target}, max_tokens={max_tokens})...")
    started = time.perf_counter()
    doc_ids: list[int] = []
    tokenized: list[list[str]] = []

    seen_ids: set[int] = set()
    for record in load_documents(docs_path):
        if not record.get("ok", True):
            continue
        did = int(record["id"])
        if did in seen_ids:
            continue
        url = str(record.get("url") or "")
        is_pm = "pubmed.ncbi.nlm.nih.gov" in url
        if is_pm != is_pubmed_target:
            continue
        text = document_text(record)
        if len(text) < min_chars:
            continue
        slug = unslug_url(url)
        if slug and slug not in text.lower():
            text = f"{slug}\n{text}"
        toks = fast_tokenize(text, is_pubmed=is_pubmed_target)
        if len(toks) > max_tokens:
            toks = toks[:max_tokens]
        if not toks:
            toks = ["__empty__"]
        seen_ids.add(did)
        doc_ids.append(did)
        tokenized.append(toks)
        if len(doc_ids) % 50000 == 0:
            print(f"  Streamed & tokenized {len(doc_ids):,} docs ({time.perf_counter() - started:.1f}s)...")

    del seen_ids
    gc.collect()

    print(f"Loaded & tokenized {len(doc_ids):,} docs in {time.perf_counter() - started:.2f}s (zero raw text kept in heap)")
    if not doc_ids:
        print("No documents found for partition!")
        return 0

    print(f"[2/2] Building and saving BM25 index to {out_dir} (scipy CSC backend, int32 dtype)...")
    started = time.perf_counter()
    params = BM25Params(k1=1.5, b=0.75, keep_diacritics=False)
    retriever = bm25s.BM25(
        k1=params.k1,
        b=params.b,
        method=params.method,
        int_dtype="int32",
        csc_backend="scipy",
    )
    retriever.index(tokenized, show_progress=True)

    del tokenized
    gc.collect()

    # Save
    retriever.save(str(out_dir / "docs.bm25"), show_progress=False)
    np.save(out_dir / "docs.rows.npy", np.array(doc_ids, dtype=np.int64).reshape(-1, 1))
    (out_dir / "docs.params.json").write_text(
        json.dumps({"k1": params.k1, "b": params.b, "method": params.method}, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "partition": partition,
        "n_docs": len(doc_ids),
        "docs_path": str(docs_path),
        "max_tokens": max_tokens,
        "build_time_s": time.perf_counter() - started,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Partition '{partition}' successfully built: {len(doc_ids):,} docs in {time.perf_counter() - started:.2f}s -> {out_dir}")
    return len(doc_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path("data/docs.jsonl"))
    parser.add_argument("--partition", choices=["pubmed", "web", "both"], required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--min-chars", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    if args.partition in ("pubmed", "both"):
        target_dir = args.out_dir if args.partition == "pubmed" and args.out_dir else Path("runs/indexes/index_pubmed")
        build_partition(args.docs, target_dir, partition="pubmed", min_chars=args.min_chars, max_tokens=args.max_tokens)

    if args.partition in ("web", "both"):
        target_dir = args.out_dir if args.partition == "web" and args.out_dir else Path("runs/indexes/index_web")
        build_partition(args.docs, target_dir, partition="web", min_chars=args.min_chars, max_tokens=args.max_tokens)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
