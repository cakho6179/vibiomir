"""Build (or inspect) the BM25 indexes from a crawled documents.jsonl.

    python scripts/r2ai2026_build_index.py --docs data/docs.jsonl \
        --index-dir runs/indexes/index --chunk-source bm25_merge

    # quick sanity check on a small slice
    python scripts/r2ai2026_build_index.py --limit-docs 500 --chunk-source doc_head
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai2026.pipeline import BaselineConfig, BaselinePipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--chunk-source", choices=["doc_head", "doc_spread", "doc_full", "bm25_merge"], default="doc_head")
    parser.add_argument("--chunk-target-tokens", type=int, default=320)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=64)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--keep-diacritics", action="store_true")
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--min-doc-chars", type=int, default=200)
    args = parser.parse_args(argv)

    config = BaselineConfig(
        docs_path=args.docs,
        index_dir=args.index_dir,
        chunk_source=args.chunk_source,
        chunk_target_tokens=args.chunk_target_tokens,
        chunk_overlap_tokens=args.chunk_overlap_tokens,
        k1=args.k1,
        b=args.b,
        keep_diacritics=args.keep_diacritics,
        min_doc_chars=args.min_doc_chars,
    )
    pipeline = BaselinePipeline(config)

    started = time.perf_counter()
    pipeline.load_corpus(show_progress=True)
    if args.limit_docs:
        keep = sorted(pipeline.state.documents)[: args.limit_docs]
        pipeline.state.documents = {doc_id: pipeline.state.documents[doc_id] for doc_id in keep}
    print(f"documents kept: {len(pipeline.state.documents)} in {time.perf_counter() - started:.1f}s")

    started = time.perf_counter()
    sizes = pipeline.build_indexes(show_progress=True)
    print(f"index sizes: {sizes} in {time.perf_counter() - started:.1f}s")

    if args.chunk_source == "bm25_merge":
        started = time.perf_counter()
        n_chunks = pipeline.build_chunks(show_progress=True) if not pipeline.state.doc_chunks else len(
            [chunk for chunks in pipeline.state.doc_chunks.values() for chunk in chunks]
        )
        print(f"chunks built: {n_chunks} in {time.perf_counter() - started:.1f}s")
        sizes = pipeline.build_indexes(show_progress=True)
        print(f"index sizes after chunk build: {sizes}")

    directory = pipeline.save_indexes()
    manifest = {
        "docs_path": str(args.docs),
        "index_dir": str(directory),
        "n_docs": len(pipeline.state.documents),
        "chunk_source": args.chunk_source,
        "k1": args.k1,
        "b": args.b,
        "keep_diacritics": args.keep_diacritics,
        "chunk_target_tokens": args.chunk_target_tokens,
        "chunk_overlap_tokens": args.chunk_overlap_tokens,
    }
    (Path(directory) / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"index written to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
