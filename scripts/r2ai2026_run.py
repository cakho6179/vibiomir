"""Run the retrieval pipeline and write a validated submission ZIP.

    python scripts/r2ai2026_run.py \
        --docs data/docs.jsonl \
        --queries data/query.parquet \
        --index-dir runs/indexes/index \
        --submission runs/submissions/submission.zip \
        --chunk-source doc_head --topk-docs 20 --chunk-budget 1400

The script always prints submission statistics and validation problems before
writing anything, so a broken submission never reaches the leaderboard.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai2026.pipeline import BaselineConfig, BaselinePipeline, load_queries  # noqa: E402
from r2ai2026.submit import submission_stats, validate_submission  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=None,
                        help="load a saved index instead of rebuilding it")
    parser.add_argument("--rankings", type=Path, default=None,
                        help="parquet with query_id/doc_id(/rank|score) to use instead of BM25")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--chunk-source", choices=["doc_head", "doc_spread", "doc_full", "bm25_merge"], default="doc_head")
    parser.add_argument("--topk-docs", type=int, default=20)
    parser.add_argument("--chunk-budget", type=int, default=1400, help="0 means the whole document")
    parser.add_argument("--spread-windows", type=int, default=4, help="slices used by --chunk-source doc_spread")
    parser.add_argument("--merge-chunks-per-doc", type=int, default=3)
    parser.add_argument("--chunk-target-tokens", type=int, default=320)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=64)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--keep-diacritics", action="store_true")
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--min-doc-chars", type=int, default=200)
    parser.add_argument("--no-write", action="store_true", help="dry run: report statistics only")
    parser.add_argument("--stats-out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = BaselineConfig(
        docs_path=args.docs,
        queries_path=args.queries,
        index_dir=args.index_dir,
        submission_path=args.submission,
        topk_docs=args.topk_docs,
        chunk_source=args.chunk_source,
        chunk_budget_tokens=args.chunk_budget,
        merge_chunks_per_doc=args.merge_chunks_per_doc,
        spread_windows=args.spread_windows,
        chunk_target_tokens=args.chunk_target_tokens,
        chunk_overlap_tokens=args.chunk_overlap_tokens,
        k1=args.k1,
        b=args.b,
        keep_diacritics=args.keep_diacritics,
        limit_queries=args.limit_queries,
        min_doc_chars=args.min_doc_chars,
    )

    pipeline = BaselinePipeline(config)
    started = time.perf_counter()
    pipeline.load_corpus(show_progress=True)
    if args.limit_docs:
        keep = sorted(pipeline.state.documents)[: args.limit_docs]
        pipeline.state.documents = {doc_id: pipeline.state.documents[doc_id] for doc_id in keep}
    print(f"documents: {len(pipeline.state.documents)} ({time.perf_counter() - started:.1f}s)")

    started = time.perf_counter()
    if args.rankings is not None:
        count = pipeline.load_rankings(args.rankings)
        print(f"loaded external rankings for {count:,} queries from {args.rankings}")
        datasets_time = time.perf_counter() - started
    elif args.index_dir and (args.index_dir / "docs.rows.npy").exists():
        pipeline.load_indexes()
        print(f"loaded index from {args.index_dir}")
        datasets_time = time.perf_counter() - started
    else:
        sizes = pipeline.build_indexes(show_progress=True)
        print(f"built index: {sizes}")
        datasets_time = time.perf_counter() - started

    started = time.perf_counter()
    records = pipeline.run()
    run_time = time.perf_counter() - started
    stats = submission_stats(records)
    print(f"index ready in {datasets_time:.1f}s, retrieved in {run_time:.1f}s")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    queries = load_queries(args.queries)
    problems = validate_submission(records, expected_ids={int(query["id"]) for query in queries})
    if problems:
        print(f"VALIDATION PROBLEMS ({len(problems)}):")
        for problem in problems[:20]:
            print(f"  - {problem}")
        if not args.no_write:
            return 1
    else:
        print("validation: ok")

    if args.no_write:
        print("dry run: nothing written")
        return 0

    stats = pipeline.write_submission(records)
    print(f"submission written: {args.submission}")
    print(json.dumps({k: round(v, 3) for k, v in stats.items()}, indent=2, ensure_ascii=False))
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
