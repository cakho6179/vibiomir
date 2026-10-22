"""Offline evaluation for the R2AI2026 pipeline (no leaderboard quota needed).

    # title-as-query retrieval quality + chunk budget curve on the current corpus
    python scripts/r2ai2026_devproxy.py --docs data/docs.jsonl \
        --index-dir runs/indexes/index --budgets 200,400,800,1400,3000,0

Reports:
* ``recall@k`` / MRR for document retrieval using document titles as queries,
* macro chunk F2 of the "one merged chunk per document" policy at several token
  budgets, measured with the official rule against pseudo-gold spans.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai2026.bm25 import rank_ids  # noqa: E402
from r2ai2026.chunking import whole_document_chunk  # noqa: E402
from r2ai2026.devproxy import retrieval_metrics, span_recovery_curve, title_queries  # noqa: E402
from r2ai2026.pipeline import BaselineConfig, BaselinePipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--keep-diacritics", action="store_true")
    parser.add_argument("--max-documents", type=int, default=400)
    parser.add_argument("--ks", default="1,5,10,20,50")
    parser.add_argument("--budgets", default="200,400,800,1400,3000,0")
    parser.add_argument("--spans", type=int, default=2)
    parser.add_argument("--span-tokens", type=int, default=180)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = BaselineConfig(
        docs_path=args.docs,
        index_dir=args.index_dir,
        k1=args.k1,
        b=args.b,
        keep_diacritics=args.keep_diacritics,
        chunk_source="doc_head",
    )
    pipeline = BaselinePipeline(config)
    pipeline.load_corpus(show_progress=True)
    if args.max_documents:
        keep = sorted(pipeline.state.documents)[: args.max_documents]
        pipeline.state.documents = {doc_id: pipeline.state.documents[doc_id] for doc_id in keep}
    print(f"documents: {len(pipeline.state.documents)}")

    if args.index_dir and (args.index_dir / "docs.rows.npy").exists():
        pipeline.load_indexes()
    else:
        print(f"index sizes: {pipeline.build_indexes(show_progress=True)}")

    documents = {doc_id: record["_text"] for doc_id, record in pipeline.state.documents.items()}
    titles = {doc_id: str(record.get("title") or "") for doc_id, record in pipeline.state.documents.items()}
    queries = title_queries(documents, titles)
    rankings: dict[int, list[int]] = {}
    ks = [int(value) for value in args.ks.split(",")]
    for doc_id, title in queries.items():
        hits = pipeline.doc_index.search(title, k=max(ks))
        rankings[doc_id] = [int(value) for value in rank_ids(hits) if int(value) in documents]
    metrics = retrieval_metrics(rankings, {doc_id: [doc_id] for doc_id in queries}, ks=ks)
    print("title-as-query retrieval:", json.dumps(metrics.as_dict(), indent=2))

    budgets = [int(value) for value in args.budgets.split(",")]
    curve = span_recovery_curve(
        documents,
        lambda text, budget: whole_document_chunk(text, max_tokens=budget),
        budgets=budgets,
        spans=args.spans,
        span_tokens=args.span_tokens,
        max_documents=args.max_documents,
    )
    print("chunk budget curve (official rule vs pseudo-gold spans):")
    for row in curve:
        print(
            f"  budget={int(row['budget_tokens']):>5}  "
            f"P={row['chunk_precision']:.3f} R={row['chunk_recall']:.3f} F2={row['chunk_f2']:.3f}"
        )

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(
            json.dumps({"retrieval": metrics.as_dict(), "chunk_budget_curve": curve}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"report written to {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
