"""Compare chunk policies at equal payload budgets (label-free, no quota).

    python scripts/r2ai2026_compare_chunk_policies.py \
        --docs data/docs.jsonl --budgets 400,800,1400,3000,0 \
        --report-out runs/reports/chunk_policy_table.json

A policy wins only if it beats the alternatives at a comparable payload size
(``mean_chars`` / ``total_mb_per_1200q``), because submission size and scoring
time are the binding constraints once chunks are long.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai2026.chunking import head_tail_chunk, spread_windows_chunk, whole_document_chunk  # noqa: E402
from r2ai2026.devproxy import policy_comparison  # noqa: E402
from r2ai2026.pipeline import BaselineConfig, BaselinePipeline  # noqa: E402


def build_policy_set(windows: int):
    return {
        "head": lambda text, budget: whole_document_chunk(text, max_tokens=budget),
        "head_tail": lambda text, budget: head_tail_chunk(text, max_tokens=budget),
        "spread": lambda text, budget: spread_windows_chunk(text, max_tokens=budget, windows=windows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--budgets", default="400,800,1400,3000,0")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--max-documents", type=int, default=500)
    parser.add_argument("--spans", type=int, default=3)
    parser.add_argument("--span-tokens", type=int, default=180)
    parser.add_argument("--min-doc-chars", type=int, default=800, help="long documents are where policies differ")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args(argv)

    pipeline = BaselinePipeline(BaselineConfig(docs_path=args.docs, min_doc_chars=args.min_doc_chars))
    pipeline.load_corpus(show_progress=True)
    documents = {doc_id: record["_text"] for doc_id, record in pipeline.state.documents.items()}
    if args.max_documents:
        documents = {doc_id: documents[doc_id] for doc_id in sorted(documents)[: args.max_documents]}
    print(f"documents: {len(documents)}  mean chars: {sum(len(t) for t in documents.values()) / max(1, len(documents)):.0f}")

    rows = policy_comparison(
        documents,
        build_policy_set(args.windows),
        budgets=[int(value) for value in args.budgets.split(",")],
        seed=args.seed,
        spans=args.spans,
        span_tokens=args.span_tokens,
        max_documents=None,
    )
    header = f"{'policy':<10} {'budget':>7} {'P':>6} {'R':>6} {'F2':>6} {'mean_chars':>11} {'MB/1200q/doc':>13}"
    print(header)
    print("-" * len(header))
    for row in rows:
        budget = int(row["budget_tokens"])
        label = "full" if budget == 0 else str(budget)
        print(
            f"{row['policy']:<10} {label:>7} {row['chunk_precision']:>6.3f} {row['chunk_recall']:>6.3f} "
            f"{row['chunk_f2']:>6.3f} {row['mean_chars']:>11.0f} {row['total_mb_per_1200q']:>13.1f}"
        )

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report written to {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
