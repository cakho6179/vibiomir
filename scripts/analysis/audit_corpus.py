"""CLI audit of ViBioMIR (R2AI2026 Shared Task 3) queries + link corpus.

Usage:
    python scripts/analysis/audit_corpus.py [--data-dir data]
                                            [--json out.json] [--markdown out.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT))

from r2ai2026.corpus_audit import (  # noqa: E402
    audit_retrieval_data,
    render_audit_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--query-file", default="query.parquet")
    parser.add_argument("--corpus-file", default="links_corpus_vi_en_zh.parquet")
    parser.add_argument("--json", dest="json_out", default="")
    parser.add_argument("--markdown", dest="markdown_out", default="")
    args = parser.parse_args()

    data_dir = (ROOT / args.data_dir).resolve()
    query_path = data_dir / args.query_file
    corpus_path = data_dir / args.corpus_file
    for path in (query_path, corpus_path):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    audit = audit_retrieval_data(query_path, corpus_path)
    markdown = render_audit_markdown(audit, title="ViBioMIR corpus audit")

    print(markdown)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[audit] json -> {out}")
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        print(f"[audit] markdown -> {out}")


if __name__ == "__main__":
    main()
