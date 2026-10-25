"""Query variants for retrieval: one row per (qid, variant) with the language it targets.

Merges the LLM query frame (query_frame.py) with the hand-written tables in configs/query_expansion.json
(clinical acronyms with VI/EN expansions, VI -> ZH medical terms). Output query_variants.parquet:
qid, variant, lang, text. Variants (rows with empty text are dropped, duplicates per qid/lang removed):

  orig    vi  the original question
  kw      vi  frame keywords
  sub<i>  vi  frame sub-questions
  hyde    vi  frame HyDE passage (meant for dense search only)
  acro    vi  question + VI expansions of the acronyms it contains (table)
  en_kw   en  frame keywords_en + English entity aliases + EN acronym expansions (table)
  zh_q    zh  frame query_zh
  zh_kw   zh  frame keywords_zh + ZH terms of the VI phrases found in the question (table)

The frame is optional: without it only orig/acro/en_kw/zh_kw (table parts) are written. Rank a variant file
with bm25_experiment.py or dense search; both keep the `variant` column.

Usage:
  python -m r2ai2026.stages.query_variants --queries query.parquet [--frame query_frame.parquet] \\
      [--table configs/query_expansion.json] --out query_variants.parquet
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

TABLE = Path(__file__).resolve().parents[2] / "configs" / "query_expansion.json"


def as_obj(value, default):
    """Frame columns are JSON strings in the parquet; accept decoded values too."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default if value is None else value


def acronyms_in(query: str, acronyms: dict) -> list[str]:
    return [a for a in acronyms if re.search(rf"(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])", query)]


def zh_terms(query: str, vi_zh: dict) -> list[str]:
    low = query.lower()
    terms = []
    for vi in sorted(vi_zh, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(vi.lower())}(?!\w)", low):
            terms += [t for t in vi_zh[vi].split() if t not in terms]
    return terms


def variants(qid: int, query: str, frame: dict | None, table: dict) -> list[tuple]:
    frame = frame or {}
    acr = acronyms_in(query, table["acronyms"])
    ents = [e for e in as_obj(frame.get("entities"), []) if isinstance(e, dict)]
    en_alias = [a for e in ents for a in e.get("aliases", []) if isinstance(a, str) and a.isascii()]
    out = [("orig", "vi", query), ("kw", "vi", frame.get("keywords") or "")]
    out += [(f"sub{i}", "vi", s) for i, s in enumerate(as_obj(frame.get("subqueries"), []))]
    out += [("hyde", "vi", frame.get("hyde") or ""),
            ("acro", "vi", " ".join([query] + [table["acronyms"][a]["vi"] for a in acr]) if acr else ""),
            ("en_kw", "en", " ".join([frame.get("keywords_en") or ""] + en_alias
                                     + [table["acronyms"][a]["en"] for a in acr])),
            ("zh_q", "zh", frame.get("query_zh") or ""),
            ("zh_kw", "zh", " ".join([frame.get("keywords_zh") or ""] + zh_terms(query, table["vi_zh"])))]
    rows, seen = [], set()
    for name, lang, text in out:
        text = " ".join(str(text).split())
        if text and (lang, text) not in seen:
            seen.add((lang, text))
            rows.append((qid, name, lang, text))
    return rows


def build(queries: pd.DataFrame, frames: pd.DataFrame | None, table: dict) -> pd.DataFrame:
    by_qid = {} if frames is None else {int(r["qid"]): r.to_dict() for _, r in frames.iterrows()}
    rows = []
    for qid, query in zip(queries["id"].astype(int), queries["query"].astype(str)):
        rows += variants(qid, query, by_qid.get(qid), table)
    return pd.DataFrame(rows, columns=["qid", "variant", "lang", "text"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=Path, required=True, help="parquet with id, query")
    ap.add_argument("--frame", type=Path, default=None, help="query_frame.parquet (optional)")
    ap.add_argument("--table", type=Path, default=TABLE)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    table = json.loads(args.table.read_text(encoding="utf-8"))
    frames = pd.read_parquet(args.frame) if args.frame else None
    df = build(pd.read_parquet(args.queries), frames, table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(json.dumps({"queries": int(df["qid"].nunique()), "rows": len(df),
                      "per_variant": df.groupby("variant").size().to_dict(),
                      "per_lang": df.groupby("lang").size().to_dict()}))


if __name__ == "__main__":
    main()
