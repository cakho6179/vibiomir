"""Section layer of the docstore: consecutive passages packed into chunk-sized sections (docstore.pack_sections).

    python scripts/data/build_sections.py [--store data/docstore/v1] [--max-chars 4800] [--min-chars 2000]

Reads passages.parquet, writes sections.parquet (doc_id, sid, pid_first, pid_last, start, end, heading) next to it
and adds the section stats to MANIFEST.json["sections"]. Offsets are body offsets, like passages.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from r2ai2026.corpus.docstore import pack_sections  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=ROOT / "data" / "docstore" / "v1")
    ap.add_argument("--max-chars", type=int, default=4800)
    ap.add_argument("--min-chars", type=int, default=2000)
    args = ap.parse_args()

    p = pd.read_parquet(args.store / "passages.parquet").sort_values(["doc_id", "pid"])
    rows = []
    doc_ids = p["doc_id"].to_numpy()
    starts, ends, heads = p["start"].tolist(), p["end"].tolist(), p["heading"].fillna("").tolist()
    i, n = 0, len(p)
    while i < n:
        j = i
        while j < n and doc_ids[j] == doc_ids[i]:
            j += 1
        spans = list(zip(starts[i:j], ends[i:j], heads[i:j]))
        for sid, (a, b, s, e) in enumerate(pack_sections(spans, args.max_chars, args.min_chars)):
            rows.append((int(doc_ids[i]), sid, a, b, s, e, spans[a][2]))
        i = j
    sec = pd.DataFrame(rows, columns=["doc_id", "sid", "pid_first", "pid_last", "start", "end", "heading"])
    sec.to_parquet(args.store / "sections.parquet", index=False)

    length = sec["end"] - sec["start"]
    vi = sec["doc_id"] <= 766_441
    stats = {"max_chars": args.max_chars, "min_chars": args.min_chars, "sections": len(sec),
             "per_doc": round(len(sec) / sec["doc_id"].nunique(), 3),
             "vi_len_quantiles": {str(q): int(v) for q, v in length[vi].quantile([.05, .25, .5, .75, .95]).items()},
             "over_max": int((length > args.max_chars).sum())}
    man_path = args.store / "MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["sections"] = stats
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
