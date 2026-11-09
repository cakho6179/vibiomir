"""Replace the doc channel with a BM25/e1 top-N list while keeping the best chunk channel.

This is the high-impact probe: the leaderboard BM25 baseline reaches Docs F2 0.5587 with
about 40 docs, while r24 reaches 0.3421 with 197.  If that doc channel reproduces, keeping
r24's 148 chunks should move FINAL from ~0.298 to ~0.40 without touching the chunk text.

    python swap_docs_keep_chunks.py --base-zip runs/mat_out/st1/r24.zip \
        --lists runs/system/v3/decide_e1/lists.parquet --job e1_seed --top-n 40 \
        --out temp_submission/e1_top40_r24chunks.zip
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT))
from r2ai2026.core.submission import submission_stats, validate_submission, write_submission_zip  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-zip", type=Path, required=True)
    ap.add_argument("--lists", type=Path, required=True)
    ap.add_argument("--job", default="e1_seed")
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with zipfile.ZipFile(args.base_zip) as z:
        base = json.loads(z.read("submission.json"))
    base = {int(x["id"]): x for x in base}
    lists = pd.read_parquet(args.lists)
    docs = lists[(lists["job"] == args.job) & (lists["kind"] == "doc") & (lists["pos"] < args.top_n)]
    by_q = {int(q): [int(d) for d in g.sort_values("pos")["doc_id"]] for q, g in docs.groupby("qid")}

    out = []
    for qid in sorted(base):
        new_docs = []
        seen = set()
        for d in by_q.get(qid, []):
            if d not in seen:
                seen.add(d)
                new_docs.append(d)
        rec = dict(base[qid])
        rec["relevant_docs"] = new_docs
        out.append(rec)

    problems = validate_submission(out, expected_ids=sorted(base))
    if problems:
        raise SystemExit(f"invalid: {problems[:5]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats = write_submission_zip(out, args.out, method="lzma")
    print(json.dumps(submission_stats(out), indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
