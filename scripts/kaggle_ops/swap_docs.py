"""Replace the relevant_docs of a built submission, keep its chunks (doc-side probes with the chunk side frozen).

    python swap_docs.py --input DIR --base r26.zip --out DIR

DIR (searched recursively) holds the base submission (a zip, or the submission.json Kaggle unpacked from it), docs.parquet
(job, qid, doc_id[, rank]) and the vibiomir-code bundle. For each job, every query gets the docs of that job in rank order
(no docs if the job has none for it) and keeps the base submission's relevant_chunks unchanged; docs and chunks are
scored separately, so the chunk score of every job equals the base one. One LZMA zip per job plus stats.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd


def find(root: str, pattern: str) -> str:
    hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    if not hits:
        raise SystemExit(f"not found under {root}: {pattern}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--base", default="r26.zip")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    code = glob.glob(f"{args.input}/**/vibiomir-code/r2ai2026", recursive=True)
    if code:
        sys.path.insert(0, str(Path(code[0]).parent))
    from r2ai2026.core.submission import submission_stats, validate_submission, write_submission_zip

    zips = sorted(glob.glob(f"{args.input}/**/{args.base}", recursive=True))
    if zips and zipfile.is_zipfile(zips[0]):
        with zipfile.ZipFile(zips[0]) as z:
            base = json.loads(z.read("submission.json").decode("utf-8"))
    else:  # Kaggle unpacks zips of mounted outputs into a folder of the same name
        base = json.loads(Path(find(args.input, f"{Path(args.base).stem}*/submission.json")).read_text(encoding="utf-8"))
    print(f"base: {len(base)} queries", flush=True)
    docs = pd.read_parquet(find(args.input, "docs.parquet"))
    sort = ["job", "qid", "rank"] if "rank" in docs else ["job", "qid"]
    docs = docs.sort_values(sort)
    qids = [int(r["id"]) for r in base]
    stats = {}
    for job, g in docs.groupby("job"):
        by = {int(q): list(dict.fromkeys(v["doc_id"].astype(int))) for q, v in g.groupby("qid")}
        out = [{**r, "relevant_docs": by.get(int(r["id"]), [])} for r in base]
        problems = validate_submission(out, expected_ids=qids)
        if problems:
            raise SystemExit(f"{job}: {problems[:5]}")
        s = write_submission_zip(out, args.out / f"{job}.zip", method="lzma")
        s.update(submission_stats(out))
        stats[job] = s
        print(job, json.dumps({k: round(v, 2) for k, v in s.items() if isinstance(v, float)}), flush=True)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
