"""Probe submissions on top of a base file: base docs + K extra docs per query, chunks from a chosen file.

    python augment_probe.py --input DIR --out DIR --job NAME=SET:K:CHUNKS [--job ...]
    (several sets: NAME=zh+vi:10+30:CHUNKS adds 10 of zh, then 30 of vi not already listed)

DIR (searched recursively) holds the base submission (base/submission.json or --base glob), additions.parquet
(qid, set, rank, doc_id) and the chunk sources named in the jobs (CHUNKS = base or a zip file name such as
v21chunk.zip). For each job: relevant_docs = base docs + the first K docs of SET (by rank) not already in the
base list, appended after them; relevant_chunks = those of CHUNKS. SET "none" adds nothing. Docs and chunks are
scored separately, so one upload measures the added docs (docs P/R) and the chunk variant (chunks P/R) at once.
Outputs NAME.zip (LZMA, 256 MiB dictionary) and stats.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

T0 = time.time()
STATE: dict = {}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def find(root: str, pattern: str) -> str:
    hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    if not hits:
        raise SystemExit(f"not found under {root}: {pattern}")
    return hits[0]


def chunks_of(records: list[dict]) -> dict[int, list]:
    return {int(r["id"]): r.get("relevant_chunks", []) for r in records}


def build(job: tuple[str, str, str, str]) -> dict:
    from r2ai2026.core.submission import submission_stats, validate_submission, write_submission_zip
    name, add_set, k, src = job
    base, adds, chunk_src = STATE["base"], STATE["adds"], STATE["chunks"][src]
    parts = [] if add_set == "none" else list(zip(add_set.split("+"), map(int, k.split("+"))))
    out, added = [], 0
    for rec in base:
        qid = int(rec["id"])
        docs = list(rec["relevant_docs"])
        have = set(docs)
        for part, n in parts:  # sets in the order given, each skipping docs already listed
            extra = [d for d in adds.get((part, qid), []) if d not in have][:n]
            docs += extra
            have.update(extra)
            added += len(extra)
        out.append({"id": qid, "relevant_docs": docs, "relevant_chunks": chunk_src[qid]})
    problems = validate_submission(out, expected_ids=[int(r["id"]) for r in base])
    if problems:
        raise SystemExit(f"{name}: {problems[:5]}")
    stats = write_submission_zip(out, Path(STATE["out"]) / f"{name}.zip", method="lzma")
    stats.update(submission_stats(out), added_docs=added, added_per_query=added / len(out), set=add_set, k=k,
                 chunks=src)
    log(f"{name}: {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()})}")
    return {name: stats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default="base/submission.json")
    ap.add_argument("--job", action="append", required=True, help="NAME=SET:K:CHUNKS")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    code = glob.glob(f"{args.input}/**/vibiomir-code/r2ai2026", recursive=True)
    if code:
        sys.path.insert(0, str(Path(code[0]).parent))
    from r2ai2026.core.submission import read_submission_zip

    jobs = []
    for spec in args.job:
        name, rest = spec.split("=", 1)
        add_set, k, src = rest.split(":")
        jobs.append((name, add_set, k, src))
    base_path = find(args.input, args.base)
    base = (read_submission_zip(base_path) if base_path.endswith(".zip")
            else json.loads(Path(base_path).read_text(encoding="utf-8")))
    log(f"base {base_path}: {len(base)} queries")
    a = pd.read_parquet(find(args.input, "additions.parquet")).sort_values(["set", "qid", "rank"])
    adds = {(s, int(q)): g["doc_id"].astype(int).tolist() for (s, q), g in a.groupby(["set", "qid"])}
    chunks = {"base": chunks_of(base)}
    for src in sorted({j[3] for j in jobs} - {"base"}):
        # Kaggle unpacks a .zip uploaded in a dataset into a folder of the same stem
        hits = (glob.glob(f"{args.input}/**/{src}", recursive=True)
                + glob.glob(f"{args.input}/**/{Path(src).stem}/**/submission.json", recursive=True)
                + glob.glob(f"{args.input}/**/{src}/**/submission.json", recursive=True))
        if not hits:
            raise SystemExit(f"chunk source not found: {src}")
        path = sorted(hits)[0]
        records = (read_submission_zip(path) if path.endswith(".zip")
                   else json.loads(Path(path).read_text(encoding="utf-8")))
        chunks[src] = chunks_of(records)
        log(f"chunks {src} <- {path}")
        log(f"chunks {src}: {sum(map(len, chunks[src].values())) / len(chunks[src]):.2f} per query")
    STATE.update(base=base, adds=adds, chunks=chunks, out=str(args.out))
    stats: dict = {}
    if args.workers <= 1:  # also the local dry-run path (no fork on Windows)
        for job in jobs:
            stats.update(build(job))
    else:
        with get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
            for s in pool.imap_unordered(build, jobs):
                stats.update(s)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    log("done")


if __name__ == "__main__":
    main()
