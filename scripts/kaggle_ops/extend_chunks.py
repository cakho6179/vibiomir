"""Fill the chunk budget: add window chunks for listed docs that have none yet, K more per query.

    python extend_chunks.py --input DIR --base p1.zip --out DIR --extra 40,55,70 [--first-lang zh] [--max-mb 98] [--window 4500] [--tag ext]

DIR (searched recursively) holds the base submission zip, the docstore v2.1 corpus parts (part-*.jsonl[.gz]) and
query.parquet. For each query the docs of relevant_docs without a chunk are taken in list order (docs of
--first-lang first), and each gets the usual chunk: a <= 4500-char window grown from the passage that shares the
most query syllables (build_submission.lexical_best / window_chunk). relevant_docs and the existing chunks are
kept as they are, so a variant changes only the chunk side of the score. One LZMA zip per K (ext<K>.zip) plus
stats.json; variants over --max-mb are reported and should not be submitted.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
import time
from pathlib import Path

import pandas as pd

T0 = time.time()
WINDOW = 4500
VI_MAX, ZH_MAX = 766441, 4392214


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def lang(d: int) -> str:
    return "vi" if d <= VI_MAX else "zh" if d <= ZH_MAX else "en"


def find(root: str, pattern: str) -> str:
    hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    if not hits:
        raise SystemExit(f"not found under {root}: {pattern}")
    return hits[0]


def load_bodies(root: str, wanted: set[int]) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    for part in sorted(glob.glob(f"{root}/**/part-*.jsonl*", recursive=True)):
        opener = gzip.open if part.endswith(".gz") else open
        with opener(part, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["id"] in wanted and d["id"] not in out:
                    out[d["id"]] = (d.get("t") or "", d.get("x") or "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--base", required=True, help="file name of the base zip, found under --input")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra", default="40,55,70", help="comma list of K (extra chunks per query)")
    ap.add_argument("--first-lang", default="", help="docs of this language get chunks first")
    ap.add_argument("--max-mb", type=float, default=98.0)
    ap.add_argument("--window", type=int, default=WINDOW, help="window size of the added chunks (chars)")
    ap.add_argument("--tag", default="ext", help="output name prefix: <tag><K>.zip")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    code = glob.glob(f"{args.input}/**/vibiomir-code/r2ai2026", recursive=True)
    if code:
        sys.path.insert(0, str(Path(code[0]).parent))
    from r2ai2026.core.submission import (read_submission_zip, submission_stats, validate_submission,
                                          write_submission_zip)
    from r2ai2026.stages.build_submission import lexical_best, rr_spans, window_chunk

    # Kaggle unpacks a .zip uploaded in a dataset into a folder of the same stem
    hits = (glob.glob(f"{args.input}/**/{args.base}", recursive=True)
            + glob.glob(f"{args.input}/**/{Path(args.base).stem}/**/submission.json", recursive=True))
    hits = [h for h in hits if Path(h).is_file()]  # a dataset folder can share the name
    if not hits:
        raise SystemExit(f"base not found: {args.base}")
    path = sorted(hits)[0]
    base = (read_submission_zip(path) if path.endswith(".zip")
            else json.loads(Path(path).read_text(encoding="utf-8")))
    log(f"base <- {path}")
    ks = sorted(int(k) for k in args.extra.split(","))
    todo: dict[int, list[int]] = {}
    for r in base:
        have = {int(c["doc_id"]) for c in r["relevant_chunks"]}
        miss = [int(d) for d in r["relevant_docs"] if int(d) not in have]
        if args.first_lang:
            miss = [d for d in miss if lang(d) == args.first_lang] + [d for d in miss if lang(d) != args.first_lang]
        todo[int(r["id"])] = miss[:ks[-1]]
    wanted = {d for v in todo.values() for d in v}
    log(f"base: {len(base)} queries; up to {ks[-1]} new chunks/query over {len(wanted):,} docs")
    q = pd.read_parquet(find(args.input, "query.parquet"))
    qtext = dict(zip(q["id"].astype(int), q["query"].astype(str)))
    bodies = load_bodies(args.input, wanted)
    log(f"bodies {len(bodies):,}/{len(wanted):,}")

    made: dict[tuple[int, int], str] = {}
    for qid, docs in todo.items():
        for d in docs:
            if d not in bodies:
                continue
            title, body = bodies[d]
            spans = rr_spans(body)
            if not spans:
                continue
            k = lexical_best(qtext[qid], title, body, spans)
            text = window_chunk(title, body, k, spans, args.window, args.window)
            if text.strip():
                made[(qid, d)] = text
    log(f"built {len(made):,} chunks")

    stats = {}
    for k in ks:
        out = []
        for r in base:
            qid = int(r["id"])
            extra = [{"doc_id": d, "chunk_text": made[(qid, d)]} for d in todo[qid] if (qid, d) in made][:k]
            out.append({"id": qid, "relevant_docs": r["relevant_docs"], "relevant_chunks": r["relevant_chunks"] + extra})
        problems = validate_submission(out, expected_ids=[int(r["id"]) for r in base])
        if problems:
            raise SystemExit(f"ext{k}: {problems[:5]}")
        s = write_submission_zip(out, args.out / f"{args.tag}{k}.zip", method="lzma")
        s.update(submission_stats(out), extra=k, over_limit=s["zip_mb"] > args.max_mb)
        stats[f"{args.tag}{k}"] = s
        log(f"{args.tag}{k}: {s['zip_mb']:.1f} MB, {s['chunks_per_query']:.1f} chunks/query")
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    log("done")


if __name__ == "__main__":
    main()
