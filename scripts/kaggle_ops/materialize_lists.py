"""Turn decided doc / chunk lists into submission zips (heavy text work on Kaggle; the decisions are made locally).

    python materialize_lists.py --input DIR --out DIR [--window 4500] [--max-mb 98]

DIR (searched recursively) holds lists.parquet, the docstore v2.1 corpus parts (part-*.jsonl[.gz]), query.parquet
and the vibiomir-code bundle. lists.parquet columns: job, qid, kind ("doc" | "chunk"), pos, doc_id and optionally
size (target chars of that chunk; missing or 0 = --window). For each job and query, relevant_docs = the "doc" rows
in pos order and relevant_chunks = one chunk per "chunk" row in pos order: a <= size char window grown from the
passage sharing the most query syllables (build_submission.lexical_best / window_chunk), as in every chunk list
submitted since #6751. Rows with src = "hf" take the text from --alt-text (hf_text.parquet of
hf_text_compare.py: doc_id, title, content; HF dataset tmh1008/ViBioMIR-crawled-corpus) when the doc is there. Queries missing from a job get empty lists. One LZMA zip
per job (<job>.zip) plus stats.json; jobs over --max-mb are flagged and should not be submitted.
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


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


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
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=4500)
    ap.add_argument("--max-mb", type=float, default=98.0)
    ap.add_argument("--alt-text", default="hf_text.parquet", help="file name searched under --input")
    ap.add_argument("--profile", action="store_true",
                    help="cut the chunk where the per-passage relevance ends (needs the passage scores)")
    ap.add_argument("--profile-file", default="pas_scores.parquet",
                    help="qid, doc_id, passage, score from rerank.py --keep-all")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="stop growing when the next passage scores below alpha x the best passage")
    ap.add_argument("--min-frac", type=float, default=0.30,
                    help="always grow to this fraction of the document first (LCS(c,g)/|g| floor)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    code = glob.glob(f"{args.input}/**/vibiomir-code/r2ai2026", recursive=True)
    if code:
        sys.path.insert(0, str(Path(code[0]).parent))
    else:
        code_zip = glob.glob(f"{args.input}/**/vibiomir-code.zip", recursive=True)
        if code_zip:
            import zipfile
            target = Path("/kaggle/working/code")
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(code_zip[0]) as z:
                z.extractall(target)
            sys.path.insert(0, str(target))
    from r2ai2026.core.submission import submission_stats, validate_submission, write_submission_zip
    from r2ai2026.stages.build_submission import lexical_best, rr_spans, window_chunk
    profile_chunk = None
    if args.profile:  # chunk_profile.py is shipped next to this script by prep_kaggle_script.py --modules
        try:
            from chunk_profile import profile_chunk
        except ImportError:
            from r2ai2026.stages.chunk_profile import profile_chunk
    stair_chunk = None

    lists = pd.read_parquet(find(args.input, "lists.parquet")).sort_values(["job", "qid", "kind", "pos"])
    q = pd.read_parquet(find(args.input, "query.parquet"))
    qids = sorted(q["id"].astype(int))
    qtext = dict(zip(q["id"].astype(int), q["query"].astype(str)))
    if "size" not in lists:
        lists["size"] = 0
    lists["size"] = lists["size"].fillna(0).astype(int).where(lambda x: x > 0, args.window)
    if "src" not in lists:
        lists["src"] = ""
    lists["src"] = lists["src"].fillna("")
    ch = lists[lists["kind"] == "chunk"]
    pairs = set(zip(ch["qid"].astype(int), ch["doc_id"].astype(int), ch["size"], ch["src"]))
    alt: dict[int, tuple[str, str]] = {}
    want_alt = {d for _, d, _, s in pairs if s == "hf"}
    if want_alt:
        a = pd.read_parquet(find(args.input, args.alt_text))
        a = a[a["doc_id"].isin(want_alt) & (a["content"].str.len() > 0)]
        alt = dict(zip(a["doc_id"].astype(int), zip(a["title"].fillna(""), a["content"])))
        log(f"alt text for {len(alt):,}/{len(want_alt):,} docs")
    log(f"{lists['job'].nunique()} jobs, {len(lists):,} rows, {len(pairs):,} distinct chunk pairs")
    modes = {s for _, _, _, s in pairs} & {"stair", "seed"}
    if modes:  # chunk_stair.py, shipped next to this script like chunk_profile.py
        try:
            from chunk_stair import stair_chunk
        except ImportError:
            from r2ai2026.stages.chunk_stair import stair_chunk
        log(f"stair-aware chunking requested: {sorted(modes)}")
    sect: dict[int, list[tuple[int, int]]] = {}
    if "stair" in modes:
        sp = pd.read_parquet(find(args.input, "sections.parquet")).sort_values(["doc_id", "sid"])
        for d, g in sp.groupby("doc_id", sort=False):
            sect[int(d)] = list(zip(g["start"].astype(int), g["end"].astype(int)))
        log(f"sections for {len(sect):,} docs (mean {sum(len(v) for v in sect.values()) / max(len(sect), 1):.2f} per doc)")
    prof: dict[tuple[int, int], list[float]] = {}
    if args.profile or modes:
        pf = pd.read_parquet(find(args.input, args.profile_file))
        pf = pf.sort_values(["qid", "doc_id", "passage"])
        for (q, d), g in pf.groupby(["qid", "doc_id"], sort=False):
            prof[(int(q), int(d))] = g["score"].astype(float).tolist()
        log(f"passage profiles for {len(prof):,} query-doc pairs "
            f"(mean {sum(len(v) for v in prof.values()) / max(len(prof), 1):.2f} passages)")
    bodies = load_bodies(args.input, {d for _, d, _, _ in pairs})
    log(f"bodies {len(bodies):,}")

    made: dict[tuple[int, int, int, str], str] = {}
    n_prof = 0
    for qid, d, size, src in pairs:
        if src == "hf" and d in alt:
            title, body = alt[d]
        elif d in bodies:
            title, body = bodies[d]
        else:
            continue
        spans = rr_spans(body)
        if not spans:
            continue
        text = None
        if src in ("stair", "seed"):
            text = stair_chunk(title, body, spans, sect.get(d) if src == "stair" else None,
                               prof.get((qid, d)), size, size, args.min_frac,
                               lexical_best(qtext[qid], title, body, spans))
            if text:
                n_prof += 1
        elif prof:
            text = profile_chunk(title, body, spans, prof.get((qid, d)), size, size,
                                 args.alpha, args.min_frac)
            if text:
                n_prof += 1
        if text is None:
            text = window_chunk(title, body, lexical_best(qtext[qid], title, body, spans), spans, size, size)
        if text.strip():
            made[(qid, d, size, src)] = text
    log(f"built {len(made):,} chunks" + (f", {n_prof:,} from the relevance profile" if prof else ""))

    stats = {}
    for job, g in lists.groupby("job", sort=True):
        by = {(int(k[0]), k[1]): list(zip(v["doc_id"].astype(int), v["size"], v["src"]))
              for k, v in g.groupby(["qid", "kind"])}
        out = [{"id": qid, "relevant_docs": [d for d, _, _ in by.get((qid, "doc"), [])],
                "relevant_chunks": [{"doc_id": d, "chunk_text": made[(qid, d, z, s)]}
                                    for d, z, s in by.get((qid, "chunk"), []) if (qid, d, z, s) in made]} for qid in qids]
        problems = validate_submission(out, expected_ids=qids)
        if problems:
            raise SystemExit(f"{job}: {problems[:5]}")
        s = write_submission_zip(out, args.out / f"{job}.zip", method="lzma")
        s.update(submission_stats(out), over_limit=s["zip_mb"] > args.max_mb)
        stats[job] = s
        log(f"{job}: {s['zip_mb']:.1f} MB, {s['docs_per_query']:.1f} docs, {s['chunks_per_query']:.1f} chunks/query")
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    log("done")


if __name__ == "__main__":
    main()
