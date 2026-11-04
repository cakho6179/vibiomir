"""Build submission zips on Kaggle from small ranking files (full-text chunks, LZMA, < 100 MB).

    python build_submission.py --data DIR --jobs jobs.json --out DIR      (or --jobs '<inline JSON>')

jobs: [{"name": "...", "systems": ["a.parquet+b.parquet", "c.parquet"], "weights": [1, 2],
             "n": [28, [150, 100]], "vi_only": true, "zip": "lzma"|"deflate", "reorder": false}, ...]
An entry of "n" is N (N docs, N whole-doc chunks) or [N_docs, N_chunks]: relevant_docs is a separate list,
so doc depth costs almost no bytes; chunks come only from the top of relevant_docs.
"cut": {"spec": "rerank_r1.parquet", "tau": -2.0, "min": 20} keeps, in fused order, only docs whose score in
that ranking is >= tau (at least "min" docs), so the doc count varies per query; N_docs is then a cap.
"keep" / "exclude" restrict the list to a stratum (see doc_lists), e.g. union docs found by >= 2 families that are
not in the top 88 of w12: {"keep": {"spec": "union_scored_v1.parquet", "query": "n_families >= 2"},
"exclude": {"systems": [...], "weights": [1, 2], "n": 88}}.
"append": {<stratum job keys incl. "n">, "k": K, "gate_min_n": G, "chunks": C} adds the first K stratum docs not
yet listed after the list, only for queries whose list has >= G docs; N_chunks then counts base docs only and the
first C appended docs also get chunks (default 0: a doc-only change). "chunk_append": {<stratum job keys>, "k": K} adds chunks
(not docs) for the first K stratum docs outside relevant_docs. A list is cut at "list_depth" (default max(n)) and each system
is read only to "depth", so a stratum needs both >= the rank it filters on; with "append", set N_docs above the
final length and "list_depth" to the base depth, or the cap truncates the appended docs.
"reorder": write queries in a greedy chain of overlapping doc sets so repeats sit inside the LZMA window.
"chunk": {"window": S, "best": file} is a window around the reranker's best passage; {"mode": "section",
"best": file, "section_max": 4800} uses the docstore section containing that passage instead, and falls back
to the window when the section is missing or too long.
A system is one or more ranking parquet files (qid, doc_id, score; found in the working tree or input mounts)
joined by "+", merged by score; several systems are fused by weighted RRF (k=60).
Chunk = whole document (title + text). The zip uses LZMA (method 14) with a 256 MiB dictionary,
because the same document recurs across queries; any zipfile reader decodes it.
Outputs: <name>_n<N>.zip or <name>_d<Nd>c<Nc>[r].zip, <name>_ranking.parquet, build_stats.json
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import lzma
import struct
import time
import zipfile
from pathlib import Path

import pandas as pd

RRF_K = 60
VI_MAX_ID = 766441
LZMA_DICT = 1 << 28
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def find(name: str) -> str:
    """Resolve a spec relative to the working tree or the platform input mounts."""
    if Path(name).exists():
        return name
    roots = [Path.cwd()]
    for p in (Path.cwd(), *Path.cwd().parents):
        q = p / "input"
        if q.is_dir():
            roots.append(q)
    for root in roots:
        hits = sorted(root.glob(f"**/{name}"))
        if hits:
            return str(hits[0])
    raise FileNotFoundError(name)


def load_system(spec: str, depth: int, vi_only: bool) -> pd.DataFrame:
    parts = pd.concat([pd.read_parquet(find(p)) for p in spec.split("+")], ignore_index=True)
    parts = parts[["qid", "doc_id", "score"]]
    if vi_only:
        parts = parts[parts["doc_id"] <= VI_MAX_ID]
    parts = parts.sort_values(["qid", "score"], ascending=[True, False]).drop_duplicates(["qid", "doc_id"])
    parts["rank"] = parts.groupby("qid").cumcount()
    return parts[parts["rank"] < depth]


def fuse(job: dict) -> pd.DataFrame:
    systems = [load_system(s, job.get("depth", 200), job.get("vi_only", False)) for s in job["systems"]]
    weights = job.get("weights") or [1.0] * len(systems)
    if len(systems) == 1:
        fused = systems[0][["qid", "doc_id", "score"]]
    else:
        fused = pd.concat([s.assign(score=w / (RRF_K + s["rank"] + 1)) for s, w in zip(systems, weights)])
        fused = fused.groupby(["qid", "doc_id"], as_index=False)["score"].sum()
    fused = fused.sort_values(["qid", "score"], ascending=[True, False])
    fused["rank"] = fused.groupby("qid").cumcount()
    return fused


def doc_lists(job: dict, fused: pd.DataFrame | None = None) -> dict[int, list[int]]:
    """Per-query doc list of a job: fused order, top depth, then "cut", "keep" and "exclude" filters.
    "keep": {"spec": file, "query": "n_families >= 2 and rank <= 500"} keeps pairs whose row in that file passes
    the pandas query; "exclude": {<job keys: systems, weights, depth, vi_only, cut>, "n": N} drops the top N of
    another list, so a stratum outside the main submission can be probed on its own."""
    fused = fuse(job) if fused is None else fused
    caps = [max(n) if isinstance(n, list) else n for n in job.get("n", [job.get("depth", 200)])]
    depth = job.get("list_depth") or max(caps)
    top = fused[fused["rank"] < depth]
    by_q = {int(q): [int(d) for d in g["doc_id"]] for q, g in top.groupby("qid")}
    if job.get("cut"):
        cut = job["cut"]
        sc = load_system(cut["spec"], 10 ** 6, False)
        ok = {(int(q), int(d)) for q, d, v in zip(sc["qid"], sc["doc_id"], sc["score"]) if v >= cut["tau"]}
        lo = cut.get("min", 0)
        by_q = {q: ds[:lo] + [d for d in ds[lo:] if (q, d) in ok] for q, ds in by_q.items()}
    if job.get("keep"):
        sc = pd.read_parquet(find(job["keep"]["spec"])).query(job["keep"]["query"])
        ok = set(zip(sc["qid"].astype(int), sc["doc_id"].astype(int)))
        by_q = {q: [d for d in ds if (q, d) in ok] for q, ds in by_q.items()}
    if job.get("exclude"):
        ex = job["exclude"]
        drop = {(q, d) for q, ds in doc_lists({**ex, "n": [ex["n"]]}).items() for d in ds[:ex["n"]]}
        by_q = {q: [d for d in ds if (q, d) not in drop] for q, ds in by_q.items()}
    if job.get("append"):
        ap = job["append"]
        extra = new_docs(by_q, doc_lists(ap), ap.get("k", 10 ** 6))
        gate = ap.get("gate_min_n", 0)
        by_q = {q: ds + extra.get(q, []) if len(ds) >= gate else ds for q, ds in by_q.items()}
    return by_q


def new_docs(by_q: dict[int, list[int]], other: dict[int, list[int]], k: int) -> dict[int, list[int]]:
    """Per query, the first k docs of ``other`` that are not in ``by_q``."""
    out = {}
    for q, ds in other.items():
        have = set(by_q.get(q, []))
        out[q] = [d for d in ds if d not in have][:k]
    return out


def base_lengths(job: dict, fused: pd.DataFrame) -> dict[int, int] | None:
    """Length of each query's list before "append" (N_chunks counts base docs only; appended docs get chunks only
    through append["chunks"])."""
    if not job.get("append"):
        return None
    return {q: len(ds) for q, ds in doc_lists({k: v for k, v in job.items() if k != "append"}, fused).items()}


def chunk_extra(job: dict, by_q: dict[int, list[int]]) -> dict[int, list[int]]:
    """"chunk_append": {<stratum job keys>, "k": K}: chunks for the first K stratum docs that are NOT in
    relevant_docs, after the usual chunks; relevant_docs is unchanged (chunk-only probe)."""
    ca = job.get("chunk_append")
    return new_docs(by_q, doc_lists(ca), ca.get("k", 40)) if ca else {}


class BigDictLZMACompressor(zipfile.LZMACompressor):
    def _init(self):
        filt = {"id": lzma.FILTER_LZMA1, "preset": 9, "dict_size": LZMA_DICT}
        props = lzma._encode_filter_properties(filt)
        self._comp = lzma.LZMACompressor(lzma.FORMAT_RAW, filters=[filt])
        return struct.pack("<BBH", 9, 4, len(props)) + props


def write_zip(records: list, path: Path, method: str = "lzma") -> None:
    """method "lzma" (smallest) or "deflate" (level 9, accepted by every reader; use it when < 100 MB)."""
    payload = json.dumps(records, ensure_ascii=False).encode("utf-8")
    if method == "deflate":
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.writestr("submission.json", payload)
    else:
        saved = zipfile.LZMACompressor
        zipfile.LZMACompressor = BigDictLZMACompressor
        try:
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_LZMA) as z:
                z.writestr("submission.json", payload)
        finally:
            zipfile.LZMACompressor = saved
    with zipfile.ZipFile(path) as z:  # round trip with the stock reader
        assert z.read("submission.json") == payload, path


RR_PASSAGE_CHARS = 1500  # r2ai2026/stages/rerank.py PASSAGE_CHARS


def rr_split(title: str, text: str) -> list[str]:
    """Exact copy of r2ai2026/stages/rerank.py split_passages (its best_passage indexes this list)."""
    pieces = []
    for para in text.split("\n"):
        while len(para) > RR_PASSAGE_CHARS:
            pieces.append(para[:RR_PASSAGE_CHARS])
            para = para[RR_PASSAGE_CHARS:]
        if para.strip():
            pieces.append(para)
    out, buf = [], ""
    for p in pieces:
        if buf and len(buf) + len(p) > RR_PASSAGE_CHARS:
            out.append(buf)
            buf = ""
        buf = f"{buf}\n{p}" if buf else p
    if buf or not out:
        out.append(buf)
    return [f"{title}\n{p}".strip() for p in out]


def rr_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) in text of each rr_split passage body (same pieces, but as contiguous spans)."""
    pieces, pos = [], 0
    for para in text.split("\n"):
        s, n = pos, len(para)
        while n > RR_PASSAGE_CHARS:
            pieces.append((s, s + RR_PASSAGE_CHARS))
            s, n = s + RR_PASSAGE_CHARS, n - RR_PASSAGE_CHARS
        if text[s:s + n].strip():
            pieces.append((s, s + n))
        pos += len(para) + 1
    out, cur, cur_len = [], None, 0
    for a, b in pieces:
        if cur and cur_len + (b - a) > RR_PASSAGE_CHARS:
            out.append(cur)
            cur = None
        if cur is None:
            cur, cur_len = (a, b), b - a
        else:
            cur, cur_len = (cur[0], b), cur_len + 1 + (b - a)
    if cur or not out:
        out.append(cur or (0, 0))
    return out


def lexical_best(query: str, title: str, text: str, spans: list[tuple[int, int]]) -> int:
    """Passage sharing the most distinct query syllables (fallback when the reranker never saw the doc)."""
    q = {w for w in query.lower().split() if len(w) > 1}
    scores = [len(q & set(f"{title}\n{text[a:b]}".lower().split())) for a, b in spans]
    return max(range(len(spans)), key=lambda k: (scores[k], -k))


def window_chunk(title: str, text: str, best: int, spans: list[tuple[int, int]], size: int, whole_max: int) -> str:
    """Contiguous window of <= size chars grown from passage ``best`` (next, then previous, alternately);
    the whole document when it is <= whole_max chars. The title is kept only when the window starts the text."""
    full = f"{title}\n{text}".strip()
    if len(full) <= whole_max:
        return full
    i = j = min(best, len(spans) - 1)
    grew = True
    while grew:
        grew = False
        for ni, nj in ((i, j + 1), (i - 1, j)):
            if 0 <= ni and nj < len(spans) and spans[nj][1] - spans[ni][0] <= size:
                i, j, grew = ni, nj, True
    start, end = spans[i][0], spans[j][1]
    return f"{title}\n{text[:end]}".strip() if start == 0 else text[start:end].strip()


def section_chunk(title: str, text: str, best: int, spans: list[tuple[int, int]],
                  sections: list[tuple[int, int]], max_chars: int) -> str | None:
    """Section containing passage ``best``, or None when it is absent or longer than ``max_chars``.
    The title is kept only when the section starts the body and the result still fits, so the chunk
    stays verbatim and never exceeds ``max_chars``."""
    if not sections or not spans:
        return None
    pos = spans[min(max(best, 0), len(spans) - 1)][0]
    for a, b in sections:
        if a <= pos < b:
            if b - a > max_chars:
                return None
            if a == 0 and len(title) + 1 + b <= max_chars:
                out = f"{title}\n{text[:b]}".strip()
            else:
                out = text[a:b].strip()
            return out or None
    return None


def chain_order(qids: list[int], sets: dict[int, set]) -> list[int]:
    """Greedy nearest neighbour by shared docs, so a doc's repeats land close together in the stream."""
    left = set(qids)
    order = [qids[0]]
    left.discard(qids[0])
    while left:
        cur = sets.get(order[-1], set())
        nxt = max(left, key=lambda q: (len(cur & sets.get(q, set())), -q))
        order.append(nxt)
        left.discard(nxt)
    return order


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="root holding corpus part-*.jsonl* and query.parquet")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(args.jobs if args.jobs.lstrip().startswith("[") else Path(find(args.jobs)).read_text())
    queries = pd.read_parquet(glob.glob(f"{args.data}/**/query.parquet", recursive=True)[0])
    qids = queries["id"].astype(int).tolist()
    qtext = dict(zip(queries["id"].astype(int), queries["query"]))

    plans = []
    need: set[int] = set()
    for job in jobs:
        fused = fuse(job)
        fused.to_parquet(args.out / f"{job['name']}_ranking.parquet", index=False)
        by_q = doc_lists(job, fused)
        extra = chunk_extra(job, by_q)
        base_len = base_lengths(job, fused)
        n_ch = max(n[1] if isinstance(n, list) else n for n in job["n"])
        k_app = job.get("append", {}).get("chunks", 0)
        need.update(d for ds in by_q.values() for d in ds[:n_ch + 50 + k_app])  # texts where chunks come from
        need.update(d for ds in extra.values() for d in ds)
        plans.append((job, by_q, extra, base_len))
        log(f"fused {job['name']}: {len(fused):,} rows")

    texts: dict[int, str] = {}
    raw: dict[int, tuple[str, str]] = {}  # (title, body) for window chunks
    binary: set[int] = set()  # binary/mis-decoded docs (scorer fails on them): kept in relevant_docs, no chunk_text
    corpus_ids: set[int] = set()
    for part in sorted(glob.glob(f"{args.data}/**/part-*.jsonl*", recursive=True)):
        opener = gzip.open if part.endswith(".gz") else open
        with opener(part, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                corpus_ids.add(d["id"])
                if d["id"] in need and d["id"] not in texts:
                    texts[d["id"]] = f"{d['t']}\n{d['x']}".strip()
                    raw[d["id"]] = (d["t"], d["x"])
                    head = d["x"][:2000]
                    if "\x00" in head or head.count("\ufffd") >= 20:
                        binary.add(d["id"])
    log(f"collected {len(texts):,}/{len(need):,} texts; {len(binary):,} binary (no chunk)")

    best: dict[tuple[int, int], int] = {}
    spec = next((j["chunk"]["best"] for j in jobs if j.get("chunk", {}).get("best")), None)
    if spec:
        bp = pd.read_parquet(find(spec))
        best = {(int(q), int(d)): int(k) for q, d, k in zip(bp["qid"], bp["doc_id"], bp["best_passage"])}
        log(f"best passages: {len(best):,}")

    sections_by_doc: dict[int, list[tuple[int, int]]] = {}
    if any(j.get("chunk", {}).get("mode") == "section" for j in jobs):
        section_paths = glob.glob(f"{args.data}/**/sections.parquet", recursive=True)
        if not section_paths:
            raise FileNotFoundError(f"sections.parquet under {args.data}")
        sec = pd.read_parquet(section_paths[0], columns=["doc_id", "start", "end"])
        sec = sec[sec["doc_id"].isin(need)]
        for d, a, b in zip(sec["doc_id"], sec["start"], sec["end"]):
            sections_by_doc.setdefault(int(d), []).append((int(a), int(b)))
        log(f"sections: {len(sec):,} spans for {len(sections_by_doc):,} docs")

    spans_cache: dict[int, list[tuple[int, int]]] = {}
    n_lexical = 0
    n_section = 0
    n_section_fallback = 0

    def chunk_text(job: dict, q: int, d: int) -> str:
        """Whole document, or with job["chunk"] = {"window": S, "whole_max": W, "best": file} a window around the
        reranker's best passage (lexical fallback). With {"mode": "section"} use the docstore section containing
        that passage, falling back to the window when absent or too long. Every chunk is verbatim document text."""
        nonlocal n_lexical, n_section, n_section_fallback
        cfg = job.get("chunk")
        if not cfg:
            return texts[d]
        title, body = raw[d]
        if d not in spans_cache:
            spans = rr_spans(body)
            ref = rr_split(title, body)
            assert len(ref) == len(spans), d
            for (a, b), r in zip(spans, ref):  # same words as the reranker's passages (newlines may differ)
                assert "".join(r.split()) == "".join(f"{title}\n{body[a:b]}".strip().split()), d
            spans_cache[d] = spans
        spans = spans_cache[d]
        k = best.get((q, d))
        if k is None:
            k = lexical_best(qtext[q], title, body, spans)
            n_lexical += 1
        mode = cfg.get("mode", "window")
        if mode == "section":
            out = section_chunk(title, body, k, spans, sections_by_doc.get(d, []),
                                cfg.get("section_max", 4800))
            if out is None:
                n_section_fallback += 1
                out = window_chunk(title, body, k, spans, cfg.get("window", 4500),
                                   cfg.get("whole_max", cfg.get("window", 4500)))
            else:
                n_section += 1
        elif mode == "window":
            out = window_chunk(title, body, k, spans, cfg["window"], cfg.get("whole_max", cfg["window"]))
        else:
            raise ValueError(f"unknown chunk mode: {mode}")
        assert out in texts[d], (q, d)
        return out

    stats = {}
    for job, by_q, extra, base_len in plans:
        for n in job["n"]:
            counts_before = (n_lexical, n_section, n_section_fallback)
            n_docs, n_chunks = n if isinstance(n, list) else (n, n)
            order = qids
            if job.get("reorder"):
                order = chain_order(qids, {q: set(by_q.get(q, [])[:n_chunks]) for q in qids})
            records = []
            for q in order:
                top = [d for d in by_q.get(q, []) if d in corpus_ids][:n_docs]
                nb = base_len.get(q, len(top)) if base_len is not None else len(top)
                k_app = job.get("append", {}).get("chunks", 0)
                chunk_docs = top[:min(n_chunks, nb)] + top[nb:nb + k_app] + extra.get(q, [])
                records.append({"id": q, "relevant_docs": top,
                                "relevant_chunks": [{"doc_id": d, "chunk_text": chunk_text(job, q, d)}
                                                    for d in chunk_docs if d in texts and d not in binary]})
            tag = f"n{n_docs}" if n_docs == n_chunks else f"d{n_docs}c{n_chunks}"
            path = args.out / f"{job['name']}_{tag}{'r' if job.get('reorder') else ''}.zip"
            write_zip(records, path, job.get("zip", "lzma"))
            mb = round(path.stat().st_size / 1e6, 1)
            stats[path.name] = {"zip_mb": mb,
                                "docs_per_query": round(sum(len(r["relevant_docs"]) for r in records) / len(records), 2),
                                "chunks_per_query": round(sum(len(r["relevant_chunks"]) for r in records) / len(records), 2)}
            stats[path.name]["chunk_chars_median"] = int(pd.Series(
                [len(c["chunk_text"]) for r in records for c in r["relevant_chunks"]] or [0]).median())
            stats[path.name].update({
                "chunk_mode": job.get("chunk", {}).get("mode", "window") if job.get("chunk") else "whole",
                "lexical_fallbacks": n_lexical - counts_before[0],
                "section_chunks": n_section - counts_before[1],
                "section_fallbacks": n_section_fallback - counts_before[2],
            })
            log(f"wrote {path.name} {mb} MB; lexical fallbacks so far {n_lexical:,}")
    (args.out / "build_stats.json").write_text(json.dumps(stats, indent=2))
    log("done")


if __name__ == "__main__":
    main()
