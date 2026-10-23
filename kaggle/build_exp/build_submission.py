"""Build submission zips on Kaggle from small ranking files (full-text chunks, LZMA, < 100 MB).

    python build_submission.py --data DIR --jobs jobs.json --out DIR      (or --jobs '<inline JSON>')

jobs: [{"name": "...", "systems": ["a.parquet+b.parquet", "c.parquet"], "weights": [1, 2],
             "n": [28, [150, 100]], "vi_only": true, "zip": "lzma"|"deflate", "reorder": false}, ...]
An entry of "n" is N (N docs, N whole-doc chunks) or [N_docs, N_chunks]: relevant_docs is a separate list,
so doc depth costs almost no bytes; chunks come only from the top of relevant_docs.
"cut": {"spec": "rerank_r1.parquet", "tau": -2.0, "min": 20} keeps, in fused order, only docs whose score in
that ranking is >= tau (at least "min" docs), so the doc count varies per query; N_docs is then a cap.
"reorder": write queries in a greedy chain of overlapping doc sets so repeats sit inside the LZMA window.
A system is one or more ranking parquet files (qid, doc_id, score; found anywhere under /kaggle/input)
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
    if Path(name).exists():
        return name
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


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
    qids = pd.read_parquet(glob.glob(f"{args.data}/**/query.parquet", recursive=True)[0])["id"].astype(int).tolist()

    plans = []
    need: set[int] = set()
    for job in jobs:
        fused = fuse(job)
        fused.to_parquet(args.out / f"{job['name']}_ranking.parquet", index=False)
        depth = max(max(n) if isinstance(n, list) else n for n in job["n"])
        top = fused[fused["rank"] < depth]
        by_q = {int(q): [int(d) for d in g["doc_id"]] for q, g in top.groupby("qid")}
        if job.get("cut"):
            cut = job["cut"]
            sc = load_system(cut["spec"], 10 ** 6, False)
            ok = {(int(q), int(d)) for q, d, v in zip(sc["qid"], sc["doc_id"], sc["score"]) if v >= cut["tau"]}
            lo = cut.get("min", 0)
            by_q = {q: ds[:lo] + [d for d in ds[lo:] if (q, d) in ok] for q, ds in by_q.items()}
        need.update(int(d) for d in top["doc_id"])
        plans.append((job, by_q))
        log(f"fused {job['name']}: {len(fused):,} rows")

    texts: dict[int, str] = {}
    binary: set[int] = set()  # binary/mis-decoded docs (scorer fails on them): kept in relevant_docs, no chunk_text
    for part in sorted(glob.glob(f"{args.data}/**/part-*.jsonl*", recursive=True)):
        opener = gzip.open if part.endswith(".gz") else open
        with opener(part, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["id"] in need and d["id"] not in texts:
                    texts[d["id"]] = f"{d['t']}\n{d['x']}".strip()
                    head = d["x"][:2000]
                    if "\x00" in head or head.count("�") >= 20:
                        binary.add(d["id"])
    log(f"collected {len(texts):,}/{len(need):,} texts; {len(binary):,} binary (no chunk)")

    stats = {}
    for job, by_q in plans:
        for n in job["n"]:
            n_docs, n_chunks = n if isinstance(n, list) else (n, n)
            order = qids
            if job.get("reorder"):
                order = chain_order(qids, {q: set(by_q.get(q, [])[:n_chunks]) for q in qids})
            records = []
            for q in order:
                top = [d for d in by_q.get(q, []) if d in texts][:n_docs]
                records.append({"id": q, "relevant_docs": top,
                                "relevant_chunks": [{"doc_id": d, "chunk_text": texts[d]} for d in top[:n_chunks]
                                                    if d not in binary]})
            tag = f"n{n_docs}" if n_docs == n_chunks else f"d{n_docs}c{n_chunks}"
            path = args.out / f"{job['name']}_{tag}{'r' if job.get('reorder') else ''}.zip"
            write_zip(records, path, job.get("zip", "lzma"))
            mb = round(path.stat().st_size / 1e6, 1)
            stats[path.name] = {"zip_mb": mb,
                                "docs_per_query": round(sum(len(r["relevant_docs"]) for r in records) / len(records), 2),
                                "chunks_per_query": round(sum(len(r["relevant_chunks"]) for r in records) / len(records), 2)}
            log(f"wrote {path.name} {mb} MB")
    (args.out / "build_stats.json").write_text(json.dumps(stats, indent=2))
    log("done")


if __name__ == "__main__":
    main()
