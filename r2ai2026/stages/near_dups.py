"""Near-duplicate clusters of Vietnamese docs (same article republished under several ids / hosts).

    python near_dups.py --data DIR --out DIR [--limit N] [--jaccard 0.8]

Candidates: docs whose normalised title (lowercase, NFC, letters/digits only, >= 15 chars) is equal, plus exact
content_key twins from docs_meta.parquet. A candidate pair joins a cluster when the Jaccard of 5-word shingles over
the first 1500 body chars is >= --jaccard (title twins) or always (content_key twins). Union-find over pairs.
Output: clusters.parquet (doc_id, cluster_id = smallest id in the cluster, size) for docs in clusters of size >= 2,
and stats.json. Used to score a cluster once and to expand every id of a selected cluster in relevant_docs.
Source is ASCII-only.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

VI_MAX_ID = 766441
HEAD_CHARS = 1500
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFC", t or "").lower()
    return " ".join(re.findall(r"\w+", t))


def shingles(text: str, k: int = 5) -> set:
    w = re.findall(r"\w+", unicodedata.normalize("NFC", text).lower())
    return {hash(" ".join(w[i:i + k])) for i in range(max(1, len(w) - k + 1))}


class DSU:
    def __init__(self):
        self.p: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jaccard", type=float, default=0.8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    by_title: dict[str, list[int]] = defaultdict(list)
    head: dict[int, str] = {}
    n = 0
    for part in sorted(glob.glob(str(args.data / "**" / "part-*.jsonl*"), recursive=True)):
        with (gzip.open if part.endswith(".gz") else open)(part, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["id"] > VI_MAX_ID:
                    continue
                key = norm_title(d["t"])
                if len(key) >= 15:
                    by_title[key].append(d["id"])
                    head[d["id"]] = (d["x"] or "")[:HEAD_CHARS]
                n += 1
                if args.limit and n >= args.limit:
                    break
        if args.limit and n >= args.limit:
            break
    log(f"docs {n:,}; titled {len(head):,}; title groups >1: {sum(len(v) > 1 for v in by_title.values()):,}")

    dsu = DSU()
    n_title_pairs = 0
    for ids in by_title.values():
        if len(ids) < 2 or len(ids) > 200:  # generic titles ("Tin tuc") are not one article
            continue
        sh = {i: shingles(head[i]) for i in ids}
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                x, y = sh[ids[a]], sh[ids[b]]
                if x and y and len(x & y) / len(x | y) >= args.jaccard:
                    dsu.union(ids[a], ids[b])
                    n_title_pairs += 1
    meta = pd.read_parquet(glob.glob(str(args.data / "**" / "docs_meta.parquet"), recursive=True)[0],
                           columns=["id", "content_key"])
    meta = meta[(meta["id"] <= VI_MAX_ID) & meta["content_key"].notna()]
    n_exact = 0
    for _, g in meta.groupby("content_key"):
        ids = g["id"].tolist()
        for other in ids[1:]:
            dsu.union(ids[0], other)
            n_exact += 1
    rows = [(d, dsu.find(d)) for d in list(dsu.p)]
    cl = pd.DataFrame(rows, columns=["doc_id", "cluster_id"])
    cl["size"] = cl.groupby("cluster_id")["doc_id"].transform("size")
    cl = cl[cl["size"] >= 2].sort_values(["cluster_id", "doc_id"])
    cl.to_parquet(args.out / "clusters.parquet", index=False)
    stats = {"docs": n, "title_pairs": n_title_pairs, "exact_pairs": n_exact, "docs_in_clusters": len(cl),
             "clusters": int(cl["cluster_id"].nunique()), "max_size": int(cl["size"].max()) if len(cl) else 0,
             "seconds": round(time.time() - T0)}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    log(json.dumps(stats))


if __name__ == "__main__":
    main()
