"""(query, best section) pairs for the decision-model gate G0 and for teacher labelling.

    python build_pairs.py --store DIR --ranking FILE --queries FILE --out FILE [--top 150] [--max-chars 1500]

For each query, the first --top docs of --ranking (qid, doc_id, rank). Each doc is cut into its docstore sections
(sections.parquet offsets into the body "x"); the section sharing the most diacritic-free query words wins (ties:
earliest). Pair text = title + section heading + section body, cut to --max-chars at a word boundary.
Output parquet: qid, doc_id, rank, sid, n_sections, overlap, text. Source is ASCII-only.
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

import numpy as np
import pandas as pd

T0 = time.time()
D_FOLD = str.maketrans({chr(0x111): "d", chr(0x110): "d"})
WORD = re.compile(r"\w+")


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def words(text: str) -> set[str]:
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c)).translate(D_FOLD)
    return {w for w in WORD.findall(t) if len(w) > 1}


def cut(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--ranking", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top", type=int, default=150)
    ap.add_argument("--from-rank", type=int, default=0, help="first rank to keep (skip the head already scored)")
    ap.add_argument("--per-query", type=int, default=0,
                    help="keep this many docs per query, evenly spaced over [from-rank, top) instead of all")
    ap.add_argument("--max-chars", type=int, default=1500)
    args = ap.parse_args()

    rk = pd.read_parquet(args.ranking, columns=["qid", "doc_id", "rank"])
    rk = rk[(rk["rank"] < args.top) & (rk["rank"] >= args.from_rank)]
    if args.per_query:  # even spacing keeps the sample proportional to the rank bands (stratified, not uniform)
        rk = rk.sort_values(["qid", "rank"]).reset_index(drop=True)
        pos = rk.groupby("qid").cumcount()
        n = rk.groupby("qid")["rank"].transform("size").clip(lower=1)
        bucket = pos * max(args.per_query - 1, 1) // (n - 1).clip(lower=1)  # 0..k-1, spread over the band
        rk = rk[~pd.DataFrame({"q": rk["qid"], "b": bucket}).duplicated()].reset_index(drop=True)
    q = pd.read_parquet(args.queries)
    qwords = {int(i): words(str(t)) for i, t in zip(q["id"], q["query"])}
    need = set(rk["doc_id"].astype(int))
    log(f"pairs {len(rk):,}; docs {len(need):,}")

    sec = pd.read_parquet(args.store / "sections.parquet")
    sec = sec[sec["doc_id"].isin(need)]
    spans: dict[int, list[tuple[int, int, int, str]]] = defaultdict(list)
    for d, s, a, b, h in zip(sec["doc_id"], sec["sid"], sec["start"], sec["end"], sec["heading"]):
        spans[int(d)].append((int(s), int(a), int(b), h or ""))

    docs: dict[int, tuple[str, str]] = {}
    for part in sorted(glob.glob(str(args.store / "**" / "part-*.jsonl*"), recursive=True)):
        with (gzip.open if part.endswith(".gz") else open)(part, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                i = int(d["id"])
                if i in need:
                    docs[i] = (d.get("t") or "", d.get("x") or "")
    log(f"docs with text {len(docs):,}/{len(need):,}")

    rows = []
    for qid, doc_id, rank in zip(rk["qid"], rk["doc_id"], rk["rank"]):
        title, body = docs.get(int(doc_id), ("", ""))
        cand = spans.get(int(doc_id)) or [(0, 0, len(body), "")]
        qw = qwords.get(int(qid), set())
        best, best_ov = cand[0], -1
        for c in cand:
            ov = len(qw & words(c[3] + " " + body[c[1]:c[2]]))
            if ov > best_ov:
                best, best_ov = c, ov
        sid, a, b, head = best
        text = " | ".join(x for x in (title.strip(), head.strip()) if x) + "\n" + body[a:b].strip()
        rows.append((int(qid), int(doc_id), int(rank), sid, len(cand), best_ov, cut(text, args.max_chars)))
    out = pd.DataFrame(rows, columns=["qid", "doc_id", "rank", "sid", "n_sections", "overlap", "text"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    log(f"wrote {len(out):,} pairs; empty text {int((out['text'].str.len() < 50).sum()):,}; "
        f"median chars {int(out['text'].str.len().median())}")


if __name__ == "__main__":
    main()
