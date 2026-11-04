"""Lexical head v2: BM25 over the document's own headings (diacritics kept) fused with the URL slug (folded).

v1 (`lex_head.py`) used the slug only: the slug is a de-accented transliteration, so "sot" matches both "sốt"
(fever) and "sót" (leftover) and Q100's head filled up with showbiz stories. Passages and sections carry the
real heading text with diacritics for 42 % / 30 % of the corpus, so:

    channel H  BM25 over the concatenated headings of the document   (diacritics kept -> no homographs)
    channel U  BM25 over the URL slug                                (folded, 100 % coverage)
    fused      RRF k=60, so the two scales never need calibrating

The point: the official BM25 baseline reaches DOCS_PRECISION 0.88 at ~40 docs/query, and a clean lexical head
is the only system we know that could reproduce it. v1's head was 57 % OUTSIDE our 1200-doc pool, so this is
also a pool expansion: those documents have never been scored by rr3/rr4/m3m.

    python scripts/analysis/lex_head2.py --top 200 --out runs/reports/lex_head2.parquet
"""
from __future__ import annotations

import argparse
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
T0 = time.time()

STOP = set("""
va cua co khong la nhung cac mot cho voi khi thi do nay de trong ra vao di den tu ve bang bi duoc se da
dang cung nhu neu ma the nao gi sao bao nhieu lam rat qua hon phai hay hoac vi boi tai sau truoc giua
minh ban ay ho chung ai day kia roi nua a o u i e y tren duoi ngoai noi
""".split())

TOKEN = re.compile(r"[0-9a-zà-ỹ]+", re.IGNORECASE)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower().replace("đ", "d")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def toks(s: str, folded: bool) -> list[str]:
    s = fold(s) if folded else unicodedata.normalize("NFKC", str(s)).lower()
    out = []
    for t in TOKEN.findall(s):
        if len(t) > 1 and not t.isdigit() and fold(t) not in STOP:
            out.append(t)
    return out


def build(units: list[tuple[int, list[str]]], vocab: set[str], k1: float, b: float, name: str):
    post: dict[str, list[tuple[int, int]]] = defaultdict(list)
    length: dict[int, int] = {}
    for did, ts in units:
        length[did] = max(1, len(ts))
        seen: dict[str, int] = {}
        for t in ts:
            if t in vocab:
                seen[t] = seen.get(t, 0) + 1
        for t, tf in seen.items():
            post[t].append((did, tf))
    avgdl = float(np.mean(list(length.values()))) if length else 1.0
    print(f"[{time.time()-T0:5.0f}s] {name}: {len(length):,} units, {len(post):,} query-terms, avgdl {avgdl:.1f}",
          flush=True)
    return post, length, avgdl


def score_units(post, length, avgdl, qtoks, n, k1, b, top):
    idf_cache = {}
    out: dict[int, float] = defaultdict(float)
    for t in qtoks:
        pl = post.get(t)
        if not pl:
            continue
        if t not in idf_cache:
            df = len(pl)
            idf_cache[t] = float(np.log(1.0 + (n - df + 0.5) / (df + 0.5)))
        idf = idf_cache[t]
        for did, tf in pl:
            dl = length[did]
            out[did] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    return sorted(out.items(), key=lambda kv: -kv[1])[:top]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", type=Path, default=ROOT / "data/docstore/v2.1/docs_meta.parquet")
    ap.add_argument("--passages", type=Path, default=ROOT / "data/docstore/v2.1/passages.parquet")
    ap.add_argument("--sections", type=Path, default=ROOT / "data/docstore/v2.1/sections.parquet")
    ap.add_argument("--queries", type=Path, default=ROOT / "data/raw/btc/query.parquet")
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/reports/lex_head2.parquet")
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.6)
    args = ap.parse_args()

    meta = pd.read_parquet(args.meta, columns=["id", "url"])
    ids = meta["id"].astype(int).values
    urls = meta["url"].astype(str).values
    q = pd.read_parquet(args.queries)

    # query tokens: heading-side keeps diacritics, slug-side is folded
    qh = [toks(t, folded=False) for t in q["query"].astype(str)]
    qf = [toks(t, folded=True) for t in q["query"].astype(str)]
    vocab_h = set(t for ts in qh for t in ts)
    vocab_u = set(t for ts in qf for t in ts)
    print(f"[{time.time()-T0:5.0f}s] {len(ids):,} docs | heading-vocab {len(vocab_h):,} | slug-vocab {len(vocab_u):,}",
          flush=True)

    # headings per doc
    hd: dict[int, list[str]] = defaultdict(list)
    for path, col in ((args.passages, "heading"), (args.sections, "heading")):
        df = pd.read_parquet(path, columns=["doc_id", col])
        s = df[col].fillna("").astype(str).str.strip()
        df = df[s != ""]
        for did, h in zip(df["doc_id"].astype(int), df[col].astype(str)):
            if h.strip():
                hd[did].append(h.strip())
    units_h = [(d, toks(" . ".join(dict.fromkeys(hs)), folded=False)) for d, hs in hd.items()]
    units_h = [(d, ts) for d, ts in units_h if ts]
    post_h, len_h, avg_h = build(units_h, vocab_h, args.k1, args.b, "headings")

    units_u = []
    for i, u in enumerate(urls):
        slug = u.split("?")[0].rstrip("/").split("/")[-1]
        ts = toks(re.sub(r"[-_+.]+", " ", slug), folded=True)[:40]
        units_u.append((int(ids[i]), ts))
    post_u, len_u, avg_u = build(units_u, vocab_u, args.k1, args.b, "slugs")

    K = 60.0
    rows = []
    for i, row in q.iterrows():
        qid = int(row["id"])
        h = score_units(post_h, len_h, avg_h, list(dict.fromkeys(qh[i])), len(len_h), args.k1, args.b, args.top)
        u = score_units(post_u, len_u, avg_u, list(dict.fromkeys(qf[i])), len(len_u), args.k1, args.b, args.top)
        rank: dict[int, float] = defaultdict(float)
        for r, (did, _) in enumerate(h):
            rank[did] += 1.0 / (K + r)
        for r, (did, _) in enumerate(u):
            rank[did] += 1.0 / (K + r)
        for r, (did, sc) in enumerate(sorted(rank.items(), key=lambda kv: -kv[1])[: args.top]):
            rows.append((qid, r, did, float(sc)))
        if qid % 200 == 0:
            print(f"[{time.time()-T0:5.0f}s]   scored {qid}/1200", flush=True)

    out = pd.DataFrame(rows, columns=["qid", "rank", "doc_id", "score"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[{time.time()-T0:5.0f}s] wrote {args.out}: {len(out):,} rows, "
          f"{len(out)/out.qid.nunique():.0f} per query", flush=True)

    url_by = dict(zip(ids, urls))
    for qid in (1, 33, 100):
        print("=" * 100)
        print(f"Q{qid}: {q.loc[q['id'] == qid, 'query'].iloc[0][:100]}")
        for _, x in out[(out.qid == qid) & (out["rank"] < 8)].iterrows():
            did = x["doc_id"]
            title = (hd.get(did) or ["<no heading>"])[0]
            s = re.sub(r"[-_]+", " ", url_by[did].split("?")[0].rstrip("/").split("/")[-1])
            print(f"   {x['score']:.5f} | {title[:60]:<60} | {s[:40]}")


if __name__ == "__main__":
    main()
