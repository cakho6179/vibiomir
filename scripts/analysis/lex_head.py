"""A clean lexical head: BM25 over the document title (URL slug), stopwords removed, diacritics folded.

Motivation: the official BM25 baseline gets DOCS_PRECISION 0.8805 at ~40 docs/query (32 gold), while every
list we have ever submitted contains only **half** of even our own `bm25_vi_doc` top-40 (the other half sits at
m3rr ranks 200-1200). Our existing BM25 channels (`bm25_vi_doc`, E1v2) index function words, so their heads
are full of off-topic matches ("em"/"con" -> children, "toi"/"khong" -> romance stories). This script builds the
channel the organisers most plausibly used: **title-level lexical matching**.

The title is taken from the URL slug (`docs_meta.url`), which is a de-accented transliteration of the real
title and needs no corpus pass. Query is folded the same way (NFKD -> strip combining marks -> d->d), so
"viêm dạ dày" matches "viem da day".

    python scripts/analysis/lex_head.py --top 200 --out runs/reports/lex_head.parquet
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

# Function words only. Content words that double as pronouns (em, con, be, toi) are KEPT: dropping them
# would erase the subject of half the queries ("con em 8 thang ...").
STOP = set("""
va cua co khong la nhung cac mot cho voi khi thi do nay de trong ra vao di den tu ve bang bi duoc se da
dang cung nhu neu ma the nao gi sao bao nhieu lam rat qua hon phai hay hoac vi boi tai sau truoc giua
minh ban ay ho chung ai day kia roi nua a o u i e y vao ra tren duoi ngoai noi
""".split())


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = s.replace("đ", "d").replace("ð", "d")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s


TOKEN = re.compile(r"[a-z0-9]+")


def tokens(s: str) -> list[str]:
    return [t for t in TOKEN.findall(fold(s)) if t not in STOP and len(t) > 1 and not t.isdigit()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", type=Path, default=ROOT / "data/docstore/v2.1/docs_meta.parquet")
    ap.add_argument("--queries", type=Path, default=ROOT / "data/raw/btc/query.parquet")
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/reports/lex_head.parquet")
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.6)
    args = ap.parse_args()

    meta = pd.read_parquet(args.meta, columns=["id", "url", "lang"])
    q = pd.read_parquet(args.queries)
    q["qtok"] = [tokens(t) for t in q["query"].astype(str)]
    vocab = set(t for ts in q["qtok"] for t in ts)
    print(f"[{time.time()-T0:5.0f}s] {len(meta):,} docs, {len(q)} queries, {len(vocab):,} query terms", flush=True)

    # --- postings over slug tokens, restricted to the query vocabulary ---
    post: dict[str, list[tuple[int, int]]] = defaultdict(list)
    length: dict[int, int] = {}
    ids = meta["id"].astype(int).values
    urls = meta["url"].astype(str).values
    langs = meta["lang"].astype(str).values
    for i in range(len(ids)):
        u = urls[i]
        slug = u.split("?")[0].rstrip("/")
        slug = re.sub(r"\.(html?|htm|php|aspx?|ldo|vov2?|shtml)$", "", slug)
        slug = slug.split("/")[-1]
        toks = [t for t in TOKEN.findall(fold(re.sub(r"[-_+.]+", " ", slug)))][:40]
        length[int(ids[i])] = max(1, len(toks))
        seen: dict[str, int] = {}
        for t in toks:
            if t in vocab:
                seen[t] = seen.get(t, 0) + 1
        for t, tf in seen.items():
            post[t].append((int(ids[i]), tf))
        if i and i % 200000 == 0:
            print(f"[{time.time()-T0:5.0f}s]   {i:,} docs", flush=True)
    n = len(ids)
    avgdl = float(np.mean(list(length.values())))
    print(f"[{time.time()-T0:5.0f}s] postings built: {len(post):,} terms, avgdl {avgdl:.1f}", flush=True)

    rows = []
    for _, r in q.iterrows():
        qid = int(r["id"])
        qt = list(dict.fromkeys(r["qtok"]))
        if not qt:
            continue
        score: dict[int, float] = defaultdict(float)
        for t in qt:
            pl = post.get(t)
            if not pl:
                continue
            df = len(pl)
            idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for did, tf in pl:
                dl = length[did]
                score[did] += idf * (tf * (args.k1 + 1)) / (tf + args.k1 * (1 - args.b + args.b * dl / avgdl))
        top = sorted(score.items(), key=lambda kv: -kv[1])[: args.top]
        for rank, (did, sc) in enumerate(top):
            rows.append((qid, rank, did, float(sc)))
        if qid % 200 == 0:
            print(f"[{time.time()-T0:5.0f}s]   scored {qid}/1200", flush=True)

    out = pd.DataFrame(rows, columns=["qid", "rank", "doc_id", "score"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[{time.time()-T0:5.0f}s] wrote {args.out} ({len(out):,} rows, "
          f"{out.qid.nunique()} queries, {len(out)/max(1,out.qid.nunique()):.0f} per query)")

    # --- a look at the head ---
    url = dict(zip(ids, urls))
    Lang = dict(zip(ids, langs))
    for qid in (1, 33, 100):
        print("=" * 100)
        print(f"Q{qid}: {q.loc[q['id'] == qid, 'query'].iloc[0][:100]}")
        for _, x in out[(out.qid == qid) & (out["rank"] < 8)].iterrows():
            s = re.sub(r"[-_]+", " ", url[x["doc_id"]].split("?")[0].rstrip("/").split("/")[-1])
            print(f"   {x['score']:7.2f} [{Lang[x['doc_id']]}] {s[:88]}")


if __name__ == "__main__":
    main()
