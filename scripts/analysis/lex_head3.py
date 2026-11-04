"""Phrase-first title/slug head: exact medical phrases in the URL slug + document headings.

`lex_head2.py` fuses unigram BM25 over headings and the slug with RRF.  Its failure mode is a query with a
strong medical phrase plus many generic words: Q844 ("noi me day ... buoi chieu/toi/sang") fills the head with
"buoi chieu", "buoi toi" articles because every generic token adds score, while the one phrase that identifies
the condition ("noi me day", 187 slugs) is diluted.

This channel makes the phrase the unit of evidence:

    P  exact n-grams (2..4 content tokens) found in the title/slug, weighted by IDF and length
    U  unigram BM25 over the same title text (the lex_head2 signal, down-weighted)
    C  IDF-weighted coverage of the query's content unigrams

A query that has a rare title phrase requires a rare phrase match; otherwise the channel falls back to U+C, so
queries without a title phrase still get a head.  Output is the same (qid, rank, doc_id, score) contract as
lex_head2, plus `phrase` for inspection.

    python scripts/analysis/lex_head3.py --top 200 --out runs/reports/lex_head3.parquet
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

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir()), Path.cwd())
T0 = time.time()

STOP = set("""
va cua co khong la nhung cac mot cho voi khi thi do nay de trong ra vao di den tu ve bang bi duoc se da
dang cung nhu neu ma the nao gi sao bao nhieu lam rat qua hon phai hay hoac vi boi tai sau truoc giua
minh ban ay ho chung ai day kia roi nua a o u i e y tren duoi ngoai noi
""".split())

TOKEN = re.compile(r"[0-9a-z]+")


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower().replace("\u0111", "d").replace("\u0110", "d")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def toks(s: str) -> list[str]:
    return [t for t in TOKEN.findall(fold(s)) if len(t) > 1 and not t.isdigit() and t not in STOP]


def ngrams(ts: list[str], nmin: int = 2, nmax: int = 4) -> set[str]:
    out: set[str] = set()
    for n in range(nmin, nmax + 1):
        for i in range(len(ts) - n + 1):
            out.add(" ".join(ts[i:i + n]))
    return out


def slug_tokens(url: str) -> list[str]:
    s = str(url).split("?")[0].rstrip("/")
    s = re.sub(r"\.(html?|htm|php|aspx?|ldo|vov2?|shtml)$", "", s, flags=re.I)
    s = s.split("/")[-1]
    return toks(re.sub(r"[-_+.]+", " ", s))[:40]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None,
                    help="docstore root with docs_meta/passages/sections/query.parquet (Kaggle dataset)")
    ap.add_argument("--meta", type=Path, default=ROOT / "data/docstore/v2.1/docs_meta.parquet")
    ap.add_argument("--passages", type=Path, default=ROOT / "data/docstore/v2.1/passages.parquet")
    ap.add_argument("--sections", type=Path, default=ROOT / "data/docstore/v2.1/sections.parquet")
    ap.add_argument("--queries", type=Path, default=ROOT / "data/raw/btc/query.parquet")
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all queries (smoke uses a small number)")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/reports/lex_head3.parquet")
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.6)
    ap.add_argument("--rare-frac", type=float, default=0.005,
                    help="a phrase is rare when its document frequency is below this fraction of the corpus")
    ap.add_argument("--cand-frac", type=float, default=0.05,
                    help="unigrams below this document-frequency fraction form the candidate set")
    ap.add_argument("--gate-min-n", type=int, default=3,
                    help="rare-phrase gate only uses phrases with at least this many tokens")
    args = ap.parse_args()
    if args.data:
        args.meta = args.data / "docs_meta.parquet"
        args.passages = args.data / "passages.parquet"
        args.sections = args.data / "sections.parquet"
        args.queries = args.data / "query.parquet"

    meta = pd.read_parquet(args.meta, columns=["id", "url"])
    ids = meta["id"].astype(int).values
    urls = meta["url"].astype(str).values
    q = pd.read_parquet(args.queries)
    if args.max_queries:
        q = q.head(args.max_queries).copy()

    # Title text per document: headings (diacritics folded) followed by the URL slug.  Headings are the real
    # title text when present; the slug is the 100%-coverage fallback and often repeats the title.
    hd: dict[int, list[str]] = defaultdict(list)
    for path in (args.passages, args.sections):
        df = pd.read_parquet(path, columns=["doc_id", "heading"])
        s = df["heading"].fillna("").astype(str).str.strip()
        df = df[s != ""]
        for did, h in zip(df["doc_id"].astype(int), df["heading"].astype(str)):
            if h.strip():
                hd[did].append(h.strip())
    print(f"[{time.time()-T0:5.0f}s] headings for {len(hd):,} docs", flush=True)

    q_tokens = [toks(t) for t in q["query"].astype(str)]
    q_phrases = [ngrams(ts) for ts in q_tokens]
    vocab_u = {t for ts in q_tokens for t in ts}
    vocab_p = {p for ps in q_phrases for p in ps}
    print(f"[{time.time()-T0:5.0f}s] queries {len(q):,} | unigrams {len(vocab_u):,} | phrases {len(vocab_p):,}",
          flush=True)

    post_u: dict[str, list[tuple[int, int]]] = defaultdict(list)
    post_p: dict[str, list[tuple[int, int]]] = defaultdict(list)
    length: dict[int, int] = {}
    for i, did in enumerate(ids):
        did = int(did)
        ts = []
        for h in hd.get(did, []):
            ts.extend(toks(h))
        ts.extend(slug_tokens(urls[i]))
        if not ts:
            ts = ["_"]
        length[did] = max(1, len(ts))
        seen_u: dict[str, int] = {}
        for t in ts:
            if t in vocab_u:
                seen_u[t] = seen_u.get(t, 0) + 1
        for t, tf in seen_u.items():
            post_u[t].append((did, tf))
        seen_p: dict[str, int] = {}
        for p in ngrams(ts):
            if p in vocab_p:
                seen_p[p] = seen_p.get(p, 0) + 1
        for p, tf in seen_p.items():
            post_p[p].append((did, tf))
        if i and i % 200000 == 0:
            print(f"[{time.time()-T0:5.0f}s]   {i:,} docs", flush=True)

    n = len(ids)
    avgdl = float(np.mean(list(length.values())))
    print(f"[{time.time()-T0:5.0f}s] postings: unigrams {len(post_u):,}, phrases {len(post_p):,}, avgdl {avgdl:.1f}",
          flush=True)

    def bm25(post, t, tf, dl, idf):
        return idf * (tf * (args.k1 + 1)) / (tf + args.k1 * (1 - args.b + args.b * dl / avgdl))

    rows = []
    for qi, row in q.iterrows():
        qid = int(row["id"])
        ut = list(dict.fromkeys(q_tokens[qi]))
        if not ut:
            continue
        idf_u = {}
        for t in ut:
            df = len(post_u.get(t, []))
            idf_u[t] = float(np.log(1.0 + (n - df + 0.5) / (df + 0.5)))
        # Score only the rare query terms.  Common terms ("benh", "dau") have huge postings and add mostly
        # noise; the phrase channel carries the title evidence.  If every term is common, use the 5 rarest.
        rare_u = [t for t in ut if len(post_u.get(t, [])) <= args.cand_frac * n]
        if not rare_u:
            rare_u = sorted(ut, key=lambda t: len(post_u.get(t, [])))[:5]
        total_idf = sum(idf_u[t] for t in rare_u) or 1.0
        score_u: dict[int, float] = defaultdict(float)
        hit_idf: dict[int, float] = defaultdict(float)
        for t in rare_u:
            for did, tf in post_u.get(t, []):
                score_u[did] += bm25(post_u, t, tf, length[did], idf_u[t])
                hit_idf[did] += idf_u[t]
        phrase_sum: dict[int, float] = defaultdict(float)
        phrase_best: dict[int, tuple[float, str]] = {}
        phrase_rare: dict[int, bool] = defaultdict(bool)
        # Rare-phrase gate: when the query contains a phrase that is rare in titles, a document must match at
        # least one rare phrase.  This is what keeps Q844's "buoi chieu" articles out of the head.
        rare = {p for p in dict.fromkeys(q_phrases[qi])
                if (p.count(" ") + 1) >= args.gate_min_n
                and len(post_p.get(p, [])) and len(post_p[p]) <= args.rare_frac * n}
        use_gate = bool(rare)
        for p in dict.fromkeys(q_phrases[qi]):
            pl = post_p.get(p, [])
            if not pl:
                continue
            df = len(pl)
            idf = float(np.log(1.0 + (n - df + 0.5) / (df + 0.5)))
            if idf <= 0:
                continue
            weight = idf * (p.count(" ") + 1)
            for did, tf in pl:
                phrase_sum[did] += weight
                phrase_rare[did] = phrase_rare[did] or p in rare
                old = phrase_best.get(did)
                if old is None or weight > old[0]:
                    phrase_best[did] = (weight, p)
        scored = []
        for did in set(score_u) | set(phrase_sum):
            if use_gate and not phrase_rare.get(did, False):
                continue
            cover = hit_idf.get(did, 0.0) / total_idf
            sc = phrase_sum.get(did, 0.0) + 0.35 * score_u.get(did, 0.0) + 2.0 * cover
            scored.append((sc, did, phrase_best.get(did, (0.0, ""))[1]))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if len(scored) < args.top and not use_gate:
            seen = {d for _, d, _ in scored}
            for t in ut:
                for did, _tf in post_u.get(t, []):
                    if did not in seen:
                        seen.add(did)
                        scored.append((0.0, did, ""))
                        if len(scored) >= args.top:
                            break
                if len(scored) >= args.top:
                    break
        for rank, (sc, did, ph) in enumerate(scored[: args.top]):
            rows.append((qid, rank, did, float(sc), ph))
        if qid % 200 == 0:
            print(f"[{time.time()-T0:5.0f}s]   scored {qid}/1200 (head {len(scored):,}, rare {len(rare)})", flush=True)

    out = pd.DataFrame(rows, columns=["qid", "rank", "doc_id", "score", "phrase"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[{time.time()-T0:5.0f}s] wrote {args.out}: {len(out):,} rows, "
          f"{len(out)/out.qid.nunique():.0f} per query", flush=True)

    # Diagnostics on the queries whose generic-token failure motivated this channel.
    url_by = dict(zip(ids, urls))
    for qid in (1, 33, 100, 137, 500, 844):
        sub = q[q["id"] == qid]
        if sub.empty:
            continue
        print("=" * 100)
        print(f"Q{qid}: {sub['query'].iloc[0][:100]}")
        for _, x in out[(out.qid == qid) & (out["rank"] < 8)].iterrows():
            did = int(x["doc_id"])
            slug = re.sub(r"[-_]+", " ", url_by[did].split("?")[0].rstrip("/").split("/")[-1])
            print(f"   {x['score']:7.3f} | {str(x['phrase'])[:34]:34s} | {slug[:65]}")


if __name__ == "__main__":
    main()
