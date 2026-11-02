"""Decide doc / chunk lists from the reranked 3-language pool; output feeds materialize_lists.py on Kaggle.

    python scripts/analysis/decide_lists.py --job rr197=rr:197:148 [--job ...] --out kaggle/datasets/mat_lists
    python scripts/analysis/decide_lists.py --report   # score distribution only, no lists

The feature files default to the production locations; --pool-file / --rr / --m3m / --judge / --dq / --e1 / --pas
point them at a run's own stage outputs instead (configs/system_v3.yaml does).

Pool = cands of kaggle/datasets/rerank_pool (VI1500/ZH500/EN500 per query) cut to --pool (default vi800,zh200,en200,
the cut measured at R .812 by #7215), scored by bge-reranker-v2-m3 (kaggle/outputs/rr3-p*/rerank.parquet).
Job spec NAME=METHOD:NDOCS:NCHUNKS orders each query's pool by METHOD and lists the first NDOCS docs and the
first NCHUNKS chunks; NAME=METHOD:NDOCS|CMETHOD:NCHUNKS[:SIZE] orders the chunks by CMETHOD instead. SIZE is the
chunk window in chars (default 4500) or fF = max(1500, F x doc length) (gold chunk ~0.7 x doc length,
runs/reports/chunk_geometry); a trailing h (f0.4h) builds ZH chunks from the external open crawl text. Methods:
  rr    reranker score
  rrf   RRF k=60 of the language-list rank and the reranker rank (weights 1:1)
  rr+N    slice probe: reranker order from rank N on
  vi zh en  slice probe: one language, reranker order
  m3all dense sparse colbert  BGE-M3 scores (m3_multi.py); m3all = 0.4 dense + 0.2 sparse + 0.4 colbert
  m3rr  RRF k=60 of the m3all rank and the reranker rank
  judge  LLM relevance judge score (stages/llm_judge.py, kaggle/outputs/judge-p*/scores.parquet)
  jrr   RRF k=60 of the judge rank and the reranker rank
  j3    RRF k=60 of the judge, m3all and reranker ranks
  dq    doc2query max-cosine (stages/dq_embed.py); dqrr = RRF of dq and reranker, d3 = RRF of dq, m3all and reranker
  BASE@vi187+zh10[+enN]  language quota: the top docs of each language by BASE (any method above)
  vitail  slice probe: VI docs at language-list rank >= 300 only, by reranker score
  zhen    slice probe: ZH and EN docs only, alternating ZH / EN by reranker rank in each language
Writes lists.parquet (job, qid, kind, pos, doc_id, size) and prints, per job, the language mix and the overlap with the
docs of #7030 (best docs F2).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BEST = ROOT / "submissions/p1_zh10_v21chunk.zip"


def load_pool(cut: dict[str, int], *, pool_file: Path | None = None, rr: str | None = None,
              m3m: str | None = None, judge: str | None = None, dq: str | None = None,
              e1: Path | None = None, pas: Path | None = None) -> pd.DataFrame:
    """Score the pool with every feature that is available. The path arguments default to the production
    locations (the kaggle/outputs/* of the kernels that built them); a run passes its own stage outputs to read
    what it just computed. Every feature stays optional: an empty glob or a missing file drops that channel."""
    cands = pd.read_parquet(pool_file or ROOT / "kaggle/datasets/rerank_pool/cands.parquet")
    cands["lang"] = cands["set"].str[:2]
    cands = cands[cands["rank"] < cands["lang"].map(cut)]
    parts = sorted(glob.glob(str(rr or ROOT / "kaggle/outputs/rr3-p*/rerank.parquet")))
    rr = pd.concat([pd.read_parquet(p, columns=["qid", "doc_id", "score"]) for p in parts], ignore_index=True)
    pool = cands.merge(rr, on=["qid", "doc_id"], how="left")
    miss = pool["score"].isna().mean()
    print(f"{len(parts)} rerank parts; pool {len(pool):,} pairs, {pool['qid'].nunique()} queries, "
          f"unscored {miss:.2%}", flush=True)
    pool["score"] = pool["score"].fillna(-99.0)
    m3p = sorted(glob.glob(str(m3m or ROOT / "kaggle/outputs/m3m-p*/m3multi.parquet")))
    if m3p:  # BGE-M3 heads (m3_multi.py); m3all = 0.4 dense + 0.2 sparse + 0.4 colbert (BGE-M3 reference weights)
        mm = pd.concat([pd.read_parquet(x, columns=["qid", "doc_id", "dense", "sparse", "colbert"]) for x in m3p])
        pool = pool.merge(mm, on=["qid", "doc_id"], how="left")
        pool["m3all"] = 0.4 * pool["dense"] + 0.2 * pool["sparse"] + 0.4 * pool["colbert"]
        print(f"{len(m3p)} m3 parts; m3 unscored {pool['m3all'].isna().mean():.2%}", flush=True)
    pool["rr_rank"] = pool.groupby("qid")["score"].rank(ascending=False, method="first") - 1
    jp = sorted(glob.glob(str(judge or ROOT / "kaggle/outputs/judge-p*/scores.parquet")))
    if jp:  # LLM relevance judge over the top of this same order (stages/llm_judge.py)
        jj = pd.concat([pd.read_parquet(x, columns=["qid", "doc_id", "judge"]) for x in jp], ignore_index=True)
        pool = pool.merge(jj.drop_duplicates(["qid", "doc_id"]), on=["qid", "doc_id"], how="left")
        pool["judge"] = pool["judge"].fillna(-99.0)
        pool["j_rank"] = pool.groupby("qid")["judge"].rank(ascending=False, method="first") - 1
        print(f"{len(jp)} judge parts; judged {float((pool['judge'] > -99).mean()):.2%} of the pool, "
              f"mean {pool.loc[pool['judge'] > -99, 'judge'].mean():.3f}", flush=True)
    dp = sorted(glob.glob(str(dq or ROOT / "kaggle/outputs/dq-score/scores.parquet")))
    if dp:  # doc2query: max cosine between the query and the generated questions of the doc (stages/dq_embed.py)
        dd = pd.concat([pd.read_parquet(x, columns=["qid", "doc_id", "dq_score"]) for x in dp], ignore_index=True)
        pool = pool.merge(dd.drop_duplicates(["qid", "doc_id"]), on=["qid", "doc_id"], how="left")
        hit = pool["dq_score"].notna()
        pool["dq_score"] = pool["dq_score"].fillna(-1.0)
        pool["dq_rank"] = pool.groupby("qid")["dq_score"].rank(ascending=False, method="first") - 1
        pool.loc[~hit, "dq_rank"] = 1000.0  # uncovered docs sink without stealing ranks from each other
        print(f"{len(dp)} dq parts; covered {hit.mean():.2%} of the pool", flush=True)
    ep = e1 or ROOT / "kaggle/outputs/vibiomir-bm25-e1v2/ranking.parquet"
    if ep.exists():  # BM25 E1v2 (VI, top-200/query) -- the lexical channel of the rd fusion (rdT2 = FINAL 0.3054)
        ee = pd.read_parquet(ep, columns=["qid", "doc_id", "rank"]).rename(columns={"rank": "e1_rank"})
        pool = pool.merge(ee.drop_duplicates(["qid", "doc_id"]), on=["qid", "doc_id"], how="left")
        hit = pool["e1_rank"].notna()
        pool["e1_rank"] = pool["e1_rank"].fillna(1000.0)
        print(f"e1v2; covered {hit.mean():.2%} of the pool", flush=True)
    pp = pas or ROOT / "runs/reports/pool_pas.parquet"
    if pp.exists():  # passage-level reranker aggregation (scripts/analysis/pas_agg.py): pmax = best passage
        pa = pd.read_parquet(pp)
        pool = pool.merge(pa.drop_duplicates(["qid", "doc_id"]), on=["qid", "doc_id"], how="left")
        pool["pmax"] = pool["pmax"].fillna(-99.0)
        pool["p_rank"] = pool.groupby("qid")["pmax"].rank(ascending=False, method="first") - 1
        print(f"pas agg; pmax median {pool['pmax'].median():.3f}", flush=True)
    return pool


def order(pool: pd.DataFrame, method: str) -> pd.DataFrame:
    if "@" in method:  # language quota: BASE@vi187+zh10 = top 187 VI and top 10 ZH, each ranked by BASE within its language
        base, quota = method.split("@")
        want = {q[:2]: int(q[2:]) for q in quota.split("+")}
        pool = order(pool, base)
        lang_rank = pool.groupby(["qid", "lang"]).cumcount()
        pool = pool[lang_rank < pool["lang"].map(want).fillna(0)]
        # interleave the languages in proportion to their quota (the scores are not comparable across languages)
        key = (pool.groupby(["qid", "lang"]).cumcount() + 0.5) / pool["lang"].map(want)
        return pool.assign(key=key).sort_values(["qid", "key"])
    if method == "rr":
        key = -pool["score"]
    elif method == "rrf":
        key = -(1 / (60 + pool["rank"]) + 1 / (60 + pool["rr_rank"]))
    elif method.startswith("rr+"):  # slice: reranker order from rank OFFSET on (rr+197 = ranks 197, 198, ...)
        pool = pool[pool["rr_rank"] >= int(method[3:])]
        key = -pool["score"]
    elif method == "m3rr":  # RRF k=60 of the m3all rank and the reranker rank
        m3_rank = pool.groupby("qid")["m3all"].rank(ascending=False, method="first") - 1
        key = -(1 / (60 + m3_rank) + 1 / (60 + pool["rr_rank"]))
    elif method == "p":  # best passage of the document, reranker score (no doc-level pass at all)
        key = -pool["pmax"]
    elif method == "m3p":  # the r8 recipe with rr3 (doc pass) swapped for pmax (best passage)
        m3_rank = pool.groupby("qid")["m3all"].rank(ascending=False, method="first") - 1
        key = -(1 / (60 + m3_rank) + 1 / (60 + pool["p_rank"]))
    elif method == "m3rp":  # both reranker readings at once: doc pass and best passage
        m3_rank = pool.groupby("qid")["m3all"].rank(ascending=False, method="first") - 1
        key = -(1 / (60 + m3_rank) + 0.5 / (60 + pool["rr_rank"]) + 0.5 / (60 + pool["p_rank"]))
    elif method.startswith("m3pw"):  # m3p with an explicit passage weight (m3pw2 = twice the weight of m3all)
        m3_rank = pool.groupby("qid")["m3all"].rank(ascending=False, method="first") - 1
        key = -(1 / (60 + m3_rank) + float(method[4:]) / (60 + pool["p_rank"]))
    elif method == "rd":  # RRF k=60 of E1v2 (weight 1), m3all (2) and the reranker (3) -- the rd fusion of rdT2
        m3_rank = pool.groupby("qid")["m3all"].rank(ascending=False, method="first") - 1
        key = -(1 / (60 + pool["e1_rank"]) + 2 / (60 + m3_rank) + 3 / (60 + pool["rr_rank"]))
    elif method.startswith("rdw"):  # same with explicit weights, e.g. rdw1-2-2
        a, b, c = (float(x) for x in method[3:].split("-"))
        m3_rank = pool.groupby("qid")["m3all"].rank(ascending=False, method="first") - 1
        key = -(a / (60 + pool["e1_rank"]) + b / (60 + m3_rank) + c / (60 + pool["rr_rank"]))
    elif method == "judge":  # LLM relevance judge, judged docs first (unjudged sink to the bottom)
        key = -pool["judge"]
    elif method == "jrr":  # RRF k=60 of the judge rank and the reranker rank
        key = -(1 / (60 + pool["j_rank"]) + 1 / (60 + pool["rr_rank"]))
    elif method == "j3":  # RRF k=60 of judge, m3all and reranker ranks
        m3_rank = pool.groupby("qid")["m3all"].rank(ascending=False, method="first") - 1
        key = -(1 / (60 + pool["j_rank"]) + 1 / (60 + m3_rank) + 1 / (60 + pool["rr_rank"]))
    elif method == "dq":  # doc2query score (uncovered docs last)
        key = -pool["dq_score"]
    elif method == "dqrr":  # RRF k=60 of the doc2query rank and the reranker rank
        key = -(1 / (60 + pool["dq_rank"]) + 1 / (60 + pool["rr_rank"]))
    elif method == "d3":  # RRF k=60 of doc2query, m3all and reranker ranks
        m3_rank = pool.groupby("qid")["m3all"].rank(ascending=False, method="first") - 1
        key = -(1 / (60 + pool["dq_rank"]) + 1 / (60 + m3_rank) + 1 / (60 + pool["rr_rank"]))
    elif method.startswith("d3w"):  # same, doc2query weighted w (d3w2 = twice the weight of each other channel)
        w = float(method[3:])
        m3_rank = pool.groupby("qid")["m3all"].rank(ascending=False, method="first") - 1
        key = -(w / (60 + pool["dq_rank"]) + 1 / (60 + m3_rank) + 1 / (60 + pool["rr_rank"]))
    elif method.startswith("dqrw"):  # RRF of doc2query (weight w) and the reranker
        w = float(method[4:])
        key = -(w / (60 + pool["dq_rank"]) + 1 / (60 + pool["rr_rank"]))
    elif method in ("m3all", "dense", "sparse", "colbert"):  # BGE-M3 head(s); unscored pairs last
        key = -pool[method].fillna(-99.0)
    elif method in ("vi", "zh", "en"):  # slice: one language, reranker order
        pool = pool[pool["lang"] == method]
        key = -pool["score"]
    elif method == "vitail":  # slice: VI docs at language-list rank >= 300, best reranker score first
        pool = pool[(pool["lang"] == "vi") & (pool["rank"] >= 300)]
        key = -pool["score"]
    elif method == "zhen":  # slice: ZH and EN docs, alternating by reranker rank within each language
        pool = pool[pool["lang"] != "vi"]
        key = pool.groupby(["qid", "lang"])["score"].rank(ascending=False, method="first") + (pool["lang"] == "en") * 0.5
    else:
        raise SystemExit(f"unknown method {method}")
    return pool.assign(key=key).sort_values(["qid", "key"])


def best_docs() -> pd.DataFrame | None:
    """The docs of the best submission so far, for the overlap report; None when that zip is not here."""
    if not BEST.exists():
        return None
    import sys
    sys.path.insert(0, str(ROOT))
    from r2ai2026.core.submission import read_submission_zip
    recs = read_submission_zip(BEST)
    return pd.DataFrame([(int(r["id"]), int(d)) for r in recs for d in r["relevant_docs"]], columns=["qid", "doc_id"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", action="append", default=[])
    ap.add_argument("--pool", default="vi800,zh200,en200")
    ap.add_argument("--out", type=Path, default=ROOT / "kaggle/datasets/mat_lists")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--pool-file", type=Path, default=None, help="pool cands.parquet (default: production)")
    ap.add_argument("--rr", default=None, help="glob of reranker doc-score parts (rr3)")
    ap.add_argument("--m3m", default=None, help="glob of BGE-M3 multi-head parts (m3_multi.py)")
    ap.add_argument("--judge", default=None, help="glob of LLM judge score parts (stages/llm_judge.py)")
    ap.add_argument("--dq", default=None, help="glob of doc2query score parts (stages/dq_embed.py)")
    ap.add_argument("--e1", type=Path, default=None, help="BM25 E1v2 ranking.parquet (the rd fusion channel)")
    ap.add_argument("--pas", type=Path, default=None, help="pool_pas.parquet (passage-level reranker aggregation)")
    args = ap.parse_args()
    cut = {s[:2]: int(s[2:]) for s in args.pool.split(",")}
    pool = load_pool(cut, pool_file=args.pool_file, rr=args.rr, m3m=args.m3m, judge=args.judge, dq=args.dq,
                     e1=args.e1, pas=args.pas)
    q = pool.groupby("lang")["score"].describe(percentiles=[.5, .9, .99])
    print(q.round(2).to_string())
    top = pool[pool["rr_rank"] < 197]
    print("reranker top-197 language mix per query:", (top.groupby("lang").size() / pool["qid"].nunique()).round(1).to_dict())
    if args.report:
        return
    best = best_docs()
    best = best.assign(in_best=1) if best is not None else None
    n_chars = pd.read_parquet(ROOT / "data/docstore/v2.1/docs_meta.parquet", columns=["id", "n_chars"]).set_index("id")["n_chars"]
    nq = pool["qid"].nunique()
    rows = []
    for spec in args.job:
        name, rest = spec.split("=")
        dspec, cspec = rest.split("|") if "|" in rest else (rest, None)
        method, nd, *nc = dspec.split(":")
        cparts = cspec.split(":") if cspec else [method, nc[0] if nc else nd]
        cmethod, ncn, size = cparts[0], int(cparts[1]), cparts[2] if len(cparts) > 2 else "4500"
        cmode = ""  # chunker variant, carried in the src column: "+stair" = section boundary, "+seed" = reranker seed
        for tag in ("stair", "seed"):
            if size.endswith("+" + tag):
                size, cmode = size[: -len(tag) - 1], tag
        o = order(pool, method)
        o["pos"] = o.groupby("qid").cumcount()
        docs = o[o["pos"] < int(nd)]
        oc = order(pool, cmethod)
        oc["pos"] = oc.groupby("qid").cumcount()
        chunks = oc[oc["pos"] < ncn].copy()
        L = chunks["doc_id"].map(n_chars).fillna(4500)
        hf = size.endswith("h")  # ZH chunks from the external open crawl text (materialize_lists.py --alt-text)
        size = size.rstrip("h")
        chunks["size"] = (np.maximum(1500, np.ceil(float(size[1:]) * L)) if size.startswith("f") else L * 0 + int(size)).astype(int)
        chunks["src"] = cmode or np.where(hf & (chunks["lang"] == "zh"), "hf", "")
        rows += [docs.assign(job=name, kind="doc", size=0, src=""), chunks.assign(job=name, kind="chunk")]
        shared = "n/a" if best is None else \
            f"{docs.merge(best, on=['qid', 'doc_id'], how='left')['in_best'].sum() / nq:.1f}"
        est = np.minimum(chunks["size"], L).sum() / nq
        print(f"{name}: {len(docs) / nq:.0f} docs/q, mix {(docs.groupby('lang').size() / nq).round(1).to_dict()}, "
              f"shared with #7030 {shared}/q; chunks {cmethod} {ncn}/q size {size}, "
              f"~{est:,.0f} chars/q", flush=True)
    out = pd.concat(rows)[["job", "qid", "kind", "pos", "doc_id", "size", "src"]]
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out / "lists.parquet", index=False)
    print(f"-> {args.out / 'lists.parquet'}: {len(out):,} rows")


if __name__ == "__main__":
    main()
