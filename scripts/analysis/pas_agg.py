"""Score a document by its BEST passage, not by one truncated reranker pass over the whole document.

The pool already carries `runs/pas_scores_rr4.parquet` -- 4.94 M rows, 1200 queries x 1200 docs x 3.43
passages, i.e. bge-reranker-v2-m3 read every passage of every pool document. The doc-level `rr3` score used by
`m3rr` is a separate, single pass over the (truncated) document, so `max over passages` is genuinely new
information, not a re-reading of the same number.

Three aggregations are compared, plus their rank fusion with the channel that currently sets the ordering:

    pmax  = max_p s(q,p)                      a document is relevant if ONE passage answers the question
    ptop2 = mean of the two best passages      needs the evidence to be more than a single lucky sentence
    pmix  = pmax + 0.25 * ptop2                a compromise

Reports, per aggregation: how many of the m3rr top-197 it keeps, and how many docs it would add -- the only
number that matters, because docsF2 is a set metric and reordering inside the top-197 cannot move it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from decide_lists import load_pool  # noqa: E402


def rrf(rank_a: pd.Series, rank_b: pd.Series, k: int = 60, w: tuple[float, float] = (1.0, 1.0)) -> pd.Series:
    return -(w[0] / (k + rank_a) + w[1] / (k + rank_b))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pas", type=Path, default=ROOT / "runs/pas_scores_rr4.parquet")
    ap.add_argument("--n", type=int, default=197)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/reports/pool_pas.parquet")
    ap.add_argument("--pool", default="vi800,zh200,en200", help="per-language pool cut, same syntax as decide_lists")
    ap.add_argument("--pool-file", type=Path, default=None, help="pool cands.parquet (default: production)")
    ap.add_argument("--rr", default=None, help="glob of reranker doc-score parts (rr3)")
    ap.add_argument("--m3m", default=None, help="glob of BGE-M3 multi-head parts (m3_multi.py)")
    args = ap.parse_args()

    cut = {s[:2]: int(s[2:]) for s in args.pool.split(",")}
    pool = load_pool(cut, pool_file=args.pool_file, rr=args.rr, m3m=args.m3m)
    pas = pd.read_parquet(args.pas)
    pas = pas.sort_values(["qid", "doc_id", "score"], ascending=[True, True, False])
    g = pas.groupby(["qid", "doc_id"], sort=False)["score"]
    agg = pd.DataFrame({"pmax": g.max(), "ptop2": g.apply(lambda s: s.iloc[:2].mean())}).reset_index()
    # load_pool may have merged an aggregate that already exists (its own default output): this script rebuilds it
    # from --pas, so drop any previous pmax/ptop2/pmix first or the merge suffixes them into _x / _y.
    pool = pool.drop(columns=[c for c in ("pmax", "ptop2", "pmix") if c in pool], errors="ignore")
    pool = pool.merge(agg, on=["qid", "doc_id"], how="left")
    print(f"passage coverage {pool['pmax'].notna().mean():.2%} of the pool", flush=True)
    pool[["pmax", "ptop2"]] = pool[["pmax", "ptop2"]].fillna(-99.0)
    pool["pmix"] = pool["pmax"] + 0.25 * pool["ptop2"]

    base = pool[pool["m3all"].notna()].copy()
    base["m3_rank"] = base.groupby("qid")["m3all"].rank(ascending=False, method="first") - 1
    top = base.sort_values(["qid", "m3_rank"]).groupby("qid").head(args.n)
    ref = set(zip(top["qid"], top["doc_id"]))
    nq = base["qid"].nunique()
    print(f"m3rr reference: {len(top):,} pairs ({args.n}/query, {nq} queries)", flush=True)

    for col in ["pmax", "ptop2", "pmix"]:
        r = base.groupby("qid")[col].rank(ascending=False, method="first") - 1
        sel = base.assign(r=r).sort_values(["qid", "r"]).groupby("qid").head(args.n)
        keep = len(ref & set(zip(sel["qid"], sel["doc_id"])))
        print(f"  {col:6s} alone : keeps {keep / len(ref):.1%} of the m3rr top-{args.n}, "
              f"swaps {args.n - keep / nq:.1f} docs/query", flush=True)

    m3_rank = base["m3_rank"]
    for col, w in [("pmax", (1.0, 1.0)), ("pmax", (1.0, 2.0)), ("ptop2", (1.0, 1.0)), ("pmix", (1.0, 1.0))]:
        pr = base.groupby("qid")[col].rank(ascending=False, method="first") - 1
        key = rrf(m3_rank, pr, w=w)
        sel = base.assign(k=key).sort_values(["qid", "k"]).groupby("qid").head(args.n)
        keep = len(ref & set(zip(sel["qid"], sel["doc_id"])))
        print(f"  rrf(m3,{col},{w[1]:.0f}) : keeps {keep / len(ref):.1%}, swaps {args.n - keep / nq:.1f} docs/query",
              flush=True)

    pool[["qid", "doc_id", "pmax", "ptop2", "pmix"]].to_parquet(args.out, index=False)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
