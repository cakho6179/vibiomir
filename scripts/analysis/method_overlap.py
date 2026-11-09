"""Free prefilter for retrieval methods: how many NEW docs each method's top-K brings, and how alike they are.

    python scripts/analysis/method_overlap.py --base submissions/p1_zh10_v21chunk.zip --k 200 --out runs/reports/method_overlap

A method is one ranked list per query: one index x one query-variant family (multi-row families such as
sub0..sub4 are fused by RRF k=60 first). "New" = not in the base submission's docs and not in pool5000 top-300.
No labels are used; the counts only decide which methods are worth a probe (one upload each).
Outputs methods.csv (per method: new docs/query by language, overlap with base) and pairs.csv (mean Jaccard of
the new-doc sets per query, for every pair of methods). --additions FILE also writes every method's top-K as
probe sets (qid, set, rank, doc_id; set = method name) for scripts/kaggle_ops/augment_probe.py.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from r2ai2026.core.submission import read_submission_zip  # noqa: E402

R = ROOT / "data" / "rankings_v21"
Q = ROOT / "kaggle" / "outputs"
SUBS = ["sub0", "sub1", "sub2", "sub3", "sub4"]
E5 = [Q / f"qv3-e5-{s}/ranking.parquet" for s in "abcd"]  # multilingual-e5-large-instruct, 4 VI shards
# name: (ranking files, variants to fuse; None = file has no variant column)
METHODS = {
    "D_orig": ([R / "m3_vi.parquet"], None),
    "D_kw": ([Q / "qv3-m3-vi-ab/ranking.parquet", Q / "qv3-m3-vi-cd/ranking.parquet"], ["kw"]),
    "D_sub": ([Q / "qv3-m3-vi-ab/ranking.parquet", Q / "qv3-m3-vi-cd/ranking.parquet"], SUBS),
    "D_hyde": ([Q / "qv3-m3-vi-ab/ranking.parquet", Q / "qv3-m3-vi-cd/ranking.parquet"], ["hyde"]),
    "E_orig": (E5, ["orig"]),
    "E_kw": (E5, ["kw"]),
    "E_sub": (E5, SUBS),
    "E_hyde": (E5, ["hyde"]),
    "Bd_orig": ([R / "bm25_vi_doc.parquet"], None),
    "Bd_kw": ([Q / "qv3-bm25-vi-doc/ranking.parquet"], ["kw", "acro"]),
    "Bd_sub": ([Q / "qv3-bm25-vi-doc/ranking.parquet"], SUBS),
    "Bs_orig": ([R / "bm25_vi_sec.parquet"], None),
    "Bs_kw": ([Q / "qv3-bm25-vi-sec/ranking.parquet"], ["kw", "acro"]),
    "Bs_sub": ([Q / "qv3-bm25-vi-sec/ranking.parquet"], SUBS),
    "Dz_vi": ([R / "m3_zh.parquet"], None),
    "Dz_q": ([Q / "qv3-m3-zh/ranking.parquet"], ["zh_q"]),
    "Dz_kw": ([Q / "qv3-m3-zh/ranking.parquet"], ["zh_kw"]),
    "Bz_q": ([Q / "qv3-bm25-zh/ranking.parquet"], ["zh_q"]),
    "Bz_kw": ([Q / "qv3-bm25-zh/ranking.parquet"], ["zh_kw"]),
    "De_vi": ([R / "m3_en.parquet"], None),
    "Be_vi": ([R / "bm25_en_viquery.parquet"], None),
}
VI_MAX, ZH_MAX = 766441, 4392214


def lang(d: int) -> str:
    return "vi" if d <= VI_MAX else "zh" if d <= ZH_MAX else "en"


def method_lists(files: list[Path], variants: list[str] | None, k: int) -> dict[int, list[int]]:
    """Top-k docs per query; shards are joined by score, variant families fused by RRF (k=60)."""
    r = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if variants is None:
        r = r.sort_values(["qid", "rank"])
        return {int(q): g["doc_id"].astype(int).tolist()[:k] for q, g in r.groupby("qid")}
    r = r[r["variant"].isin(variants)]
    # rank inside each (qid, variant) after joining shards (dense shards each restart at rank 0)
    r = r.sort_values(["qid", "variant", "score"], ascending=[True, True, False])
    r["rank"] = r.groupby(["qid", "variant"]).cumcount()
    r = r[r["rank"] < 1000]
    r["rrf"] = 1.0 / (60 + r["rank"])
    s = r.groupby(["qid", "doc_id"], as_index=False)["rrf"].sum().sort_values(["qid", "rrf"], ascending=[True, False])
    return {int(q): g["doc_id"].astype(int).tolist()[:k] for q, g in s.groupby("qid")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--pool-depth", type=int, default=300)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--additions", type=Path, default=None, help="write method top-K lists as probe sets")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base = {int(r["id"]): set(map(int, r["relevant_docs"])) for r in read_submission_zip(args.base)}
    pool = pd.read_parquet(R / "pool5000_vi.parquet", columns=["qid", "doc_id", "rank"])
    pool = pool[pool["rank"] < args.pool_depth].groupby("qid")["doc_id"].apply(lambda s: set(map(int, s)))
    known = {q: base[q] | pool.get(q, set()) for q in base}
    qids = sorted(base)

    rows, new_sets, sets = [], {}, []
    for name, (files, variants) in METHODS.items():
        missing = [f for f in files if not f.exists()]
        if missing:
            print(f"skip {name}: missing {missing}")
            continue
        lists = method_lists(files, variants, args.k)
        if args.additions:
            sets += [(q, name, r, d) for q, docs in lists.items() for r, d in enumerate(docs)]
        new = {q: [d for d in lists.get(q, []) if d not in known[q]] for q in qids}
        new_sets[name] = {q: set(v) for q, v in new.items()}
        n = len(qids)
        row = {"method": name, "queries_with_list": sum(1 for q in qids if lists.get(q)),
               "listed": sum(len(lists.get(q, [])) for q in qids) / n,
               "in_base": sum(len(set(lists.get(q, [])) & base[q]) for q in qids) / n,
               "new": sum(len(v) for v in new.values()) / n}
        for lg in ("vi", "zh", "en"):
            row[f"new_{lg}"] = sum(sum(lang(d) == lg for d in v) for v in new.values()) / n
        rows.append(row)
        print(json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in row.items()}))
    pd.DataFrame(rows).round(2).to_csv(args.out / "methods.csv", index=False)
    if args.additions:
        pd.DataFrame(sets, columns=["qid", "set", "rank", "doc_id"]).to_parquet(args.additions, index=False)
        print(f"wrote {len(sets):,} rows to {args.additions}")

    pairs = []
    for a, b in itertools.combinations(new_sets, 2):
        jac = [len(new_sets[a][q] & new_sets[b][q]) / max(len(new_sets[a][q] | new_sets[b][q]), 1) for q in qids]
        pairs.append({"a": a, "b": b, "jaccard_new": round(sum(jac) / len(jac), 3)})
    pd.DataFrame(pairs).sort_values("jaccard_new", ascending=False).to_csv(args.out / "pairs.csv", index=False)
    union = {q: set().union(*(s[q] for s in new_sets.values())) for q in qids}
    print("union of new docs per query:", round(sum(map(len, union.values())) / len(qids), 1))


if __name__ == "__main__":
    main()
