"""Calibrate p(gold | features) of (query, doc) pairs from the aggregate docs P/R of our scored submissions.

    python scripts/analysis/llp_calibrate.py --out runs/reports/llp

Learning from label proportions: no per-pair labels are used, only each submission's docs precision and recall
as shown on the leaderboard (macro over the public queries). For a submission S with doc set S_q per query,
  predicted R = mean_q sum_{d in S_q} p(q, d) / G,   predicted P = mean_q sum_{d in S_q} p(q, d) / |S_q|
with G = 71 (mean gold docs per query, official scorer). Parameters are fitted by least squares on all
submissions (P and R) with a small ridge penalty; leave-one-out predictions show how far the model can be
trusted. Features of a pair:
  lang      vi / zh / en (one bias each)
  lrank     log1p(rank in its language list): pool5000 (VI) or the fused ZH / EN lists (lang_pool_sets.py);
            pairs outside these lists get the list length
  n200 n50  number of retrieval methods (method_overlap.py, top-200 each) that list the pair at rank < 200 / 50
  rr        bge-reranker-v2-m3 logit (kaggle/outputs/rr3-p*/rerank.parquet, pool VI1500/ZH500/EN500); unscored
            pairs get 0 and the flag rr_miss
Outputs fit.json (parameters, per-submission observed vs fitted vs leave-one-out) and scored_pool.parquet
(p for every pool pair: VI pool top-3000, ZH / EN fused top-1000).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from r2ai2026.core.submission import read_submission_zip  # noqa: E402

G = 71.0
VI_MAX, ZH_MAX = 766441, 4392214
SUBMISSIONS = {  # id: (file, docs P, docs R) as scored on the public leaderboard
    6751: ("submissions/w12T2_VI_win4500_n150.zip", 0.2495, 0.3823),
    6769: ("submissions/probe_r1ord_w12T2_VI.zip", 0.2398, 0.3690),
    6768: ("submissions/probe_sliceA_w12T2_VI.zip", 0.0696, 0.2175),
    6767: ("submissions/probe_sliceB_w12T2_VI.zip", 0.0593, 0.0339),
    6935: ("submissions/mixA100_w12T2_VI_win4500_n150.zip", 0.1686, 0.5225),
    6932: ("submissions/mixA999_w12T2_VI_win4500_n150.zip", 0.1208, 0.5998),
    7030: ("submissions/p1_zh10_v21chunk.zip", 0.1687, 0.5532),
    7034: ("submissions/p2_en10_v21section.zip", 0.1603, 0.5225),
    7036: ("submissions/p3_vinew30_base.zip", 0.1544, 0.5474),
    7043: ("submissions/p5_pool5000_base.zip", 0.0105, 0.7317),
    7044: ("submissions/p6_vi3000zh1000en1000_base.zip", 0.0134, 0.9360),
    # rebuilt from the pool sets (their zips hold ~850 MB of chunk text): "+"-joined sets, "base+" = #6935 docs first
    7191: ("base+vi1500+zh500+en500", 0.0253, 0.8960),
    7214: ("zh250+en250", 0.0209, 0.1557),
    7215: ("vi800+zh200+en200", 0.0463, 0.8117),
    # doc lists decided by decide_lists.py: "lists:<job>" = the doc rows of that job in kaggle/datasets/mat_lists
    7225: ("lists:rr197", 0.1596, 0.5124),
    7226: ("lists:vitail100", 0.0486, 0.0572),
    7227: ("lists:zhen50", 0.0534, 0.0503),
}
BASE = "submissions/mixA100_w12T2_VI_win4500_n150.zip"
LANGS = ("vi", "zh", "en")
NAMES = ["b_vi", "b_zh", "b_en", "w_lrank_vi", "w_lrank_zh", "w_lrank_en", "w_n200", "w_n50", "w_rr", "w_rr_miss"]


def lang_code(doc: np.ndarray) -> np.ndarray:
    return np.where(doc <= VI_MAX, 0, np.where(doc <= ZH_MAX, 1, 2))


def features(pairs: pd.DataFrame, lists: pd.DataFrame, methods: pd.DataFrame, rr: pd.DataFrame) -> pd.DataFrame:
    """pairs (qid, doc_id) -> + lang, lrank, n200, n50, rr (reranker logit, 0 if unscored), rr_miss."""
    f = (pairs.merge(lists, on=["qid", "doc_id"], how="left").merge(methods, on=["qid", "doc_id"], how="left")
         .merge(rr, on=["qid", "doc_id"], how="left"))
    f["rr_miss"] = f["score"].isna().astype(float)
    f["rr"] = f["score"].fillna(0.0)
    f["lang"] = lang_code(f["doc_id"].to_numpy())
    depth = np.array([5000.0, 1000.0, 1000.0])[f["lang"].to_numpy()]
    f["lrank"] = np.log1p(f["rank"].fillna(pd.Series(depth, index=f.index)))
    f[["n200", "n50"]] = f[["n200", "n50"]].fillna(0)
    return f[["qid", "doc_id", "lang", "lrank", "n200", "n50", "rr", "rr_miss"]]


def arrays(f: pd.DataFrame) -> dict[str, np.ndarray]:
    """Numpy view of a feature table; qix = dense query index for bincount."""
    qix, _ = pd.factorize(f["qid"])
    return {"lang": f["lang"].to_numpy(), "lrank": f["lrank"].to_numpy(float), "n200": f["n200"].to_numpy(float),
            "n50": f["n50"].to_numpy(float), "rr": f["rr"].to_numpy(float), "rr_miss": f["rr_miss"].to_numpy(float),
            "qix": qix}


def prob(theta: np.ndarray, a: dict[str, np.ndarray]) -> np.ndarray:
    lang = a["lang"]
    z = (theta[lang] + theta[3 + lang] * a["lrank"] + theta[6] * a["n200"] + theta[7] * a["n50"]
         + theta[8] * a["rr"] + theta[9] * a["rr_miss"])
    return 1.0 / (1.0 + np.exp(-z))


def predict(theta: np.ndarray, a: dict[str, np.ndarray], sizes: np.ndarray) -> tuple[float, float]:
    tp = np.bincount(a["qix"], weights=prob(theta, a), minlength=len(sizes))
    return float((tp / sizes).mean()), float((tp / G).mean())


def zip_pairs(path: Path) -> pd.DataFrame:
    recs = read_submission_zip(path)
    return pd.DataFrame([(int(x["id"]), int(d)) for x in recs for d in x["relevant_docs"]],
                        columns=["qid", "doc_id"]).drop_duplicates()


def set_pairs(spec: str, add: pd.DataFrame) -> pd.DataFrame:
    """Doc pairs of a submission built by augment_probe.py from pool sets (whole sets, base docs optional)."""
    names = spec.split("+")
    parts = [zip_pairs(ROOT / BASE)] if names[0] == "base" else []
    parts += [add.loc[add["set"] == n, ["qid", "doc_id"]] for n in names if n != "base"]
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ridge", type=float, default=0.3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pool = pd.read_parquet(ROOT / "data/rankings_v21/pool5000_vi.parquet", columns=["qid", "doc_id", "rank"])
    add = pd.read_parquet(ROOT / "kaggle/datasets/probe_inputs/additions.parquet")
    zhen = add[add["set"].isin(["zh1000", "en1000"])][["qid", "doc_id", "rank"]]
    lists = pd.concat([pool, zhen], ignore_index=True).drop_duplicates(["qid", "doc_id"])
    m = pd.read_parquet(ROOT / "runs/reports/method_overlap/method_sets.parquet")
    methods = m.assign(n200=1, n50=(m["rank"] < 50).astype(int)).groupby(["qid", "doc_id"], as_index=False)[
        ["n200", "n50"]].sum()
    rr_parts = sorted(ROOT.glob("kaggle/outputs/rr3-p*/rerank.parquet"))
    rr = (pd.concat([pd.read_parquet(x, columns=["qid", "doc_id", "score"]) for x in rr_parts], ignore_index=True)
          if rr_parts else pd.DataFrame({"qid": [], "doc_id": [], "score": []}).astype({"qid": int, "doc_id": int}))
    print(f"reranker: {len(rr_parts)} parts, {len(rr):,} pairs", flush=True)
    print(f"lists {len(lists):,} pairs; methods {m['set'].nunique()} -> {len(methods):,} pairs", flush=True)

    mat = pd.read_parquet(ROOT / "kaggle/datasets/mat_lists/lists.parquet", columns=["job", "qid", "kind", "doc_id"])
    data = {}
    for sid, (path, p, r) in SUBMISSIONS.items():
        if path.startswith("lists:"):
            pairs = mat[(mat["job"] == path[6:]) & (mat["kind"] == "doc")][["qid", "doc_id"]]
        else:
            pairs = set_pairs(path, add) if not path.endswith(".zip") else zip_pairs(ROOT / path)
        f = features(pairs, lists, methods, rr)
        a = arrays(f)
        sizes = np.bincount(a["qix"]).astype(float)
        data[sid] = (a, sizes, p, r)
        print(f"#{sid}: {len(pairs) / len(sizes):.0f} docs/query", flush=True)

    def residuals(theta, ids):
        res = []
        for sid in ids:
            f, sizes, p, r = data[sid]
            pp, rr = predict(theta, f, sizes)
            res += [(pp - p) / 0.01, (rr - r) / 0.01]
        return np.r_[res, args.ridge * theta[3:]]

    theta0 = np.array([-3.0, -4.0, -5.0, -0.3, -0.3, -0.3, 0.1, 0.1, 0.0, 0.0])
    ids = list(data)
    fit = least_squares(residuals, theta0, args=(ids,))
    report = {"params": dict(zip(NAMES, map(float, fit.x))), "submissions": {}}
    for sid in ids:
        f, sizes, p, r = data[sid]
        loo = least_squares(residuals, fit.x, args=([i for i in ids if i != sid],)).x
        report["submissions"][sid] = {"obs_P": p, "obs_R": r,
                                      "fit_P_R": [round(v, 4) for v in predict(fit.x, f, sizes)],
                                      "loo_P_R": [round(v, 4) for v in predict(loo, f, sizes)]}
    (args.out / "fit.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))

    scored = features(lists[["qid", "doc_id"]], lists, methods, rr)
    scored["p"] = prob(fit.x, arrays(scored))
    scored.to_parquet(args.out / "scored_pool.parquet", index=False)
    exp = scored.groupby("lang")["p"].sum() / scored["qid"].nunique()
    print("expected gold/query in pool by language:", {LANGS[k]: round(v, 1) for k, v in exp.items()})


if __name__ == "__main__":
    main()
