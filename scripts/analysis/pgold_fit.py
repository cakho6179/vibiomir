"""Small calibrated p(gold) for every pool pair, fitted on the docs precision of submitted doc lists, then a per-query
doc count chosen by expected F2 (no fixed N).

    python scripts/analysis/pgold_fit.py --out runs/reports/pgold

Model: p = sigmoid(a_lang + b * rr3 + c * m3all + d * m3rel), 6 parameters, plus gamma for the spread of G_q (g_exp) and sigma for the
query-level gold noise the features miss (f2_exp); m3rel = m3all minus its query/language mean. rr3 = bge-reranker-v2-m3 logit,
m3all = BGE-M3 0.4 dense + 0.2 sparse + 0.4 ColBERT (decide_lists.load_pool). Raw scores, not ranks: a query whose docs
all score high is expected to hold more gold, which is what lets the doc count vary per query.
Fit: expected macro docs P, R and F2 of the FIT lists against the leaderboard values (R and F2 pin down how the
expected gold spreads over queries); HOLDOUT lists are only predicted (acceptance: |error| <~ 0.01). Only aggregate P/R of our own submissions.
Per-query N: docs sorted by p; G_q from g_exp; N_q = argmax over n of
E[F2] under that noise (f2_exp).
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
sys.path.insert(0, str(Path(__file__).parent))
from decide_lists import load_pool  # noqa: E402

D = ROOT / "kaggle/datasets"
POOL_R = 0.8117  # #7215: docs recall of the whole VI800/ZH200/EN200 pool
FIT = {  # submission id: (lists.parquet, job, leaderboard docs P, R, F2 macro)
    7230: (D / "mat_lists4", "r8_m3rr197", 0.1792, 0.5709, 0.3421),
    7225: (D / "mat_lists", "rr197", 0.1596, 0.5124, 0.3055),
    7226: (D / "mat_lists", "vitail100", 0.0486, 0.0572, 0.0506),
    7227: (D / "mat_lists", "zhen50", 0.0534, 0.0503, 0.0415),
    7231: (D / "mat_lists3", "r6_zh25", 0.1068, 0.0503, 0.0476),
    7215: (None, "pool", 0.0463, 0.8117, 0.1751),
    7517: (D / "pn_docs", "pN1", 0.2551, 0.4926, 0.3776),  # per-query N of the first fit: pins the spread of G_q
}
HOLDOUT = {
    7324: (D / "mat_lists10", "r13", 0.1739, 0.5575, 0.3328),
    7426: (D / "mat_lists19", "r30", 0.1644, 0.5298, 0.3158),
    7423: (D / "mat_lists19", "r31", 0.1777, 0.5662, 0.3393),
}
LANGS = ("vi", "zh", "en")
# Gauss-Hermite nodes for the query-level gold factor u = exp(sigma * z - sigma^2 / 2), z ~ N(0, 1), E[u] = 1
GH_Z, GH_W = np.polynomial.hermite_e.hermegauss(9)
GH_W = GH_W / GH_W.sum()


def doc_sets(pool: pd.DataFrame, spec: dict) -> dict[int, pd.DataFrame]:
    out = {}
    for sid, (folder, job, *_) in spec.items():
        if folder is None:
            out[sid] = pool[["qid", "doc_id"]]
            continue
        if (folder / "docs.parquet").exists():  # swap_docs input: doc lists only
            l = pd.read_parquet(folder / "docs.parquet", columns=["job", "qid", "doc_id"]).assign(kind="doc")
        else:
            l = pd.read_parquet(folder / "lists.parquet", columns=["job", "qid", "kind", "doc_id"])
        out[sid] = l[(l["job"] == job) & (l["kind"] == "doc")][["qid", "doc_id"]].drop_duplicates()
    return out


def prob(theta: np.ndarray, f: pd.DataFrame) -> np.ndarray:
    a = np.select([f["lang"] == x for x in LANGS], theta[:3])
    return 1 / (1 + np.exp(-(a + theta[3] * f["score"].to_numpy() + theta[4] * f["m3z"].to_numpy()
                             + theta[5] * f["m3rel"].to_numpy())))


def f2_exp(tp: np.ndarray, gq: np.ndarray, n: np.ndarray, sigma: float) -> np.ndarray:
    """E[F2] when the query's true gold (and so the gold among its docs) is the model's times a factor u, E[u] = 1.
    P and R keep their expectation, F2 is concave in u: per-query noise the features miss lowers F2 only."""
    u = np.exp(sigma * GH_Z - sigma ** 2 / 2).reshape((-1,) + (1,) * np.ndim(tp))
    return (GH_W.reshape(u.shape) * 5 * u * tp / (4 * u * gq + n)).sum(0)


def macro(theta: np.ndarray, f: pd.DataFrame, g: pd.Series) -> tuple[float, float, float]:
    """Expected macro docs P, R, F2 of a list; g = expected gold per query (g_exp)."""
    q = f.assign(p=prob(theta, f)).groupby("qid")["p"].agg(["sum", "size"])
    gq = g.reindex(q.index).to_numpy()
    tp, n = q["sum"].to_numpy(), q["size"].to_numpy()
    sigma = theta[7] if len(theta) > 7 else 0.0
    return float((tp / n).mean()), float(np.minimum(tp / gq, 1).mean()), float(f2_exp(tp, gq, n, sigma).mean())


def g_exp(theta: np.ndarray, pool: pd.DataFrame) -> pd.Series:
    """Expected gold per query. theta[6] = gamma shrinks the per-query spread toward the mean (1 = pool sum / pool
    recall as is): #7517 showed the raw pool sums spread the gold over queries too much (recall .53 predicted, .49 seen)."""
    s = pool.assign(p=prob(theta, pool)).groupby("qid")["p"].sum()
    gamma = theta[6] if len(theta) > 6 else 1.0
    return s.mean() * (s / s.mean()) ** gamma / POOL_R


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/reports/pgold")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pool = load_pool({"vi": 800, "zh": 200, "en": 200})
    pool["m3z"] = (pool["m3all"] - pool["m3all"].mean()) / pool["m3all"].std()  # scale only, same for every query
    # the same score relative to the query's own pool (query-level offsets of BGE-M3 removed)
    pool["m3rel"] = pool["m3z"] - pool.groupby(["qid", "lang"])["m3z"].transform("mean")
    feats = pool[["qid", "doc_id", "lang", "score", "m3z", "m3rel"]]
    sets = {}
    for spec in (FIT, HOLDOUT):
        for sid, pairs in doc_sets(pool, spec).items():
            f = pairs.merge(feats, on=["qid", "doc_id"], how="left")
            miss = f["score"].isna().mean()
            sets[sid] = f.dropna()
            print(f"#{sid}: {len(pairs) / pairs['qid'].nunique():.0f} docs/q, outside pool {miss:.1%}", flush=True)

    def res(theta):
        g = g_exp(theta, pool)
        out = []
        for s in FIT:
            pred = macro(theta, sets[s], g)
            out += [(pred[i] - FIT[s][2 + i]) / 0.005 for i in range(3)]
        return out + list(0.1 * theta[3:6])

    fit = least_squares(res, np.array([-3.0, -3.5, -4.0, 0.3, 0.3, 0.3, 1.0, 0.3]),
                        bounds=([-50] * 6 + [0, 0], [50] * 6 + [2, 3]))
    theta = fit.x
    g = g_exp(theta, pool)
    names = ["a_vi", "a_zh", "a_en", "b_rr3", "c_m3all", "d_m3rel", "gamma", "sigma"]
    report = {"theta": dict(zip(names, map(float, theta))), "fit": {}, "holdout": {}}
    for name, spec in (("fit", FIT), ("holdout", HOLDOUT)):
        for s in spec:
            report[name][s] = {"obs_P_R_F2": list(spec[s][2:]), "pred": [round(v, 4) for v in macro(theta, sets[s], g)]}
    print(json.dumps(report, indent=1))

    pool["p"] = prob(theta, pool)
    pool = pool.sort_values(["qid", "p"], ascending=[True, False])
    rows = []
    for qid, grp in pool.groupby("qid", sort=False):
        p = grp["p"].to_numpy()
        gq = g[qid]
        n = np.arange(1, len(p) + 1)
        f2 = f2_exp(np.cumsum(p), gq, n, theta[7])
        k = int(f2.argmax()) + 1
        rows.append((qid, k, gq, float(f2[k - 1])))
    nq = pd.DataFrame(rows, columns=["qid", "n_docs", "g_exp", "f2_exp"])
    print(nq[["n_docs", "g_exp", "f2_exp"]].describe(percentiles=[.1, .5, .9]).round(3).to_string())
    pool.to_parquet(args.out / "pool_p.parquet", index=False)
    nq.to_parquet(args.out / "n_per_query.parquet", index=False)
    (args.out / "fit.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
