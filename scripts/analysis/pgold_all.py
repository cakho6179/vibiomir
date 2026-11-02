"""p(gold) for every candidate, fitted on the aggregate docs P/R/F2 of ALL distinct scored doc lists (sub_doc_lists.py),
validated leave-one-list-out; then a per-query doc count by expected F2 (no fixed N).

    python scripts/analysis/pgold_all.py --lists runs/reports/pgold_lists --out runs/reports/pgold_all [--loo]

Scored region = the 2,500 first-stage candidates per query (rerank_pool/cands.parquet: vi1500/zh500/en500, rank within
the language). logit p = a_lang + e * log1p(rank)  (every candidate)
                         + in_pool * (h + b * rr3 + c * m3z + d * m3rel)  (the 1,200 the reranker scored)
A listed doc outside the candidates gets a per-language constant p (out_lang) and still counts in n.
G_q = (sum of p over the candidates)^gamma-spread / r_cand, r_cand = fitted recall of the candidates.
E[F2] carries a query-level gold factor u, E[u] = 1, sd ~ sigma (pgold_fit.f2_exp).
Only aggregate P/R/F2 of our own submissions are used, never per-pair labels.
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
from pgold_fit import f2_exp  # noqa: E402

LANGS = ("vi", "zh", "en")
OUT = ROOT / "kaggle/outputs"
# optional extra features (--feats): name -> (scope, loader). "pool" features exist for the 1,200 reranked docs only
# (0 elsewhere after z-scoring); "cands" features exist for every candidate.
BM_CAP = 2000


def _rr4() -> pd.DataFrame:
    parts = sorted(OUT.glob("rr4-p*/rerank.parquet"))
    return pd.concat([pd.read_parquet(x, columns=["qid", "doc_id", "score"]) for x in parts]).rename(columns={"score": "rr4"})


def _rr5(name: str) -> pd.DataFrame:
    """A second-opinion reranker over the top 400 candidates of base4 (kaggle/outputs/<name>*/rerank.parquet)."""
    parts = sorted(OUT.glob(f"{name}*/rerank.parquet"))
    return pd.concat([pd.read_parquet(x, columns=["qid", "doc_id", "score"]) for x in parts]).rename(columns={"score": name})


def _m3heads() -> pd.DataFrame:
    cols = ["qid", "doc_id", "dense", "sparse", "colbert", "p_dense", "p_sparse", "p_colbert"]
    return pd.concat([pd.read_parquet(x, columns=cols) for x in sorted(OUT.glob("m3m-p*/m3multi.parquet"))])


def _bm25(which: str) -> pd.DataFrame:
    """BM25 over the query-frame v3 variants (original query + LLM-written rewrites): best rank over variants and
    indexes, as -log1p(rank); docs missing from every list get the cap."""
    dirs = {"vi": ["qv3-bm25-vi-doc", "qv3-bm25-vi-sec"], "zh": ["qv3-bm25-zh"], "en": ["qv3-bm25-en"]}[which]
    r = pd.concat([pd.read_parquet(OUT / d / "ranking.parquet", columns=["qid", "variant", "rank", "doc_id"]) for d in dirs])
    return r.groupby(["qid", "doc_id"], as_index=False)["rank"].min().rename(columns={"rank": f"bm_{which}"})


def _bm25_llm() -> pd.DataFrame:
    """Same, LLM-written variants only (not the original query)."""
    dirs = ["qv3-bm25-vi-doc", "qv3-bm25-vi-sec", "qv3-bm25-zh", "qv3-bm25-en"]
    r = pd.concat([pd.read_parquet(OUT / d / "ranking.parquet", columns=["qid", "variant", "rank", "doc_id"]) for d in dirs])
    r = r[r["variant"] != "orig"]
    return r.groupby(["qid", "doc_id"], as_index=False)["rank"].min().rename(columns={"rank": "bm_llm"})


META = ROOT / "data/docstore/v2.1/docs_meta.parquet"
HOSTS = ["nhathuoclongchau.com.vn", "suckhoedoisong.vn", "medlatec.vn", "www.120ask.com", "www.vinmec.com",
         "thanhnien.vn", "tiemchunglongchau.com.vn", "hellobacsi.com", "suckhoecongdongonline.vn"]


def _meta() -> pd.DataFrame:
    """Doc metadata: log length, thin flag, share of page furniture (menus, ads) in the raw text."""
    m = pd.read_parquet(META, columns=["id", "n_chars", "thin", "furniture_chars"]).rename(columns={"id": "doc_id"})
    return pd.DataFrame({"doc_id": m["doc_id"], "lnchars": np.log1p(m["n_chars"]), "thin": m["thin"].astype(float),
                         "furn": m["furniture_chars"] / (m["n_chars"] + m["furniture_chars"]).clip(lower=1)})


def _host() -> pd.DataFrame:
    """One indicator per frequent host of the submitted lists (the rest is the baseline)."""
    m = pd.read_parquet(META, columns=["id", "host"]).rename(columns={"id": "doc_id"})
    return pd.DataFrame({"doc_id": m["doc_id"], **{f"h_{i}": (m["host"] == h).astype(float) for i, h in enumerate(HOSTS)}})


FEATS = {"rr4": ("pool", _rr4, ["rr4"]),
         "m3heads": ("pool", _m3heads, ["dense", "sparse", "colbert", "p_dense", "p_sparse", "p_colbert"]),
         "bm": ("cands", lambda: pd.concat([_bm25(x) for x in ("vi", "zh", "en")]).groupby(["qid", "doc_id"], as_index=False)
                .min().assign(bm=lambda d: d[["bm_vi", "bm_zh", "bm_en"]].min(axis=1))[["qid", "doc_id", "bm"]], ["bm"]),
         "bm_llm": ("cands", _bm25_llm, ["bm_llm"]),
         "meta": ("cands", _meta, ["lnchars", "thin", "furn"]),
         "host": ("cands", _host, [f"h_{i}" for i in range(len(HOSTS))]),
         **{n: ("cands", lambda n=n: _rr5(n), [n]) for n in ("rr5vr", "rr5gte", "rr5pho")},
         **{f"{n}s": ("split", lambda n=n: _rr5(n), [n]) for n in ("rr5vr", "rr5gte", "rr5pho")},
         # control: the same scored set with the scores shuffled within each query (no ranking information left)
         "rr5vrs_shuf": ("split", lambda: _rr5("rr5vr").assign(
             rr5vr=lambda d: d.groupby("qid")["rr5vr"].transform(lambda s: s.sample(frac=1, random_state=0).to_numpy())),
             ["rr5vr"])}
NAMES = ["a_vi", "a_zh", "a_en", "e_lrank", "h_pool", "b_rr3", "c_m3all", "d_m3rel",
         "out_vi", "out_zh", "out_en", "logit_rcand", "gamma", "sigma"]
X0 = np.array([-2.0, -2.5, -3.0, -0.5, 0.0, 0.25, 0.7, 2.0, -6.0, -6.0, -6.0, 1.5, 1.0, 0.5])
LO = np.array([-50] * 11 + [-5, 0, 0])
HI = np.array([50] * 11 + [8, 2, 3])


def doc_lang(doc_id: np.ndarray) -> np.ndarray:
    """Language of an official doc id from its id block (VI <= 766441 < ZH <= 4392214 < EN)."""
    return np.where(doc_id <= 766441, 0, np.where(doc_id <= 4392214, 1, 2))


def build_cands(feats: list[str]) -> tuple[pd.DataFrame, list[str]]:
    c = pd.read_parquet(ROOT / "kaggle/datasets/rerank_pool/cands.parquet")
    c["lang"] = c["set"].str[:2]
    pool = load_pool({"vi": 800, "zh": 200, "en": 200})
    pool["m3z"] = (pool["m3all"] - pool["m3all"].mean()) / pool["m3all"].std()
    pool["m3rel"] = pool["m3z"] - pool.groupby(["qid", "lang"])["m3z"].transform("mean")
    c = c.merge(pool[["qid", "doc_id", "score", "m3z", "m3rel"]], on=["qid", "doc_id"], how="left")
    c["in_pool"] = c["score"].notna().astype(float)
    c[["score", "m3z", "m3rel"]] = c[["score", "m3z", "m3rel"]].fillna(0.0)
    c["li"] = c["lang"].map({x: i for i, x in enumerate(LANGS)}).astype(int)
    extra = []
    for name in feats:
        scope, load, cols = FEATS[name]
        f = load()
        on = ["qid", "doc_id"] if "qid" in f else ["doc_id"]  # per-pair scores or per-doc metadata
        c = c.merge(f.drop_duplicates(on)[on + cols], on=on, how="left")
        if scope == "split":  # scored-set indicator, within-query rank signal and per-query level apart
            for col in cols:  # (filling the outside with the query min made _w a disguised indicator: a shuffled
                g = c.groupby("qid")[col]  # control fitted as well as the real scores)
                z = (c[col] - g.transform("mean")) / g.transform("std")
                lev = g.transform("mean")
                c[f"{col}_in"] = c[col].notna().astype(float)
                c[f"{col}_w"] = z.fillna(0.0)
                c[f"{col}_lev"] = ((lev - lev.mean()) / lev.std()).where(c[col].notna(), 0.0).fillna(0.0)
                print(f"feature {col}: split, coverage {c[col].notna().mean():.1%}", flush=True)
                extra += [f"{col}_in", f"{col}_w", f"{col}_lev"]
            continue
        for col in cols:
            if col.startswith("bm"):  # rank feature: -log1p(rank), missing = beyond the list
                c[col] = -np.log1p(c[col].fillna(BM_CAP).clip(upper=BM_CAP))
            m = c["in_pool"] > 0 if scope == "pool" else np.ones(len(c), bool)
            v = c.loc[m, col]
            c[col] = ((c[col] - v.mean()) / v.std()).where(m, 0.0).fillna(0.0)
            print(f"feature {col}: scope {scope}, coverage {c.loc[m, col].ne(0).mean():.1%}", flush=True)
            extra.append(col)
    return c.sort_values(["qid", "lang", "rank"]).reset_index(drop=True), extra


class Model:
    def __init__(self, cands: pd.DataFrame, lists: pd.DataFrame, scores: pd.DataFrame, extra: list[str]):
        self.c = cands
        self.E = cands[extra].to_numpy() if extra else np.zeros((len(cands), 0))
        self.qids = np.sort(cands["qid"].unique())
        self.qi = np.searchsorted(self.qids, cands["qid"].to_numpy())
        self.X = cands[["in_pool", "score", "m3z", "m3rel"]].to_numpy()
        self.lr = np.log1p(cands["rank"].to_numpy())
        self.li = cands["li"].to_numpy()
        key = pd.Series(np.arange(len(cands)), index=pd.MultiIndex.from_arrays([cands["qid"], cands["doc_id"]]))
        self.sets = {}
        for sid, l in lists.groupby("sid"):
            idx = key.reindex(pd.MultiIndex.from_arrays([l["qid"], l["doc_id"]])).to_numpy()
            inside = ~np.isnan(idx)
            out = l[~inside]
            self.sets[sid] = dict(
                idx=idx[inside].astype(int),
                out_q=np.searchsorted(self.qids, out["qid"].to_numpy()),
                out_l=doc_lang(out["doc_id"].to_numpy()),
                n=np.bincount(np.searchsorted(self.qids, l["qid"].to_numpy()), minlength=len(self.qids)),
                has=np.bincount(np.searchsorted(self.qids, l["qid"].to_numpy()), minlength=len(self.qids)) > 0,
                out_share=float((~inside).mean()))
        self.obs = scores.set_index("sid")[["P", "R", "F2"]]

    def p(self, t: np.ndarray) -> np.ndarray:
        z = (t[self.li] + t[3] * self.lr
             + self.X[:, 0] * (t[4] + t[5] * self.X[:, 1] + t[6] * self.X[:, 2] + t[7] * self.X[:, 3])
             + self.E @ t[14:])
        return 1 / (1 + np.exp(-z))

    def gq(self, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        s = np.bincount(self.qi, weights=p, minlength=len(self.qids))
        r = 1 / (1 + np.exp(-t[11]))
        return s.mean() * (s / s.mean()) ** t[12] / r

    def predict(self, t: np.ndarray, sid: int, p: np.ndarray | None = None, g: np.ndarray | None = None):
        p = self.p(t) if p is None else p
        g = self.gq(t, p) if g is None else g
        s = self.sets[sid]
        tp = np.bincount(self.qi[s["idx"]], weights=p[s["idx"]], minlength=len(self.qids))
        pout = 1 / (1 + np.exp(-t[8:11]))
        tp += np.bincount(s["out_q"], weights=pout[s["out_l"]], minlength=len(self.qids))
        h = s["has"]  # a query with an empty list scores 0 on every measure
        tp, n, gg = tp[h], s["n"][h], g[h]
        P, R, F = (np.zeros(len(self.qids)) for _ in range(3))
        P[h], R[h], F[h] = tp / n, np.minimum(tp / gg, 1), f2_exp(tp, gg, n, t[13])
        return float(P.mean()), float(R.mean()), float(F.mean())

    def residuals(self, t: np.ndarray, sids: list[int]) -> list[float]:
        p = self.p(t)
        g = self.gq(t, p)
        out = []
        for sid in sids:
            pred = self.predict(t, sid, p, g)
            out += [(pred[i] - self.obs.loc[sid].iloc[i]) / 0.005 for i in range(3)]
        return out + list(0.05 * t[3:8]) + list(0.05 * t[14:])

    def fit(self, sids: list[int], x0: np.ndarray | None = None) -> np.ndarray:
        k = self.E.shape[1]
        x0 = np.r_[X0, np.zeros(k)] if x0 is None else x0
        return least_squares(lambda t: self.residuals(t, sids), x0,
                             bounds=(np.r_[LO, [-20] * k], np.r_[HI, [20] * k])).x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lists", type=Path, default=ROOT / "runs/reports/pgold_lists")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/reports/pgold_all")
    ap.add_argument("--loo", action="store_true", help="leave-one-list-out check of every list")
    ap.add_argument("--exclude", default="", help="comma list of sids left out of the fit")
    ap.add_argument("--feats", default="", help=f"extra features, comma list of {sorted(FEATS)}")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lists = pd.read_parquet(args.lists / "lists.parquet")
    scores = pd.read_parquet(args.lists / "scores.parquet")
    excl = {int(x) for x in args.exclude.split(",") if x}
    cands, extra = build_cands([x for x in args.feats.split(",") if x])
    m = Model(cands, lists, scores, extra)
    sids = [s for s in m.sets if s not in excl]
    for s in m.sets:
        print(f"#{s}: {m.sets[s]['n'].sum() / max(m.sets[s]['has'].sum(), 1):.0f} docs/q, "
              f"outside the candidates {m.sets[s]['out_share']:.1%}", flush=True)
    t = m.fit(sids)
    rep = {"theta": dict(zip(NAMES + extra, map(float, t))), "lists": {}}
    res = np.array(m.residuals(t, sids)[:3 * len(sids)]) * 0.005
    rep["rmse_P_R_F2"] = [float(np.sqrt((res[i::3] ** 2).mean())) for i in range(3)]
    print("fit rmse P/R/F2:", [round(v, 4) for v in rep["rmse_P_R_F2"]], flush=True)
    for s in m.sets:
        rep["lists"][s] = {"obs": [float(v) for v in m.obs.loc[s]], "pred": [round(v, 4) for v in m.predict(t, s)],
                           "in_fit": s in sids}
    if args.loo:
        for s in sids:
            ts = m.fit([x for x in sids if x != s], t)
            rep["lists"][s]["loo"] = [round(v, 4) for v in m.predict(ts, s)]
            print(f"LOO #{s}: obs F2 {m.obs.loc[s, 'F2']:.4f} loo {rep['lists'][s]['loo'][2]:.4f}", flush=True)
    print(json.dumps(rep["theta"], indent=1))
    for s, v in rep["lists"].items():
        print(s, v["obs"], v["pred"], v.get("loo", ""))
    (args.out / "fit.json").write_text(json.dumps(rep, indent=1))

    # per-query N over the candidates in p order (docs outside the candidates have no score to rank them)
    p = m.p(t)
    g = m.gq(t, p)
    c = m.c[["qid", "doc_id", "lang", "rank", "in_pool"]].assign(p=p)
    c = c.sort_values(["qid", "p"], ascending=[True, False])
    rows = []
    for (qid, grp), gq in zip(c.groupby("qid", sort=True), g):
        cs = np.cumsum(grp["p"].to_numpy())
        n = np.arange(1, len(cs) + 1)
        f2 = f2_exp(cs, gq, n, t[13])
        k = int(f2.argmax()) + 1
        rows.append((qid, k, gq, float(f2[k - 1]), float(1 - grp["in_pool"].to_numpy()[:k].mean())))
    nq = pd.DataFrame(rows, columns=["qid", "n_docs", "g_exp", "f2_exp", "share_outside_pool"])
    print(nq.drop(columns="qid").describe(percentiles=[.1, .5, .9]).round(3).to_string())
    c.to_parquet(args.out / "cands_p.parquet", index=False)
    nq.to_parquet(args.out / "n_per_query.parquet", index=False)


if __name__ == "__main__":
    main()
