"""Rebuild the doc/chunk lists of #6536 (w12T2) and #6537 (rdT2) and compare them (offline, no labels)."""
import numpy as np
import pandas as pd

O = "kaggle/outputs/"
R = "kaggle/rankings_ds/"


def lists(path, cap=150, tau=-2.0, lo=20, n_chunks=90):
    f = pd.read_parquet(path).sort_values(["qid", "rank"])
    f = f[f["rank"] < cap]
    ok = set(zip(rr.qid, rr.doc_id)) if tau is None else set(zip(rr.qid[rr.score >= tau], rr.doc_id[rr.score >= tau]))
    out = {}
    for q, g in f.groupby("qid"):
        ds = g.doc_id.tolist()
        ds = ds[:lo] + [d for d in ds[lo:] if (q, d) in ok]
        out[q] = (ds, ds[:n_chunks])
    return out


rr = pd.read_parquet(R + "rerank_r1.parquet")
rr = rr[rr.doc_id <= 766441]
rscore = dict(zip(zip(rr.qid, rr.doc_id), rr.score))
w12 = pd.read_parquet(O + "vibiomir-build-depth-d/w12T2_VI_ranking.parquet")
wrank = dict(zip(zip(w12.qid, w12.doc_id), w12["rank"]))
A = lists(O + "vibiomir-build-depth-d/w12T2_VI_ranking.parquet")
B = lists(O + "vibiomir-build-depth-c/rdT2_VI_ranking.parquet")

rows = []
only_b_rank, only_b_score, only_a_score, both_score = [], [], [], []
for q in A:
    da, ca = A[q]; db, cb = B.get(q, ([], []))
    sa, sb = set(da), set(db)
    rows.append((len(da), len(db), len(sa & sb), len(sa - sb), len(sb - sa), len(set(ca) & set(cb)), len(ca), len(cb)))
    for d in sb - sa:
        only_b_rank.append(wrank.get((q, d), 10 ** 4)); only_b_score.append(rscore.get((q, d), np.nan))
    for d in sa - sb:
        only_a_score.append(rscore.get((q, d), np.nan))
    for d in sa & sb:
        both_score.append(rscore.get((q, d), np.nan))
r = np.array(rows, dtype=float).mean(0)
print(f"docs/q  6536 {r[0]:.1f}  6537 {r[1]:.1f}  shared {r[2]:.1f}  only6536 {r[3]:.1f}  only6537 {r[4]:.1f}")
print(f"chunks/q 6536 {r[6]:.1f}  6537 {r[7]:.1f}  shared chunk docs {r[5]:.1f}")
ob = np.array(only_b_rank)
print("only-6537 docs: w12 rank quantiles (10/50/90%)", np.percentile(ob, [10, 50, 90]).round(0),
      " share beyond w12 top-150:", round((ob >= 150).mean(), 3))
for name, s in [("shared", both_score), ("only 6536", only_a_score), ("only 6537", only_b_score)]:
    s = np.array(s); s = s[~np.isnan(s)]
    print(f"reranker logit {name:10s}: median {np.median(s):.2f}  mean {s.mean():.2f}  n={len(s)}")
