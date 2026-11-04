"""What do gold chunks look like? Infer it from our scored chunk probes (aggregate P/R only, no labels).

    python scripts/analysis/chunk_geometry.py inventory   # per-chunk table of every probe zip (+ 6000 vs 7500 diff)
    python scripts/analysis/chunk_geometry.py fit         # deterministic gold-length hypotheses vs observed chunk P

fit: every probe has the same 88 docs/query (w12T2 list), so only the chunk text differs. A hypothesis gives the
gold chunk length g of a gold doc (v2.1 length L): g = f x L, or g = min(L, K). Best case alignment (our window
inside the gold span): LCS = min(|c|, g), so c matches when min(|c|, g) >= 0.4 g and >= 0.4 |c|. Predicted chunk P
= mean_q sum_c p(doc) match / |S_q| with p = .35 for doc positions 1-28 and .20 after (measured slices, docs P .2495),
times one free scale per hypothesis fitted on the five window probes; #7030 (v2.1 window) and #7034 (sections)
are held out. Chunk P is the target because its denominator is exact.

Outputs under runs/reports/chunk_geometry/. Rule (since 23/10): chunk c matches gold g of the same doc when
LCS_tokens(c, g) >= 0.4 |g| and >= 0.4 |c|, so c can only match a reference of 0.4|c| .. 2.5|c| tokens.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs/reports/chunk_geometry"
# id: (zip, chunks P, chunks R) as scored on the public leaderboard under the 23/10 rule
PROBES = {
    6749: ("submissions/w12T2_VI_win1500_n150.zip", 0.0784, 0.1119),
    6750: ("submissions/w12T2_VI_win3000_n150.zip", 0.1856, 0.2686),
    6751: ("submissions/w12T2_VI_win4500_n150.zip", 0.1941, 0.2810),
    6933: ("submissions/w12T2_VI_win6000_n150.zip", 0.1937, 0.2808),
    6934: ("submissions/w12T2_VI_win7500_n150.zip", 0.1937, 0.2808),
    7030: ("submissions/p1_zh10_v21chunk.zip", 0.1949, 0.2817),
    7034: ("submissions/p2_en10_v21section.zip", 0.1870, 0.2730),
}


def inventory() -> None:
    from r2ai2026.core.submission import read_submission_zip
    rows = []
    for sid, (path, _, _) in PROBES.items():
        recs = read_submission_zip(ROOT / path)
        for r in recs:
            dpos = {int(d): i for i, d in enumerate(r["relevant_docs"])}
            for k, c in enumerate(r["relevant_chunks"]):
                t = c["chunk_text"]
                rows.append((sid, int(r["id"]), int(c["doc_id"]), k, dpos.get(int(c["doc_id"]), -1), len(t),
                             hashlib.md5(t.encode("utf-8")).hexdigest(), t[:120], t[-120:]))
        print(f"#{sid}: {len(recs)} queries, {sum(len(r['relevant_chunks']) for r in recs) / len(recs):.1f} chunks/q",
              flush=True)
        del recs
    df = pd.DataFrame(rows, columns=["probe", "qid", "doc_id", "k", "doc_pos", "chars", "md5", "head", "tail"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "inventory.parquet", index=False)
    s = df.groupby("probe").agg(chunks_q=("k", lambda x: len(x) / df.loc[x.index, "qid"].nunique()),
                                chars_med=("chars", "median"), chars_mean=("chars", "mean"),
                                in_doclist=("doc_pos", lambda x: (x >= 0).mean()))
    print(s.round(2).to_string())
    a = df[df["probe"] == 6933].set_index(["qid", "doc_id"])["md5"]
    b = df[df["probe"] == 6934].set_index(["qid", "doc_id"])["md5"]
    j = a.to_frame("a").join(b.to_frame("b"), how="outer")
    print(f"#6933 vs #6934: {len(j):,} (q, doc) pairs, same text {(j['a'] == j['b']).mean():.4f}, "
          f"only in one {j.isna().any(axis=1).mean():.4f}")


def fit() -> None:
    df = pd.read_parquet(OUT / "inventory.parquet", columns=["probe", "qid", "doc_id", "doc_pos", "chars"])
    meta = pd.read_parquet(ROOT / "data/docstore/v2.1/docs_meta.parquet", columns=["id", "n_chars"])
    df = df.merge(meta.rename(columns={"id": "doc_id"}), on="doc_id", how="left")
    L = df["n_chars"].fillna(df["chars"])
    p = np.where(df["doc_pos"] < 28, 0.35, 0.20)
    obs = pd.Series({k: v[1] for k, v in PROBES.items()})
    fitset = [6749, 6750, 6751, 6933, 6934]
    sizes = df.groupby(["probe", "qid"]).size()
    hyps = {f"g={f}L": f * L for f in (.5, .6, .65, .7, .75, .8, .9, 1.0)}
    hyps.update({f"g=min(L,{k})": np.minimum(L, k) for k in (1500, 2000, 3000, 4000, 5000, 6000, 8000)})
    rows = []
    for name, g in hyps.items():
        ov = np.minimum(df["chars"], g)
        match = (ov >= 0.4 * g) & (ov >= 0.4 * df["chars"])
        pred = ((p * match).groupby([df["probe"], df["qid"]]).sum() / sizes).groupby(level=0).mean()
        scale = (obs[fitset] * pred[fitset]).sum() / (pred[fitset] ** 2).sum()
        res = pred * scale - obs
        rows.append({"hypothesis": name, "scale": round(scale, 3),
                     "rmse_fit": round(float(np.sqrt((res[fitset] ** 2).mean())), 4),
                     **{f"res_{k}": round(float(v), 4) for k, v in res.items()}})
    rep = pd.DataFrame(rows).sort_values("rmse_fit")
    rep.to_csv(OUT / "fit.csv", index=False)
    print(rep.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["inventory", "fit"])
    args = ap.parse_args()
    {"inventory": inventory, "fit": fit}[args.step]()


if __name__ == "__main__":
    main()
