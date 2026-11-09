"""Build the pN3_base (#7589) and N1_n125 materialiser lists from the p(gold) model output.

    python scripts/analysis/pn3_lists.py --out kaggle/datasets/mat_lists22/lists.parquet [--model base2]

Inputs (runs/reports/pgold_feats/<model>/ from pgold_all.py): cands_p.parquet (qid, doc_id, p over 2,500 candidates
per query), n_per_query.parquet (n_docs = argmax E[F2] depth, g_exp = expected gold count) and fit.json (sigma).
  pN3_base  docs   = the n_docs candidates of highest p, in p order
            chunks = the n_c highest-p candidates, one 4,500-char "seed" window each, where n_c maximises the
                     expected chunk F2 with p_c = 0.77 p (a gold doc yields a matching chunk ~77% of the time) and
                     a chunk gold count of 75/71 x g_exp (mean |gold chunks| / |gold docs| on the public split)
  N1_n125   docs   = the round(1.25 n_docs) highest-p candidates; chunks = those of pN3_base
Only aggregate statistics enter: p(gold) is fitted on the public P/R of our own submissions (pgold_all.py), never on
per-pair labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(Path(__file__).parent))
from pgold_fit import f2_exp  # noqa: E402

CONV, GC = 0.77, 75 / 71


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="base2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depth", type=float, default=1.25, help="doc depth factor of the N1 job")
    args = ap.parse_args()
    src = ROOT / "runs/reports/pgold_feats" / args.model
    c = pd.read_parquet(src / "cands_p.parquet").sort_values(["qid", "p"], ascending=[True, False])
    c["pos"] = c.groupby("qid").cumcount()
    nq = pd.read_parquet(src / "n_per_query.parquet").set_index("qid")
    sigma = json.loads((src / "fit.json").read_text())["theta"]["sigma"]

    n_chunks = {}
    for q, g in c.groupby("qid", sort=True):
        p = g["p"].to_numpy()[:1200] * CONV
        f = f2_exp(np.cumsum(p), GC * nq["g_exp"][q], np.arange(1, len(p) + 1), sigma)
        n_chunks[q] = int(f.argmax()) + 1
    docs = c[c["pos"] < c["qid"].map(nq["n_docs"])]
    ch = c[c["pos"] < c["qid"].map(n_chunks)]
    deep = c[c["pos"] < (c["qid"].map(nq["n_docs"]) * args.depth).round()]
    chunk_rows = {"kind": "chunk", "pos": ch["pos"], "doc_id": ch["doc_id"], "qid": ch["qid"], "size": 4500,
                  "src": "seed"}
    out = pd.concat([
        pd.DataFrame({"job": "pN3_base", "qid": docs["qid"], "kind": "doc", "pos": docs["pos"],
                      "doc_id": docs["doc_id"], "size": 0, "src": ""}),
        pd.DataFrame({"job": "pN3_base", **chunk_rows}),
        pd.DataFrame({"job": "N1_n125", "qid": deep["qid"], "kind": "doc", "pos": deep["pos"],
                      "doc_id": deep["doc_id"], "size": 0, "src": "doc"}),
        pd.DataFrame({"job": "N1_n125", **chunk_rows}),
    ], ignore_index=True)[["job", "qid", "kind", "pos", "doc_id", "size", "src"]]
    out = out.astype({"job": str, "qid": "int64", "kind": str, "pos": "int64", "doc_id": "int64", "size": "int64",
                      "src": str})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print((out.groupby(["job", "kind"]).size() / out["qid"].nunique()).round(1).to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
