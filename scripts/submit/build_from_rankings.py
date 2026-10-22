"""Fuse Kaggle rankings and write submission zips (chunk = whole document, uncapped).

A SYSTEM is one or more ranking.parquet files (qid, rank, doc_id, score) joined by "+": shard
parts of the same model are merged by score. Several systems are fused by weighted RRF.
    python scripts/submit/build_from_rankings.py --system kaggle/outputs/vibiomir-bm25-e1/ranking.parquet \
        --system "kaggle/outputs/vibiomir-dense-m3-s01/ranking.parquet+...s23/...+..." --weights 1,1 \
        --n 28,45,70 --name e1_m3_rrf
-> runs/submissions/<name>_n<N>.zip + runs/submissions/<name>_ranking.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import orjson
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT))
from r2ai2026.paths import DATA, QUERIES  # noqa: E402
from r2ai2026.submit import validate_submission, write_submission_zip  # noqa: E402

RRF_K = 60
OUT = ROOT / "runs" / "submissions"


def load_system(spec: str, depth: int) -> pd.DataFrame:
    parts = pd.concat([pd.read_parquet(p) for p in spec.split("+")], ignore_index=True)
    parts = parts.sort_values(["qid", "score"], ascending=[True, False]).drop_duplicates(["qid", "doc_id"])
    parts["rank"] = parts.groupby("qid").cumcount()
    return parts[parts["rank"] < depth]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", action="append", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--depth", type=int, default=200, help="ranks of each system used for fusion")
    ap.add_argument("--n", default="28,45,70")
    ap.add_argument("--name", required=True)
    ap.add_argument("--zip", choices=["deflate", "lzma"], default="lzma",
                    help="lzma keeps full text under the ~100 MB upload limit")
    args = ap.parse_args()
    weights = [float(w) for w in args.weights.split(",")] if args.weights else [1.0] * len(args.system)
    assert len(weights) == len(args.system)

    systems = [load_system(s, args.depth) for s in args.system]
    if len(systems) == 1:
        fused = systems[0][["qid", "doc_id", "score"]]
    else:
        fused = pd.concat([s.assign(score=w / (RRF_K + s["rank"] + 1)) for s, w in zip(systems, weights)])
        fused = fused.groupby(["qid", "doc_id"], as_index=False)["score"].sum()
    fused = fused.sort_values(["qid", "score"], ascending=[True, False])
    fused["rank"] = fused.groupby("qid").cumcount()
    OUT.mkdir(parents=True, exist_ok=True)
    fused.to_parquet(OUT / f"{args.name}_ranking.parquet", index=False)

    ns = [int(n) for n in args.n.split(",")]
    by_q = {q: g["doc_id"].tolist() for q, g in fused[fused["rank"] < max(ns)].groupby("qid")}
    need = {d for docs in by_q.values() for d in docs}
    texts = {}
    with open(DATA / "docs_clean.jsonl", "rb") as f:
        for line in f:
            d = orjson.loads(line)
            if d["id"] in need:
                texts[d["id"]] = f"{d.get('title') or ''}\n{d.get('text') or ''}".strip()
    qids = pd.read_parquet(QUERIES)["id"].astype(int).tolist()
    for n in ns:
        records = []
        for q in qids:
            top = [int(d) for d in by_q.get(q, [])[:n] if d in texts]
            records.append({"id": q, "relevant_docs": top,
                            "relevant_chunks": [{"doc_id": d, "chunk_text": texts[d]} for d in top]})
        problems = validate_submission(records, expected_ids=qids)
        if problems:
            raise SystemExit(f"invalid submission n={n}: {problems[:5]}")
        stats = write_submission_zip(records, OUT / f"{args.name}_n{n}.zip", method=args.zip)
        print(f"{args.name}_n{n}.zip", stats)


if __name__ == "__main__":
    main()
