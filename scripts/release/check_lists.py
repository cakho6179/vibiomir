"""Compare rebuilt doc/chunk lists with the lists the submitted zips were built from (or with a submitted zip).

    python scripts/release/check_lists.py NEW.parquet kaggle/datasets/mat_lists22/lists.parquet --job pN3_base
    python scripts/release/check_lists.py NEW.parquet submissions/pN3_base_pqchunks.zip --job pN3_base

Per kind (doc, chunk): queries whose doc_id sequence is identical, queries whose doc_id set is identical (the
metric is set based: P/R/F2 do not depend on the order) and the total symmetric difference in docs.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd


def from_zip(path: Path) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".json") or n.endswith(".jsonl"))
        raw = z.read(name).decode("utf-8")
    items = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(x) for x in raw.splitlines() if x]
    for it in items:
        q = int(it["id"])
        rows += [(q, "doc", i, int(d)) for i, d in enumerate(it.get("relevant_docs", []))]
        rows += [(q, "chunk", i, int(c["doc_id"])) for i, c in enumerate(it.get("relevant_chunks", []))]
    return pd.DataFrame(rows, columns=["qid", "kind", "pos", "doc_id"])


def load(path: Path, job: str) -> pd.DataFrame:
    if path.suffix == ".zip":
        return from_zip(path)
    d = pd.read_parquet(path)
    return d[d["job"] == job] if "job" in d else d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new", type=Path)
    ap.add_argument("ref", type=Path)
    ap.add_argument("--job", required=True)
    args = ap.parse_args()
    a, b = load(args.new, args.job), load(args.ref, args.job)
    if a.empty or b.empty:
        raise SystemExit(f"job {args.job}: {len(a)} new rows, {len(b)} reference rows")
    ok = True
    for kind in ("doc", "chunk"):
        x = a[a["kind"] == kind].sort_values(["qid", "pos"]).groupby("qid")["doc_id"].apply(tuple)
        y = b[b["kind"] == kind].sort_values(["qid", "pos"]).groupby("qid")["doc_id"].apply(tuple)
        qs = sorted(set(x.index) | set(y.index))
        seq = sum(x.get(q, ()) == y.get(q, ()) for q in qs)
        sets = sum(set(x.get(q, ())) == set(y.get(q, ())) for q in qs)
        sym = sum(len(set(x.get(q, ())) ^ set(y.get(q, ()))) for q in qs)
        n_a, n_b = sum(map(len, x)), sum(map(len, y))
        print(f"{args.job} {kind:5s}: rows {n_a:,} vs {n_b:,} | same order {seq}/{len(qs)} | same set {sets}/{len(qs)} "
              f"| docs differing {sym}")
        ok &= sets == len(qs)
    print("MATCH (set level)" if ok else "DIFFERENT")


if __name__ == "__main__":
    main()
