"""Build materialiser lists that append doc2query picks to the pN3_base (#7589) layout.

    python scripts/analysis/dq_lists.py --out kaggle/datasets/mat_lists31/lists.parquet \
        --job F1_dqd20:dense=20 --job F2_dqd20b20:dense=20,bm25=20 [--chunks 40] [--gate 200]

The pN3 base is recovered from mat_lists29 job S1_dqd20: its docs without src "doc" are the pN3 docs, and its
chunk rows of those docs are the pN3 chunks minus the 20 lowest-p ones (the budget S1/S2 freed under the ~100 MB cap).
Each job appends, per query, the top picks of each named channel that are not in the base list (channels interleaved
in the order given, duplicates skipped), then gives the first `--chunks` added docs a 4500-char "seed" chunk and drops
that many extra lowest-p base chunks beyond the 20 already dropped.

Tail gate: docs that only got questions from the dqt kernels (the "tail", whose gold density S1/S2 did not measure)
are kept only when the other channel also ranks them in its top `--gate` for that query (0 disables the gate).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
CHANNELS = {"dense": ROOT / "runs/reports/dq_cross_all/scores.parquet",
            "bm25": ROOT / "runs/reports/dq_bm25_all/scores.parquet",
            "dense_head": ROOT / "runs/reports/dq_cross/scores.parquet",
            "bm25_head": ROOT / "runs/reports/dq_bm25/scores.parquet"}
OTHER = {"dense": "bm25", "bm25": "dense"}


def ranked(path: Path) -> dict[int, list[int]]:
    sc = pd.read_parquet(path, columns=["qid", "doc_id", "dq_rank"]).sort_values(["qid", "dq_rank"])
    return sc.groupby("qid")["doc_id"].apply(list).to_dict()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=ROOT / "kaggle/datasets/mat_lists29/lists.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--job", action="append", required=True, help="NAME:channel=n[,channel=n]")
    ap.add_argument("--chunks", type=int, default=20, help="added docs that get a chunk (and base chunks dropped)")
    ap.add_argument("--gate", type=int, default=200)
    args = ap.parse_args()

    L = pd.read_parquet(args.base)
    s1 = L[L["job"] == "S1_dqd20"]
    docs = s1[(s1["kind"] == "doc") & (s1["src"] != "doc")].sort_values(["qid", "pos"])
    added_s1 = set(zip(s1.loc[s1["src"] == "doc", "qid"], s1.loc[s1["src"] == "doc", "doc_id"]))
    ch = s1[s1["kind"] == "chunk"]
    ch = ch[[(q, d) not in added_s1 for q, d in zip(ch["qid"], ch["doc_id"])]].sort_values(["qid", "pos"])
    base_docs = docs.groupby("qid")["doc_id"].apply(list).to_dict()
    base_chunks = dict(tuple(ch.groupby("qid")))
    tail = set(pd.concat([pd.read_parquet(f, columns=["doc_id"])
                          for f in glob.glob(str(ROOT / "kaggle/outputs/dqt-p*/questions.parquet"))])["doc_id"])
    cache: dict[str, dict[int, list[int]]] = {}

    def lists(name: str) -> dict[int, list[int]]:
        if name not in cache:
            cache[name] = ranked(CHANNELS[name])
        return cache[name]

    out = []
    for spec in args.job:
        name, chans = spec.split(":")
        want = [(c, int(n)) for c, n in (x.split("=") for x in chans.split(","))]
        n_tail = n_add = 0
        for qid, base in base_docs.items():
            seen = set(base)
            picks: dict[str, list[int]] = {}
            for c, n in want:
                gate = set(lists(OTHER[c])[qid][: args.gate]) if args.gate and c in OTHER else None
                got = []
                for d in lists(c).get(qid, []):
                    if len(got) == n:
                        break
                    if d in seen or d in got or (gate is not None and d in tail and d not in gate):
                        continue
                    got.append(d)
                picks[c] = got
                seen |= set(got)  # a doc picked by an earlier channel does not use up a later channel's quota
            seen = set(base)
            add: list[int] = []
            for i in range(max(n for _, n in want)):
                for c, _ in want:
                    if i < len(picks[c]):
                        add.append(picks[c][i])
            n_add += len(add)
            n_tail += sum(d in tail for d in add)
            out.append(pd.DataFrame({"job": name, "qid": qid, "kind": "doc", "pos": range(len(base) + len(add)),
                                     "doc_id": base + add, "size": 0, "src": [""] * len(base) + ["doc"] * len(add)}))
            bc = base_chunks.get(qid, ch.iloc[:0])
            keep = bc.iloc[: max(len(bc) - max(args.chunks - 20, 0), 0)]
            cadd = add[: args.chunks]
            out.append(pd.DataFrame({"job": name, "qid": qid, "kind": "chunk",
                                     "pos": range(len(keep) + len(cadd)),
                                     "doc_id": list(keep["doc_id"]) + cadd,
                                     "size": list(keep["size"]) + [4500] * len(cadd),
                                     "src": list(keep["src"]) + ["seed"] * len(cadd)}))
        print(f"{name}: {n_add / len(base_docs):.1f} docs added/q, tail share {n_tail / max(n_add, 1):.2f}", flush=True)
    res = pd.concat(out, ignore_index=True).astype(L.dtypes.to_dict())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(args.out, index=False)
    print(res.groupby(["job", "kind"]).size().to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
