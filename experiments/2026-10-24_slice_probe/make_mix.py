"""Base docs + the top-k of a union stratum, chunks of the base zip unchanged (no corpus needed).

    python experiments/2026-10-24_slice_probe/make_mix.py --base submissions/w12T2_VI_win4500_n150.zip --k 225,100

relevant_docs = base docs (base order) followed by the first k sliceA docs of the query in union-rank order
(sliceA: >= 2 families, not in the base, union rank <= 500; density 7.0% on #6768, above F2/5).
--gate-min-n N adds it only where the base kept >= N docs (#6932: adding it everywhere lowered macro Docs F2; the
gated probe splits that loss between the two halves of the queries).
Writes submissions/mixA{k}[g{N}]_{base stem}.zip and mix_stats.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_probes import HERE, ROOT, base_frame, probe_lists, read_submission_zip, write_submission_zip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=ROOT / "submissions" / "w12T2_VI_win4500_n150.zip")
    ap.add_argument("--k", default="225", help="comma-separated stratum depths; 999 = whole stratum")
    ap.add_argument("--gate-min-n", type=int, default=0,
                    help="add the stratum only to queries whose base list has >= this many docs (0 = all)")
    ap.add_argument("--out", type=Path, default=ROOT / "submissions")
    args = ap.parse_args()

    records = read_submission_zip(args.base)
    stratum = probe_lists(base_frame(records))["sliceA"]
    by_q = stratum.groupby("qid")["doc_id"].apply(list).to_dict()
    stats = {}
    for k in map(int, args.k.split(",")):
        new = [{**r, "relevant_docs": r["relevant_docs"] + [int(d) for d in by_q.get(int(r["id"]), [])[:k]]}
               if len(r["relevant_docs"]) >= args.gate_min_n else r for r in records]
        gate = f"g{args.gate_min_n}" if args.gate_min_n else ""
        out = args.out / f"mixA{k}{gate}_{args.base.stem}.zip"
        zs = write_submission_zip(new, out, method="lzma")
        stats[out.name] = {"docs_per_q": round(sum(len(r["relevant_docs"]) for r in new) / len(new), 2),
                           "zip_mb": round(zs["zip_mb"], 1)}
        print(out.name, json.dumps(stats[out.name]), flush=True)
    (HERE / f"mix_stats{gate}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
