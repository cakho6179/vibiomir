"""Scorebook: every scored submission with the depth it was built at, so P/R can be turned into counts.

    python scripts/analysis/scorebook.py [--json runs/reports/codabench_scores.json] [--md docs/SCOREBOOK.md]

Reads the file written by fetch_scores.py (leaderboard + our submissions with all 7 metric columns) and the depth
table below (docs and chunks per query, measured on the zips themselves; update it when a new job is submitted).
For each submission it prints, next to the official P/R:

  TP   = P x N          correct items submitted (N = docs or chunks per query)
  |G|  = TP / R         implied gold size (docs ~57-72, chunks ~75 by the official scorer)
  conv = chunks_R / docs_R   share of our gold docs that also produced a matched gold chunk

P and R are macro-averaged per query, so TP and |G| are approximations: read them as trends, not as exact counts.
The point of the table is the P(N) curve (how fast precision decays with depth) and the chunk conversion rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())

# The chunk match rule changed on 24/10 01:12 UTC (added "LCS(c,g)/len(c) >= 0.4", i.e. a chunk must
# also be *covered* by the gold span, not just cover it).  The same file #6536 was re-submitted under
# both rules: 0.3108 -> 0.2687, chkF2 0.3083 -> 0.2242.  Scores before the cut are NOT comparable with
# scores after it, so the table below is split.  (This is what made us believe 0.2983 was a regression
# from 0.3108 -- it was not, 0.3108 simply predates the rule.)
RULE_CUT = "2026-10-24T01:12"

DEPTH: dict[str, tuple[float, float]] = {  # file -> (docs per query, chunks per query)
    "q3eonly_n28.zip": (28, 28),
    "rerank_r1_n28.zip": (28, 28),
    "w12_VI_n28.zip": (28, 28),
    "b3_VI_n28.zip": (28, 28),
    "e3v2_VI_n28.zip": (28, 28),
    "w12_VI_d150c28.zip": (150, 28),
    "w12T2_VI_d150c90r.zip": (88.2, 75.4),
    "rdT2_VI_d150c90r.zip": (103.4, 77.9),
    "rd_VI_d150c90r.zip": (150, 90),
    "pool_VI_d500c75r.zip": (500, 75),
    "pool_VI_d300c75r.zip": (300, 75),
    "pool_VI_d500c75r_nb.zip": (500, 75),
    "w12T2_VI_win1500_n150.zip": (88.2, 88.2),
    "w12T2_VI_win3000_n150.zip": (88.2, 88.2),
    "w12T2_VI_win4500_n150.zip": (88.2, 88.2),
    "probe_sliceB_w12T2_VI.zip": (58, 150),
    "probe_sliceA_w12T2_VI.zip": (225, 150),
    "probe_r1ord_w12T2_VI.zip": (88, 150),
    "mixA999_w12T2_VI_win4500_n150.zip": (314, 88),
    "w12T2_VI_win6000_n150.zip": (88, 88),
    "w12T2_VI_win7500_n150.zip": (88, 88),
    "mixA100_w12T2_VI_win4500_n150.zip": (187, 88),
    "chunkmix40_win4500_n150.zip": (88, 128),
    "p1_zh10_v21chunk.zip": (197, 88),
    "p2_en10_v21section.zip": (197, 88),
    "p3_vinew30_base.zip": (217, 88),
    "p5_pool5000_base.zip": (4972, 88),
    "p6_vi3000zh1000en1000_base.zip": (5001, 88),
    "p7_ext35.zip": (2502, 123),
    "pA_zh250en250_ext60.zip": (500, 148),
    "pB_vi800zh200en200_extn60.zip": (1200, 148),
    "r1_rr197_rrchunk148.zip": (197, 148),
    "r2_vitail100_rrchunk148f04.zip": (100, 148),
    "r3_zhen50_rrchunk250f04.zip": (50, 250),
    "r8_m3rr197.zip": (197, 148),
    "r6_zh25.zip": (25, 148),
    "r11_chkf07_m3rr197.zip": (197, 148),
    "r12_chkeq197.zip": (197, 197),
    "r13_rd197_rdchunk148.zip": (197, 148),
}

IDEAS: dict[str, tuple[str, ...]] = {  # reading groups: what each submission was testing
    "BM25 / lexical": ("b3_VI_n28.zip", "e3v2_VI_n28.zip", "rerank_r1_n28.zip"),
    "depth + threshold": ("w12T2_VI_d150c90r.zip", "rdT2_VI_d150c90r.zip", "pool_VI_d500c75r_nb.zip"),
    "chunk window": ("w12T2_VI_win1500_n150.zip", "w12T2_VI_win3000_n150.zip", "w12T2_VI_win4500_n150.zip",
                     "w12T2_VI_win6000_n150.zip", "w12T2_VI_win7500_n150.zip"),
    "extra docs / languages": ("mixA100_w12T2_VI_win4500_n150.zip", "mixA999_w12T2_VI_win4500_n150.zip",
                               "p1_zh10_v21chunk.zip", "p2_en10_v21section.zip", "p3_vinew30_base.zip",
                               "p5_pool5000_base.zip", "p6_vi3000zh1000en1000_base.zip", "p7_ext35.zip"),
    "pool ceiling / no base": ("pA_zh250en250_ext60.zip", "pB_vi800zh200en200_extn60.zip"),
    "order by reranker vs fusion": ("r1_rr197_rrchunk148.zip", "r8_m3rr197.zip"),
    "slice probes": ("probe_sliceA_w12T2_VI.zip", "probe_sliceB_w12T2_VI.zip", "probe_r1ord_w12T2_VI.zip",
                     "r2_vitail100_rrchunk148f04.zip", "r3_zhen50_rrchunk250f04.zip", "r6_zh25.zip"),
}


def row(s: dict) -> dict:
    f = s["file"].split("\\")[-1].split("/")[-1]
    nd, nc = DEPTH.get(f, (float("nan"), float("nan")))
    p, r = s.get("DOCS_PRECISION"), s.get("DOCS_RECALL")
    cp, cr = s.get("CHUNKS_PRECISION"), s.get("CHUNKS_RECALL")
    ok = s.get("FINAL_SCORE") is not None
    tp = p * nd if ok and nd == nd else float("nan")
    g = tp / r if ok and r else float("nan")
    tpc = cp * nc if ok and nc == nc else float("nan")
    gc = tpc / cr if ok and cr else float("nan")
    return {"file": f, "id": s["id"], "when": s["when"][:10], "status": s["status"], "nd": nd, "nc": nc,
            "rule": "old" if s["when"][:16] < RULE_CUT else "new",
            "final": s.get("FINAL_SCORE"), "df2": s.get("DOCS_F2MACRO"), "cf2": s.get("CHUNKS_F2MACRO"),
            "dp": p, "dr": r, "cp": cp, "cr": cr, "tp": tp, "g": g, "tpc": tpc, "gc": gc,
            "conv": (cr / r) if ok and r else float("nan")}


def fmt(v: float, w: int = 5, d: int = 3) -> str:
    if v is None or v != v:
        return " " * (w - 3) + "n/a"
    return f"{v:{w}.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=ROOT / "runs/reports/codabench_scores.json")
    ap.add_argument("--md", type=Path, default=None, help="also write the tables as markdown here")
    args = ap.parse_args()
    d = json.loads(args.json.read_text(encoding="utf-8"))
    rows = [row(s) for s in d["ours"]]
    scored = [x for x in rows if x["final"] is not None]
    scored.sort(key=lambda x: -x["final"])
    head = (f"phase {d['phase']} · {d['teams']} teams · our rank {d['rank']} · "
            f"{len(rows)} submissions ({len(scored)} scored, {len(rows) - len(scored)} failed)")

    lines = [head, "",
             f"Chunk rule changed {RULE_CUT[:10]} (added LCS(c,g)/len(c) >= 0.4). Rows marked `old` are not "
             "comparable with `new` rows: #6536 re-submitted unchanged went 0.3108 -> 0.2687.", "",
             "| # | file | date | rule | N doc | FINAL | docsF2 | chkF2 | dP | dR | TP | |G| | cP | cR | TPc | conv |",
             "|---:|---|---|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i, x in enumerate(scored, 1):
        lines.append(f"| {i} | {x['file']} | {x['when']} | {x['rule']} | {x['nd']:.0f} | {fmt(x['final'], 6, 4)} | {fmt(x['df2'], 6, 4)} | "
                     f"{fmt(x['cf2'], 6, 4)} | {fmt(x['dp'])} | {fmt(x['dr'])} | {fmt(x['tp'], 5, 1)} | "
                     f"{fmt(x['g'], 5, 1)} | {fmt(x['cp'])} | {fmt(x['cr'])} | {fmt(x['tpc'], 5, 1)} | "
                     f"{fmt(x['conv'], 5, 2)} |")
    failed = [x["file"] for x in rows if x["final"] is None]
    lines += ["", f"Failed ({len(failed)}): " + ", ".join(failed)]

    new_only = [x for x in scored if x["rule"] == "new"]
    lines += ["", "## Current rule only (the only rows that can guide a submission)", "",
              "| # | file | date | N doc | N chk | FINAL | docsF2 | chkF2 | dP | dR | cP | cR | conv |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i, x in enumerate(new_only, 1):
        lines.append(f"| {i} | {x['file']} | {x['when']} | {x['nd']:.0f} | {x['nc']:.0f} | {fmt(x['final'], 6, 4)} | "
                     f"{fmt(x['df2'], 6, 4)} | {fmt(x['cf2'], 6, 4)} | {fmt(x['dp'])} | {fmt(x['dr'])} | "
                     f"{fmt(x['cp'])} | {fmt(x['cr'])} | {fmt(x['conv'], 5, 2)} |")

    lines += ["", "## P(N): precision vs docs per query", "",
              "| N doc | file | dP | dR | TP |", "|---:|---|---:|---:|---:|"]
    for x in sorted([x for x in scored if x["nd"] == x["nd"]], key=lambda x: x["nd"]):
        lines.append(f"| {x['nd']:.0f} | {x['file']} | {fmt(x['dp'])} | {fmt(x['dr'])} | {fmt(x['tp'], 5, 1)} |")

    lines += ["", "## Reading groups", ""]
    by_file = {x["file"]: x for x in scored}
    for name, files in IDEAS.items():
        got = [by_file[f] for f in files if f in by_file]
        if not got:
            continue
        lines += [f"### {name}", "", "| file | N doc | FINAL | dP | dR | TP | conv |", "|---|---:|---:|---:|---:|---:|---:|"]
        for x in sorted(got, key=lambda x: -x["final"]):
            lines.append(f"| {x['file']} | {x['nd']:.0f} | {fmt(x['final'], 6, 4)} | {fmt(x['dp'])} | "
                         f"{fmt(x['dr'])} | {fmt(x['tp'], 5, 1)} | {fmt(x['conv'], 5, 2)} |")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.md:
        args.md.write_text(f"# Scorebook — {head}\n\n" + text.split("\n", 2)[2], encoding="utf-8")
        print(f"\n-> {args.md}")


if __name__ == "__main__":
    main()
