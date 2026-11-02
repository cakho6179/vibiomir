"""Doc lists of scored submissions, read from their zips (streamed: the chunk text is never parsed), plus their scores.

    python scripts/analysis/sub_doc_lists.py --out runs/reports/pgold_lists

Writes lists.parquet (sid, qid, doc_id, rank) and scores.parquet (sid, file, docs P, R, F2) for every submission in
SUBS. Scores come from runs/reports/codabench_scores.json (fetch_scores.py). Doc ids are the official BTC ids, the same
in every docstore version, so lists from older docstores are comparable. Only aggregate scores, never per-pair labels.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "submissions"
# one zip per distinct doc list (submissions with identical docs P/R are the same list with other chunks)
SUBS = {
    6749: S / "w12T2_VI_win1500_n150.zip", 6537: S / "rd_VI_d150c90r_nb.zip",
    6767: S / "probe_sliceB_w12T2_VI.zip", 6768: S / "probe_sliceA_w12T2_VI.zip", 6769: S / "probe_r1ord_w12T2_VI.zip",
    6932: S / "mixA999_w12T2_VI_win4500_n150.zip", 6935: S / "mixA100_w12T2_VI_win4500_n150.zip",
    7030: S / "p1_zh10_v21chunk.zip", 7034: S / "p2_en10_v21section.zip", 7036: S / "p3_vinew30_base.zip",
    7043: S / "p5_pool5000_base.zip", 7044: S / "p6_vi3000zh1000en1000_base.zip", 7191: S / "p7_ext35.zip",
    7214: S / "pA_zh250en250_ext60.zip", 7215: S / "pB_vi800zh200en200_extn60.zip",
    7225: S / "r1_rr197_rrchunk148.zip", 7226: S / "r2_vitail100_rrchunk148f04.zip",
    7227: S / "r3_zhen50_rrchunk250f04.zip", 7230: S / "r8_m3rr197.zip", 7231: S / "r6_zh25.zip",
    7324: S / "r13_rd197_rdchunk148.zip", 7517: S / "pN1_pgold_r26chunks.zip",
    7534: S / "X40_lexhead_r26chunks.zip", 7537: S / "ab_yes_direct_tail.zip", 7548: S / "ab_no_direct_tail.zip",  # #7539 failed server-side
    7584: S / "pN2_bmllm_r26chunks.zip", 7589: S / "pN3_base_pqchunks.zip",
    7608: S / "pH_host_hostx7589.zip", 7609: S / "p9b_top_docsonly.zip", 7632: S / "pVR_slice_docsonly.zip", 7634: S / "pR4_slice_docsonly.zip", 7635: S / "pPho_slice_docsonly.zip",
}
# lists whose zip is not kept locally but whose doc list is (materialize_lists output)
PARQ = {7423: (ROOT / "kaggle/datasets/mat_lists19/lists.parquet", "r31"),
        7426: (ROOT / "kaggle/datasets/mat_lists19/lists.parquet", "r30")}
REC = re.compile(rb'\{"id": (\d+), "relevant_docs": \[([0-9, ]*)\]')


def read_docs(path: Path) -> pd.DataFrame:
    rows, tail = [], b""
    with zipfile.ZipFile(path) as z, z.open("submission.json") as fh:
        while block := fh.read(1 << 25):
            buf = tail + block
            end = 0
            for m in REC.finditer(buf):
                ids = [int(x) for x in m.group(2).split(b",") if x.strip()]
                rows += [(int(m.group(1)), d, r) for r, d in enumerate(dict.fromkeys(ids))]
                end = m.end()
            tail = buf[max(end, len(buf) - 200_000):]  # a record cut by the block boundary is matched next time
    df = pd.DataFrame(rows, columns=["qid", "doc_id", "rank"])
    return df.drop_duplicates(["qid", "doc_id"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/reports/pgold_lists")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sc = {}  # the backup keeps the scores of the old submissions deleted on 29/10 to free server space
    for f in ("codabench_scores_backup_2026-10-29.json", "codabench_scores.json"):
        sc.update({s["id"]: s for s in json.loads((ROOT / "runs/reports" / f).read_text(encoding="utf-8"))["ours"]
                   if s["status"] == "Finished" and s.get("DOCS_PRECISION") is not None})
    lists, scores = [], []
    for sid, path in SUBS.items():
        if sid not in sc or not path.exists():
            print(f"#{sid}: skipped ({'no score' if sid not in sc else 'no zip'})", flush=True)
            continue
        d = read_docs(path).assign(sid=sid)
        s = sc[sid]
        scores.append((sid, path.name, s["DOCS_PRECISION"], s["DOCS_RECALL"], s["DOCS_F2MACRO"]))
        lists.append(d)
        print(f"#{sid} {path.name}: {d['qid'].nunique()} queries, {len(d) / max(d['qid'].nunique(), 1):.1f} docs/q, "
              f"P/R/F2 {s['DOCS_PRECISION']}/{s['DOCS_RECALL']}/{s['DOCS_F2MACRO']}", flush=True)
    for sid, (path, job) in PARQ.items():
        l = pd.read_parquet(path, columns=["job", "qid", "kind", "doc_id"])
        d = l[(l["job"] == job) & (l["kind"] == "doc")].drop_duplicates(["qid", "doc_id"])
        d = d.assign(sid=sid, rank=d.groupby("qid").cumcount())
        s = sc[sid]
        scores.append((sid, f"{path.parent.name}:{job}", s["DOCS_PRECISION"], s["DOCS_RECALL"], s["DOCS_F2MACRO"]))
        lists.append(d)
        print(f"#{sid} {job}: {len(d) / d['qid'].nunique():.1f} docs/q", flush=True)
    pd.concat(lists, ignore_index=True)[["sid", "qid", "doc_id", "rank"]].to_parquet(args.out / "lists.parquet", index=False)
    pd.DataFrame(scores, columns=["sid", "file", "P", "R", "F2"]).to_parquet(args.out / "scores.parquet", index=False)


if __name__ == "__main__":
    main()
