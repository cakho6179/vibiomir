"""Single entry point: rebuild the private submissions from the released artifacts.

    python reproduce.py                       # lists of the 5 private submissions + checks (CPU, ~1 min)
    python reproduce.py --refit               # first refit p(gold) from the scored lists (~4 min more)
    python reproduce.py --zip                 # also build the reported submission #7892 (chunk text + LZMA zip, ~25 min)
    python reproduce.py --zip --jobs all      # all 5 zips
    python reproduce.py --zip --submitted DIR # also diff the built zips against submitted ones, query by query

Each built zip is checked against the sha256 of submission.json in the submitted zip (no submitted file needed).

Inputs (README section 2): artifacts_A_pgold.zip (+ C for the doc2query jobs, + B and the docstore corpus for --zip),
unzipped at the repo root. Output: out/final/ (lists.parquet, report.json, zips/).

Steps
  1. p(gold) per candidate and per-query depth N_q        scripts/analysis/pgold_all.py   (only with --refit)
  2. doc / chunk lists of pN3_base (#7892), N1_n125 (#7916) scripts/analysis/pn3_lists.py
  3. doc2query jobs S1 (#7893), S2 (#7894), F2 (#7917)     scripts/analysis/dq_lists.py
  4. check every list against the lists the submitted zips were built from (kaggle/datasets/mat_lists*)
  5. --zip: chunk text (4,500-char windows, chunk_stair "seed") + LZMA zip with the exact code of the Kaggle
     kernel that built the submissions (kaggle/kernels_generic/vibiomir-mat-st3-acc2, pinned r2ai2026 bundle)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OUT = ROOT / "out" / "final"

# job -> (private submission id, submitted file, reference lists the zip was built from)
JOBS = {
    "pN3_base": (7892, "pN3_base_pqchunks.zip", "kaggle/datasets/mat_lists22/lists.parquet"),
    "S1_dqd20": (7893, "S1_dqd20.zip", "kaggle/datasets/mat_lists29/lists.parquet"),
    "S2_dqb20": (7894, "S2_dqb20.zip", "kaggle/datasets/mat_lists29/lists.parquet"),
    "N1_n125": (7916, "N1_n125.zip", "kaggle/datasets/mat_lists30/lists.parquet"),
    "F2_dqd20b20g": (7917, "F2_dqd20b20g.zip", "kaggle/datasets/mat_lists31/lists.parquet"),
}
# sha256 of submission.json inside each submitted zip (the zip headers carry timestamps, the json does not)
SUBMITTED_JSON_SHA256 = {
    "pN3_base": "acfa3707419de6803a7cb74e85d872c6cfc1557cf19c45726cf85d5f418fbefa",
    "S1_dqd20": "7ed0be3cc470c8ff078b92f7490ca55efcc4e5b317cf3d3aab6c6b8631c3ab31",
    "S2_dqb20": "d5aa35f2cef2ccfbf0ef318eb10602a5f097a4330e87bfd1814308f5997f8d37",
    "N1_n125": "854a766ecc7c3578c132f164d0b1c12608f5c67e01b484cbc0a07c82be930a8d",
    "F2_dqd20b20g": "07b5cba143ac4b6fb5785eae0fe567be9f3281180797ce041f560f8dc5b47dad",
}
NEEDS = {
    "A": ["kaggle/datasets/rerank_pool/cands.parquet", "runs/reports/pgold_lists/lists.parquet",
          "runs/reports/pgold_feats/base2/cands_p.parquet", "data/raw/btc/query.parquet"],
    "C": ["runs/reports/dq_cross_all/scores.parquet", "runs/reports/dq_bm25_all/scores.parquet",
          "runs/reports/dq_cross/scores.parquet", "runs/reports/dq_bm25/scores.parquet"],
    "B": ["kaggle/datasets/pas_scores/pas_scores.parquet", "kaggle/chunk_v21_assets/vibiomir-code.zip",
          "data/docstore/v2.1/corpus/part-00.jsonl.gz"],
}
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def run(*args: str) -> None:
    log(" ".join(args))
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def require(group: str) -> None:
    missing = [p for p in NEEDS[group] if not (ROOT / p).exists()]
    if missing:
        hint = {"A": "artifacts_A_pgold.zip", "C": "artifacts_C_doc2query.zip",
                "B": "artifacts_B_materialize.zip and docstore_v2.1/ -> data/docstore/v2.1/"}[group]
        raise SystemExit(f"missing {missing}: unzip {hint} at the repo root (README section 2)")


def compare_lists(new: pd.DataFrame, ref: pd.DataFrame) -> dict:
    out = {}
    for kind in ("doc", "chunk"):
        a = new[new["kind"] == kind].sort_values(["qid", "pos"]).groupby("qid")["doc_id"].apply(tuple)
        b = ref[ref["kind"] == kind].sort_values(["qid", "pos"]).groupby("qid")["doc_id"].apply(tuple)
        qs = sorted(set(a.index) | set(b.index))
        out[kind] = {"queries": len(qs), "same_order": int(sum(a.get(q, ()) == b.get(q, ()) for q in qs)),
                     "same_set": int(sum(set(a.get(q, ())) == set(b.get(q, ())) for q in qs)),
                     "rows": [int(sum(map(len, a))), int(sum(map(len, b)))]}
    return out


def read_zip(path: Path) -> dict[int, dict]:
    with zipfile.ZipFile(path) as z:
        items = json.loads(z.read("submission.json").decode("utf-8"))
    return {int(x["id"]): x for x in items}


def link_dir(target: Path, link: Path) -> None:
    if os.name == "nt":  # a directory junction needs no admin rights
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        link.symlink_to(target, target_is_directory=True)


def build_zips(lists: pd.DataFrame, jobs: list[str], out: Path) -> dict[str, Path]:
    require("B")
    stage, tools, zips = out / "mat_in", out / "tools", out / "zips"
    corpus = stage / "corpus"
    if corpus.is_symlink() or getattr(corpus, "is_junction", lambda: False)():  # drop the link only, never the docstore behind it
        os.rmdir(corpus) if os.name == "nt" else corpus.unlink()
    for d in (stage, tools):
        if d.exists():
            shutil.rmtree(d)
    stage.mkdir(parents=True)
    lists[lists["job"].isin(jobs)].to_parquet(stage / "lists.parquet", index=False)
    shutil.copy(ROOT / "data/raw/btc/query.parquet", stage / "query.parquet")
    shutil.copy(ROOT / "kaggle/datasets/pas_scores/pas_scores.parquet", stage / "pas_scores.parquet")
    with zipfile.ZipFile(ROOT / "kaggle/chunk_v21_assets/vibiomir-code.zip") as z:  # pinned r2ai2026 (06f71bf-dirty)
        z.extractall(stage / "vibiomir-code")
    link_dir((ROOT / "data/docstore/v2.1/corpus").resolve(), stage / "corpus")
    run("scripts/release/extract_bundle.py", "kaggle/kernels_generic/vibiomir-mat-st3-acc2", "--out", str(tools))
    run(str(tools / "materialize_lists.py"), "--input", str(stage), "--out", str(zips))
    return {j: zips / f"{j}.zip" for j in jobs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refit", action="store_true", help="refit p(gold) instead of using the released base2 fit")
    ap.add_argument("--zip", action="store_true", help="also build the submission zip(s)")
    ap.add_argument("--jobs", default="pN3_base", help="jobs to zip: comma list or 'all' (default: the reported one)")
    ap.add_argument("--submitted", type=Path, default=ROOT / "submissions",
                    help="folder with the submitted zips, for the zip comparison")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    require("A")
    report: dict = {"refit": args.refit}

    model = "base2"
    if args.refit:  # the 5 lists scored after the fit was frozen are left out, as when it was made
        run("scripts/analysis/pgold_all.py", "--lists", "runs/reports/pgold_lists",
            "--out", "runs/reports/pgold_feats/base2_refit", "--exclude", "7584,7589,7608,7609,7632")
        model = "base2_refit"
    pn3 = out / "lists_pn3.parquet"
    run("scripts/analysis/pn3_lists.py", "--model", model, "--out", str(pn3))
    parts = [pd.read_parquet(pn3)]
    have_dq = all((ROOT / p).exists() for p in NEEDS["C"])
    if have_dq:
        run("scripts/analysis/dq_lists.py", "--base", str(pn3), "--out", str(out / "lists_dq29.parquet"),
            "--job", "S1_dqd20:dense_head=20", "--job", "S2_dqb20:bm25_head=20", "--gate", "0")
        run("scripts/analysis/dq_lists.py", "--base", str(pn3), "--out", str(out / "lists_dq31.parquet"),
            "--job", "F2_dqd20b20g:dense=20,bm25=20", "--chunks", "40", "--gate", "200")
        parts += [pd.read_parquet(out / "lists_dq29.parquet"), pd.read_parquet(out / "lists_dq31.parquet")]
    else:
        log("artifacts_C_doc2query not unzipped: skipping S1 / S2 / F2")
    lists = pd.concat(parts, ignore_index=True)
    lists = lists[lists["job"].isin(JOBS)]
    lists.to_parquet(out / "lists.parquet", index=False)

    print(f"\n{'job':14s} {'sub':>6s}  {'docs/q':>7s} {'chunks/q':>8s}  docs same order / set   chunks same order / set")
    report["lists"] = {}
    for job, (sid, _, ref) in JOBS.items():
        new = lists[lists["job"] == job]
        if new.empty:
            continue
        r = pd.read_parquet(ROOT / ref)
        c = compare_lists(new, r[r["job"] == job])
        report["lists"][job] = {"submission": sid, **c}
        d, k = c["doc"], c["chunk"]
        print(f"{job:14s} #{sid:5d}  {d['rows'][0] / 1200:7.1f} {k['rows'][0] / 1200:8.1f}  "
              f"{d['same_order']:4d} / {d['same_set']:4d} of {d['queries']}     "
              f"{k['same_order']:4d} / {k['same_set']:4d} of {k['queries']}")

    if args.zip:
        jobs = list(JOBS) if args.jobs == "all" else args.jobs.split(",")
        jobs = [j for j in jobs if j in set(lists["job"])]
        built = build_zips(lists, jobs, out)
        report["zips"] = {}
        for job, path in built.items():
            with zipfile.ZipFile(path) as z:
                digest = hashlib.sha256(z.read("submission.json")).hexdigest()
            entry = {"zip": str(path.relative_to(ROOT)), "mb": round(path.stat().st_size / 2**20, 1),
                     "submission_json_sha256": digest,
                     "same_as_submitted": digest == SUBMITTED_JSON_SHA256[job]}
            sub = args.submitted / JOBS[job][1]
            if sub.exists():  # optional: per-query diff against the submitted file
                a, b = read_zip(path), read_zip(sub)
                entry.update(submitted=str(sub), identical_queries=f"{sum(a.get(q) == b.get(q) for q in b)}/{len(b)}")
            report["zips"][job] = entry
            log(f"{job}: {entry}")
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log(f"done -> {out / 'report.json'}")


if __name__ == "__main__":
    main()
