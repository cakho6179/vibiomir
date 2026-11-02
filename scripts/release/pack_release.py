"""Pack the data + code release that backs the BTC product description (docs/THUYET_MINH_SAN_PHAM.md).

    python scripts/release/pack_release.py                 # zips of the small groups + MANIFEST (sha256 of everything)
    python scripts/release/pack_release.py --hash-big      # also hash the docstore / index files (~12 GB, slow)

Writes release/ (outside git):
  vibiomir_code_<commit>.zip        git archive of HEAD (the code; untracked files are not in it)
  artifacts_<group>.zip             one zip per small group, paths relative to the repo root: unzip at the repo root
  MANIFEST.tsv                      group, path, bytes, sha256 for every shipped file, the big groups and the
                                    submitted zips (hash only)
The big groups (docstore corpus, retrieval index) are not copied: upload those folders as they are.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "release"

# group -> glob patterns relative to the repo root
SMALL = {
    # tier A: rebuild the #7892 lists on a CPU (pgold_all -> pn3_lists) and check them against the submitted ones
    "A_pgold": [
        "data/raw/btc/query.parquet",
        "kaggle/datasets/rerank_pool/cands.parquet",
        "kaggle/outputs/rr3-p*/rerank.parquet", "kaggle/outputs/rr3-p*/*.log",
        "kaggle/outputs/m3m-p*/m3multi.parquet", "kaggle/outputs/m3m-p*/*.log",
        "runs/reports/pgold_lists/lists.parquet", "runs/reports/pgold_lists/scores.parquet",
        "runs/reports/pgold_feats/base2/*",
        "kaggle/datasets/mat_lists22/*", "kaggle/datasets/mat_lists29/*", "kaggle/datasets/mat_lists30/*",
        "kaggle/datasets/mat_lists31/*",
    ],
    # chunk text: passage relevance profiles + the code bundle the materialize kernels import
    "B_materialize": [
        "kaggle/datasets/pas_scores/*",
        "kaggle/outputs/rr4-p*/rerank.parquet", "kaggle/outputs/rr4-p*/passages.parquet", "kaggle/outputs/rr4-p*/*.log",
        "kaggle/chunk_v21_assets/*",
    ],
    # doc2query questions and their scores
    "C_doc2query": [
        "kaggle/outputs/dq-g*/questions.parquet", "kaggle/outputs/dq-g*/stats.json", "kaggle/outputs/dq-g*/*.log",
        "kaggle/outputs/dqt-p*/questions.parquet", "kaggle/outputs/dqt-p*/*.log",
        "kaggle/outputs/dq-all/*", "kaggle/outputs/dq-all2/*",
        "runs/reports/dq_cross/*", "runs/reports/dq_cross_all/*", "runs/reports/dq_bm25/*", "runs/reports/dq_bm25_all/*",
    ],
    # tier B outputs: query frame, query variants, first-stage rankings, candidate pool inputs
    "D_retrieval": [
        "kaggle/outputs/query-frame-v3/*",
        "kaggle/datasets/qvariants_v3_acc1/*",
        "kaggle/outputs/qv3-bm25-*/ranking.parquet", "kaggle/outputs/qv3-bm25-*/*.log",
        "kaggle/outputs/qv3-m3-*/ranking.parquet", "kaggle/outputs/qv3-m3-*/*.log",
        "data/rankings_v21/*.parquet",
        "kaggle/datasets/probe_inputs/additions.parquet",
    ],
}
BIG = {
    "E_docstore": ["data/docstore/v2.1/corpus/*", "data/docstore/v2.1/*.parquet", "data/docstore/v2.1/*.json"],
    "F_index": ["data/docstore/v2.1/index/**/*"],
}
SUBMITTED = ["submissions/pN3_base_pqchunks.zip", "submissions/S1_dqd20.zip", "submissions/S2_dqb20.zip",
             "submissions/N1_n125.zip", "submissions/F2_dqd20b20g.zip"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def files(patterns: list[str]) -> list[Path]:
    out: dict[str, Path] = {}
    for pat in patterns:
        for p in ROOT.glob(pat):
            if p.is_file():
                out[p.relative_to(ROOT).as_posix()] = p
    return [out[k] for k in sorted(out)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hash-big", action="store_true")
    ap.add_argument("--skip-zip", action="store_true", help="only (re)write MANIFEST.tsv")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rows = []
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                            check=True).stdout.strip()
    code = OUT / f"vibiomir_code_{commit}.zip"
    if not args.skip_zip:
        subprocess.run(["git", "archive", "--format=zip", "-o", str(code), "HEAD"], cwd=ROOT, check=True)
    rows.append(("code", code.name, code.stat().st_size, sha256(code)))
    for group, pats in SMALL.items():
        fs = files(pats)
        if not fs:
            raise SystemExit(f"{group}: no files")
        target = OUT / f"artifacts_{group}.zip"
        if not args.skip_zip:
            with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
                for p in fs:
                    z.write(p, p.relative_to(ROOT).as_posix())
        size = 0
        for p in fs:
            rel = p.relative_to(ROOT).as_posix()
            rows.append((group, rel, p.stat().st_size, sha256(p)))
            size += p.stat().st_size
        rows.append((group + "_zip", target.name, target.stat().st_size, sha256(target)))
        print(f"{group}: {len(fs)} files, {size / 2**20:,.1f} MiB -> {target.name}", flush=True)
    for group, pats in BIG.items():
        fs = files(pats)
        size = sum(p.stat().st_size for p in fs)
        for p in fs:
            rows.append((group, p.relative_to(ROOT).as_posix(), p.stat().st_size, sha256(p) if args.hash_big else "-"))
        print(f"{group}: {len(fs)} files, {size / 2**30:,.2f} GiB (not copied; upload the folder)", flush=True)
    for s in SUBMITTED:
        p = ROOT / s
        rows.append(("submitted", s, p.stat().st_size, sha256(p)))
    with open(OUT / "MANIFEST.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("group\tpath\tbytes\tsha256\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"-> {OUT / 'MANIFEST.tsv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
