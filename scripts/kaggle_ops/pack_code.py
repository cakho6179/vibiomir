"""Stage the system source as a Kaggle dataset folder, so kernels run the same entry point as any machine.

    python scripts/kaggle_ops/pack_code.py --user tthiutoko [--out kaggle/datasets/code_<user>]

Copies r2ai2026/, configs/, the stage scripts under kaggle/*_exp/ and kaggle/common/ (no bundles, outputs or
data) into OUT/vibiomir-code.zip plus dataset-metadata.json and CODE_VERSION (git commit + dirty flag). Upload
with `kaggle datasets create -p OUT` (first time) or `kaggle datasets version -p OUT -m <commit>`.
A kernel then runs: unzip to /kaggle/working/code, `cd` there, `python -m r2ai2026 run --config ... --var ...`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ["r2ai2026/*.py", "configs/*.yaml", "kaggle/*_exp/*.py", "kaggle/common/*.py", "pyproject.toml"]


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / "kaggle" / "datasets" / f"code_{args.user}"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted({p for pat in INCLUDE for p in ROOT.glob(pat) if "__pycache__" not in p.parts})
    version = f"{git('rev-parse', '--short', 'HEAD')}{'-dirty' if git('status', '--porcelain') else ''}"
    with zipfile.ZipFile(out / "vibiomir-code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr("CODE_VERSION", version + "\n")
    meta = {"title": "vibiomir-code", "id": f"{args.user}/vibiomir-code", "licenses": [{"name": "other"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(files)} files, version {version} -> {out}")


if __name__ == "__main__":
    main()
