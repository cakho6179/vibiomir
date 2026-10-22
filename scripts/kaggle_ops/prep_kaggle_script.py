"""Wrap any local experiment script into a Kaggle kernel bundle for one account.

The script source is embedded in run.py, written to /kaggle/tmp at runtime and run with the
given arguments. "{data}" in the arguments becomes the mounted corpus dataset root and
"{input}" becomes /kaggle/input. Pure-ASCII sources only (Kaggle mis-decodes others).

    python scripts/kaggle_ops/prep_kaggle_script.py kaggle/translate_exp/translate_queries.py acc3 \
        vibiomir-translate --gpu -- --data {data} --out /kaggle/working
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_accounts import ACCOUNTS  # noqa: E402

RUN = '''import glob, subprocess, sys
from pathlib import Path
SRC = {src!r}
script = Path("/kaggle/tmp/{name}")
script.parent.mkdir(parents=True, exist_ok=True)
script.write_text(SRC, encoding="utf-8")
hits = glob.glob("/kaggle/input/**/part-00.jsonl*", recursive=True)
data = str(Path(hits[0]).parents[1]) if hits else "/kaggle/input"
pip = {pip!r}
if pip:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pip], check=True)
args = [a.replace("{{data}}", data).replace("{{input}}", "/kaggle/input") for a in {args!r}]
subprocess.run([sys.executable, str(script), *args], check=True)
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("script", type=Path)
    ap.add_argument("acc", choices=sorted(ACCOUNTS))
    ap.add_argument("slug")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--pip", default="", help="comma-separated packages to pip install first")
    ap.add_argument("--datasets", default="vibiomir-corpus-clean", help="comma-separated slugs of this account")
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:cut])
    a.args = argv[cut + 1:]  # everything after "--" goes to the wrapped script
    src = a.script.read_text(encoding="utf-8")
    if not src.isascii():
        raise SystemExit(f"{a.script} has non-ASCII characters")
    user = ACCOUNTS[a.acc]
    out = ROOT / "kaggle" / "kernels_generic" / f"{a.slug}-{a.acc}"
    out.mkdir(parents=True, exist_ok=True)
    args = a.args
    pip = [p for p in a.pip.split(",") if p]
    (out / "run.py").write_text(RUN.format(src=src, name=a.script.name, pip=pip, args=args), encoding="utf-8")
    meta = {"id": f"{user}/{a.slug}", "title": a.slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": a.gpu, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [f"{user}/{d}" for d in a.datasets.split(",") if d],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
