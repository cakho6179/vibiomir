"""Build the Kaggle bundles for the BM25 experiments E1/E2/E3.

Dataset  kaggle/corpus_dataset            -> <user>/vibiomir-corpus-clean (corpus shards + query.parquet)
Kernels  kaggle/bm25_exp/kernels/<exp>/   -> <user>/vibiomir-bm25-<exp> (CPU, internet for pip)
Each kernel's run.py is bm25_experiment.py with a Kaggle entry block appended.

    python scripts/kaggle_ops/prep_kaggle_bm25.py --user tthiutoko
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
DATASET = ROOT / "kaggle" / "corpus_dataset"
EXP_DIR = ROOT / "kaggle" / "bm25_exp"
SLUG = "vibiomir-corpus-clean"

ENTRY = '''

if __name__ == "__main__":
    import subprocess
    import sys
    EXP = "{exp}"
    if EXP == "e3":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyvi", "jieba"], check=True)
    import glob
    query = glob.glob("/kaggle/input/**/query.parquet", recursive=True)[0]
    sys.argv = ["bm25", EXP, "/kaggle/input", "/kaggle/working", "--queries", str(query)]
    main()
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", required=True)
    ap.add_argument("--suffix", default="", help="kernel slug suffix, e.g. v2")
    ap.add_argument("--extra-datasets", default="", help="comma-separated extra dataset slugs (delta corpora)")
    args = ap.parse_args()
    datasets = [f"{args.user}/{SLUG}"] + [f"{args.user}/{d}" for d in args.extra_datasets.split(",") if d]

    meta = {"title": "ViBioMIR corpus clean", "id": f"{args.user}/{SLUG}",
            "licenses": [{"name": "other"}]}
    (DATASET / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    source = (EXP_DIR / "bm25_experiment.py").read_text(encoding="utf-8")
    body, guard = source.rsplit('\nif __name__ == "__main__":', 1)
    assert guard.strip() == "main()", "unexpected main guard in bm25_experiment.py"
    for exp in ("e1", "e2", "e3"):
        name = f"{exp}{args.suffix}"
        out = EXP_DIR / "kernels" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "run.py").write_text(body.rstrip() + "\n" + ENTRY.format(exp=exp), encoding="utf-8")
        kmeta = {"id": f"{args.user}/vibiomir-bm25-{name}", "title": f"vibiomir-bm25-{name}",
                 "code_file": "run.py", "language": "python", "kernel_type": "script",
                 "is_private": True, "enable_gpu": False, "enable_tpu": False,
                 "enable_internet": True, "dataset_sources": datasets,
                 "competition_sources": [], "kernel_sources": [], "model_sources": []}
        (out / "kernel-metadata.json").write_text(json.dumps(kmeta, indent=2), encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
