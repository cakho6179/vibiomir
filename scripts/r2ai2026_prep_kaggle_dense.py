"""Build Kaggle GPU kernel bundles for dense retrieval (kaggle/dense_exp/dense_embed.py).

Each account runs a pair of corpus shards so a model covers all 8 shards across 4 accounts;
per-shard rankings merge exactly afterwards (cosine scores are comparable).
    python scripts/r2ai2026_prep_kaggle_dense.py m3 q3e
-> kaggle/dense_exp/kernels/<model>-<acc>/{run.py, kernel-metadata.json}, push with --accelerator NvidiaTeslaT4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from r2ai2026_kaggle_accounts import ACCOUNTS  # noqa: E402

EXP = ROOT / "kaggle" / "dense_exp"
SHARDS = {"acc2": "0,1", "acc1": "2,3", "acc3": "4,5", "acc4": "6,7"}
MODELS = {
    "m3": {"model": "BAAI/bge-m3", "prompt": "", "batch": 64},
    "q3e": {"model": "Qwen/Qwen3-Embedding-0.6B", "batch": 64,
            "prompt": "Instruct: Given a medical question in Vietnamese, retrieve documents that answer the question\nQuery:"},
}

RUN = '''import glob, subprocess, sys
from pathlib import Path
SRC = {src!r}
script = Path("/kaggle/tmp/dense_embed.py")
script.parent.mkdir(parents=True, exist_ok=True)
script.write_text(SRC, encoding="utf-8")
data = Path(glob.glob("/kaggle/input/**/part-00.jsonl*", recursive=True)[0]).parents[1]
subprocess.run([sys.executable, str(script), "--model", {model!r}, "--data", str(data), "--out", "/kaggle/working",
                "--shards", {shards!r}, "--batch", "{batch}", "--query-prompt", {prompt!r}], check=True)
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+", choices=sorted(MODELS))
    args = ap.parse_args()
    src = (EXP / "dense_embed.py").read_text(encoding="utf-8")
    for tag in args.models:
        cfg = MODELS[tag]
        for acc, shards in SHARDS.items():
            user = ACCOUNTS[acc]
            out = EXP / "kernels" / f"{tag}-{acc}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "run.py").write_text(RUN.format(src=src, model=cfg["model"], shards=shards,
                                                   batch=cfg["batch"], prompt=cfg["prompt"]), encoding="utf-8")
            slug = f"vibiomir-dense-{tag}-s{shards.replace(',', '')}"
            meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
                    "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
                    "enable_internet": True, "dataset_sources": [f"{user}/vibiomir-corpus-clean"],
                    "competition_sources": [], "kernel_sources": [], "model_sources": []}
            (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(acc, slug)


if __name__ == "__main__":
    main()
