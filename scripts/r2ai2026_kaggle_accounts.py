"""Run the Kaggle CLI as one of several accounts, and retarget bundles between accounts.

The CLI 2.x reads ~/.kaggle/access_token before kaggle.json, so per-account auth is done
by pointing USERPROFILE/HOME at ~/.kaggle_accounts/home<N> (which only holds that
account's kaggle.json). Account "main" uses the normal environment.

    python scripts/r2ai2026_kaggle_accounts.py acc1 kernels list -m
    python scripts/r2ai2026_kaggle_accounts.py --retarget SRC_DIR DST_DIR hytahy
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ACCOUNTS = {"main": "snnguynvnk19hl", "acc1": "hytahy", "acc2": "tthiutoko",
            "acc3": "noskaiser", "acc4": "lanpana"}
HOMES = Path.home() / ".kaggle_accounts"


def env_for(acc: str) -> dict[str, str]:
    env = dict(os.environ)
    if acc != "main":
        home = HOMES / f"home{acc[-1]}"
        env.update(USERPROFILE=str(home), HOME=str(home), KAGGLE_CONFIG_DIR=str(home / ".kaggle"))
        # the CLI keeps resumable-upload records under TEMP keyed by file path: keep accounts apart
        (home / "tmp").mkdir(exist_ok=True)
        env.update(TEMP=str(home / "tmp"), TMP=str(home / "tmp"))
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def kaggle(acc: str, *args: str) -> str:
    proc = subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env_for(acc))
    return (proc.stdout + proc.stderr).strip()


def retarget(src: Path, dst: Path, user: str) -> Path:
    """Copy a dataset/kernel folder and rewrite its metadata to belong to ``user``."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for name in ("dataset-metadata.json", "kernel-metadata.json"):
        meta_path = dst / name
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["id"] = f"{user}/{meta['id'].split('/', 1)[1]}"
            if "dataset_sources" in meta:
                meta["dataset_sources"] = [f"{user}/{s.split('/', 1)[1]}" for s in meta["dataset_sources"]]
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dst


if __name__ == "__main__":
    if sys.argv[1] == "--retarget":
        print(retarget(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]))
    else:
        print(kaggle(sys.argv[1], *sys.argv[2:]))
