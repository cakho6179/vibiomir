"""Kaggle kernel shim: unpack the vibiomir-code dataset and run the system entry point on it.

    python run_system.py --config configs/system_v1.yaml [--stages a,b] [--var name=value ...] [--smoke]

The code zip (scripts/kaggle_ops/pack_code.py) is found anywhere under /kaggle/input and unpacked to
/kaggle/working/code; stage outputs land in /kaggle/working/code/<work>/ and so become kernel output.
Extra arguments are passed through to `python -m r2ai2026 run`. Source is ASCII-only.
"""
from __future__ import annotations

import glob
import subprocess
import sys
import zipfile
from pathlib import Path

CODE = Path("/kaggle/working/code")


def main() -> None:
    zips = glob.glob("/kaggle/input/**/vibiomir-code.zip", recursive=True)
    if not zips:
        raise SystemExit("vibiomir-code.zip not found under /kaggle/input")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(CODE)
    print("code version:", (CODE / "CODE_VERSION").read_text().strip(), flush=True)
    cmd = [sys.executable, "-m", "r2ai2026", "run", *sys.argv[1:]]
    sys.exit(subprocess.run(cmd, cwd=str(CODE)).returncode)


if __name__ == "__main__":
    main()
