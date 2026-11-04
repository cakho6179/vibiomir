"""Download a kernel's output files into the project (works around a Kaggle CLI 2.x path bug).

    python scripts/kaggle_ops/fetch_out.py acc1 hytahy/vibiomir-dq-g kaggle/outputs/dq-g0

`kaggle kernels output -p DIR` honours DIR for the log but writes the data files into
"<PortableGit>/kaggle/working" (the CLI resolves the file list against a stale cwd). This wrapper
runs the download, then moves anything that landed there into DIR, and prints what arrived.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_accounts import kaggle  # noqa: E402

STRAY = Path(r"C:/Users/NOS/AppData/Local/Programs/WorkBuddyAI/resources/vendor/PortableGit/kaggle/working")


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    acc, ref, dst = sys.argv[1], sys.argv[2], Path(sys.argv[3]).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    before = {p.name: p.stat().st_mtime for p in STRAY.glob("*")} if STRAY.is_dir() else {}
    print(kaggle(acc, "kernels", "output", ref, "-p", str(dst), "--force")[-400:])
    moved = []
    if STRAY.is_dir():
        for p in sorted(STRAY.glob("*")):
            if p.is_file() and before.get(p.name) != p.stat().st_mtime:
                shutil.move(str(p), str(dst / p.name))
                moved.append(p.name)
    print(f"dst={dst}")
    for p in sorted(dst.glob("*")):
        print(f"  {p.name:32s} {p.stat().st_size / 1e6:8.2f} MB")
    if moved:
        print(f"recovered from stray dir: {', '.join(moved)}")


if __name__ == "__main__":
    main()
