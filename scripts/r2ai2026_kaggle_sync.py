"""Monitor all 5 Kaggle CPU crawler kernels and auto-download results upon completion.

Usage:
    python scripts/r2ai2026_kaggle_sync.py
    # or watch in loop:
    python scripts/r2ai2026_kaggle_sync.py --loop --interval 120
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.r2ai2026_merge_crawled import merge_parquet

KERNELS = [
    "vibiomir-crawl-p1-vinmec",
    "vibiomir-crawl-p2-longchau",
    "vibiomir-crawl-p3-vnhealth",
    "vibiomir-crawl-p4-120ask",
    "vibiomir-crawl-p5-zhmed",
]


def check_and_sync(out_dir: Path, username: str = "snnguynvnk19hl") -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    statuses = {}

    for k in KERNELS:
        ref = f"{username}/{k}"
        res = subprocess.run(["kaggle", "kernels", "status", ref], capture_output=True, text=True)
        out = res.stdout.strip()
        status = "UNKNOWN"
        for candidate in ["RUNNING", "COMPLETE", "FAILED", "QUEUED", "CANCELLED"]:
            if candidate in out:
                status = candidate
                break
        statuses[k] = status
        print(f"  {k:30s} : {status}")

        if status == "COMPLETE":
            target_file = out_dir / f"{k}.parquet"
            if target_file.exists() and target_file.stat().st_size == 0:
                target_file.unlink()

            if not target_file.exists():
                print(f"    Downloading output for {ref} to {out_dir}...")
                for attempt in range(1, 4):
                    try:
                        subprocess.run(["kaggle", "kernels", "output", ref, "-p", str(out_dir)], check=True)
                        break
                    except subprocess.CalledProcessError as err:
                        print(f"    Attempt {attempt} failed: {err}")
                        if target_file.exists() and target_file.stat().st_size == 0:
                            target_file.unlink()
                        time.sleep(3)

                if target_file.exists() and target_file.stat().st_size > 0:
                    print(f"    Downloaded {target_file.name} ({target_file.stat().st_size / 1024 / 1024:.2f} MB)")
                    docs_path = ROOT / "data" / "docs.jsonl"
                    sqlite_path = ROOT / "data" / "crawl.sqlite"
                    merge_parquet(target_file, docs_path, sqlite_path)
                else:
                    print(f"    WARNING: {target_file.name} not found or empty after download attempts.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("kaggle/crawled_output"))
    parser.add_argument("--username", type=str, default="snnguynvnk19hl")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args(argv)

    print("Checking Kaggle CPU crawler status...")
    while True:
        statuses = check_and_sync(args.out_dir, username=args.username)
        if not args.loop:
            break
        if all(s in ("COMPLETE", "FAILED", "CANCELLED") for s in statuses.values()):
            print("All 5 kernels have terminated.")
            break
        print(f"Waiting {args.interval}s before next check...\n")
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
