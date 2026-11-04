"""Wait for kernels on several accounts, then download their outputs into kaggle/outputs/<slug>.

    python scripts/kaggle_ops/wait_fetch.py acc2:vibiomir-mat-r6 acc1:vibiomir-judge-smoke [--poll 120]

Thin wrapper over `kernels status` + fetch_out.py (which works around the CLI writing data files into
"<PortableGit>/kaggle/working"). Exits when every job reached COMPLETE / ERROR / CANCEL, so it can be
left running in the background and will notify on completion.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_accounts import ACCOUNTS, kaggle  # noqa: E402


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jobs", nargs="+", help="ACC:SLUG[:outdir]")
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()

    jobs = []
    for spec in args.jobs:
        parts = spec.split(":")
        acc, slug = parts[0], parts[1]
        out = Path(parts[2]) if len(parts) > 2 else ROOT / "kaggle" / "outputs" / slug
        jobs.append({"acc": acc, "slug": slug, "ref": f"{ACCOUNTS[acc]}/{slug}", "out": out, "state": "running"})

    while any(j["state"] == "running" for j in jobs):
        for j in jobs:
            if j["state"] != "running":
                continue
            out = kaggle(j["acc"], "kernels", "status", j["ref"])
            if not any(k in out for k in ("COMPLETE", "ERROR", "CANCEL")):
                continue
            log(f"{j['ref']}: {out.split('status')[-1].strip()} -> {j['out']}")
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "fetch_out.py"),
                                   j["acc"], j["ref"], str(j["out"])],
                                  capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(proc.stdout[-800:], flush=True)
            j["state"] = "downloaded" if "COMPLETE" in out else "error"
        left = [j["slug"] for j in jobs if j["state"] == "running"]
        if left:
            log(f"waiting on {', '.join(left)}")
            time.sleep(args.poll)
    log("ALL DONE " + ", ".join(f"{j['slug']}={j['state']}" for j in jobs))


if __name__ == "__main__":
    main()
