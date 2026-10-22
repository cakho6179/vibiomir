"""Wait until an account's corpus dataset is ready, then push the given kernel bundles (GPU).

    python scripts/r2ai2026_push_when_ready.py acc1 kaggle/dense_exp/kernels/m3-acc1 ... [--poll 180]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2ai2026_kaggle_accounts import ACCOUNTS, kaggle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acc")
    ap.add_argument("bundles", nargs="+")
    ap.add_argument("--poll", type=int, default=180)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dataset", default="vibiomir-corpus-clean")
    args = ap.parse_args()
    ref = f"{ACCOUNTS[args.acc]}/{args.dataset}"
    while "ready" not in kaggle(args.acc, "datasets", "status", ref):
        time.sleep(args.poll)
    print(time.strftime("%H:%M:%S"), ref, "ready", flush=True)
    for b in args.bundles:
        extra = [] if args.cpu else ["--accelerator", "NvidiaTeslaT4"]
        print(b, "->", kaggle(args.acc, "kernels", "push", "-p", b, *extra).splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()
