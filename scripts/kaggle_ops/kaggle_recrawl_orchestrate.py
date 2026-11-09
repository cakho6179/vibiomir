"""Drive the Kaggle re-crawl shards to completion: push when a CPU slot is free (max 5
batch CPU sessions per account, measured 2026-10-20), download finished outputs, merge them.

State: kaggle/recrawl/orchestrator.json  {shard: queued|running|merged|error}
    python scripts/kaggle_ops/kaggle_recrawl_orchestrate.py --user snnguynvnk19hl --running lc1,lc2,lc3,lc4,misc1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
BASE = ROOT / "kaggle" / "recrawl"
STATE = BASE / "orchestrator.json"


def kaggle(*args: str) -> str:
    proc = subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (proc.stdout + proc.stderr).strip()


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", required=True)
    ap.add_argument("--running", default="", help="shards already pushed")
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()

    shards = sorted(p.name for p in (BASE / "kernels").iterdir() if p.is_dir())
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    for s in shards:
        state.setdefault(s, "running" if s in args.running.split(",") else "queued")

    def save() -> None:
        STATE.write_text(json.dumps(state, indent=2))

    save()
    while any(v in ("queued", "running") for v in state.values()):
        for s in [s for s, v in state.items() if v == "running"]:
            ref = f"{args.user}/vibiomir-recrawl-{s}"
            out = kaggle("kernels", "status", ref)
            if "COMPLETE" in out or "ERROR" in out or "CANCEL" in out:
                dest = BASE / "outputs" / s
                dest.mkdir(parents=True, exist_ok=True)
                log(f"{s}: {out.split('status')[-1].strip()} -> downloading")
                log(kaggle("kernels", "output", ref, "-p", str(dest))[-300:])
                if "COMPLETE" in out and (dest / "docs.jsonl").exists():
                    merge = subprocess.run(
                        [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "crawl" / "merge_kaggle_recrawl.py"), str(dest)],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
                    log(f"{s}: merge -> {(merge.stdout + merge.stderr).strip()[-400:]}")
                    state[s] = "merged" if merge.returncode == 0 else "error"
                else:
                    state[s] = "error"
                save()
        for s in [s for s, v in state.items() if v == "queued"]:
            out = kaggle("kernels", "push", "-p", str(BASE / "kernels" / s))
            if "successfully pushed" in out:
                state[s] = "running"
                log(f"{s}: pushed")
                save()
            else:
                if "Maximum batch CPU session" not in out:
                    log(f"{s}: push failed: {out[-300:]}")
                break  # no free slot (or a real error): retry next round
        log("status " + json.dumps(state))
        if any(v in ("queued", "running") for v in state.values()):
            time.sleep(args.poll)
    log("ALL DONE " + json.dumps(state))


if __name__ == "__main__":
    main()
