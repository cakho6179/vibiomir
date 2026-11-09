"""Wait for Kaggle kernels on several accounts, download each output when it finishes, and
merge re-crawl outputs into the local corpus. Polls rarely (default every 20 min).

Jobs are ACC:KERNEL_SLUG[:merge]; state lives in kaggle/jobs_state.json.
    python scripts/kaggle_ops/kaggle_wait_fetch.py main:vibiomir-recrawl-lc1:merge acc2:vibiomir-bm25-e1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_accounts import ACCOUNTS, kaggle  # noqa: E402

STATE = ROOT / "kaggle" / "jobs_state.json"
OUTPUTS = ROOT / "kaggle" / "outputs"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--poll", type=int, default=1200)
    args = ap.parse_args()

    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    jobs = {}
    for spec in args.jobs:
        acc, slug, *flags = spec.split(":")
        ref = f"{ACCOUNTS[acc]}/{slug}"
        jobs[ref] = (acc, slug, "merge" in flags)
        state.setdefault(ref, "running")
    pending = lambda: [r for r in jobs if state[r] == "running"]  # noqa: E731

    while pending():
        for ref in pending():
            acc, slug, merge = jobs[ref]
            out = kaggle(acc, "kernels", "status", ref)
            if not any(k in out for k in ("COMPLETE", "ERROR", "CANCEL")):
                continue
            dest = OUTPUTS / slug
            dest.mkdir(parents=True, exist_ok=True)
            log(f"{ref}: {out.split('status')[-1].strip()} -> downloading")
            log(kaggle(acc, "kernels", "output", ref, "-p", str(dest))[-200:])
            state[ref] = "downloaded" if "COMPLETE" in out else "error"
            if merge and state[ref] == "downloaded" and (dest / "docs.jsonl").exists():
                proc = subprocess.run(
                    [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "crawl" / "merge_kaggle_recrawl.py"), str(dest)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
                log(f"{ref}: merge -> {(proc.stdout + proc.stderr).strip()[-300:]}")
                state[ref] = "merged" if proc.returncode == 0 else "merge_error"
            STATE.write_text(json.dumps(state, indent=2))
        left = pending()
        log(f"{len(jobs) - len(left)}/{len(jobs)} done; waiting on {', '.join(r.split('/')[1] for r in left) or '-'}")
        if left:
            time.sleep(args.poll)
    log("ALL DONE " + json.dumps({r: state[r] for r in jobs}))


if __name__ == "__main__":
    main()
