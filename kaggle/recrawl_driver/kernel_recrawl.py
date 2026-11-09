"""Kaggle driver for scripts/r2ai2026_recrawl.py (the tool is loaded unchanged from the dataset).

Locates the dataset copy of the recrawler (code/scripts + code/r2ai2026 + targets.tsv), points its
TARGETS / DOCS_RAW / STATE at the dataset and /kaggle/working, and enforces a global request spacing
(--min-interval seconds between any two fetches of this kernel, so Wayback stays <= 1 request/second).
Every argument this driver does not know is passed on to the recrawler (--method, --hosts, ...).
"""

import argparse
import asyncio
import glob
import importlib.util
import sys
import time
from pathlib import Path


def find_root(explicit: str) -> Path:
    if explicit:
        return Path(explicit).resolve()
    hits = sorted(glob.glob("/kaggle/input/**/code/scripts/r2ai2026_recrawl.py", recursive=True))
    if not hits:
        raise SystemExit("r2ai2026_recrawl.py not found under /kaggle/input")
    return Path(hits[0]).resolve().parents[2]


def install_spacing(rc, interval: float) -> None:
    orig = rc.Run.fetch_html
    lock = asyncio.Lock()
    state = {"next": 0.0}

    async def spaced(self, session, url):
        async with lock:
            now = time.monotonic()
            slot = max(now, state["next"])
            state["next"] = slot + interval
        if slot > now:
            await asyncio.sleep(slot - now)
        return await orig(self, session, url)

    rc.Run.fetch_html = spaced


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--root", default="")
    ap.add_argument("--min-interval", type=float, default=1.0)
    own, passthrough = ap.parse_known_args()

    root = find_root(own.root)
    out = Path(own.out)
    out.mkdir(parents=True, exist_ok=True)
    script = root / "code" / "scripts" / "r2ai2026_recrawl.py"
    spec = importlib.util.spec_from_file_location("r2ai2026_recrawl", script)
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)

    rc.TARGETS = root / "targets.tsv"
    rc.DOCS_RAW = out / "docs.jsonl"
    rc.STATE = out / "recrawl_state.tsv"
    install_spacing(rc, own.min_interval)
    print(f"root={root} out={out} min_interval={own.min_interval} args={passthrough}", flush=True)
    sys.argv = [str(script)] + passthrough
    rc.main()


if __name__ == "__main__":
    main()
