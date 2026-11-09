"""Download kernel output with resume.

The CLI's downloader aborts the whole listing when a stream breaks (`IncompleteRead`) and, in this
network, streams break at 37-75 MB, so the 90-105 MB submission zips never arrive. This wrapper keeps
a `.part` file, resumes with an HTTP Range request, and re-lists the session to get a fresh signed URL
when the old one expires.

    python scripts/kaggle_ops/dl_resume.py acc2 tthiutoko/vibiomir-mat-st1 runs/mat_out/st1 [--pattern r24]
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kaggle_accounts import env_for  # noqa: E402


def _resume(resp, outfile: str, entity_name=None, quiet=True, resume=True, tries: int = 12) -> None:
    part = Path(outfile + ".part")
    url = resp.url
    total = int(resp.headers.get("Content-Length") or 0)
    if part.exists():
        have = part.stat().st_size
    else:
        have = 0
    if total:
        total += have  # Content-Length of a resumed request is the remainder
    stream = None if have else resp
    for attempt in range(tries):
        try:
            if stream is None:
                headers = {"Range": f"bytes={have}-"} if have else {}
                r = requests.get(url, headers=headers, stream=True, timeout=300)
                if r.status_code == 416:
                    break
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored Range: restart
                    have = 0
                    part.write_bytes(b"")
                if total == 0:
                    total = int(r.headers.get("Content-Length") or 0) + have
                stream = r
            with open(part, "ab") as f:
                for chunk in stream.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
                        have += len(chunk)
            stream = None
            if not total or have >= total:
                break
            print(f"  short read at {have:,}/{total:,}, resuming", flush=True)
        except Exception as exc:  # noqa: BLE001 - any transport error is retried the same way
            stream = None
            have = part.stat().st_size if part.exists() else 0
            print(f"  attempt {attempt + 1}: {type(exc).__name__} at {have:,} B, retrying", flush=True)
            time.sleep(min(2 + attempt, 10))
    if not total or have >= total:
        os.replace(part, outfile)
        print(f"  saved {outfile} ({have:,} B)", flush=True)
    else:
        raise SystemExit(f"incomplete: {have:,}/{total:,} for {outfile}")


def main() -> None:
    acc, kernel, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    pattern = None
    if "--pattern" in sys.argv:
        pattern = sys.argv[sys.argv.index("--pattern") + 1]
    os.environ.update(env_for(acc))
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.download_file = _resume  # type: ignore[method-assign]
    pat = re.compile(pattern) if pattern else None
    for attempt in range(4):  # a broken listing is retried, files already complete are skipped
        try:
            api.kernels_output(kernel, outdir, file_pattern=pattern, force=False, quiet=False)
            break
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"listing attempt {attempt + 1} failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(5)
    print("done", flush=True)


if __name__ == "__main__":
    main()
