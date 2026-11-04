"""Isolate the upload transport without consuming a submission slot.

    python scripts/submit/probe_upload.py [path/to/file.zip]

Creates a Codabench dataset row (which only yields a presigned PUT URL) and then tries to PUT
the file with three transports: httpx direct, httpx via the environment proxy, and curl.
No submission is created, so the daily quota is untouched.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_scores import BASE, COMPETITION_ID, auth_token  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("submissions/r6_zh25.zip")
    size = path.stat().st_size
    print(f"file={path} size={size/1e6:.2f} MB", flush=True)
    with httpx.Client(timeout=120) as c:
        tok = auth_token(c)
        c.headers["Authorization"] = f"Token {tok}"
        r = c.post(f"{BASE}/api/datasets/", json={"type": "submission", "competition": COMPETITION_ID,
                                                  "request_sassy_file_name": path.name, "file_size": size})
        print("dataset create:", r.status_code, r.text[:200], flush=True)
        r.raise_for_status()
        ds = r.json()
    url = ds["sassy_url"]
    print("host:", url.split("/")[2], flush=True)
    headers = {"Content-Type": "application/zip"}
    if "blob.core.windows.net" in url:
        headers.update({"x-ms-blob-type": "BlockBlob", "x-ms-version": "2018-03-28"})

    for trust in (False, True):
        t0 = time.time()
        try:
            with path.open("rb") as fh:
                with httpx.Client(timeout=600, trust_env=trust) as c:
                    resp = c.put(url, content=fh, headers=headers)
            print(f"httpx trust_env={trust}: {resp.status_code} in {time.time()-t0:.0f}s "
                  f"{resp.text[:200]}", flush=True)
        except Exception as exc:
            print(f"httpx trust_env={trust}: {type(exc).__name__}: {str(exc)[:300]} "
                  f"after {time.time()-t0:.0f}s", flush=True)

    hdrs = []
    for k, v in headers.items():
        hdrs += ["-H", f"{k}: {v}"]
    t0 = time.time()
    cmd = ["curl", "-sS", "-X", "PUT", "--data-binary", f"@{path}", "-H", "Expect:",
           *hdrs, "-w", "\nHTTP=%{http_code} uploaded=%{size_upload} speed=%{speed_upload}",
           "--max-time", "900", url]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=960)
    print(f"curl: rc={out.returncode} {out.stdout.strip()[-400:]} err={out.stderr.strip()[:300]} "
          f"in {time.time()-t0:.0f}s", flush=True)
    Path("runs/reports/probe_upload.json").write_text(json.dumps({"file": str(path), "size": size,
                                                                  "url_host": url.split("/")[2]}, indent=1))


if __name__ == "__main__":
    main()
