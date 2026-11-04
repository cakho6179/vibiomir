"""Resumable download of one file from a Kaggle kernel's output (the CLI aborts on IncompleteRead).

    python scripts/kaggle_ops/dl_out.py acc2 tthiutoko/vibiomir-mat-r11 r12.zip kaggle/outputs/vibiomir-mat-r11

The Kaggle CLI streams the whole output in one request; on this link it dies after 30-80 MB
("IncompleteRead(... more expected)"). This asks the API for the presigned file URL and hands it to
curl with `-C -`, so every retry continues where the previous one stopped.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_accounts import env_for  # noqa: E402

SNIPPET = r"""
import os, sys, json
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
owner, slug, want = sys.argv[1], sys.argv[2], sys.argv[3]
api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as client:
    req = ApiListKernelSessionOutputRequest(); req.user_name = owner; req.kernel_slug = slug
    files = client.kernels.kernels_api_client.list_kernel_session_output(req).files
    hit = [f for f in files if f.file_name == want or f.file_name.endswith("/" + want)]
    if not hit:
        print(json.dumps({"error": "not found", "files": [f.file_name for f in files]})); raise SystemExit(1)
    print(json.dumps({"url": hit[0].url, "size": getattr(hit[0], "total_bytes", None) or getattr(hit[0], "size", None)}))
"""


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    acc, ref, want, dst_dir = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]).resolve()
    owner, _, slug = ref.partition("/")
    env = env_for(acc)
    proc = subprocess.run([sys.executable, "-c", SNIPPET, owner, slug, want],
                          capture_output=True, text=True, env=env)
    out = (proc.stdout or "").strip().splitlines()
    if not out or not out[-1].startswith("{"):
        raise SystemExit(f"api lookup failed: {proc.stdout[-300:]} {proc.stderr[-300:]}")
    import json
    info = json.loads(out[-1])
    if "error" in info:
        raise SystemExit(f"{want} not in output; available: {info['files']}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    dest = dst_dir / want
    want_bytes = info.get("size") or 0
    for attempt in range(1, 21):
        have = dest.stat().st_size if dest.exists() else 0
        if want_bytes and have >= want_bytes:
            break
        print(f"attempt {attempt}: have {have / 1e6:.1f} MB of {want_bytes / 1e6:.1f} MB", flush=True)
        subprocess.run(["curl", "-sS", "--noproxy", "*", "-L", "-C", "-", "--retry", "5", "--retry-delay", "3",
                        "--max-time", "900", "-o", str(dest), info["url"]], env=env)
    have = dest.stat().st_size if dest.exists() else 0
    ok = bool(want_bytes) and have >= want_bytes
    print(f"{dest}: {have / 1e6:.1f} MB (expected {want_bytes / 1e6:.1f} MB) -> {'OK' if ok else 'INCOMPLETE'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
