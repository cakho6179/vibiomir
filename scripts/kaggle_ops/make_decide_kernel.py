"""Build an offline Kaggle kernel (internet off) that runs r2ai2026/stages/decide.py with the vendored anyjev wheel.

    python scripts/kaggle_ops/make_decide_kernel.py --user noskaiser --slug vibiomir-decide-q3-8b \
        --dataset noskaiser/vibiomir-g0-pairs --model-source qwen-lm/qwen-3/transformers/8b/1 \
        -- --model "/kaggle/input/**/transformers/8b/**" --device-map auto --shard 0 --nshards 3 --limit 6000

The kernel writes decide.py and the wheel (base64) to /kaggle/tmp and runs
`decide.py --data /kaggle/input --out /kaggle/working --offline <args>`. Output: kaggle/kernels_generic/<slug>/.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHIM = '''import base64, subprocess, sys
from pathlib import Path
d = Path("/kaggle/tmp"); d.mkdir(parents=True, exist_ok=True)
(d / "decide.py").write_text(SRC, encoding="utf-8")
(d / WHEEL_NAME).write_bytes(base64.b64decode(WHEEL_B64))
import torch, transformers
print("torch", torch.__version__, "transformers", transformers.__version__, "gpus", torch.cuda.device_count(), flush=True)
subprocess.run([sys.executable, str(d / "decide.py"), "--data", "/kaggle/input", "--out", "/kaggle/working",
                "--offline", *ARGS], check=True)
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--model-source", action="append", default=[])
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    args = [x for x in a.args if x != "--"]
    src = (ROOT / "r2ai2026/stages/decide.py").read_text(encoding="utf-8")
    wheel = sorted((ROOT / "vendor").glob("anyjev-*.whl"))[-1]
    if not src.isascii():
        raise SystemExit("decide.py must stay ASCII")
    code = (f"SRC = {src!r}\nWHEEL_NAME = {wheel.name!r}\n"
            f"WHEEL_B64 = {base64.b64encode(wheel.read_bytes()).decode()!r}\nARGS = {args!r}\n" + SHIM)
    out = ROOT / "kaggle" / "kernels_generic" / a.slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.py").write_text(code, encoding="utf-8")
    meta = {"id": f"{a.user}/{a.slug}", "title": a.slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": a.dataset, "competition_sources": [],
            "kernel_sources": [], "model_sources": a.model_source}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(out)


if __name__ == "__main__":
    main()
