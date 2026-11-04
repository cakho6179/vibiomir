"""Universal Entrypoint for ViBioMIR System (R2AI 2026 Stage 3).

This script provides a single entry point to execute either:
1. Fast Verification Mode (Verify private submissions #7892, #7893, #7894, #7916, #7917 in ~1 min on CPU).
2. End-to-End Pipeline Mode (Execute full retrieval, reranking, and submission build from raw queries).
3. Smoke Mode (Rapid 30-second pipeline test on 5 queries, suitable for laptops and CPU-only environments).

Usage Examples:
    # 1. Quick verification of the official submitted files (CPU, ~1 min):
    python run.py --verify

    # 2. Run smoke test on laptop CPU:
    python run.py --smoke --allow-cpu

    # 3. Run full end-to-end pipeline from precomputed index:
    python run.py --config configs/system_v2_1_precomputed.yaml

    # 4. Dry-run pipeline to inspect stages and parameters:
    python run.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ViBioMIR Universal Entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification check on the 5 private submissions (#7892 etc.) via reproduce.py",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/system_v2_1_precomputed.yaml",
        help="Path to YAML pipeline configuration (default: configs/system_v2_1_precomputed.yaml)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test mode on 5 queries (fast testing on laptop/CPU)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        default=True,
        help="Allow running GPU stages on CPU fallback if CUDA is unavailable (default: True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would execute without running them",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated subset of stages to execute (e.g., 'bm25,dense,cands')",
    )

    args, extra = parser.parse_known_args()

    if args.verify:
        print("[run.py] Executing Verification Mode (reproduce.py)...", flush=True)
        cmd = [sys.executable, str(ROOT / "reproduce.py"), *extra]
        ret = subprocess.run(cmd, cwd=str(ROOT))
        sys.exit(ret.returncode)

    # Otherwise execute full system runner pipeline
    print(f"[run.py] Executing Pipeline Mode with config: {args.config}...", flush=True)
    runner_args = ["--config", args.config]
    if args.smoke:
        runner_args.append("--smoke")
    if args.allow_cpu:
        runner_args.append("--allow-cpu")
    if args.dry_run:
        runner_args.append("--dry-run")
    if args.stages:
        runner_args.extend(["--stages", args.stages])
    runner_args.extend(extra)

    from r2ai2026.system.runner import main as run_system_main

    sys.exit(run_system_main(runner_args))


if __name__ == "__main__":
    main()
