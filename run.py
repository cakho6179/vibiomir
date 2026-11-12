"""Universal Entrypoint for ViBioMIR System (R2AI 2026 Stage 3).

Modes:
1. --verify: rebuild the lists of the 5 private submissions (#7892, #7893, #7894, #7916, #7917) from the shipped
   artifacts and check them against the submitted ones (reproduce.py; CPU, ~30 s; add --zip for the zip, ~25 min).
2. --demo: replay of the production run for one question, checked against the submitted lists
   (scripts/demo/trace_query.py; CPU, a few seconds; writes out/demo/q<id>.html).
3. Pipeline mode (default): the stage runner (r2ai2026/system/runner.py) on a config. The configs in configs/ are
   the earlier system versions v2.1 / v3; the final #7892 / #7916 come from the Kaggle kernels (README, tier C) plus
   reproduce.py. Stages marked gpu (Qwen3.5-9B, BGE-M3, reranker) need a GPU in practice.

Usage Examples:
    python run.py --verify
    python run.py --demo --qid 401
    python run.py --web              # local demo website over the 1,200 questions
    python run.py --demo --prepare --qids 1,401,556
    python run.py --demo --grid      # once: data of the 100-document grid shown from step 6 of the demo
    python run.py --dry-run          # print the stages of the default config
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
        "--demo",
        action="store_true",
        help="Replay one question (--qid N) through the production run; --prepare --qids adds chunk text",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Local demo website over the 1,200 questions (http://127.0.0.1:8765; --port to change)",
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

    if args.web:
        cmd = [sys.executable, str(ROOT / "scripts" / "demo" / "web.py"), *extra]
        sys.exit(subprocess.run(cmd, cwd=str(ROOT)).returncode)

    if args.demo:
        cmd = [sys.executable, str(ROOT / "scripts" / "demo" / "trace_query.py"), *extra]
        sys.exit(subprocess.run(cmd, cwd=str(ROOT)).returncode)

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
