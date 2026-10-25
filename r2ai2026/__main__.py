"""python -m r2ai2026 run --config configs/<system>.yaml  (see r2ai2026/system/runner.py, docs/SYSTEM_RUN.md)."""
import sys

if __name__ == "__main__":
    if sys.argv[1:2] != ["run"]:
        sys.exit("usage: python -m r2ai2026 run --config configs/system_v1.yaml [--stages ...] [--dry-run]")
    from r2ai2026.system.runner import main

    sys.exit(main(sys.argv[2:]))
