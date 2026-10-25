"""Enable python -m r2ai2026 execution.

``python -m r2ai2026 run ...`` is the end-to-end system (r2ai2026/system.py); other commands go to the
older tooling CLI (r2ai2026/cli.py).
"""
import sys

if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        from r2ai2026.system import main as system_main

        sys.exit(system_main(sys.argv[2:]))
    from r2ai2026.cli import main

    sys.exit(main())
