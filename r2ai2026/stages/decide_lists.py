"""Stage wrapper: decide the doc / chunk lists of a run from the scored 3-language pool.

    python r2ai2026/stages/decide_lists.py --job NAME=METHOD:NDOCS|CMETHOD:NCHUNKS:SIZE --out DIR
                                           [--pool vi800,zh200,en200] [--pool-file F] [--rr GLOB]
                                           [--m3m GLOB] [--e1 F] [--pas F] [--report]

The decision logic (orders, slice probes, language quotas, chunk sizing) lives in
scripts/analysis/decide_lists.py: six analysis scripts import ``load_pool`` / ``order`` from there, so it is not
moved. This stage only makes it callable from the runner with the same arguments; with no path overrides it reads
the production features (kaggle/outputs/rr3-p*, m3m-p*, the pool cands and runs/reports/pool_pas.parquet), and with
them it reads the features a run just computed. See configs/system_v3.yaml for the wired paths.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMPL = ROOT / "scripts" / "analysis" / "decide_lists.py"


def run_main() -> None:
    """Load the production recipe from its file and run it (a distinct module name, so a future
    r2ai2026/stages/decide_lists.py can never shadow the implementation)."""
    sys.path.insert(0, str(ROOT))  # the recipe imports r2ai2026.core.submission for its overlap report
    spec = importlib.util.spec_from_file_location("_decide_lists_impl", IMPL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    run_main()
