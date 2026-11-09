"""Stage wrapper: build submission zips out of the decided doc / chunk lists (the heavy text step).

    python r2ai2026/stages/materialize_lists.py --lists DIR --queries F --corpus DIR --out DIR
                                                [--window 4500] [--max-mb 98] [--extra F] [--profile ...]

The production script (scripts/kaggle_ops/materialize_lists.py) expects one input root holding lists.parquet,
query.parquet and the corpus parts; on Kaggle that root is assembled by the dataset. Locally the stage assembles it
instead: lists.parquet and query.parquet are copied into <out>/input/ (both small) and the corpus is linked as
<out>/input/store, so the 4 GB of text is read in place, never copied. The link is a junction on Windows and a
symlink elsewhere, the same trick runner.link_dir uses; glob('**') follows it (checked on Python 3.13 / NTFS).

--extra copies one more file into the staged root, for the modes that look one up: sections.parquet and
pas_scores.parquet for --profile or the src=stair / src=seed lists, hf_text.parquet for src=hf. The corpus link
also exposes the docstore's own sections.parquet. Everything else (the chunk builders, the LZMA zip, the
validation) is the production code, run unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMPL = ROOT / "scripts" / "kaggle_ops" / "materialize_lists.py"


def log(msg: str) -> None:
    print(f"[materialize] {msg}", flush=True)


def link_dir(src: Path, dst: Path) -> None:
    """Directory link needing no admin rights: a junction on Windows, a symlink elsewhere."""
    if dst.exists():
        return
    if platform.system() == "Windows":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], check=True, capture_output=True)
    else:
        os.symlink(src, dst, target_is_directory=True)


def stage_input(lists: Path, queries: Path, corpus: Path, out: Path, extra: list[Path]) -> Path:
    """Build the root the production script expects: the decided lists, the queries and a link to the corpus."""
    staging = out / "input"
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(lists / "lists.parquet", staging / "lists.parquet")
    shutil.copyfile(queries, staging / "query.parquet")
    link_dir(corpus.resolve(), staging / "store")
    for f in extra:
        shutil.copyfile(f, staging / f.name)
    log(f"staged {staging}: lists.parquet, query.parquet, store -> {corpus}")
    return staging


def run_impl(argv: list[str]) -> None:
    sys.path.insert(0, str(ROOT))  # the script imports r2ai2026.core.* and r2ai2026.stages.build_submission
    spec = importlib.util.spec_from_file_location("_materialize_lists_impl", IMPL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(IMPL), *argv]
    module.main()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lists", type=Path, required=True, help="directory holding lists.parquet (the decide output)")
    ap.add_argument("--queries", type=Path, required=True, help="query.parquet (id, query)")
    ap.add_argument("--corpus", type=Path, required=True, help="corpus root with part-*.jsonl[.gz]; linked, not copied")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=4500, help="default chunk size in chars (the lists' size wins)")
    ap.add_argument("--max-mb", type=float, default=98.0, help="flag jobs above this zip size")
    ap.add_argument("--alt-text", default="hf_text.parquet")
    ap.add_argument("--extra", action="append", default=[], type=Path,
                    help="one more file to place in the staged root, e.g. pas_scores.parquet (repeatable)")
    ap.add_argument("--profile", action="store_true", help="cut the chunk where the passage relevance ends")
    ap.add_argument("--profile-file", default="pas_scores.parquet")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--min-frac", type=float, default=0.30)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    staging = stage_input(args.lists, args.queries, args.corpus, args.out, args.extra)
    argv = ["--input", str(staging), "--out", str(args.out), "--window", str(args.window),
            "--max-mb", str(args.max_mb), "--alt-text", args.alt_text]
    if args.profile:
        argv += ["--profile", "--profile-file", args.profile_file, "--alpha", str(args.alpha),
                 "--min-frac", str(args.min_frac)]
    run_impl(argv)


if __name__ == "__main__":
    main()
