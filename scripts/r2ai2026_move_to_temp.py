"""Move superseded files into vibiomir/_temp/<tag>/ (same relative path), never delete.

    python scripts/r2ai2026_move_to_temp.py --tag 2026-10-20 [--dry-run]
    python scripts/r2ai2026_move_to_temp.py --tag 2026-10-20 --restore    # move everything back

Every move is recorded in _temp/<tag>/MANIFEST.tsv (src, dst, bytes, reason).
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (glob relative to ROOT, reason)
PLAN = [
    ("data/docs_cleaned.jsonl", "old 758k snapshot (19/10 13:01), superseded by docs_clean.jsonl v2"),
    ("data/query_translated.parquet", "old translations are garbage; replaced by vibiomir-translate (Qwen2.5-7B)"),
    ("data/real_ids.txt", "fake-doc id list; docs.jsonl already rebuilt fake-free"),
    ("data/docs.jsonl.lock", "stale lock file"),
    ("runs/indexes", "local indexes on old corpora; BM25/dense now run on Kaggle"),
    ("runs/submissions/submission_*.zip", "old submissions (19/10-20/10 early), scored <= 0.065"),
    ("runs/submissions/kaggle_gpu_v1", "old submission 0.0648"),
    ("runs/submissions/e1v2_m3_rrf_n*.zip", "scored/superseded zip (ranking parquet kept)"),
    ("runs/submissions/e1v2_m3_w12_n45.zip", "deflate zip, superseded by LZMA builds"),
    ("runs/submissions/e1v2_m3_w12_n70.zip", "deflate 159 MB, rejected by upload limit (413)"),
    ("runs/submissions/e1v2_m3_w12_n100.zip", "deflate 231 MB, over upload limit"),
    ("runs/submissions/e1_m3_q3e_w122_n*.zip", "scored 0.1948 / deflate over limit"),
    ("runs/submissions/m3only_n*.zip", "scored 0.1906 (ranking parquet kept)"),
    ("runs/submissions/q3eonly_n28.zip", "built, not needed (Qwen3 dilutes fusion)"),
    ("runs/submissions/probe_xl_enzh_n28.zip", "probe scored 0.0089 (gold ~99% Vietnamese)"),
    ("kaggle/outputs/vibiomir-bm25-e1", "BM25 on corpus v1, superseded by e1v2"),
    ("kaggle/outputs/vibiomir-bm25-e2", "BM25 on corpus v1, superseded by e2v2"),
    ("kaggle/outputs/vibiomir-bm25-e3", "BM25 on corpus v1, superseded by e3v2"),
    ("kaggle/outputs/vibiomir-bm25-*/submission_*.zip", "kernel-built zips (rankings kept)"),
    ("kaggle/outputs/vibiomir-recrawl-*", "recrawl outputs already merged into data/docs.jsonl + docs_clean"),
    ("kaggle/outputs/_logs", "debug logs of failed kernels"),
    ("kaggle/crawled_output", "first crawl round, merged into docs.jsonl (~1.1k leftover rows rejected by cleaner)"),
    ("kaggle/vibiomir_data", "old candidates dataset for the 19/10 GPU rerank"),
    ("kaggle/corpus_by_acc", "hardlinked upload copies; datasets already on Kaggle"),
    ("kaggle/delta_by_acc", "hardlinked upload copies; datasets already on Kaggle"),
    ("research/results/_data/*.pkl", "analysis caches from 18/10 (old corpus snapshot)"),
    ("research/results/_data/links_corpus_vi_en_zh.parquet", "duplicate of data/links_corpus_vi_en_zh.parquet"),
    ("docs/EXPERIMENT_RESULTS_2026-10-19.md", "speculative 'expected' scores, no measurements"),
    ("docs/r2ai2026_hierarchical_retrieval_decision_architecture.md", "assumes k_gold~2, contradicted by data"),
    ("docs/r2ai2026_system_architecture_deep_dive.md", "tri-stream EN design, contradicted (gold ~99% VI)"),
    ("docs/baseline_v0_2026-10-19.md", "duplicate of research/reports/baseline_v0_2026-10-19.md"),
]


def size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()
    temp = ROOT / "_temp" / args.tag
    manifest = temp / "MANIFEST.tsv"

    if args.restore:
        for line in manifest.read_text(encoding="utf-8").splitlines()[1:]:
            src, dst = line.split("\t")[:2]
            if Path(dst).exists():
                Path(src).parent.mkdir(parents=True, exist_ok=True)
                os.replace(dst, src)
                print("restored", src)
        return

    rows, total = [], 0
    for pattern, reason in PLAN:
        for hit in sorted(glob.glob(str(ROOT / pattern))):
            src = Path(hit)
            rel = src.relative_to(ROOT)
            dst = temp / rel
            n = size(src)
            total += n
            rows.append(f"{src}\t{dst}\t{n}\t{reason}")
            print(f"{n / 1e6:9.1f} MB  {rel}")
            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(src, dst)  # same volume: instant rename
    print(f"total {total / 1e9:.2f} GB in {len(rows)} items{' (dry run)' if args.dry_run else ''}")
    if not args.dry_run and rows:
        temp.mkdir(parents=True, exist_ok=True)
        new = not manifest.exists()
        with manifest.open("a", encoding="utf-8") as f:
            if new:
                f.write("src\tdst\tbytes\treason\n")
            f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
