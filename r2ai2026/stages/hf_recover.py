"""Recover text for Vietnamese BTC links missing from docstore v1, from a public HF crawl of the same link list.

    python hf_recover.py --data DIR --out DIR [--repo tmh1008/ViBioMIR-crawled-corpus] [--min-chars 300]

Source: https://huggingface.co/datasets/tmh1008/ViBioMIR-crawled-corpus (public, no license stated; found via
research/refs/neiths_med-doc-retrieval). Rows: id, url, domain, status_code, success, title, content.
Keeps rows with id <= 766441 (Vietnamese part of the link list), id not in docs_meta.parquet, success true,
status 200 and >= --min-chars of content. Also compares lengths with our text on a sample of shared ids.
Outputs: recovered_hf.jsonl.gz ({"id","url","t","x"}, the docstore raw format), stats.json. Source is ASCII-only.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

VI_MAX_ID = 766441
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", default="tmh1008/ViBioMIR-crawled-corpus")
    ap.add_argument("--min-chars", type=int, default=300)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(glob.glob(str(args.data / "**" / "docs_meta.parquet"), recursive=True)[0])
    have = set(meta["id"].astype(int))
    log(f"docstore ids {len(have):,}")
    files = [f for f in HfApi().list_repo_files(args.repo, repo_type="dataset") if f.endswith(".parquet")]
    log(f"{len(files)} parquet files in {args.repo}")

    n_rows = n_vi = n_ok = 0
    kept, seen = [], set()
    for k, f in enumerate(files):
        path = hf_hub_download(args.repo, f, repo_type="dataset")
        df = pd.read_parquet(path)
        Path(path).unlink(missing_ok=True)  # Kaggle disk is small
        n_rows += len(df)
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
        df = df[df["id"].notna() & (df["id"] <= VI_MAX_ID)]
        n_vi += len(df)
        ok = df["success"].astype(bool) & (pd.to_numeric(df["status_code"], errors="coerce") == 200)
        df = df[ok & (df["content"].fillna("").str.len() >= args.min_chars)]
        n_ok += len(df)
        for i, u, t, x in zip(df["id"].astype(int), df["url"], df["title"].fillna(""), df["content"]):
            if i not in have and i not in seen:
                seen.add(i)
                kept.append({"id": i, "url": u, "t": t, "x": x})
        if k % 100 == 0:
            log(f"{k}/{len(files)} files; rows {n_rows:,}; vi ok {n_ok:,}; new {len(kept):,}")

    with gzip.open(args.out / "recovered_hf.jsonl.gz", "wt", encoding="utf-8") as fo:
        for r in kept:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens = pd.Series([len(r["x"]) for r in kept])
    stats = {"repo": args.repo, "rows": n_rows, "vi_rows": n_vi, "vi_ok": n_ok, "new_vi_docs": len(kept),
             "new_len_median": int(lens.median()) if len(lens) else 0, "seconds": round(time.time() - T0)}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    log(json.dumps(stats))


if __name__ == "__main__":
    main()
