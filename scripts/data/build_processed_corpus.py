"""Build processed corpus v3 from data/docs_clean.jsonl (see r2ai2026/processing.py).

Pass 1 learns per-host template lines from a reservoir sample; pass 2 writes parquet shards.
docs_clean.jsonl and existing indexes are not touched.

    python scripts/data/build_processed_corpus.py [--extra kaggle/outputs/vibiomir-cc-recover0/docs.jsonl ...]

Outputs (data/processed_v3/):
  docs-XX.parquet   id, host, lang, title, text, n_chars, removed_chars, content_key, dup_of
  templates.json    host -> [[doc share, template line], ...]
  report.json       counts, characters removed, duplicates, per language/host
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import orjson
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())))
from r2ai2026.paths import DOCS_CLEAN, DOCS_PROCESSED  # noqa: E402
from r2ai2026.processing import TemplateModel, prepare_body, process_document  # noqa: E402

SHARD_DOCS = 150_000


def iter_docs(paths: list[Path]):
    """(id, host, title, text) from docs_clean (and optional extra jsonl sources); first id wins."""
    seen: set[int] = set()
    for path in paths:
        with open(path, "rb") as f:
            for line in f:
                d = orjson.loads(line)
                doc_id = int(d["id"])
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                host = d.get("host") or ""
                if not host and d.get("url"):
                    from urllib.parse import urlsplit
                    host = urlsplit(d["url"]).netloc.lower()
                yield doc_id, host, d.get("title") or d.get("t") or "", d.get("text") or d.get("x") or ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extra", nargs="*", default=[], help="extra jsonl sources (e.g. Common Crawl recovery)")
    ap.add_argument("--out", type=Path, default=DOCS_PROCESSED)
    ap.add_argument("--min-share", type=float, default=0.2)
    args = ap.parse_args()
    sources = [DOCS_CLEAN] + [Path(p) for p in args.extra]
    args.out.mkdir(parents=True, exist_ok=True)

    model = TemplateModel(min_share=args.min_share)
    for _, host, title, text in iter_docs(sources):
        model.observe(host, prepare_body(title, text)[1])
    model.fit()
    (args.out / "templates.json").write_text(
        json.dumps({h: [[round(s, 3), ln] for s, ln in ex] for h, ex in model.examples.items()},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"templates learned for {len(model.lines)} hosts", flush=True)

    first_of: dict[str, int] = {}
    stats = Counter()
    by_lang, by_host_removed, by_host_chars = Counter(), Counter(), Counter()
    buf, shard = [], 0

    def flush() -> None:
        nonlocal buf, shard
        if buf:
            pd.DataFrame(buf).to_parquet(args.out / f"docs-{shard:02d}.parquet", index=False)
            print(f"  wrote docs-{shard:02d}.parquet ({len(buf):,})", flush=True)
            shard += 1
            buf = []

    for doc_id, host, title, text in iter_docs(sources):
        rec = process_document(doc_id, host, title, text, model)
        rec["dup_of"] = first_of.setdefault(rec["content_key"], doc_id)
        stats["docs"] += 1
        stats["dups"] += rec["dup_of"] != doc_id
        stats["chars"] += rec["n_chars"]
        stats["removed"] += rec["removed_chars"]
        stats["empty"] += rec["n_chars"] == 0
        by_lang[rec["lang"]] += 1
        by_host_removed[host] += rec["removed_chars"]
        by_host_chars[host] += rec["n_chars"] + rec["removed_chars"]
        buf.append(rec)
        if len(buf) >= SHARD_DOCS:
            flush()
    flush()

    top = sorted(by_host_removed, key=lambda h: -by_host_removed[h])[:30]
    report = {"sources": [str(s) for s in sources], "docs": stats["docs"], "by_lang": dict(by_lang),
              "exact_duplicates": stats["dups"], "empty_after_processing": stats["empty"],
              "chars_kept": stats["chars"], "chars_removed": stats["removed"],
              "removed_share": round(stats["removed"] / max(stats["chars"] + stats["removed"], 1), 4),
              "top_hosts_removed": {h: {"removed_share": round(by_host_removed[h] / max(by_host_chars[h], 1), 4),
                                        "removed_mchars": round(by_host_removed[h] / 1e6, 2)} for h in top}}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "top_hosts_removed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
