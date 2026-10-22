"""Measure per-host template boilerplate in data/docs_clean.jsonl (read-only audit).

A line is boilerplate for a host when the same normalised line appears in >= --min-share of a
sample of that host's documents (and in >= --min-docs of them): menus, "related news" lists,
footers, share/ads text. Writes runs/reports/boilerplate_audit.tsv (per host) and
runs/reports/boilerplate_lines.tsv (the template lines themselves, top per host).

    python scripts/r2ai2026_audit_boilerplate.py [--sample 3000] [--min-share 0.05]
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import orjson
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2ai2026.paths import DATA, LINKS, REPORTS  # noqa: E402

WS = re.compile(r"\s+")


def norm_line(line: str) -> str:
    return WS.sub(" ", line).strip().lower()


def line_key(line: str) -> bytes:
    return hashlib.blake2b(norm_line(line).encode("utf-8"), digest_size=8).digest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=3000, help="docs per host used to learn templates")
    ap.add_argument("--min-share", type=float, default=0.05)
    ap.add_argument("--min-docs", type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(0)

    links = pd.read_parquet(LINKS)
    host = dict(zip(links["id"].astype(int), links["url"].map(lambda u: urlsplit(u).netloc.lower())))
    # reservoir sample of docs per host
    seen: Counter = Counter()
    sample: dict[str, list] = defaultdict(list)
    chars: Counter = Counter()
    with open(DATA / "docs_clean.jsonl", "rb") as f:
        for line in f:
            d = orjson.loads(line)
            h = host.get(d["id"], "?")
            text = d.get("text") or ""
            chars[h] += len(text)
            seen[h] += 1
            if len(sample[h]) < args.sample:
                sample[h].append(text)
            else:
                j = rng.randrange(seen[h])
                if j < args.sample:
                    sample[h][j] = text

    rows, tmpl_rows = [], []
    for h, texts in sample.items():
        df: Counter = Counter()
        example: dict[bytes, str] = {}
        for t in texts:
            keys = set()
            for ln in t.split("\n"):
                if ln.strip():
                    k = line_key(ln)
                    keys.add(k)
                    example.setdefault(k, norm_line(ln)[:160])
            df.update(keys)
        need = max(args.min_docs, args.min_share * len(texts))
        tmpl = {k for k, c in df.items() if c >= need}
        total = removed = 0
        for t in texts:
            for ln in t.split("\n"):
                total += len(ln)
                if ln.strip() and line_key(ln) in tmpl:
                    removed += len(ln)
        rows.append({"host": h, "docs": seen[h], "sampled": len(texts), "template_lines": len(tmpl),
                     "boilerplate_share": round(removed / max(total, 1), 4),
                     "mean_chars": round(chars[h] / seen[h])})
        for k in sorted(tmpl, key=lambda k: -df[k])[:15]:
            tmpl_rows.append({"host": h, "doc_share": round(df[k] / len(texts), 3), "line": example[k]})

    out = pd.DataFrame(rows).sort_values("docs", ascending=False)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORTS / "boilerplate_audit.tsv", sep="\t", index=False)
    pd.DataFrame(tmpl_rows).to_csv(REPORTS / "boilerplate_lines.tsv", sep="\t", index=False)
    w = (out["boilerplate_share"] * out["docs"] * out["mean_chars"]).sum() / (out["docs"] * out["mean_chars"]).sum()
    print(out.head(40).to_string(index=False))
    print(f"\nchar-weighted boilerplate share over corpus: {w:.3f}")


if __name__ == "__main__":
    main()
