"""Remove fabricated (URL-slug / template) documents from docs.jsonl and mark them in crawl.sqlite.

Two passes over docs.jsonl. Pass 1 remembers, per id, the byte offset of the LAST non-synthetic
ok record. Pass 2 writes exactly those lines to ``--out``. Nothing is modified in place;
swap the files yourself after checking the printed counts.

    python scripts/r2ai2026_purge_synthetic.py --src data/docs.jsonl --out data/docs_real.jsonl
    python scripts/r2ai2026_purge_synthetic.py --sqlite-only --keep-ids data/real_ids.txt
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2ai2026.paths import CRAWL_DB, DOCS_RAW  # noqa: E402


def synthetic_kind(text: str) -> str | None:
    """Name of the fabrication template the text matches, else None."""
    head = text[:400]
    if head.startswith("疾病与临床医学咨询"):
        return "zh_template"
    if head.startswith("Chuyên mục: Y khoa lâm sàng, dược phẩm"):
        return "vi_template"
    if head.startswith("Title: ") and "Domain: Clinical Medicine, Pharmacology" in text[:600]:
        return "pubmed_template"
    if len(text) < 1200 and "Thông tin y khoa lâm sàng:" in text[-400:]:
        return "slug_cloudflare"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DOCS_RAW)
    ap.add_argument("--out", type=Path, default=DOCS_RAW.with_name("docs_real.jsonl"))
    ap.add_argument("--ids-out", type=Path, default=DOCS_RAW.with_name("real_ids.txt"))
    ap.add_argument("--sqlite", type=Path, default=CRAWL_DB)
    ap.add_argument("--update-sqlite", action="store_true", help="mark non-real ids ok=0 in crawl.sqlite")
    args = ap.parse_args()

    offsets: dict[int, int] = {}
    seen_fake: dict[int, str] = {}
    lines = bad = 0
    with open(args.src, "rb") as f:
        pos = 0
        for line in f:
            lines += 1
            try:
                d = orjson.loads(line)
            except Exception:
                bad += 1
                pos += len(line)
                continue
            if d.get("ok"):
                kind = synthetic_kind(d.get("text") or "")
                if kind:
                    seen_fake[d["id"]] = kind
                else:
                    offsets[d["id"]] = pos
            pos += len(line)
    only_fake = [i for i in seen_fake if i not in offsets]
    print(f"lines={lines:,} bad_json={bad} real_ids={len(offsets):,} "
          f"fake_ids={len(seen_fake):,} fake_without_real={len(only_fake):,}")

    order = sorted(offsets.values())
    written = 0
    with open(args.src, "rb") as src, open(args.out, "wb") as out:
        for off in order:
            src.seek(off)
            out.write(src.readline())
            written += 1
    print(f"wrote {written:,} real records -> {args.out}")
    args.ids_out.write_text("\n".join(str(i) for i in sorted(offsets)) + "\n", encoding="utf-8")

    if args.update_sqlite:
        keep = set(offsets)
        con = sqlite3.connect(args.sqlite)
        cur = con.cursor()
        rows = cur.execute("SELECT id FROM docs").fetchall()
        drop = [(i,) for (i,) in rows if i not in keep]
        cur.executemany(
            "UPDATE docs SET ok=0, n_chars=0, error='synthetic_removed' WHERE id=?", drop)
        con.commit()
        n_ok = cur.execute("SELECT count(*) FROM docs WHERE ok=1").fetchone()[0]
        con.close()
        print(f"sqlite: marked {len(drop):,} ids ok=0; ok=1 now {n_ok:,}")


if __name__ == "__main__":
    main()
