"""Trim the per-query chunk list of a submission zip without loading the whole JSON.

    python scripts/submit/trim_submission.py --in kaggle/outputs/vibiomir-mat-r16/r16.zip
        --out submissions/r16b_m3rr197_prof200.zip --chunks 200

The submission JSON is ~1 GB, so it is streamed out of the zip and read with json.JSONDecoder.raw_decode
over a sliding buffer: one query object at a time, never the whole array. Trimming keeps the FIRST n
chunks of each query, which is the order materialize_lists wrote (chunk score order), so it is the same as
having built the list shorter.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT))


def iter_top_array(f):
    """Yield the objects of the top-level JSON array of an open text stream, one at a time."""
    dec = json.JSONDecoder()
    buf = ""
    while True:  # reach the opening bracket
        more = f.read(1 << 22)
        if not more:
            return
        buf += more
        i = buf.find("[")
        if i >= 0:
            buf = buf[i + 1:]
            break
    while True:
        buf = buf.lstrip().lstrip(",").lstrip()
        if not buf:
            more = f.read(1 << 22)
            if not more:
                return
            buf += more
            continue
        if buf[0] == "]":
            return
        try:
            obj, end = dec.raw_decode(buf)
        except ValueError:
            more = f.read(1 << 22)
            if not more:
                return
            buf += more
            continue
        yield obj
        buf = buf[end:]
        if len(buf) < (1 << 21):
            buf += f.read(1 << 22)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--docs", type=int, default=0, help="also cap docs per query (0 = keep)")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".json.tmp")
    n_q = n_ch = 0
    with zipfile.ZipFile(args.src) as z:
        inner = next(n for n in z.namelist() if n.endswith(".json"))
        with z.open(inner) as raw, io.TextIOWrapper(raw, encoding="utf-8") as f, \
                tmp.open("w", encoding="utf-8") as out:
            out.write("[")
            for i, rec in enumerate(iter_top_array(f)):
                if args.docs:
                    rec["relevant_docs"] = rec["relevant_docs"][:args.docs]
                rec["relevant_chunks"] = rec["relevant_chunks"][:args.chunks]
                n_q += 1
                n_ch += len(rec["relevant_chunks"])
                if i:
                    out.write(",")
                out.write(json.dumps(rec, ensure_ascii=False))
            out.write("]")
    print(f"{n_q} queries, {n_ch / max(n_q, 1):.1f} chunks/query, json {tmp.stat().st_size / 1e6:.1f} MB")
    # the project writer stores LZMA with a 256 MiB dictionary, which is what gets ~2.5x over deflate on a
    # corpus where documents repeat across queries; zipfile's own LZMACompressor only has an 8 MiB dict
    from r2ai2026.core import submission as S  # noqa: E402

    payload = tmp.read_bytes()
    saved = zipfile.LZMACompressor
    zipfile.LZMACompressor = S._BigDictLZMACompressor
    try:
        with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_LZMA) as z:
            z.writestr("submission.json", payload)
    finally:
        zipfile.LZMACompressor = saved
    tmp.unlink()
    print(f"{args.out}: {args.out.stat().st_size / 1_048_576:.1f} MiB")


if __name__ == "__main__":
    main()
