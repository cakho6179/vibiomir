"""Build docstore v<N> straight from data/raw (docs/DATA_PIPELINE.md §3, r2ai2026/docstore.py).

    python scripts/data/build_docstore.py --version v1

S0  crawl: last ok record per id, junk filter with duplicate counts over the crawl only (reproduces docs_clean);
    recovered/*.jsonl: ids missing from the crawl, junk filter with duplicate counts over crawl + recovered,
    one record per id (source priority, then longest text).
S1  prepare_body (normalise, drop leading title line); per-host template lines learned on every body.
S2  metadata: lang, host, source, n_chars, thin, content_key, dup_of; furniture spans (template + clean_v4 kinds).
S3  passages: heading-aware spans that tile the body exactly.

Outputs in data/docstore/<version>/ (offsets are character positions in x; the title t is separate):
  corpus/part-XX.jsonl.gz   {"id", "t", "x"}: x = canonical body WITH furniture (format the Kaggle kernels read)
  docs_meta.parquet         id, url, host, lang, source, n_chars, thin, content_key, dup_of, furniture_chars, n_passages
  passages.parquet          doc_id, pid, start, end, heading, content_chars
  furniture.parquet         doc_id, start, end, kind
  templates.json, MANIFEST.json (inputs sha256, git commit, parameters, counts)
data/raw is only read.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT))
from r2ai2026.cleaning import junk_reason  # noqa: E402
from r2ai2026.docstore import (PUBMED_HOST, THIN_CHARS, as_doc, content_chars, furniture_spans, host_of,  # noqa: E402
                               passage_spans, pick_recovered, source_of, text_hash)
from r2ai2026.paths import DOCS_RAW, DOCSTORE, RECOVERED  # noqa: E402
from r2ai2026.processing import TemplateModel, content_key, lang_of, prepare_body  # noqa: E402

N_PARTS = 8
EXPECTED_CRAWL_DOCS = 1_183_751  # data/docs_clean.jsonl, 2026-10-20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()[:16]


def iter_crawl(path: Path):
    with open(path, "rb") as f:
        for line in f:
            try:
                d = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            if d.get("ok") and d.get("url"):
                yield d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--template-share", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N crawl records")
    args = ap.parse_args()
    out = DOCSTORE / args.version
    if (out / "MANIFEST.json").exists():
        sys.exit(f"{out} already built; pick a new --version")
    (out / "corpus").mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    junk: dict[str, Counter] = defaultdict(Counter)

    # ---- S0 crawl: last ok record per id + duplicate-template counts over the crawl
    log("S0 pass 1: crawl index")
    last: dict[int, int] = {}
    hashes: dict[int, bytes] = {}
    for n, d in enumerate(iter_crawl(DOCS_RAW)):
        if args.limit and n >= args.limit:
            break
        last[d["id"]] = n
        if host_of(d["url"]) != PUBMED_HOST:
            hashes[n] = text_hash(d.get("text") or "")
    keep_n = set(last.values())
    hashes = {n: h for n, h in hashes.items() if n in keep_n}
    crawl_counts = Counter(hashes.values())
    log(f"  {len(last):,} ids")

    # ---- S0 recovered: every file in data/raw/recovered, ids missing from the crawl decided after the crawl pass
    recovered: dict[int, list[dict]] = defaultdict(list)
    rec_files = sorted(RECOVERED.glob("*.jsonl"))
    for path in rec_files:
        src = source_of(path)
        with open(path, "rb") as f:
            for line in f:
                if line.strip() and (doc := as_doc(orjson.loads(line), src)):
                    recovered[doc["id"]].append(doc)
        log(f"  recovered {path.name}: source={src}")
    all_counts = crawl_counts.copy()
    for docs in recovered.values():
        for doc in docs:
            if doc["host"] != PUBMED_HOST:
                all_counts[text_hash(doc["text"])] += 1

    # ---- S0 + S1: write kept docs to a stage file, learn templates
    stage = out / "_stage.jsonl"
    model = TemplateModel(min_share=args.template_share, seed=args.seed)
    kept_crawl: set[int] = set()

    def stage_doc(f, doc: dict) -> None:
        title, body = prepare_body(doc["title"], doc["text"])
        model.observe(doc["host"], body)
        f.write(orjson.dumps({"id": doc["id"], "url": doc["url"], "host": doc["host"], "source": doc["source"],
                              "t": title, "x": body}) + b"\n")

    log("S0 pass 2: crawl junk filter + stage")
    with open(stage, "wb") as f:
        for n, d in enumerate(iter_crawl(DOCS_RAW)):
            if args.limit and n >= args.limit:
                break
            if last[d["id"]] != n:
                continue
            doc = as_doc(d, "crawl")
            is_pm = doc["host"] == PUBMED_HOST
            why = junk_reason(doc["text"], is_pubmed=is_pm, dup_count=1 if is_pm else crawl_counts[hashes[n]])
            if why:
                junk["crawl"][why] += 1
                continue
            kept_crawl.add(doc["id"])
            stage_doc(f, doc)
            if len(kept_crawl) % 200_000 == 0:
                log(f"  staged {len(kept_crawl):,}")
        del hashes
        for doc_id in sorted(recovered):
            if doc_id in kept_crawl:
                stats["recovered_id_in_crawl"] += 1
                continue
            good = []
            for doc in recovered[doc_id]:
                is_pm = doc["host"] == PUBMED_HOST
                why = junk_reason(doc["text"], is_pubmed=is_pm, dup_count=1 if is_pm else all_counts[text_hash(doc["text"])])
                if why:
                    junk[doc["source"]][why] += 1
                else:
                    good.append(doc)
            if good:
                doc = pick_recovered(good)
                stats[f"recovered_{doc['source']}"] += 1
                stage_doc(f, doc)
    n_docs = len(kept_crawl) + sum(v for k, v in stats.items() if k.startswith("recovered_") and k != "recovered_id_in_crawl")
    log(f"  crawl kept {len(kept_crawl):,} (expected {EXPECTED_CRAWL_DOCS:,}); total {n_docs:,}; junk {dict(junk)}")
    model.fit()
    (out / "templates.json").write_text(json.dumps({h: [[round(s, 3), ln] for s, ln in ex]
                                                    for h, ex in model.examples.items()}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    log(f"S1 templates for {len(model.lines)} hosts")

    # ---- S2 + S3: metadata, furniture, passages; corpus parts in the Kaggle format
    log("S2/S3: furniture + passages + corpus parts")
    per_part = -(-n_docs // N_PARTS)
    meta_schema = pa.schema([("id", pa.int64()), ("url", pa.string()), ("host", pa.string()), ("lang", pa.string()),
                             ("source", pa.string()), ("n_chars", pa.int32()), ("thin", pa.bool_()),
                             ("content_key", pa.string()), ("dup_of", pa.int64()), ("furniture_chars", pa.int32()),
                             ("n_passages", pa.int32())])
    pas_schema = pa.schema([("doc_id", pa.int64()), ("pid", pa.int32()), ("start", pa.int32()), ("end", pa.int32()),
                            ("heading", pa.string()), ("content_chars", pa.int32())])
    fur_schema = pa.schema([("doc_id", pa.int64()), ("start", pa.int32()), ("end", pa.int32()), ("kind", pa.string())])
    writers = {name: pq.ParquetWriter(out / f"{name}.parquet", schema, compression="zstd")
               for name, schema in (("docs_meta", meta_schema), ("passages", pas_schema), ("furniture", fur_schema))}
    bufs: dict[str, list[dict]] = {name: [] for name in writers}

    def flush(force: bool = False) -> None:
        for name, rows in bufs.items():
            if rows and (force or len(rows) >= 200_000):
                writers[name].write_table(pa.Table.from_pylist(rows, schema=writers[name].schema))
                rows.clear()

    first_of: dict[str, int] = {}
    by_lang, by_source = Counter(), Counter()
    part, part_f = -1, None
    with open(stage, "rb") as f:
        for i, line in enumerate(f):
            if i // per_part != part:
                if part_f:
                    part_f.close()
                part = i // per_part
                part_f = gzip.open(out / "corpus" / f"part-{part:02d}.jsonl.gz", "wb", compresslevel=5)
            d = orjson.loads(line)
            doc_id, body = d["id"], d["x"]
            part_f.write(orjson.dumps({"id": doc_id, "t": d["t"], "x": body}) + b"\n")
            lines = body.split("\n")
            fur = furniture_spans(lines, model.template_mask(d["host"], lines))
            pas = passage_spans(lines, max_chars=args.max_chars, min_chars=args.min_chars)
            key = content_key(body)
            lang = lang_of(doc_id)
            fur_chars = sum(e - s for s, e, _ in fur)
            bufs["docs_meta"].append({"id": doc_id, "url": d["url"], "host": d["host"], "lang": lang,
                                      "source": d["source"], "n_chars": len(body), "thin": len(body) < THIN_CHARS,
                                      "content_key": key, "dup_of": first_of.setdefault(key, doc_id),
                                      "furniture_chars": fur_chars, "n_passages": len(pas)})
            bufs["passages"].extend({"doc_id": doc_id, "pid": k, "start": s, "end": e, "heading": h,
                                     "content_chars": content_chars(s, e, fur)} for k, (s, e, h) in enumerate(pas))
            bufs["furniture"].extend({"doc_id": doc_id, "start": s, "end": e, "kind": k} for s, e, k in fur)
            stats["docs"] += 1
            stats["passages"] += len(pas)
            stats["chars"] += len(body)
            stats["furniture_chars"] += fur_chars
            stats["thin"] += len(body) < THIN_CHARS
            stats["exact_duplicates"] += first_of[key] != doc_id
            by_lang[lang] += 1
            by_source[d["source"]] += 1
            flush()
            if (i + 1) % 100_000 == 0:
                log(f"  {i + 1:,} docs, {stats['passages']:,} passages")
    part_f.close()
    flush(force=True)
    for w in writers.values():
        w.close()
    stage.unlink()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "r2ai2026", "scripts", "kaggle/common"],
                                cwd=ROOT, capture_output=True, text=True).stdout.strip())
    manifest = {
        "version": args.version, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": commit, "git_dirty": dirty,
        "inputs": {str(p.relative_to(DOCS_RAW.parents[1])): sha16(p) for p in [DOCS_RAW, *rec_files]},
        "params": {"max_chars": args.max_chars, "min_chars": args.min_chars, "template_share": args.template_share,
                   "seed": args.seed, "thin_chars": THIN_CHARS, "parts": N_PARTS, "limit": args.limit},
        "counts": {**stats, "crawl_kept": len(kept_crawl), "crawl_expected": EXPECTED_CRAWL_DOCS,
                   "by_lang": dict(by_lang), "by_source": dict(by_source),
                   "furniture_share": round(stats["furniture_chars"] / max(stats["chars"], 1), 4),
                   "junk": {k: dict(v) for k, v in junk.items()}, "template_hosts": len(model.lines)},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
