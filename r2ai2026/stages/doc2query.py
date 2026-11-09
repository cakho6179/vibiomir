"""doc2query: write the patient-style questions that a head document answers.

    python doc2query.py --data DIR --out DIR [--model Qwen/Qwen3-4B] [--per-doc 3] [--part 0/4] [--limit 200]

Why: the organizers' KB keeps `similar_queries` per chunk (doc2query style) and the admin BM25 reaches docs precision
0.88, while body-text systems sit at 0.18. If the gold labels were produced by matching a query against those synthetic
questions, then a query-to-question similarity is a much sharper signal than a query-to-body one, and the passage that
produced the best question is also the chunk to submit.

Input (found under --data): head_docs.parquet from scripts/kaggle_ops/doc2query_inputs.py
  doc_id, qid, pid, start, end, heading, rr_score, n_head, chars
  and the docstore corpus parts part-*.jsonl[.gz] with {"id", "t", "x"}.
For every head document the passage [start:end] (plus title and heading) is shown to the model, which returns N questions.
--part i/k keeps every k-th document of the (sorted) list, so several accounts can share the work.

Writes questions.parquet (doc_id, qid, pid, idx, question) and stats.json (docs, questions, parse rate, docs/s).
Source stays ASCII (Kaggle decodes non-ASCII kernel source badly); the questions themselves are Vietnamese.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

T0 = time.time()
SYSTEM = (
    "You write search queries for a Vietnamese medical search engine. Given a passage from a Vietnamese health "
    "article, write {n} different questions that a patient would type to find this passage. Rules: each question is "
    "8-20 words, in natural spoken Vietnamese; name the disease, symptom or drug explicitly; ask about ONE thing "
    "(symptoms, causes, treatment, medication and dosage, diet, prevention, tests, or when to see a doctor); do not "
    "copy sentences from the passage; do not number the questions; no markdown, no links, no brackets, no quotes. "
    "Return ONLY a JSON array of {n} strings.")
USER = "TITLE: {title}\nSECTION: {heading}\nPASSAGE:\n{passage}"


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def find(root: str, pattern: str) -> str:
    hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    if not hits:
        raise SystemExit(f"not found under {root}: {pattern}")
    return hits[0]


def load_passages(root: str, wanted: set[int]) -> dict[int, tuple[str, str]]:
    """id -> (title, body) for the wanted ids, streamed over the corpus parts."""
    out: dict[int, tuple[str, str]] = {}
    for part in sorted(glob.glob(f"{root}/**/part-*.jsonl*", recursive=True)):
        opener = gzip.open if part.endswith(".gz") else open
        with opener(part, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["id"] in wanted and d["id"] not in out:
                    out[d["id"]] = (d.get("t") or "", d.get("x") or "")
        log(f"{Path(part).name}: {len(out):,}/{len(wanted):,} bodies")
    return out


BAD = ("http", "](", "[", "**")


def keep(q: str) -> bool:
    """Drop link echoes and fragments the model copies from the source page."""
    return len(q) >= 15 and not any(b in q for b in BAD)


def parse_array(raw: str) -> list[str]:
    """A JSON array of questions; tolerate numbering, bullets and a cut tail."""
    m = re.search(r"\[.*\]", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group(0), strict=False)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except json.JSONDecodeError:
            pass
    out = []
    for line in raw.splitlines():
        line = re.sub(r'^\s*(?:[-*\u2022]|\d+[.)])\s*', "", line).strip().strip(",").strip('"').strip()
        if len(line) > 12 and "?" in line:
            out.append(line)
    return out


def load(name: str):
    """Tokenizer + model in fp16 (T4 has no bf16); split over GPUs only when there are several."""
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    gpu = torch.cuda.is_available()
    multi = gpu and torch.cuda.device_count() > 1
    kw = dict(dtype=torch.float16 if gpu else torch.float32, device_map="auto" if multi else None)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, **kw)
    model.eval()
    if gpu and not multi:
        model.to("cuda")
    return tok, model


def generate(tok, model, prompts: list[str], max_new: int, per_doc: int) -> list[tuple[list[str], str]]:
    """(questions, raw completion) per prompt, so a parse failure can be diagnosed later."""
    msgs = [[{"role": "system", "content": SYSTEM.format(n=per_doc)}, {"role": "user", "content": p}] for p in prompts]
    try:
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for m in msgs]
    except TypeError:
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
    with torch.inference_mode():
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    out = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    res = []
    for t in out:
        clean = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
        res.append(([q for q in parse_array(clean) if keep(q)][:per_doc], clean[:400]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--per-doc", type=int, default=3)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--chars", type=int, default=1100)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=2000, help="checkpoint every N documents (0 = never)")
    ap.add_argument("--keep-raw", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    i, k = (int(x) for x in args.part.split("/"))
    ngpu = int(os.environ.get("DQ_NGPU", torch.cuda.device_count()))  # DQ_NGPU: test the split on a CPU box
    if ngpu > 1 and not os.environ.get("DQ_WORKER"):
        # a 4B model fits one T4: one process per GPU on its own share (index % (k*ngpu) == i + g*k) runs ngpu times
        # faster than one model split over the GPUs, where only one GPU works at a time
        procs = []
        for g in range(ngpu):
            argv = [a for a in sys.argv[1:]]
            j = argv.index("--part") if "--part" in argv else None
            if j is None:
                argv += ["--part", f"{i + g * k}/{k * ngpu}"]
            else:
                argv[j + 1] = f"{i + g * k}/{k * ngpu}"
            o = argv.index("--out")
            argv[o + 1] = str(args.out / f"w{g}")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), DQ_WORKER="1")
            procs.append(subprocess.Popen([sys.executable, sys.argv[0], *argv], env=env))
        codes = [pr.wait() for pr in procs]
        parts = [args.out / f"w{g}" / "questions.parquet" for g in range(ngpu)]
        pd.concat([pd.read_parquet(x) for x in parts if x.exists()], ignore_index=True).to_parquet(
            args.out / "questions.parquet", index=False)
        log(f"workers exited {codes}; merged {sum(x.exists() for x in parts)} question files")
        if any(codes):
            raise SystemExit(1)
        return

    head = pd.read_parquet(find(str(args.data), "head_docs.parquet")).sort_values("doc_id").reset_index(drop=True)
    head = head[head.index % k == i]
    if args.limit:
        head = head.head(args.limit)
    log(f"part {i}/{k}: {len(head):,} docs")

    rows = []
    if "text" in head.columns:  # cleaned passage shipped in the dataset: no corpus scan needed
        for r in head.itertuples():
            seg = str(r.text or "").strip()[:args.chars]
            if len(seg) < 200:
                continue
            rows.append({"doc_id": int(r.doc_id), "qid": int(r.qid), "pid": int(r.pid),
                         "prio": float(getattr(r, "prio", 0.0) or 0.0),
                         "prompt": USER.format(title=str(r.title or "")[:200],
                                               heading=str(r.heading or "")[:120], passage=seg)})
    else:
        bodies = load_passages(str(args.data), set(head["doc_id"].astype(int)))
        for r in head.itertuples():
            body = bodies.get(int(r.doc_id))
            if not body:
                continue
            title, text = body
            seg = text[int(r.start):int(r.end)][:args.chars].strip()
            if len(seg) < 200:
                continue
            rows.append({"doc_id": int(r.doc_id), "qid": int(r.qid), "pid": int(r.pid),
                         "prio": float(getattr(r, "prio", 0.0) or 0.0),
                         "prompt": USER.format(title=title[:200], heading=str(r.heading or "")[:120], passage=seg)})
    log(f"{len(rows):,} docs with a usable passage")

    tok, model = load(args.model)
    log(f"loaded {args.model}")
    # high prio (p(gold)) first, so a run cut short still covers the docs that matter; inside blocks of 2,000 docs
    # similar lengths share a batch
    by_prio = sorted(range(len(rows)), key=lambda j: -rows[j]["prio"])
    order = [j for lo in range(0, len(by_prio), 2000)
             for j in sorted(by_prio[lo:lo + 2000], key=lambda j: len(rows[j]["prompt"]))]

    # resume: a Kaggle session can die at the 12 h wall, so completed docs are checkpointed
    out, raw, fail, done = [], {}, {}, 0
    ckpt = args.out / "progress.parquet"
    if ckpt.exists():
        prev = pd.read_parquet(ckpt)
        out = prev.to_dict("records")
        raw = {int(d): [] for d in prev["doc_id"].unique()}
        for rec in prev.itertuples():
            raw[int(rec.doc_id)].append(rec.question)
        order = [j for j in order if rows[j]["doc_id"] not in raw]
        log(f"resumed: {len(raw):,} docs already done, {len(order):,} left")

    for lo in range(0, len(order), args.batch):
        idx = order[lo:lo + args.batch]
        for j, (qs, text) in zip(idx, generate(tok, model, [rows[j]["prompt"] for j in idx],
                                               args.max_new, args.per_doc)):
            raw[rows[j]["doc_id"]] = qs
            if not qs and len(fail) < 300:
                fail[str(rows[j]["doc_id"])] = text
            for n, q in enumerate(qs):
                out.append({"doc_id": rows[j]["doc_id"], "qid": rows[j]["qid"], "pid": rows[j]["pid"],
                            "idx": n, "question": q})
        done += len(idx)
        if args.save_every and done % args.save_every < args.batch and out:
            pd.DataFrame(out).to_parquet(ckpt, index=False)
            log(f"checkpoint {len(raw):,} docs")
        if done % (args.batch * 10) < args.batch:
            log(f"{done:,}/{len(order):,} new, {len(raw):,} total docs, {len(out):,} questions, "
                f"{done / max(time.time() - T0, 1):.2f} docs/s")

    df = pd.DataFrame(out)
    df.to_parquet(args.out / "questions.parquet", index=False)
    stats = {"part": args.part, "model": args.model, "per_doc": args.per_doc, "docs": len(rows),
             "docs_with_questions": len(raw), "parsed": sum(bool(v) for v in raw.values()),
             "questions": len(df), "questions_per_doc": round(len(df) / max(len(raw), 1), 2),
             "median_chars": int(df["question"].str.len().median()) if len(df) else 0,
             "docs_per_s": round(done / max(time.time() - T0, 1), 3), "seconds": round(time.time() - T0)}
    (args.out / "fail.json").write_text(json.dumps(fail, ensure_ascii=False), encoding="utf-8")
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    if args.keep_raw:
        (args.out / "raw.json").write_text(json.dumps({str(a): b for a, b in raw.items()}, ensure_ascii=False))
    log(json.dumps(stats))


if __name__ == "__main__":
    main()
