"""Pointwise LLM relevance judge: does this passage answer this patient's question?

    python llm_judge.py --data DIR --out DIR [--model Qwen/Qwen3.5-4B] [--part 0/4] [--chars 700]

Why: the pool already contains 67 of the ~71 gold documents per query (recall 0.936 at 1,200 docs) but
only 40 make the top 197, so the whole problem is ranking, not retrieval. A cross-encoder scores a
1,500-char passage out of context; a 4B instruct model reading the whole question can reject the
"same disease, wrong aspect" documents that fill ranks 50-300.

Scoring is a single forward pass per pair - the answer is read from the logits of "1" versus "0" at the
last position, so nothing is generated and the cost is pure prefill. Score = logit(1) - logit(0).

Input (under --data): judge_cands.parquet (qid, doc_id, bp, rank, rr, m3all) from
scripts/kaggle_ops/llm_judge_inputs.py, plus the docstore v2.1 corpus parts part-*.jsonl[.gz].
The passage is split with rerank.split_passages and `bp` selects the one the cross-encoder preferred,
so the judge sees exactly the evidence the current system already relies on.

--part i/k keeps queries with index % k == i (queries sorted by id), so several accounts share the work.
Output: scores.parquet (qid, doc_id, judge) and stats.json (pairs/s, score distribution by rank bucket).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before the first CUDA call

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerank import split_passages  # noqa: E402

T0 = time.time()
SYSTEM = (
    "You are a medical search assessor. You get a patient's question and one passage from a medical "
    "document. Decide whether the passage contains information that helps answer that question. "
    "The passage helps when it discusses the same disease or condition AND the aspect the patient asks "
    "about (symptoms, causes, treatment, drugs and dosage, diet, prevention, tests, or when to see a "
    "doctor). A passage about the same disease but a different aspect does not help. "
    "Answer with one character only: 1 if it helps, 0 if it does not.")
USER = "QUESTION: {q}\n\nPASSAGE: {p}\n\nAnswer:"


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def find(root: str, pattern: str) -> str:
    hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    if not hits:
        raise SystemExit(f"not found under {root}: {pattern}")
    return hits[0]


def load_bodies(root: str, wanted: set[int]) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    for part in sorted(glob.glob(f"{root}/**/part-*.jsonl*", recursive=True)):
        opener = gzip.open if part.endswith(".gz") else open
        with opener(part, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("id") in wanted and d["id"] not in out:
                    out[d["id"]] = (d.get("t") or "", d.get("x") or "")
        log(f"{Path(part).name}: {len(out):,}/{len(wanted):,} bodies")
        if len(out) == len(wanted):
            break
    return out


def load(name: str):
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


def answer_ids(tok, text: str) -> list[int]:
    """Last token id of a bare answer string, in both the spaced and unspaced spelling."""
    ids = []
    for s in (text, " " + text):
        enc = tok.encode(s, add_special_tokens=False)
        if enc:
            ids.append(int(enc[-1]))
    return sorted(set(ids))


def score_batch(tok, model, prompts: list[str], id1: list[int], id0: list[int]) -> list[float]:
    msgs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}] for p in prompts]
    try:
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for m in msgs]
    except TypeError:
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
    # only the last position is needed: the full (batch, seq, 151k) logits tensor is 7 GB at batch 32
    with torch.inference_mode():
        for kw in ("logits_to_keep", "num_logits_to_keep", None):
            try:
                out = (model(**enc) if kw is None else model(**enc, **{kw: 1})).logits[:, -1, :].float()
                break
            except TypeError:
                continue
    l1 = torch.logsumexp(out[:, id1], dim=1)
    l0 = torch.logsumexp(out[:, id0], dim=1)
    return (l1 - l0).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chars", type=int, default=700)
    ap.add_argument("--qchars", type=int, default=400)
    ap.add_argument("--limit", type=int, default=None, help="keep only the first N pairs (smoke test)")
    ap.add_argument("--save-every", type=int, default=20000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    i, k = (int(x) for x in args.part.split("/"))

    cands = pd.read_parquet(find(str(args.data), "judge_cands.parquet"))
    qtab = pd.read_parquet(find(str(args.data), "queries.parquet"))
    qmap = dict(zip(qtab["qid"].astype(int), qtab["query"].astype(str)))
    qids = sorted(cands["qid"].unique())
    mine = {q for j, q in enumerate(qids) if j % k == i}
    cands = cands[cands["qid"].isin(mine)].sort_values(["qid", "rank"]).reset_index(drop=True)
    if args.limit:
        cands = cands.head(args.limit)
    log(f"part {i}/{k}: {len(cands):,} pairs over {len(mine)} queries")

    bodies = load_bodies(str(args.data), set(cands["doc_id"].astype(int)))
    log(f"{len(bodies):,} bodies")

    rows = []
    for r in cands.itertuples():
        body = bodies.get(int(r.doc_id))
        if not body:
            continue
        title, text = body
        passages = split_passages(title, text)
        p = passages[min(int(r.bp), len(passages) - 1)][:args.chars].strip()
        if len(p) < 120:
            continue
        rows.append({"qid": int(r.qid), "doc_id": int(r.doc_id), "rank": int(r.rank),
                     "prompt": USER.format(q=qmap.get(int(r.qid), "")[:args.qchars], p=p)})
    log(f"{len(rows):,} pairs with a usable passage")

    tok, model = load(args.model)
    id1, id0 = answer_ids(tok, "1"), answer_ids(tok, "0")
    log(f"loaded {args.model}; id1={id1} id0={id0}")

    order = sorted(range(len(rows)), key=lambda j: len(rows[j]["prompt"]))
    out = []
    for lo in range(0, len(order), args.batch):
        idx = order[lo:lo + args.batch]
        for j, s in zip(idx, score_batch(tok, model, [rows[j]["prompt"] for j in idx], id1, id0)):
            out.append({"qid": rows[j]["qid"], "doc_id": rows[j]["doc_id"], "rank": rows[j]["rank"],
                        "judge": round(s, 4)})
        if lo % (args.batch * 20) < args.batch:
            rate = len(out) / max(time.time() - T0, 1)
            left = (len(order) - len(out)) / max(rate, 1e-6) / 60
            log(f"{len(out):,}/{len(order):,} pairs, {rate:.1f}/s, ~{left:.0f} min left")
        if args.save_every and out and len(out) % args.save_every < args.batch:
            pd.DataFrame(out).to_parquet(args.out / "scores.parquet", index=False)

    df = pd.DataFrame(out)
    df.to_parquet(args.out / "scores.parquet", index=False)
    by = df.groupby(pd.cut(df["rank"], [0, 10, 50, 100, 200, 300]), observed=True)["judge"].mean()
    stats = {"part": args.part, "model": args.model, "pairs": int(len(df)),
             "queries": int(df["qid"].nunique()), "chars": args.chars,
             "pairs_per_s": round(len(df) / max(time.time() - T0, 1), 1),
             "mean_judge": round(float(df["judge"].mean()), 3),
             "p1_share": round(float((df["judge"] > 0).mean()), 3),
             "mean_by_rank": {str(a): round(float(v), 3) for a, v in by.items()},
             "seconds": round(time.time() - T0)}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    log(json.dumps(stats))


if __name__ == "__main__":
    main()
