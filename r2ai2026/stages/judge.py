"""Zero-shot / teacher yes-no judge with a Qwen3-Reranker model over (query, section) pairs.

    python judge.py --data DIR --out DIR [--model Qwen/Qwen3-Reranker-4B] [--shard 0 --nshards 1]
                    [--asks A,B] [--max-len 512] [--batch 16]

DIR holds pairs.parquet (build_pairs.py: qid, doc_id, rank, text) and query.parquet. Each ask is one instruction:
  A  same main disease / condition / drug / topic as the question (subject gate)
  B  contains information that helps answer at least one asked point
Score = logit("yes") - logit("no") at the last position (Qwen3-Reranker template, thinking block left empty).
With 2 GPUs the shard is split in two halves, one subprocess per GPU. Output: scores_<shard>.parquet
(qid, doc_id, rank, <ask>...) and stats_<shard>.json (pairs, seconds, tokens/s). Source is ASCII-only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

T0 = time.time()
ASKS = {
    "A": "Given a Vietnamese patient's health question, judge whether the document is mainly about the same "
         "disease, condition, drug or health topic that the question is about.",
    "B": "Given a Vietnamese patient's health question, judge whether the document contains information that "
         "helps answer at least one of the points asked in the question.",
}
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
          "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def find(root: Path, name: str) -> str:
    return sorted(glob.glob(str(root / "**" / name), recursive=True))[0]


def score(pairs: pd.DataFrame, queries: dict[int, str], args) -> tuple[pd.DataFrame, int]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre, suf = tok.encode(PREFIX, add_special_tokens=False), tok.encode(SUFFIX, add_special_tokens=False)
    room = args.max_len - len(pre) - len(suf)
    out = pairs[["qid", "doc_id", "rank"]].copy()
    n_tok = 0
    for ask in args.asks.split(","):
        ids = []
        for q, t in zip(pairs["qid"], pairs["text"]):
            body = tok.encode(f"<Instruct>: {ASKS[ask]}\n<Query>: {queries[int(q)]}\n<Document>: {t}",
                              add_special_tokens=False)[:room]
            ids.append(pre + body + suf)
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        res = [0.0] * len(ids)
        for b in range(0, len(order), args.batch):
            idx = order[b:b + args.batch]
            enc = tok.pad({"input_ids": [ids[i] for i in idx]}, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                # last position only: full (batch, seq, vocab) logits ran out of T4 memory on the longest batches
                last = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            diff = (last[:, yes] - last[:, no]).cpu().tolist()
            for i, d in zip(idx, diff):
                res[i] = d
            n_tok += int(enc["attention_mask"].sum())
            if b // args.batch % 500 == 0:
                log(f"ask {ask}: {b}/{len(ids)}; {n_tok / (time.time() - T0):.0f} tok/s")
        out[ask] = res
    return out, n_tok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    ap.add_argument("--asks", default="A,B")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--half", type=int, default=-1, help="internal: which GPU half of the shard")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_parquet(find(args.data, "pairs*.parquet"))
    qids = sorted(pairs["qid"].unique())
    mine = set(qids[args.shard::args.nshards])  # shard by query so a query's list is scored in one place
    pairs = pairs[pairs["qid"].isin(mine)].reset_index(drop=True)
    if args.limit:
        pairs = pairs.head(args.limit)

    import torch
    n_gpu = torch.cuda.device_count()
    if args.half < 0 and n_gpu > 1:  # one subprocess per GPU, then merge
        procs = []
        for h in range(2):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(h))
            cmd = [sys.executable, __file__] + [a for a in sys.argv[1:]] + ["--half", str(h)]
            procs.append(subprocess.Popen(cmd, env=env))
        if any(p.wait() for p in procs):
            raise SystemExit("a GPU worker failed")
        parts = [pd.read_parquet(args.out / f"scores_{args.shard}_h{h}.parquet") for h in range(2)]
        st = [json.loads((args.out / f"stats_{args.shard}_h{h}.json").read_text()) for h in range(2)]
        pd.concat(parts).to_parquet(args.out / f"scores_{args.shard}.parquet", index=False)
        stats = {"pairs": sum(s["pairs"] for s in st), "tokens": sum(s["tokens"] for s in st),
                 "seconds": round(time.time() - T0), "model": args.model, "asks": args.asks}
        stats["tok_per_s"] = round(stats["tokens"] / max(stats["seconds"], 1))
        (args.out / f"stats_{args.shard}.json").write_text(json.dumps(stats, indent=1))
        log(json.dumps(stats))
        return

    if args.half >= 0:
        pairs = pairs.iloc[args.half::2].reset_index(drop=True)
    q = pd.read_parquet(find(args.data, "query.parquet"))
    queries = {int(i): str(t) for i, t in zip(q["id"], q["query"])}
    log(f"shard {args.shard}/{args.nshards} half {args.half}: {len(pairs):,} pairs")
    res, n_tok = score(pairs, queries, args)
    tag = f"{args.shard}_h{args.half}" if args.half >= 0 else f"{args.shard}"
    res.to_parquet(args.out / f"scores_{tag}.parquet", index=False)
    stats = {"pairs": len(res), "tokens": n_tok, "seconds": round(time.time() - T0)}
    (args.out / f"stats_{tag}.json").write_text(json.dumps(stats, indent=1))
    log(json.dumps(stats))


if __name__ == "__main__":
    main()
