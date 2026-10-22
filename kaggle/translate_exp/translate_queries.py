"""Translate the Vietnamese queries to English and Chinese with an open-weight LLM (Kaggle GPU).

    python translate_queries.py --data DIR --out DIR [--model Qwen/Qwen2.5-7B-Instruct] [--max-queries N]

Writes query_en.parquet and query_zh.parquet (columns id, query) in OUT, ready to be fed to the
BM25 kernel with --queries. fp16, device_map="auto" (spreads the 7B model over both T4s).
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

T0 = time.time()
LANGS = {"en": "English", "zh": "Simplified Chinese"}
SYSTEM = ("You are a professional medical translator. Translate the user's Vietnamese medical question into {lang}. "
          "Keep medical terms, drug names and numbers exact. Output only the translation, nothing else.")


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-queries", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    queries = pd.read_parquet(glob.glob(f"{args.data}/**/query.parquet", recursive=True)[0])
    if args.max_queries:
        queries = queries.head(args.max_queries)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    gpu = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if gpu else torch.float32,
                                                 device_map="auto" if gpu else None).eval()
    log(f"loaded {args.model}")
    for code, lang in LANGS.items():
        out = []
        texts = queries["query"].tolist()
        for lo in range(0, len(texts), args.batch):
            prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM.format(lang=lang)},
                                                {"role": "user", "content": q}],
                                               tokenize=False, add_generation_prompt=True)
                       for q in texts[lo:lo + args.batch]]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=160, do_sample=False)
            out.extend(t.strip() for t in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
            log(f"{code}: {len(out)}/{len(texts)}")
        pd.DataFrame({"id": queries["id"], "query": out}).to_parquet(args.out / f"query_{code}.parquet", index=False)
        log(f"{code} sample: {texts[0]!r} -> {out[0]!r}")
    log("done")


if __name__ == "__main__":
    main()
