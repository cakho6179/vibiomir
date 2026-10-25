"""Smoke test of candidate open LLMs for the query frame on Kaggle T4s (fp16, no bf16).

    python llm_smoke.py --data DIR --out DIR [--models a,b,c] [--n 48] [--batch 16] [--clear-cache]

Same prompt and parser as query_frame.py, on N queries spread over the length range. Per model: load time,
seconds per query, JSON parse rate, entities per query and the share copied verbatim from the query, subqueries
per query, empty/garbled outputs (fp16 overflow shows up as empty text or repeated "!"), peak GPU memory.
Writes smoke.json (one row per model) and smoke_samples.parquet (model, qid, query, raw) for reading by hand.
Source is ASCII-only.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
import transformers

from query_frame import generate, load, normalise, parse

DEFAULT = "Qwen/Qwen3.5-9B,Qwen/Qwen3.5-4B,Qwen/Qwen3-8B,Qwen/Qwen3-14B"


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default=DEFAULT)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=700)
    ap.add_argument("--clear-cache", action="store_true", help="delete the HF hub cache after each model (Kaggle only)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        import fla  # noqa: F401  fast Gated DeltaNet kernels for Qwen3.5 linear-attention layers
        has_fla = True
    except ImportError:
        has_fla = False
    log(f"transformers {transformers.__version__}, torch {torch.__version__}, gpus {torch.cuda.device_count()}, "
        f"fla {has_fla}")

    q = pd.read_parquet(glob.glob(f"{args.data}/**/query.parquet", recursive=True)[0])
    q = q.assign(n=q["query"].astype(str).str.len()).sort_values("n")
    q = q.iloc[[round(i * (len(q) - 1) / (args.n - 1)) for i in range(args.n)]]  # spread over lengths
    texts, qids = q["query"].astype(str).tolist(), q["id"].tolist()

    rows, samples = [], []
    for name in args.models.split(","):
        row = {"model": name}
        try:
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            tok, model = load(name)
            row["load_s"] = round(time.time() - t0)
            t0 = time.time()
            raw = []
            for lo in range(0, len(texts), args.batch):
                raw += generate(tok, model, texts[lo:lo + args.batch], args.max_new)
            row["s_per_q"] = round((time.time() - t0) / len(texts), 2)
            frames = [parse(r) for r in raw]
            ok = [normalise(f, t) for f, t in zip(frames, texts) if f]
            ents = [e for f in ok for e in f["entities"]]
            row.update({
                "parsed": round(len(ok) / len(texts), 3),
                "entities_per_q": round(len(ents) / max(len(ok), 1), 2),
                "entities_in_query": round(sum(e["in_query"] for e in ents) / max(len(ents), 1), 3),
                "aliases_per_entity": round(sum(len(e["aliases"]) for e in ents) / max(len(ents), 1), 2),
                "subqueries_per_q": round(sum(len(f["subqueries"]) for f in ok) / max(len(ok), 1), 2),
                "hyde_words": round(sum(len(f["hyde"].split()) for f in ok) / max(len(ok), 1)),
                "garbled": sum(1 for r in raw if not r.strip() or r.count("!") > 20),
                "peak_gb": round(sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
                                 / 2**30, 1) if torch.cuda.is_available() else 0,
            })
            samples += [{"model": name, "qid": int(i), "query": t, "raw": r} for i, t, r in zip(qids, texts, raw)]
            del model
        except Exception as e:  # one failing model must not stop the comparison
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if args.clear_cache:  # Kaggle disk is small; never on a workstation
            shutil.rmtree(Path.home() / ".cache" / "huggingface" / "hub", ignore_errors=True)
        rows.append(row)
        log(json.dumps(row))
        (args.out / "smoke.json").write_text(json.dumps(rows, indent=1))
        pd.DataFrame(samples).to_parquet(args.out / "smoke_samples.parquet", index=False)


if __name__ == "__main__":
    main()
