"""Feasibility probe: can a Kaggle TPU VM generate doc2query questions with an open <=15B LLM?

    python tpu_probe.py --data /kaggle/input --model Qwen/Qwen3-4B --batch 8 --steps 2

Prints library versions and devices, then times greedy generation on one XLA device with static shapes (prompts padded
to --prompt-len, a static KV cache), two batches: the first includes the XLA compile. Writes probe.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=160)
    args = ap.parse_args()
    res = {"python": sys.version.split()[0]}
    for mod in ("torch", "torch_xla", "jax", "transformers"):
        try:
            m = __import__(mod)
            res[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            res[mod] = f"missing: {e}"
    log(json.dumps(res))
    try:
        import jax
        res["jax_devices"] = [str(d) for d in jax.devices()]
        log(f"jax devices: {res['jax_devices']}")
    except Exception as e:  # noqa: BLE001
        log(f"jax: {e}")
    import pandas as pd
    import torch
    import torch_xla.core.xla_model as xm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = xm.xla_device()
    log(f"xla device {dev}")
    head = pd.read_parquet(sorted(glob.glob(f"{args.data}/**/head_docs.parquet", recursive=True))[0])
    head = head.sort_values("prio", ascending=False).head(args.batch * args.steps)
    sys_msg = ("You write search queries for a Vietnamese medical search engine. Given a passage from a health article, "
               "write 3 different questions that a patient would type to find this passage, in natural spoken "
               "Vietnamese, 8-20 words each. Return ONLY a JSON array of 3 strings.")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    log("model on device")
    texts = []
    for r in head.itertuples():
        m = [{"role": "system", "content": sys_msg},
             {"role": "user", "content": f"TITLE: {str(r.title)[:200]}\nPASSAGE:\n{str(r.text)[:1100]}"}]
        try:
            texts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        except TypeError:
            texts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
    times = []
    for s in range(args.steps):
        batch = texts[s * args.batch:(s + 1) * args.batch]
        enc = tok(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=args.prompt_len).to(dev)
        t = time.time()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, cache_implementation="static")
        xm.mark_step()
        out = tok.batch_decode(gen[:, args.prompt_len:].cpu(), skip_special_tokens=True)
        times.append(round(time.time() - t, 1))
        log(f"batch {s}: {times[-1]}s; sample: {out[0][:300]!r}")
    res.update({"batch": args.batch, "times": times, "docs_per_s_steady": round(args.batch / times[-1], 3)})
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "probe.json").write_text(json.dumps(res, indent=1))
    log(json.dumps(res))


if __name__ == "__main__":
    main()
