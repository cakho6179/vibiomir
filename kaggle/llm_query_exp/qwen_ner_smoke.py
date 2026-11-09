"""Smoke test: Qwen3.5-9B text-only inference on one Vietnamese medical query.

This kernel only checks that the official open-weight model loads on a Kaggle
GPU and can return strict JSON. The full query-NER run is a separate kernel.
Source is ASCII-only because Kaggle decodes kernel source badly.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import torch

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes"],
        check=True,
    )
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import Qwen3_5ForConditionalGeneration as ModelClass
    except Exception:
        from transformers import AutoModelForImageTextToText as ModelClass

    model_id = "Qwen/Qwen3.5-9B"
    log(f"loading {model_id}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = ModelClass.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    log("model loaded")

    query = (
        "Em b\u1ecb vi\u00eam xung huy\u1ebft 1/3 d\u01b0\u1edbi th\u1ef1c qu\u1ea3n, vi\u00eam d\u1ea1 d\u00e0y tr\u1ee3t xung huy\u1ebft m\u1ea1n t\u00ednh, "
        "s\u1eb9o lo\u00e9t h\u00e0nh t\u00e1 tr\u00e0ng. Cho em h\u1ecfi c\u00e1c b\u1ec7nh n\u00e0y c\u00f3 n\u1eb7ng kh\u00f4ng?"
    )
    system = (
        "You are a Vietnamese clinical information extractor. Return ONE JSON object, no markdown. "
        "Schema: {\"entities\":[{\"text\":\"exact substring of the query\","
        "\"type\":\"disease|symptom|drug|test|treatment|cause\","
        "\"canonical_vi\":\"short normalized Vietnamese term\"}]}. "
        "Only extract spans that appear verbatim in the query. Do not invent entities."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": query}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    log(f"input tokens: {inputs['input_ids'].shape[1]}")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated = out[0][inputs["input_ids"].shape[1]:]
    answer = processor.decode(generated, skip_special_tokens=True)
    print("RAW OUTPUT:", answer, flush=True)
    try:
        parsed = json.loads(answer)
        print("PARSED:", json.dumps(parsed, ensure_ascii=False), flush=True)
    except Exception as exc:
        print("JSON ERROR:", repr(exc), flush=True)
    print("GPU:", torch.cuda.get_device_name(0), "VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))


if __name__ == "__main__":
    main()
