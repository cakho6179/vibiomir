"""bm25_experiment.py: a saved index scores exactly like the in-memory one; variant query files rank per variant."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "r2ai2026" / "stages" / "bm25_experiment.py"
DOCS = [(1, "Viem da day", "dau bung tren ron, viem da day man tinh"), (2, "Gan", "men gan cao, gan nhiem mo"),
        (3, "Da day", "trao nguoc da day thuc quan, o chua"), (800001, "zh", "gan zang")]


def run(*args):
    subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], check=True, capture_output=True)


def test_save_then_load_index(tmp_path):
    corpus = tmp_path / "store" / "corpus"
    corpus.mkdir(parents=True)
    with open(corpus / "part-00.jsonl", "w", encoding="utf-8") as f:
        for i, t, x in DOCS:
            f.write(json.dumps({"id": i, "t": t, "x": x}) + "\n")
    pd.DataFrame({"id": [7, 8], "query": ["dau da day", "men gan"]}).to_parquet(tmp_path / "store" / "query.parquet")
    run("e1", tmp_path / "store", tmp_path / "a", "--lang", "vi", "--ns", "", "--topk", "5",
        "--save-index", tmp_path / "idx")
    meta = json.loads((tmp_path / "idx" / "meta.json").read_text())
    assert meta["langs"] == ["vi"] and not meta["chunked"]
    run("e1", tmp_path / "store", tmp_path / "b", "--ns", "", "--topk", "5", "--load-index", tmp_path / "idx")
    a, b = (pd.read_parquet(tmp_path / d / "ranking.parquet") for d in "ab")
    pd.testing.assert_frame_equal(a, b)
    assert 800001 not in set(a["doc_id"]) and a[a.qid == 7]["doc_id"].iloc[0] in (1, 3)

    variants = pd.DataFrame({"qid": [7, 7], "variant": ["raw", "kw"], "text": ["dau da day", "trao nguoc"]})
    variants.to_parquet(tmp_path / "v.parquet")
    run("e1", tmp_path / "store", tmp_path / "c", "--ns", "", "--topk", "5", "--load-index", tmp_path / "idx",
        "--queries", tmp_path / "v.parquet")
    c = pd.read_parquet(tmp_path / "c" / "ranking.parquet")
    assert list(c.columns) == ["qid", "variant", "rank", "doc_id", "score"]
    assert c[c.variant == "kw"]["doc_id"].iloc[0] == 3
