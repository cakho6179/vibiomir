import numpy as np
import pandas as pd

from r2ai2026.stages import dense_search as ds


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_best_passage_per_doc_and_exact_merge():
    q = unit([[1, 0, 0], [0, 1, 0]])
    # file A: doc 10 has two passages, doc 11 one; file B: doc 20
    emb_a, ids_a = unit([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]]), np.array([10, 10, 11])
    emb_b, ids_b = unit([[0.8, 0.6, 0]]), np.array([20])
    a = ds.top_docs(q, emb_a, ids_a, topk=2, device="cpu")
    b = ds.top_docs(q, emb_b, ids_b, topk=2, device="cpu")
    assert [d for d, _ in a[0]] == [10, 11] and abs(a[0][0][1] - 1.0) < 1e-6  # doc 10 once, best passage
    variants = pd.DataFrame({"qid": [5, 5], "variant": ["orig", "zh_q"]})
    r = ds.merge(variants, [a, b], topk=2)
    assert list(r.columns) == ["qid", "variant", "rank", "doc_id", "score"]
    assert r[r.variant == "orig"]["doc_id"].tolist() == [10, 20]
    assert r[r.variant == "zh_q"]["doc_id"].tolist() == [11, 20]
