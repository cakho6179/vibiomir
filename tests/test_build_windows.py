import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_submission", ROOT / "r2ai2026" / "stages" / "build_submission.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)


def check(title: str, text: str) -> None:
    spans = bs.rr_spans(text)
    ref = bs.rr_split(title, text)
    assert len(spans) == len(ref)
    for (a, b), r in zip(spans, ref):
        assert "".join(r.split()) == "".join(f"{title}\n{text[a:b]}".strip().split())
    full = f"{title}\n{text}".strip()
    for size in (1500, 3000, 4500):
        for k in range(len(spans)):
            out = bs.window_chunk(title, text, k, spans, size, size)
            assert out in full
            assert len(out) <= len(title) + 101 + max(size, bs.RR_PASSAGE_CHARS) or len(full) <= size  # title at doc start; blank lines inside a span


def test_windows_synthetic():
    check("Sỏi thận", "\n".join(["đoạn " * 200, "", "x" * 4000, "   ", "1. Điều trị", "uống nước " * 300]))
    check("", "")
    check("T", "\n\nmột dòng")


def test_windows_real_docs():
    path = ROOT / "data" / "docs_clean.jsonl"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 997:
                continue
            d = json.loads(line)
            check(d.get("title") or "", d.get("text") or "")
            if i > 400_000:
                break


def test_lexical_best_prefers_query_words():
    text = "bệnh tim mạch và huyết áp\n" + "x " * 900 + "\nsỏi thận gây đau lưng dữ dội"
    spans = bs.rr_spans(text)
    assert bs.lexical_best("sỏi thận đau lưng", "", text, spans) == len(spans) - 1
