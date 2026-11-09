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


def test_section_chunk_uses_section_containing_best_passage():
    body = "A" * 1500 + "\n" + "B" * 1500 + "\n" + "C" * 1500
    spans = bs.rr_spans(body)
    sections = [(0, 1500), (1501, 3001), (3002, 4502)]
    out = bs.section_chunk("T", body, 1, spans, sections, 4800)
    assert out == "B" * 1500
    # The first section keeps the title; later sections do not.
    assert bs.section_chunk("T", body, 0, spans, sections, 4800) == f"T\n{'A' * 1500}".strip()


def test_section_chunk_falls_back_when_section_too_long_or_missing():
    body = "A" * 100 + "\n" + "B" * 100
    spans = bs.rr_spans(body)
    assert bs.section_chunk("T", body, 1, spans, [(0, 5000)], 4800) is None
    assert bs.section_chunk("T", body, 1, spans, [], 4800) is None


def test_section_chunk_drops_title_when_it_would_exceed_max():
    body = "A" * 100
    spans = bs.rr_spans(body)
    out = bs.section_chunk("T" * 50, body, 0, spans, [(0, 100)], 100)
    assert out == body
    assert len(out) <= 100
