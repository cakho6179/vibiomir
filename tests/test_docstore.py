import json
import random
from pathlib import Path

from r2ai2026.docstore import (as_doc, content_chars, furniture_spans, passage_spans, pick_recovered, source_of)
from r2ai2026.processing import TemplateModel, prepare_body

ROOT = Path(__file__).resolve().parents[1]


def assert_tiles(body: str, spans):
    assert spans[0][0] == 0 and spans[-1][1] == len(body)
    for (s1, e1, _), (s2, _, _) in zip(spans, spans[1:]):
        assert e1 == s2 and s1 < e1
    assert "".join(body[s:e] for s, e, _ in spans) == body


def test_passages_tile_body_with_headings_and_giant_lines():
    lines = ["Mở đầu " + "a " * 100, "1. Triệu chứng", "Đau lưng " + "b " * 900, "x" * 4000,
             "2. Chẩn đoán", "Siêu âm " + "c " * 300]
    body = "\n".join(lines)
    spans = passage_spans(lines, max_chars=1500, min_chars=100)
    assert_tiles(body, spans)
    assert max(e - s for s, e, _ in spans) <= 1500 + 1
    assert any(h == "1. Triệu chứng" for _, _, h in spans)


def test_passages_empty_body():
    assert passage_spans([""]) == [(0, 0, "")]


def test_passages_tile_random_texts():
    rng = random.Random(0)
    for _ in range(200):
        lines = [" ".join("w" * rng.randint(1, 12) for _ in range(rng.randint(1, 400))) for _ in range(rng.randint(1, 30))]
        assert_tiles("\n".join(lines), passage_spans(lines, max_chars=rng.choice([300, 1500]), min_chars=200))


def test_furniture_spans_mark_without_removing():
    lines = ["Sốt xuất huyết là bệnh truyền nhiễm do virus Dengue gây ra.", "Liên hệ hotline 1900 56 56 56 để đặt lịch.",
             "Menu chung"]
    body = "\n".join(lines)
    spans = furniture_spans(lines, [False, False, True])
    assert [(body[s:e], k) for s, e, k in spans] == [(lines[1], "cta"), (lines[2], "template")]
    assert content_chars(0, len(body), spans) == len(body) - len(lines[1]) - len(lines[2])


def test_template_mask_matches_apply():
    model = TemplateModel(min_share=0.5, min_docs=2)
    for i in range(4):
        model.observe("h", f"Trang chủ\nBài {i} nội dung riêng dài dài\nBản quyền")
    model.fit()
    text = "Trang chủ\nBài 9 nội dung riêng dài dài\nBản quyền"
    assert model.template_mask("h", text.split("\n")) == [True, False, True]
    assert model.apply("h", text) == "Bài 9 nội dung riêng dài dài"


def test_recovered_record_selection():
    assert source_of(Path("wayback_vi_2026-10-22.jsonl")) == "wayback"
    assert source_of(Path("cc_kernels_2026-10-22.jsonl")) == "cc"
    a = as_doc({"id": 5, "url": "https://x.vn/a", "t": "T", "x": "short"}, "cc")
    b = as_doc({"id": 5, "url": "https://x.vn/a", "ok": True, "title": "T", "text": "longer text"}, "wayback")
    c = as_doc({"id": 5, "url": "https://x.vn/a", "t": "T", "x": "the longest text of all"}, "cc")
    assert as_doc({"id": 5, "ok": False}, "direct") is None
    assert a["host"] == "x.vn"
    assert pick_recovered([a, b, c]) is b
    assert pick_recovered([a, c]) is c


def test_real_docs_tile():
    path = ROOT / "data" / "docs_clean.jsonl"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for i, line in zip(range(300), f):
            d = json.loads(line)
            _, body = prepare_body(d.get("title") or "", d.get("text") or "")
            assert_tiles(body, passage_spans(body.split("\n")))


def test_binary_documents_are_junk():
    from r2ai2026.cleaning import is_binary_text, junk_reason
    docx = "PK\x03\x04\x14\x00\x06\x00" + "x" * 500
    mojibake = "��标题" * 30 + "y" * 500
    assert is_binary_text(docx) and is_binary_text(mojibake)
    assert junk_reason(docx, is_pubmed=False) == "binary"
    assert not is_binary_text("Sốt xuất huyết là bệnh truyền nhiễm. " * 20)
