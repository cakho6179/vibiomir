from r2ai2026.processing import (
    TemplateModel,
    content_key,
    lang_of,
    normalize_text,
    prepare_body,
    process_document,
    split_passages,
)


def test_normalize_text_collapses_and_dedupes_lines():
    raw = "Viêm&nbsp;dạ  dày​\r\n\r\nViêm dạ dày\n  \nTriệu chứng &amp; điều trị\n"
    assert normalize_text(raw) == "Viêm dạ dày\nTriệu chứng & điều trị"


def test_normalize_text_composes_vietnamese_to_nfc():
    decomposed = "Viếm"  # e + circumflex + acute
    assert normalize_text(decomposed) == "Viếm"


def test_lang_of_uses_official_id_blocks():
    assert lang_of(583) == "vi"
    assert lang_of(766_441) == "vi"
    assert lang_of(766_444) == "zh"
    assert lang_of(4_421_055) == "en"


def test_prepare_body_drops_repeated_title():
    title, body = prepare_body("Prolactin in male reproduction", "Prolactin in male reproduction\nAbstract text.")
    assert title == "Prolactin in male reproduction"
    assert body == "Abstract text."


def test_template_model_removes_host_boilerplate_only():
    model = TemplateModel(min_share=0.5, min_docs=2)
    for i in range(10):
        model.observe("site.vn", f"Kích thước chữ\nMặc định\nBài viết số {i} về bệnh tiểu đường.")
    model.observe("other.vn", "Kích thước chữ\nNội dung khác.")
    model.fit()
    assert model.apply("site.vn", "Kích thước chữ\nMặc định\nNội dung mới về huyết áp.") == "Nội dung mới về huyết áp."
    assert model.apply("other.vn", "Kích thước chữ\nNội dung khác.") == "Kích thước chữ\nNội dung khác."


def test_template_model_never_empties_a_document():
    model = TemplateModel(min_share=0.5, min_docs=2, keep_min_share=0.2)
    for _ in range(5):
        model.observe("site.vn", "Menu A\nMenu B")
    model.fit()
    assert model.apply("site.vn", "Menu A\nMenu B") == "Menu A\nMenu B"


def test_process_document_and_duplicate_key():
    model = TemplateModel().fit()
    rec = process_document(1, "a.vn", "Tiêu đề", "Tiêu đề\nNội dung  bài viết.", model)
    assert rec["text"] == "Nội dung bài viết."
    assert rec["lang"] == "vi" and rec["removed_chars"] == 0
    assert content_key("Nội dung bài viết.") == content_key("nội  dung bài\nviết.")


def test_split_passages_respects_limit_and_prefixes_title():
    text = "\n".join(["x" * 700] * 5)
    passages = split_passages("T", text, max_chars=1500)
    assert all(p.startswith("T\n") for p in passages)
    assert all(len(p) <= 1502 for p in passages)
    assert "".join(p[2:].replace("\n", "") for p in passages) == "x" * 3500
