import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle" / "common"))

from docorg import clean_v4, section_chunks  # noqa: E402


def test_clean_v4_drops_page_furniture_keeps_content():
    text = "\n".join([
        "Tốt nghiệp Đại học Dược Hà Nội, hơn 10 năm kinh nghiệm. Hiện là giảng viên.",
        "Cẩm Ly",
        "- 21/06/2022 | Dịch vụ xét nghiệm sốt xuất huyết tại nhà",
        "Sốt xuất huyết là bệnh truyền nhiễm do virus Dengue gây ra.",
        "Liên hệ hotline 1900 56 56 56 để đặt lịch xét nghiệm.",
        "Peppermint Oil https://www.webmd.com/x Ngày tham khảo: 04/06/2021",
        "Dược sĩ Đại học Nguyễn Thanh Hải",
    ])
    assert clean_v4(text) == "Sốt xuất huyết là bệnh truyền nhiễm do virus Dengue gây ra."


def test_clean_v4_never_returns_empty():
    assert clean_v4("Xem thêm") == "Xem thêm"


def test_section_chunks_keep_title_and_heading_with_body():
    body = "x " * 400
    text = f"Mở đầu {body}\n1. Triệu chứng\nĐau lưng {body}\n2. Chẩn đoán\nSiêu âm {body}"
    chunks = section_chunks("Sỏi thận", text, max_chars=4500, min_chars=600)
    assert len(chunks) == 3
    assert all(c.startswith("Sỏi thận\n") for c in chunks)
    assert chunks[1].split("\n")[1] == "1. Triệu chứng"


def test_section_chunks_respect_max_chars():
    text = "\n".join(["Một đoạn văn dài về bệnh tiểu đường và cách điều trị. " * 20] * 30)
    chunks = section_chunks("T", text, max_chars=4500, min_chars=600)
    assert max(len(c) for c in chunks) <= 4500 + 2
