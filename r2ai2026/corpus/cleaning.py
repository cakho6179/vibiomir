"""Text cleaning and normalization module for ViBioMIR.

Safely removes boilerplate (breadcrumbs, date/related article lists, hotlines,
social buttons, repetitive medical disclaimers) while keeping 100% of clinical
content verbatim.
"""

from __future__ import annotations

import re
from typing import Sequence

NOISE_LINE_PATTERNS = [
    re.compile(r"^-\s*\d{1,2}/\d{1,2}/\d{4}\s*\|", re.IGNORECASE),
    re.compile(
        r"^(tin tức|trang chủ|chuyên khoa|bài viết liên quan|tin liên quan|xem thêm:?|đọc thêm:?|bình luận\s*(\(\d+\))?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(đường dây nóng:?\s*\d+|hotline:?\s*[\d\.\s]+|tổng đài tư vấn:?\s*[\d\.\s]+|đặt lịch khám ngay|tải ứng dụng (myvinmec|ncovi|bluezone)|liên hệ ngay với chúng tôi)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(bài viết chỉ mang tính chất tham khảo|không thay thế cho việc chẩn đoán|thông tin trên website chỉ mang tính chất tham khảo|mọi thông tin trên website chỉ dùng cho mục đích tham khảo)",
        re.IGNORECASE,
    ),
    re.compile(r"^(免责声明|版权声明|相关推荐|点击查看更多|分享到微信|来源：)", re.IGNORECASE),
    re.compile(r"^(404 not found|không tìm thấy trang|nội dung đang được cập nhật)$", re.IGNORECASE),
]


def clean_document_text(text: str, is_pubmed: bool = False) -> str:
    """Clean raw crawled text, removing boilerplate while preserving verbatim sentences."""
    if not text:
        return ""
    if is_pubmed:
        return text.strip()

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        sline = line.strip()
        if not sline:
            continue
        is_noise = False
        for pat in NOISE_LINE_PATTERNS:
            if pat.search(sline):
                is_noise = True
                break
        if not is_noise:
            cleaned_lines.append(sline)

    return "\n".join(cleaned_lines).strip()


BLOCK_PAGE_RE = re.compile(
    r"attention required|just a moment|access denied|enable javascript and cookies|checking your browser|"
    r"không tìm thấy trang|trang bạn tìm kiếm không|the content on this page is not available in|"
    r"404 not found|page not found|页面不存在|您访问的页面不存在",
    re.IGNORECASE,
)

MIN_WEB_CHARS = 200
BINARY_HEAD_CHARS = 2000
BINARY_MIN_REPLACEMENT = 20


def is_binary_text(text: str) -> bool:
    """A binary file (docx/jpeg/png...) or a wrongly decoded page saved as text: NUL bytes or many U+FFFD near the start.

    Such "documents" are millions of junk tokens; one of them as a chunk made the leaderboard scorer fail (22/10).
    """
    head = (text or "")[:BINARY_HEAD_CHARS]
    return "\x00" in head or head.count("\ufffd") >= BINARY_MIN_REPLACEMENT
DUP_TEMPLATE_MIN_IDS = 3


def junk_reason(text: str, *, is_pubmed: bool, dup_count: int = 1) -> str | None:
    """Why a crawled document is not genuine page content, or None when it is usable.

    Catches what the crawl leaves behind besides real articles: block/404 pages,
    JS-shell stubs, slug-only stubs and one identical text served under many URLs.
    PubMed abstracts are legitimately short, so only an empty one is rejected.
    """
    n = len((text or "").strip())
    if is_pubmed:
        return "empty" if n < 30 else None
    if n < MIN_WEB_CHARS:
        return "too_short"
    if is_binary_text(text):
        return "binary"
    if n < 3000 and BLOCK_PAGE_RE.search(text[:800]):
        return "blocked_or_404"
    if dup_count >= DUP_TEMPLATE_MIN_IDS:
        return "duplicate_template"
    return None


def is_valid_document(text: str, min_chars: int = 150) -> bool:
    """Check if document has sufficient substantive content."""
    if not text:
        return False
    return len(text.strip()) >= min_chars
