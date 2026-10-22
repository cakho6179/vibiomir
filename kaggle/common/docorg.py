"""Document organisation for retrieval (index side only; submissions keep the full original text).

clean_v4(text)            drop page furniture that survived extraction: related-article lists, author bios
                          and signatures, reference/URL lines, call-to-action and share widgets, Q&A boilerplate.
section_chunks(title, text, max_chars, min_chars)
                          heading-aware chunks like the organisers' pipeline (mentor slides 14, 22-24):
                          one chunk per section = title + heading + body; small sections merge forward,
                          long ones split on paragraph boundaries with the heading repeated.

Non-ASCII literals are escaped automatically when the module is embedded into a Kaggle kernel.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------- clean_v4

RELATED_DATED = re.compile(r"^-\s*\d{1,2}/\d{1,2}/\d{4}\s*\|")          # medlatec "- 21/06/2022 | <title>"
TABLE_RELATED = re.compile(r"^\|\s*\|")                                  # "| | <related headline>"
URL_LINE = re.compile(r"https?://|www\.\w|Ngày tham khảo", re.I)
REF_HEADING = re.compile(r"^(tài liệu tham khảo|nguồn tham khảo|tham khảo|references?|nguồn tài liệu)\s*:?\s*$", re.I)
CTA = re.compile(
    r"hotline|tổng đài|1900\s?\d|đặt lịch (khám|hẹn|xét nghiệm|tư vấn)|xem thêm|bài viết liên quan|tin liên quan|"
    r"có thể bạn quan tâm|chia sẻ bài viết|đăng ký (nhận|tư vấn)|tải ứng dụng|quét mã|theo dõi (fanpage|kênh)|"
    r"cảm ơn bạn đã (chia sẻ|gửi|đặt) (thắc mắc|câu hỏi)|được giải đáp bởi|để chắc chắn, bạn có thể đến|"
    r"mời bạn (đọc|đón) xem|video đang được quan tâm",
    re.I)
WIDGET = re.compile(r"^(chia sẻ|copy link|bình luận|in bài|lưu bài|theo dõi|quay lại|trang chủ|\[cập nhật\])\s*$", re.I)
BIO = re.compile(r"(tốt nghiệp|năm kinh nghiệm|hiện (đang )?là|giảng viên|chuyên viên|dược sĩ|bác sĩ chuyên khoa)", re.I)
SIGNATURE = re.compile(r"^(dược sĩ|bác sĩ|ths\.?|bs\.?|ds\.?|nguồn( tổng hợp)?|biên tập|tác giả)\b", re.I)


def _short(line: str, words: int) -> bool:
    return len(line.split()) <= words


def _name_like(line: str) -> bool:
    """A bare person name: 2-5 capitalised words, no sentence punctuation."""
    w = line.split()
    return 2 <= len(w) <= 5 and all(t[:1].isupper() for t in w) and not re.search(r"[.?!:;,]", line)


def clean_v4(text: str) -> str:
    lines = [l.strip() for l in text.split("\n")]
    lines = [l for l in lines if l]
    # references block: from a reference heading to the end
    for i, l in enumerate(lines):
        if i > 2 and REF_HEADING.match(l):
            lines = lines[:i]
            break
    # author bio at the head (first 3 lines): bio sentence or bare name
    head = 0
    while head < min(3, len(lines)) and (BIO.search(lines[head]) and _short(lines[head], 45)
                                         or _name_like(lines[head])):
        head += 1
    if head and head < len(lines):
        lines = lines[head:]
    # signature / source lines at the tail
    while len(lines) > 1 and (_name_like(lines[-1]) or SIGNATURE.match(lines[-1]) and _short(lines[-1], 12)):
        lines.pop()
    out = []
    for l in lines:
        if RELATED_DATED.match(l) or TABLE_RELATED.match(l) or WIDGET.match(l):
            continue
        if URL_LINE.search(l) and _short(l, 40):
            continue
        if CTA.search(l) and _short(l, 60):
            continue
        out.append(l)
    return "\n".join(out) if out else text


# ---------------------------------------------------------------- section_chunks

NUMBERED = re.compile(r"^(\d{1,2}(\.\d{1,2})*\.?|[IVX]{1,4}\.|[a-z]\))\s+\S")


def is_heading(line: str, nxt: str | None) -> bool:
    """Heuristic heading: short line, no closing sentence punctuation, followed by a longer line."""
    if len(line) > 150 or len(line.split()) > 25 or line.startswith(("-", "+", "*", "•", "●")):
        return False
    if NUMBERED.match(line) and not line.endswith((".", ";", ",")):
        return True
    if line.endswith((".", ";", ",", ":", "!", "”", '"')):
        return False
    return nxt is not None and len(nxt) > len(line) * 1.5 and len(line.split()) >= 2


def split_sections(text: str) -> list[tuple[str, list[str]]]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    sections: list[tuple[str, list[str]]] = [("", [])]
    for i, l in enumerate(lines):
        if is_heading(l, lines[i + 1] if i + 1 < len(lines) else None):
            sections.append((l, []))
        else:
            sections[-1][1].append(l)
    return [(h, b) for h, b in sections if h or b]


def _pack(lines: list[str], size: int) -> list[str]:
    out, buf = [], ""
    for l in lines:
        while len(l) > size:  # one giant paragraph: cut at a space
            cut = l.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            if buf:
                out.append(buf)
                buf = ""
            out.append(l[:cut])
            l = l[cut:].lstrip()
        if buf and len(buf) + len(l) + 1 > size:
            out.append(buf)
            buf = ""
        buf = f"{buf}\n{l}" if buf else l
    if buf:
        out.append(buf)
    return out


def section_chunks(title: str, text: str, max_chars: int = 4500, min_chars: int = 600) -> list[str]:
    """Chunks of <= max_chars (+ title), each = title + section heading + section body."""
    title = (title or "").strip()
    merged: list[tuple[str, list[str]]] = []
    for head, body in split_sections(text):
        if merged and sum(len(x) + 1 for x in merged[-1][1]) + len(merged[-1][0]) < min_chars:
            prev_head, prev_body = merged[-1]  # previous section too small: absorb this one
            merged[-1] = (prev_head, prev_body + ([head] if head else []) + body)
        else:
            merged.append((head, list(body)))
    chunks = []
    for head, body in merged:
        budget = max(500, max_chars - len(head) - 1)
        for piece in _pack(body, budget) or [""]:
            parts = [p for p in (title, head, piece) if p]
            chunks.append("\n".join(parts))
    return chunks or [title or text[:max_chars]]
