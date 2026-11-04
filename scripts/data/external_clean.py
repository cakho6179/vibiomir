"""Conservative boilerplate cleaner for the external tmh1008 crawl.

The HF crawl stores the whole page, including navigation, related links and
footers.  We do not want that noise in the docstore, especially for Chinese
pages where the article is often followed by a long site menu.

The cleaner is deliberately conservative:

* it only cuts at explicit boilerplate markers, never at arbitrary sentences;
* it removes whole lines that are navigation/footer lines;
* it keeps the text before the first marker, so an article body is never
  shortened in the middle;
* it returns ``""`` when too little remains, and the caller drops the row.

It is not a general HTML cleaner and it does not try to recover a missing
article.  A row that is only a menu stays a menu and must be rejected.
"""
from __future__ import annotations

import re

from r2ai2026.corpus.processing import prepare_body

# Markers that start a footer / related-links block.  Cut at the first one.
_ZH_CUT = [
    r"用药\s*药品库",
    r"查看说明书",
    r"医生在线免费咨询",
    r"看病\s*男科",
    r"[\u4e00-\u9fffA-Za-z0-9]{2,20}\s*本品用于",
    r"对医师的追问",
    r"赠送医生锦旗",
    r"支付金额",
    r"请选择投诉理由",
    r"文章录入",
    r"网友评论",
    r"绿色通道",
    r"版权所有",
    r"更多>>",
    r"查看更多关于",
    r"以上是对",
    r"健康资讯推荐",
    r"医生回答>>",
    r"挂号科室",
    r"你的浏览器不支持",
    r"Copyright",
    r"相关阅读",
    r"推荐阅读",
    r"上一篇",
    r"下一篇",
    r"免责声明",
    r"本站信息仅供参考",
    r"大医精诚",
    r"中医宝典\s*$",
]

_VI_CUT = [
    r"Nguồn\s*:",
    r"Nguồn ảnh",
    r"Bài viết liên quan",
    r"Tin liên quan",
    r"Đọc nhiều",
    r"Bình luận",
    r"Chia sẻ",
    r"Quảng cáo",
    r"Hotline",
    r"Liên hệ ngay",
    r"Copyright",
    r"©",
    r"Lựa chọn dịch vụ",
    r"Quý khách",
    r"Để đặt lịch",
    r"Tham vấn",
    r"Tác giả",
]

_ZH_PREFIX = [
    r"^首页\s+中医\s+穴位.*?搜索\s*",
    r"^中医宝典\s*>\s*",
    r"^A\+医学百科\s*>>\s*",
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+我要投诉\s*",
    r"^栏目关注：\s*",
]

_VI_PREFIX = [
    r"^Menu\s*",
    r"^Trang chủ\s*",
    r"^Tin tức\s*",
]

_NAV_LINE = re.compile(
    r"^\s*(?:"
    r"首页|中医|穴位|中药|方剂|疾病|诊断|药品|急救|登录|注册|相关阅读|推荐阅读|上一篇|下一篇|"
    r"联系我们|网站地图|免责声明|版权声明|关于我们|"
    r"Trang chủ|Giới thiệu|Bệnh thường gặp|Video|Liên hệ|Đăng nhập|Đăng ký|Menu|"
    r"Bài viết nổi bật|Đọc nhiều nhất|Bài viết mới nhất"
    r")\s*$",
    re.IGNORECASE,
)

# Lines that are only a copyright/contact/boilerplate notice.
_BOILER_LINE = re.compile(
    r"^\s*(?:Copyright|©|版权所有|免责声明|本站信息仅供参考|"
    r"文章录入|责任编辑|网友评论|绿色通道|"
    r"Nguồn\s*:|Hotline|Liên hệ|Quảng cáo)\b",
    re.IGNORECASE,
)


def _strip_table_nav(text: str) -> str:
    """Drop a leading ``|`` navigation table, keeping the first prose cell."""
    if text.count("|") < 4:
        return text
    parts = re.split(r"\|", text)
    for i, part in enumerate(parts):
        s = part.strip()
        if len(s) >= 40 and re.search(r"[。！？!?]", s):
            return " ".join(p.strip() for p in parts[i:] if p.strip())
    return text


def _cut_at_markers(text: str, markers: list[str]) -> str:
    best: int | None = None
    for pat in markers:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m and (best is None or m.start() < best):
            best = m.start()
    if best is not None and best >= 200:
        return text[:best]
    return text


def _cut_hard(text: str, markers: list[str]) -> str:
    """Cut at the first marker, regardless of position (host-specific noise)."""
    best: int | None = None
    for pat in markers:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m and (best is None or m.start() < best):
            best = m.start()
    if best is not None:
        return text[:best]
    return text


def clean_external_body(title: str, text: str, lang: str, host: str = "") -> str:
    """Return a cleaned body or ``""`` when nothing usable remains."""
    title, body = prepare_body(title, text)
    if not body.strip():
        return ""

    if lang == "zh":
        for pat in _ZH_PREFIX:
            body = re.sub(pat, "", body, count=1, flags=re.IGNORECASE | re.DOTALL)
        if host.endswith("120ask.com"):
            # Drop the doctor-intro prefix when the structured Q&A starts later.
            positions = [body.find(m) for m in ("健康咨询描述", "病情分析") if body.find(m) >= 0]
            if positions and min(positions) > 0:
                body = body[min(positions):]
            body = _cut_hard(body, [
                r"医生在线免费咨询",
                r"查看更多关于",
                r"以上是对",
                r"医生回答>>",
                r"更多>>",
                r"[\u4e00-\u9fffA-Za-z0-9]{2,20}\s*查看说明书",
                r"[\u4e00-\u9fffA-Za-z0-9]{2,20}\s*本品用于",
                r"用药\s*药品库",
            ])
        body = _cut_at_markers(body, _ZH_CUT)
        body = _strip_table_nav(body)
    else:
        for pat in _VI_PREFIX:
            body = re.sub(pat, "", body, count=1, flags=re.IGNORECASE)
        body = _cut_at_markers(body, _VI_CUT)

    lines = []
    for line in body.split("\n"):
        s = line.strip()
        if not s:
            continue
        if _NAV_LINE.match(s) or _BOILER_LINE.match(s):
            continue
        # A table/navigation row: keep it only if it contains prose.
        if s.startswith("|") and len(s) < 160 and not re.search(r"[。！？.!?]", s):
            continue
        lines.append(s)

    body = "\n".join(lines).strip()
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = re.sub(r"[ \t]{2,}", " ", body).strip()
    if len(body) < 200:
        return ""
    return body
