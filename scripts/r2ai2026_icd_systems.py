"""Gán nhóm hệ cơ quan (chương ICD-10) cho query và tiêu đề doc, rồi đo mức trùng nhóm của các ranking.

Câu hỏi: hệ nào có precision LB cao hơn thì có trùng chương ICD với query nhiều hơn không? Nếu có,
"trùng chương" là một đặc trưng tốt cho decision model (slide mentor tr.28-31: ưu tiên, không lọc cứng).

Từ điển: ontology ICD-10 tam ngữ (Bộ Y tế icd.kcb.vn) trong repo tham khảo neiths/med-doc-retrieval,
cộng một danh sách từ ngữ đời thường theo hệ cơ quan. Chỉ đọc cột id/title của corpus (nhẹ, chạy local).

    python scripts/r2ai2026_icd_systems.py --top 28 --out runs/reports/icd_systems.md
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = ROOT / "research/refs/neiths_med-doc-retrieval/data/lexicon/icd10_ontology.json"
RANKINGS = ROOT / "kaggle/rankings_ds"

# (chương, khoảng mã) theo slide mentor tr.45-46
CHAPTERS = [
    ("I", "A00", "B99"), ("II", "C00", "D48"), ("III", "D50", "D89"), ("IV", "E00", "E90"),
    ("V", "F00", "F99"), ("VI", "G00", "G99"), ("VII", "H00", "H59"), ("VIII", "H60", "H95"),
    ("IX", "I00", "I99"), ("X", "J00", "J99"), ("XI", "K00", "K93"), ("XII", "L00", "L99"),
    ("XIII", "M00", "M99"), ("XIV", "N00", "N99"), ("XV", "O00", "O99"), ("XVI", "P00", "P96"),
    ("XVII", "Q00", "Q99"), ("XVIII", "R00", "R99"), ("XIX", "S00", "T98"),
]
CHAPTER_NAMES = {
    "I": "nhiễm trùng", "II": "khối u", "III": "máu/miễn dịch", "IV": "nội tiết/chuyển hoá",
    "V": "tâm thần", "VI": "thần kinh", "VII": "mắt", "VIII": "tai", "IX": "tuần hoàn",
    "X": "hô hấp", "XI": "tiêu hoá", "XII": "da", "XIII": "cơ xương khớp", "XIV": "tiết niệu/sinh dục",
    "XV": "thai sản", "XVI": "chu sinh", "XVII": "dị tật bẩm sinh", "XVIII": "triệu chứng chung",
    "XIX": "chấn thương/ngộ độc",
}
# Từ ngữ đời thường, không có trong tên ICD chính thức. Chỉ cụm đủ đặc hiệu (tránh "da", "tai" đơn lẻ).
LAY_TERMS = {
    "I": ["sốt xuất huyết", "tay chân miệng", "thuỷ đậu", "thủy đậu", "sởi", "lao phổi", "hiv", "giun", "sán",
          "nhiễm khuẩn", "virus", "zona", "cúm"],
    "II": ["ung thư", "khối u", "u ác", "u lành", "di căn", "hoá trị", "hóa trị", "xạ trị", "polyp"],
    "III": ["thiếu máu", "tiểu cầu", "bạch cầu", "máu khó đông", "thalassemia", "miễn dịch"],
    "IV": ["tiểu đường", "đái tháo đường", "đường huyết", "tuyến giáp", "bướu cổ", "mỡ máu", "cholesterol",
           "gout", "gút", "béo phì", "nội tiết", "vitamin", "suy dinh dưỡng"],
    "V": ["trầm cảm", "lo âu", "mất ngủ", "rối loạn tâm thần", "tâm lý", "stress", "căng thẳng", "tự kỷ",
          "tăng động", "nghiện"],
    "VI": ["đau đầu", "động kinh", "co giật", "parkinson", "alzheimer", "sa sút trí tuệ", "thần kinh",
           "tê bì", "đau nửa đầu", "rối loạn tiền đình", "tiền đình"],
    "VII": ["mắt", "cận thị", "viễn thị", "loạn thị", "đục thuỷ tinh thể", "đục thủy tinh thể", "glaucoma",
            "võng mạc", "giác mạc", "kết mạc", "thị lực"],
    "VIII": ["ù tai", "điếc", "viêm tai", "tai giữa", "thính lực"],
    "IX": ["huyết áp", "tim mạch", "nhồi máu", "đột quỵ", "tai biến", "suy tim", "mạch vành", "rối loạn nhịp",
           "nhịp tim", "suy giãn tĩnh mạch", "giãn tĩnh mạch", "trĩ", "xơ vữa"],
    "X": ["ho", "hen suyễn", "hen phế quản", "phổi", "phế quản", "viêm họng", "viêm mũi", "xoang", "amidan",
          "hô hấp", "khó thở", "viêm phổi", "copd"],
    "XI": ["dạ dày", "tá tràng", "đại tràng", "ruột", "gan", "mật", "tuỵ", "tụy", "tiêu hoá", "tiêu hóa",
           "táo bón", "tiêu chảy", "trào ngược", "thực quản", "răng", "chân răng", "nướu", "miệng", "đau bụng",
           "đầy hơi", "khó tiêu", "vi khuẩn hp"],
    "XII": ["mụn", "trứng cá", "nổi mẩn", "mề đay", "ngứa", "chàm", "vảy nến", "viêm da", "nấm da", "da liễu",
            "rụng tóc", "nám", "tàn nhang", "sẹo"],
    "XIII": ["xương khớp", "đau lưng", "thoát vị đĩa đệm", "đau khớp", "viêm khớp", "thoái hoá", "thoái hóa",
             "loãng xương", "cột sống", "đau vai gáy", "gai cột sống", "cơ xương"],
    "XIV": ["thận", "tiết niệu", "bàng quang", "tiểu buốt", "tiểu rắt", "sỏi thận", "tuyến tiền liệt",
            "kinh nguyệt", "buồng trứng", "tử cung", "âm đạo", "phụ khoa", "vô sinh", "hiếm muộn", "tinh trùng",
            "dương vật", "tinh hoàn", "vú", "hành kinh", "chu kỳ kinh"],
    "XV": ["mang thai", "có thai", "thai kỳ", "thai nhi", "bà bầu", "mẹ bầu", "sinh mổ", "sinh thường",
           "sảy thai", "dọa sảy", "doạ sảy", "ốm nghén", "sau sinh", "tiền sản giật", "tuần thai"],
    "XVI": ["trẻ sơ sinh", "sơ sinh", "vàng da sơ sinh", "sinh non"],
    "XVII": ["bẩm sinh", "dị tật", "hở hàm ếch", "hội chứng down"],
    "XIX": ["gãy xương", "chấn thương", "bỏng", "ngộ độc", "côn trùng cắn", "chó cắn", "rắn cắn", "bong gân"],
}

_TOK = re.compile(r"[0-9a-zà-ỹđ]+")


def norm(s: str) -> list[str]:
    s = unicodedata.normalize("NFC", s or "").lower()
    return _TOK.findall(s)


def chapter_of(code: str) -> str | None:
    c = code[:3]
    for ch, lo, hi in CHAPTERS:
        if lo <= c <= hi:
            return ch
    return None


def build_lexicon() -> dict[tuple[str, ...], str]:
    lex: dict[tuple[str, ...], str] = {}
    data = json.loads(ONTOLOGY.read_text(encoding="utf-8"))
    for e in data["entries"]:
        ch = chapter_of(e["icd10_code"])
        if ch is None:
            continue
        for name in (e.get("name_vn"), e.get("parent_type")):
            if not name:
                continue
            name = re.sub(r"\(.*?\)", " ", name).split(",")[0]
            toks = tuple(norm(name))
            if 2 <= len(toks) <= 8:
                lex.setdefault(toks, ch)
    for ch, terms in LAY_TERMS.items():  # từ đời thường ghi đè tên ICD trùng
        for t in terms:
            lex[tuple(norm(t))] = ch
    return lex


def tag(text: str, lex: dict, maxn: int = 8) -> Counter:
    toks = norm(text)
    out: Counter = Counter()
    i = 0
    while i < len(toks):
        for n in range(min(maxn, len(toks) - i), 0, -1):
            ch = lex.get(tuple(toks[i:i + n]))
            if ch:
                out[ch] += 1
                i += n
                break
        else:
            i += 1
    return out


def load_ranking(name: str) -> pd.DataFrame:
    files = sorted(RANKINGS.glob(f"{name}.parquet")) or sorted(RANKINGS.glob(f"{name}_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files])
    return df.sort_values(["qid", "score"], ascending=[True, False])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=28)
    ap.add_argument("--systems", default="bm25_e1v2,m3,q3e,m3c,m3L,bm25_b1,bm25_b3,bm25_b5,rerank_r1")
    ap.add_argument("--out", default=str(ROOT / "runs/reports/icd_systems.md"))
    args = ap.parse_args()

    lex = build_lexicon()
    q = pd.read_parquet(ROOT / "data/raw/btc/query.parquet")
    qtag = {int(r.id): tag(r.query, lex) for r in q.itertuples()}

    rankings = {}
    for s in args.systems.split(","):
        df = load_ranking(s)
        df = df.groupby("qid").head(args.top)
        rankings[s] = df
    need = set().union(*(set(df.doc_id.astype(int)) for df in rankings.values()))

    titles: dict[int, str] = {}
    for f in sorted((ROOT / "data/processed_v3").glob("docs-*.parquet")):
        t = pq.read_table(f, columns=["id", "title"]).to_pandas()
        t = t[t.id.isin(need)]
        titles.update(zip(t.id.astype(int), t.title))
    dtag = {d: tag(titles.get(d, ""), lex) for d in need}

    lines = [f"# Nhóm hệ cơ quan (chương ICD-10) — query vs top-{args.top}", ""]
    tagged_q = [k for k, v in qtag.items() if v]
    lines.append(f"Query có ≥1 chương: {len(tagged_q)}/{len(qtag)} ({len(tagged_q)/len(qtag):.1%}); "
                 f"nhiều chương: {sum(len(v) > 1 for v in qtag.values())}. Lexicon {len(lex)} cụm.")
    cnt = Counter(ch for v in qtag.values() for ch in v)
    lines += ["", "| Chương | Hệ | #query |", "|---|---|---|"]
    lines += [f"| {ch} | {CHAPTER_NAMES[ch]} | {n} |" for ch, n in cnt.most_common()]

    lines += ["", f"| Hệ thống | doc có chương | trùng chương (doc có chương, query có chương) | "
              f"doc khác chương hoàn toàn / query |", "|---|---|---|---|"]
    for s, df in rankings.items():
        has = agree = total = 0
        off = []
        for qid, g in df.groupby("qid"):
            qt = set(qtag.get(int(qid), {}))
            o = 0
            for d in g.doc_id.astype(int):
                total += 1
                dt = set(dtag.get(d, {}))
                if dt:
                    has += 1
                    if qt:
                        if dt & qt:
                            agree += 1
                        else:
                            o += 1
            if qt:
                off.append(o)
        pairs = sum(off) + agree
        lines.append(f"| {s} | {has/total:.1%} | {agree/max(pairs,1):.1%} | {sum(off)/max(len(off),1):.2f} |")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
