"""High-speed corpus cleaner for ViBioMIR.

Cleans web boilerplate (breadcrumbs, date/related article lists, hotline,
disclaimers, social links) and filters out empty error pages (<150 chars),
while preserving verbatim sentence structure and 100% keeping raw data intact.

Usage:
    python scripts/r2ai2026_clean_corpus.py \
        --input data/docs.jsonl \
        --output data/docs_cleaned.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

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


def clean_text(text: str, is_pubmed: bool = False) -> str:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/docs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/docs_cleaned.jsonl"))
    parser.add_argument("--min-chars", type=int, default=150)
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8")

    if not args.input.exists():
        print(f"Error: input file {args.input} does not exist!")
        return 1

    if args.input.resolve() == args.output.resolve():
        print("Error: output file cannot be the same as input file! Preserving raw data is mandatory.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting corpus cleaning...")
    print(f"  Input (RAW, READ-ONLY): {args.input}")
    print(f"  Output (CLEANED):        {args.output}")

    started = time.perf_counter()
    total = 0
    kept = 0
    dropped = 0
    raw_bytes = 0
    cleaned_bytes = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            raw_bytes += len(line.encode("utf-8"))
            try:
                rec = json.loads(line)
            except Exception:
                dropped += 1
                continue

            url = rec.get("url", "")
            is_pm = "pubmed.ncbi.nlm.nih.gov" in url
            txt = rec.get("text", "") or ""

            cleaned = clean_text(txt, is_pubmed=is_pm)
            if len(cleaned) < args.min_chars:
                dropped += 1
                continue

            rec["text"] = cleaned
            out_line = json.dumps(rec, ensure_ascii=False) + "\n"
            fout.write(out_line)
            cleaned_bytes += len(out_line.encode("utf-8"))
            kept += 1

            if total % 100000 == 0:
                elapsed = time.perf_counter() - started
                print(f"  Processed {total:,} docs ({kept:,} kept, {dropped:,} dropped) - {total / elapsed:,.0f} docs/s...")

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 60)
    print("CORPUS CLEANING COMPLETED!")
    print(f"Total documents processed: {total:,}")
    print(f"  Kept clean documents:    {kept:,} ({kept / total * 100:.1f}%)")
    print(f"  Dropped junk/short docs: {dropped:,} ({dropped / total * 100:.1f}%)")
    print(f"Raw file size:             {raw_bytes / 1e9:.2f} GB")
    print(f"Cleaned file size:         {cleaned_bytes / 1e9:.2f} GB")
    print(f"Total runtime:             {elapsed:.2f}s ({total / elapsed:,.0f} docs/s)")
    print(f"Cleaned corpus saved to:   {args.output}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
