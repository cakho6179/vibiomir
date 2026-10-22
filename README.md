# ViBioMIR — R2AI2026 Shared Task 3

Truy hồi tài liệu y khoa đa ngữ (VI/ZH/EN) cho 1.200 câu hỏi tiếng Việt + trích chunk. Metric: FINAL = trung bình
F2 tài liệu và F2 chunk (`knowledge/pages/01_Evaluation.md`). Public phase đóng **31/10/2026 23:59 ICT** (API: 2026-10-31T16:59Z); private phase 01/11 00:00 → 03/11 23:59 ICT.
Điểm và trạng thái hiện tại: `EXPERIMENTS.md` (bảng A) · bài chờ nộp: `submissions/MANIFEST.md` · lộ trình: `docs/ROADMAP_SYSTEM.md`.

## Cấu trúc
```
vibiomir/
├── README.md            ← file này
├── EXPERIMENTS.md       sổ số đo: điểm LB (A), kernel (B), phân tích offline (C), dữ liệu (E)
├── WORKLOG.md           nhật ký hành động theo ngày
├── r2ai2026/            thư viện Python dùng chung (paths, text, cleaning, fetch, scoring, submit…)
├── scripts/             lệnh chạy local, theo giai đoạn
│   ├── data/            dựng corpus dẫn xuất từ data/raw (clean → processed_v3 → gói dataset Kaggle)
│   ├── crawl/           crawl / recrawl / gộp kết quả crawl (tạo ra data/raw/crawl, data/raw/recovered)
│   ├── kaggle_ops/      chuẩn bị kernel, nhiều tài khoản, chờ & tải kết quả
│   ├── submit/          dựng zip từ ranking, gom bài nộp về submissions/
│   ├── analysis/        đo đạc, kiểm toán corpus, phân tích theo hệ ICD
│   └── hunter_bridge/   cầu nối sang competition_hunter (workspace cha)
├── kaggle/              nguồn kernel chạy trên Kaggle (bm25_exp, dense_exp, rerank_exp, build_exp, cc_recover…)
│   ├── kaggle_jobs.json sổ kernel đã đẩy (skill kaggle-multi-account)
│   └── outputs/         kết quả kernel tải về (ngoài git)
├── experiments/         mỗi thí nghiệm một thư mục: câu hỏi, cách làm, kết quả, quyết định — xem REGISTRY.md
├── docs/                kế hoạch & thiết kế đang dùng; docs/archive/ = kế hoạch cũ
├── research/            báo cáo nghiên cứu, probe crawl; research/refs/ = repo đội khác (chỉ đọc, ngoài git)
├── knowledge/           tài liệu BTC (trang Codabench, leaderboard snapshot)
├── tests/               pytest
├── data/                (junction, ngoài git) raw/ = dữ liệu gốc CHỈ ĐỌC; còn lại là dẫn xuất
├── runs/                (ngoài git) báo cáo, ranking, log chạy local
├── submissions/         bài chờ nộp (zip ngoài git; MANIFEST.md/manifest.json trong git)
├── temp_submission/     bài đã có điểm (ngoài git)
└── _temp/               file thừa chờ xoá — không bao giờ xoá thẳng (ngoài git)
```
Bản đồ chi tiết "giai đoạn → file → lệnh": `docs/CODEMAP.md`.

## Luồng dữ liệu
```
data/raw/btc (links, query) ──crawl──► data/raw/crawl/docs.jsonl ─┐
data/raw/recovered/*.jsonl (Common Crawl, Wayback, PubMed, direct) ┴─► scripts/data → docs_clean
   → gói dataset Kaggle (kernel đọc docs_clean; processed_v3 chỉ ở local — xem docs/DATA_PIPELINE.md) → kernel: BM25 / BGE-M3 / reranker → ranking.parquet → kernel build_submission → zip
   → submissions/ (user nộp tay trên Codabench) → điểm ghi EXPERIMENTS.md bảng A
```

## Quy tắc làm việc
1. **Dữ liệu gốc bất biến**: `data/raw/` chỉ đọc (sha256 trong `data/raw/MANIFEST.md`); dữ liệu mới = file mới trong `data/raw/recovered/`.
2. **Việc nặng chạy trên Kaggle** (index, embed, rerank, build zip); local chỉ phân tích nhẹ. Python hệ thống 3.13, không venv.
3. **Bài nộp do user tự nộp**; agent chỉ dựng zip (≤ ~100 MB, LZMA) và cập nhật `submissions/MANIFEST.md`.
4. **Mỗi thí nghiệm có ID và thư mục** trong `experiments/`; số đo ghi `EXPERIMENTS.md`, hành động ghi `WORKLOG.md`.
5. **Không xoá file**: chuyển vào `_temp/<ngày>/`. Mọi thay đổi mã đi qua git commit.
6. Chỉ model mở ≤ 15B phát hành trước 01/08/2026 trong hệ thống nộp; không dùng MedKB API cho bài nộp.

## Lệnh hay dùng
```powershell
python -m pytest -q tests                                  # kiểm thử
python scripts/submit/collect_submissions.py               # gom zip mới về submissions/ + MANIFEST
python scripts/analysis/icd_systems.py --top 28            # đo trùng chương ICD theo hệ
$env:PYTHONIOENCODING="utf-8"; python ~/.claude/skills/kaggle-multi-account/scripts/kaggle_acc.py --ledger kaggle/kaggle_jobs.json status
```
