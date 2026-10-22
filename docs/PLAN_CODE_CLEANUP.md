# Kế hoạch dọn mã nguồn — bản để hình dung, CHƯA làm (22/10)

Nguồn: bản đồ mã do Haiku lập (22/10). Nguyên tắc cứng: **không xoá**, chỉ chuyển vào `_temp/<ngày>/cleanup/`; không đụng
kernel đang chạy; không đụng `.competition_hunter/`.

## 1. Hiện trạng [ĐO]
| Khu | Số file .py | Vấn đề |
|---|---|---|
| `r2ai2026/` (thư viện) | 25 | ổn; một số module cũ (pipeline/devproxy/fetch) có bản trùng ở `scripts/` |
| `scripts/` | 50 | tiền tố `r2ai2026_` dài; **18 file không còn nơi nào nhắc tới**; nhiều cặp phiên bản (decision_v4 / v4_1, clean_corpus / build_clean_corpus, kaggle_sync / wait_fetch) |
| `kaggle/` nguồn kernel | 16 | 3 cách đặt tên kernel; helper chép lại nhiều nơi |
| `kaggle/` bundle sinh tự động | ~215 file, 78 thư mục | bản sao gói `r2ai2026/` trong `recrawl_missing*`, `recrawl/dataset` (cũ), shim `kernels_generic` (43) |
| `research/results/_data/` | 41 | probe một lần (cc_check, cc_check2, enc_test2…) |
| Helper chép lại | — | `log()` ×8, `find()` ×3, `VI_MAX_ID` ×4, RRF ×8, ghi zip ×5, đọc corpus ×6 |
| Ngôn ngữ docstring | — | trộn Việt/Anh; nguồn kernel **bắt buộc ASCII** |
| Git | — | `vibiomir/` **chưa có git** ⇒ dọn mà không có lịch sử là rủi ro |

## 2. Cấu trúc đích (hình dung)
```
vibiomir/
├── r2ai2026/              thư viện dùng chung (giữ nguyên tên)
│   ├── common.py          ◄ MỚI: log, find, VI_MAX_ID, đọc corpus, RRF, ghi zip LZMA — một bản duy nhất
│   └── docstore/          ◄ MỚI (Pha 1–4 của DOCSTORE_DESIGN.md)
├── scripts/               CLI chạy local, chia theo giai đoạn
│   ├── data/              build_processed_corpus, pack_corpus_delta, merge_recovered…
│   ├── crawl/             recrawl, wayback, pubmed
│   ├── analysis/          icd_systems, probe_frame, host_report…
│   └── submit/            collect_submissions, build_from_rankings
├── kaggle/
│   ├── src/               nguồn kernel (ASCII), mỗi giai đoạn 1 file: bm25, dense_embed, rerank, build_submission,
│   │                      cc_recover, cc_columnar, translate_queries, docstore_* (mới)
│   ├── bundles/           thư mục kernel sinh bởi wrap (không sửa tay)
│   └── kaggle_jobs.json   sổ kernel
├── research/              phân tích + repo tham khảo (chỉ đọc)
├── docs/CODEMAP.md        ◄ MỚI: "giai đoạn → file chuẩn → lệnh chạy" (1 trang)
└── tests/
```

## 3. Các bước (mỗi bước độc lập, đảo ngược được)
| # | Việc | Rủi ro | Ghi chú |
|---|---|---|---|
| 0 | ✅ 22/10: `git init` cho `vibiomir/` (commit bffa280), dữ liệu gốc chuyển vào `data/raw/` (chỉ đọc, có MANIFEST sha256), `paths.py` trỏ về `raw/` | — | xong |
| 1 | Chuyển vào `_temp`: 41 probe `research/results/_data/`, `research/_reveng.py`, `recrawl/dataset` + `recrawl/by_account` (bản cũ), kernel BM25 v1 (e1/e2/e3), `__pycache__` | rất thấp | không file nào tham chiếu |
| 2 | Chuyển vào `_temp`: 18 script không còn tham chiếu (build_sota_v2, build_decision_v4, clean_corpus, kaggle_sync…) — Haiku kiểm lại từng file bằng grep trước | thấp | |
| 3 | Viết `docs/CODEMAP.md` | không | làm trước cả bước 4 để có bản đồ |
| 4 | Gom `scripts/` vào thư mục con, bỏ tiền tố `r2ai2026_`; ghi bảng tên cũ → mới trong CODEMAP | trung bình (WORKLOG trỏ tên cũ) | làm khi **không có** kernel nào đang chờ |
| 5 | Nguồn kernel → `kaggle/src/`, bundle → `kaggle/bundles/` | trung bình (sổ kernel trỏ đường dẫn bundle) | cập nhật sổ cùng lúc; sau khi 7 kernel crawl xong |
| 6 | `r2ai2026/common.py`: gom helper; **chỉ thay dần** khi một file được sửa vì lý do khác (không refactor ồ ạt) | thấp | kernel import qua bundle nhiều file (wrap đã hỗ trợ, như cc_columnar → cc_recover) |
| 7 | Quy ước: docstring/comment trong code bằng tiếng Anh (ASCII cho kernel); tài liệu `.md` bằng tiếng Việt | không | áp dụng cho file mới/sửa |

Thứ tự đề xuất: 0 → 1 → 2 → 3 ngay (nhanh, an toàn); 4–5 sau khi crawl xong; 6–7 liên tục.
