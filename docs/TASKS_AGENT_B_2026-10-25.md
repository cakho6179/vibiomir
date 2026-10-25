# Giao việc cho agent B (25/10 → 31/10)

Quyết định của user, 25/10:
- **docstore v2.1 là dữ liệu duy nhất** từ nay đến hết cuộc thi.
- **Dựng full trên v2.1 và lưu lại toàn bộ:** embedding dense (cả 3 ngôn ngữ) và index BM25, chạy trên Kaggle, tải kết quả về. Mọi kiểu truy vấn sau đó chỉ tốn công encode/score query.
- Vì vậy đề xuất "không re-embed v2.1" không áp dụng. Lý do: embedding lưu một lần sẽ dùng cho mọi biến thể truy vấn (câu hỏi con, HyDE, ZH), và probe ZH/EN cần doc ZH mới của v2.1.

## Chia phần (để không ghi đè nhau)

| | Agent A (Claude) | Agent B |
|---|---|---|
| Phạm vi | tầng lấy doc + truy vấn + decision model | **chunk** (điểm Chunks F2), dữ liệu, tái lập, working notes |
| File sở hữu | `r2ai2026/stages/{bm25_experiment,dense_embed,query_frame,decide,judge,build_pairs,entity_index}.py`, `r2ai2026/system/runner.py`, `configs/system_v2_1*.yaml` (phần stage lấy doc) | `r2ai2026/stages/build_submission.py` (phần chunk), `scripts/data/*`, `data/docstore/v2.1/*` (đóng băng; chỉ sửa khi user yêu cầu), `docs/DOCSTORE_*`, `docs/EXTERNAL_*`, file mới `docs/WORKING_NOTES*.md`, `docs/DATA_CARD.md` |
| Tài khoản Kaggle | GPU: acc1 hytahy, acc3 noskaiser, acc4 lanpana, acc5/main snnguynvnk19hl; GPU của acc2 trong lượt embed v2.1 | **CPU của acc2 tthiutoko** (5 phiên). Cần GPU thì báo trước |
| Không làm | – | không chạy dense / BM25 / query-frame / decide; không sửa các file sở hữu của A. Thấy lỗi ở đó thì ghi vào WORKLOG để A sửa (ví dụ bug alias trong `normalise()`: ghi rõ file:dòng) |

File dùng chung (`WORKLOG.md`, `EXPERIMENTS.md`, `docs/CODEMAP.md`): chỉ **thêm** dòng mới, không viết lại dòng của người khác. Commit chỉ file của mình.

## Việc của agent B, theo thứ tự

1. **Probe chunk-only trên v2.1** (đề xuất số 2 của B; giữ nguyên).
   - Lấy đúng ứng viên và rerank của #6935 (0.2805; Docs .3256 / Chunks .2353), chỉ dựng lại chunk trên text v2.1.
   - Map đoạn best-passage cũ sang body mới để lấy offset; không tìm thấy thì dùng cửa sổ 4.500 ký tự.
   - Dựng zip trên Kaggle CPU acc2. **Không nộp**: đưa user duyệt file trước.
   - Báo: số doc đổi offset, số doc phải dùng fallback, kích thước zip.
2. **Chunk theo section** (đề xuất số 4 của B).
   - Chunk = section chứa best passage (≤ 4.800 ký tự), thêm làm một chế độ mới trong `build_submission.py` (`chunk: {mode: section}`). Giữ chế độ `window` như cũ, có test.
   - Một probe riêng trên cùng ứng viên của #6935, cũng cần user duyệt. Mục tiêu là tách được Chunks F2 thấp là do recall hay do biên chunk.
3. **Hồ sơ dữ liệu cho working notes.** Tạo `docs/DATA_CARD.md` gồm:
   - Nguồn của từng phần trong v2.1 (crawl, wayback, cc, direct, pubmed, lc, tmh), số doc mỗi nguồn, commit/sha của nguồn ngoài.
   - Giấy phép và trích dẫn. HF tmh không có license: liệt kê để hỏi BTC cùng email đang chờ gửi.
   - Cách dựng lại (`scripts/data/build_docstore_v2_1.py` + `validate_docstore.py`).
4. **Kiểm tra tái lập.** Rà toàn repo tìm đường dẫn hard-code (`/kaggle/...`, `D:\...`) trong `r2ai2026/` và `configs/`, rồi viết danh sách vào WORKLOG. Chỉ sửa trực tiếp những file B sở hữu.
5. **Khung working notes** (`docs/WORKING_NOTES_DRAFT.md`): mục lục, mô tả dữ liệu, mô tả chunk. Để chỗ trống cho phần truy hồi và decision model; A sẽ điền.

## Mốc so sánh
Best: #6935 = 0.2805 (Docs F2 .3256, Chunks F2 .2353). Mỗi probe trả lời được: tăng Docs hay Chunks, và có vượt mốc này không. Mọi lượt nộp đều phải được user duyệt từng file.
