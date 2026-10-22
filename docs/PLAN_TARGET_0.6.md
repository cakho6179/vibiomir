# Kế hoạch lên FINAL 0.6 (lập 2026-10-21) — ĐÃ THAY bằng `docs/ROADMAP_SYSTEM.md` (22/10)

> 22/10: ràng buộc "N = 28 cố định" bên dưới đã được user đính chính — số doc/query biến thiên, chọn bằng decision model.

Nguyên tắc: **mọi tác vụ nặng chạy trên Kaggle** (5 account; mỗi account ≤5 CPU + ≤2 GPU session đồng thời).
Local chỉ: viết code, test nhỏ, ghép ranking (parquet vài MB), ghi sổ. Mỗi lượt nộp phải trả lời đúng một câu hỏi;
user nộp tay (10 lượt/ngày, upload ≤ ~100 MB ⇒ zip LZMA). Số đo ghi `EXPERIMENTS.md`, việc làm ghi `WORKLOG.md`.

## Hiện trạng (đo)
- Tốt nhất 0.1997 = RRF(BM25 E1v2, BGE-M3) 1:2, N=28: P 0.354, R 0.203. Gold ≈ 46 doc/query, ~99% tiếng Việt.
- BGE-M3 đơn P 0.345; BM25 đơn P 0.178 (BM25 của BTC P 0.88!); Qwen3 làm loãng.
- Mô hình gold [GIẢ THUYẾT]: gold = (BM25 của BTC ∪ dense ~BGE-M3 của BTC) trên chunk của họ, lọc bởi người/LLM.

## RÀNG BUỘC CỐ ĐỊNH (user, 2026-10-21): N = 28 doc/query — không nộp được N > 28.
Gold ≈ 46 ⇒ recall tối đa ≈ 28/46 ≈ 0.61. FINAL ≈ docs F2 (chunk = toàn văn) ⇒ 0.6 cần P ≈ 0.9+ ở top-28
(P 0.88 / R 0.535 → F2 0.58 — đúng mức BM25 của BTC). Mục tiêu thực tế: **tối đa precision@28** bằng cách tái
tạo BM25 + dense của BTC; pha P1 (độ sâu N) và các zip N>28 bị hủy. Mục "Phép tính mục tiêu" bên dưới chỉ còn
giá trị tham khảo.

## Phép tính mục tiêu
F2 = 5PR/(4P+R). FINAL ≈ docs F2 (chunk = toàn văn). Ví dụ: P 0.45 / R 0.70 → 0.63; P 0.30 / R 0.80 → 0.60;
P 0.35 / R 0.55 → 0.49. ⇒ cần **recall ≥ 0.7** ⇒ N lớn (≈70–110) **và** precision giữ ≥ 0.3 ở độ sâu đó.
Điều này chỉ có được nếu thứ hạng của mình trùng pool gold sâu, tức tái tạo được BM25 + dense của BTC.

## Các pha

### P1 — Độ sâu N (ngay, 0 compute)
- Nộp `runs/submissions/e1v2_m3_w12_lz_n70.zip` (LZMA, 68 MB): xác nhận server nhận LZMA + đo P ở N=70.
- Nếu P(70) ≥ 0.25 → dựng N=100, 120 (trên Kaggle, LZMA).

### P2 — Rerank + chỉ doc tiếng Việt (đã dựng, chờ nộp)
- `rerank_r1_VI_n28`, `w12_rr_VI_n28/_n70`, `w12_VI_n28`: đo rerank (bge-reranker-v2-m3) và lọc VI.

### P3 — Tái tạo BM25 của BTC (Kaggle CPU, song song 4 account)
Kernel `bm25_experiment.py` mở rộng tham số; mỗi biến thể xuất ranking + zip n28 (deflate < 100 MB):
| id | Biến thể | Giả thuyết |
|---|---|---|
| b1 | chunk 1000 ký tự / overlap 200 (kiểu LangChain), âm tiết, doc = max chunk | stack RAG phổ biến |
| b2 | như b1 + tách từ pyvi | stack RAG tiếng Việt phổ biến |
| b3 | doc-level, k1 0.9 b 0.4 (mặc định Pyserini) | dùng Pyserini/Anserini |
| b4 | chunk 512 ký tự / overlap 100, âm tiết | chunk nhỏ |
| e3v2 | (có sẵn) doc-level + pyvi | tách từ |
Chọn biến thể có P cao nhất → dùng trong fusion. Biến thể nào P ≥ 0.4 là dấu hiệu đúng hướng.

### P4 — Tái tạo dense của BTC (Kaggle GPU)
- M3 trên chunk 1000/200 không prefix tiêu đề, corpus **chỉ tiếng Việt** (728k doc ⇒ nhẹ hơn 40%), dùng processed v3.
- So với M3 hiện tại (1500 ký tự + tiêu đề). Thêm M3 sparse/colbert nếu còn quota.

### P5 — Dựng bài nộp trên Kaggle
- Kernel `build_submission` nhận dataset rankings nhỏ → ghép RRF → zip LZMA (đọc corpus trên Kaggle, không đọc 4.9 GB ở local).

### P6 — Kết hợp cuối
- Fusion các thành phần tốt nhất (trọng số theo P đo được), rerank, chọn N theo đường cong P(N) đo được.

## Quy tắc ra quyết định
- Mỗi biến thể được giữ nếu cải thiện P ở N=28 so với thành phần cùng loại.
- Không nộp hai bản trả lời cùng một câu hỏi. Ghi mọi kết quả vào EXPERIMENTS.md ngay khi user gửi.
