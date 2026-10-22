# 2026-10-22_pool_ceiling — Tập ứng viên hiện có chứa bao nhiêu % gold?

- **Trạng thái:** đang chạy
- **Câu hỏi:** Recall doc tối đa của tập ứng viên hiện tại (BM25 E1v2 top-200 ∪ BGE-M3 5 shard × top-200 ∪ M3L) ở độ sâu 300 và 500 là bao nhiêu? BM25 top-2000 thêm được bao nhiêu?
- **Giả thuyết:** Top 1 (ongthomiu) đạt R 0.665 ở P 0.275 (≈ 170 doc/query) ⇒ cần tập ứng viên recall ≥ 0.75. Slide BTC 22/10: BM25 lấy 2.000 doc → dense lọc 400 → rerank đoạn.
- **Quyết định phụ thuộc:** R@500 ≥ 0.75 ⇒ tập hiện tại đủ, dồn sức vào xếp hạng/cắt; R@500 < 0.6 ⇒ ưu tiên mở rộng ứng viên (BM25 top-2000, M3 sâu hơn, truy vấn viết lại).

## Cách làm
- `tthiutoko/vibiomir-build-pool` (acc2): job `pool_VI` = RRF(E1v2 : M3 : M3L = 1:2:1), depth 800/hệ, VI, n = [[300,75],[500,75]], xếp lại query. Chỉ đọc Docs R (Docs P và chunk không quan trọng ở bài thử).
- `tthiutoko/vibiomir-bm25-e1k2000` (acc2): BM25 E1 `--topk 2000 --vi-only` (tham số mới trong `kaggle/bm25_exp/bm25_experiment.py`).
- Đo offline không nhãn: độ phủ hợp ứng viên (số doc khác nhau/query, phần BM25-2000 không có trong M3).

## Kết quả [ĐO]
(chờ)

## Kết luận
(chờ)
