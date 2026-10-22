# Bài nộp ViBioMIR — một chỗ duy nhất

Sinh bởi `scripts/r2ai2026_collect_submissions.py`; sửa `status`/`note` trong `manifest.json`, điểm LB ghi ở `EXPERIMENTS.md` bảng A rồi chạy lại script.

| File | Trạng thái | Ghi chú / hệ thống | MB | doc/q | chunk/q | FINAL | Docs P / R | Chunks P / R | Nằm ở | Nguồn |
|---|---|---|---|---|---|---|---|---|---|---|
| `e1v2_m3_w12_lz_n70.zip` | chưa nộp | w12 70/70 LZMA (trước VI-only) — đã có depth probe thay | 67.6 |  |  |  |  |  | submissions | runs/submissions/e1v2_m3_w12_lz_n70.zip |
| `m3L_VI_n28.zip` | chưa nộp | BGE-M3 chunk 4500; 96.6 MB | 96.6 | 28.0 |  |  |  |  | submissions | kaggle/outputs/vibiomir-build-v2/m3L_VI_n28.zip |
| `m3c_VI_n28.zip` | chưa nộp | BGE-M3 chunk 1000 không tiêu đề | 76.2 | 28.0 |  |  |  |  | submissions | kaggle/outputs/vibiomir-build-v2/m3c_VI_n28.zip |
| `rerank_r1_VI_n28.zip` | chưa nộp | bge-reranker-v2-m3 trên E1v2∪M3, VI | 32.6 |  |  |  |  |  | submissions | runs/submissions/rerank_r1_VI_n28.zip |
| `w12_VI_d100c28.zip` | chưa nộp | Depth probe: w12 VI, 100 doc + 28 chunk ⇒ P(100) | 36.6 | 99.98 | 28.0 |  |  |  | submissions | kaggle/outputs/vibiomir-build-depth-a/w12_VI_d100c28.zip |
| `w12_VI_d150c28.zip` | chưa nộp | Depth probe: 150 doc + 28 chunk ⇒ P(150) — ưu tiên 1 | 36.9 | 149.94 | 28.0 |  |  |  | submissions | kaggle/outputs/vibiomir-build-depth-a/w12_VI_d150c28.zip |
| `w12_VI_d150c60.zip` | chưa nộp | 150 doc + 60 chunk toàn văn ⇒ lợi chunk theo độ sâu | 68.7 | 149.94 | 60.0 |  |  |  | submissions | kaggle/outputs/vibiomir-build-depth-a/w12_VI_d150c60.zip |
| `w12_VI_d150c90.zip` | chưa nộp | 150 doc + 90 chunk, 99.7 MB (dưới trần 102.6 MB đội khác chứng minh) — ứng viên mạnh | 99.7 | 149.94 | 89.99 |  |  |  | submissions | kaggle/outputs/vibiomir-build-depth-b/w12_VI_d150c90.zip |
| `w12_VI_d200c28.zip` | chưa nộp | Depth probe: 200 doc + 28 chunk ⇒ P(200) | 37.1 | 199.9 | 28.0 |  |  |  | submissions | kaggle/outputs/vibiomir-build-depth-a/w12_VI_d200c28.zip |
| `w12_VI_n28.zip` | chưa nộp | bản 0.1997 chỉ giữ doc VI | 33.7 |  |  |  |  |  | submissions | runs/submissions/w12_VI_n28.zip |
| `w12_rr_VI_n28.zip` | chưa nộp | RRF w12 + reranker, VI | 33.7 |  |  |  |  |  | submissions | runs/submissions/w12_rr_VI_n28.zip |
| `w12_rr_VI_n70.zip` | chưa nộp | như trên, 70 doc/chunk | 72.6 |  |  |  |  |  | submissions | runs/submissions/w12_rr_VI_n70.zip |
| `rdT1_VI_d150c90r.zip` | chưa nộp | như rd + ngưỡng ≥ −1 (TB 86 doc, 69 chunk/query) — chặt hơn | 68.6 | 86.24 | 69.09 |  |  |  | submissions | kaggle/outputs/vibiomir-build-depth-d/rdT1_VI_d150c90r.zip |
| `rdT2_VI_d150c90r.zip` | chưa nộp | như rd + cắt theo ngưỡng reranker ≥ −2 (TB 103 doc, 78 chunk/query) — số doc biến thiên | 71.5 | 103.37 | 77.9 |  |  |  | submissions | kaggle/outputs/vibiomir-build-depth-c/rdT2_VI_d150c90r.zip |
| `rd_VI_d150c90r.zip` | chưa nộp | RRF(E1v2, M3, reranker) 1:2:3, 150 doc + 90 chunk, xếp lại query (85.4 MB) — reranker trong thứ tự | 85.4 | 149.94 | 89.99 |  |  |  | submissions | kaggle/outputs/vibiomir-build-depth-c/rd_VI_d150c90r.zip |
| `w12T2_VI_d150c90r.zip` | chưa nộp | w12 (không reranker trong thứ tự) + ngưỡng ≥ −2 (TB 88 doc, 75 chunk) — tách hiệu ứng cắt | 71.3 | 88.23 | 75.42 |  |  |  | submissions | kaggle/outputs/vibiomir-build-depth-d/w12T2_VI_d150c90r.zip |
| `b1_n28.zip` | không nộp | BM25 chunk 1000/200 — dừng hướng BM25 (22/10: các biến thể đều ~0.09–0.11) | 68.3 |  |  |  |  |  | temp_submission/khong_nop | kaggle/outputs/vibiomir-bm25-b1/submission_b1_n28.zip |
| `b2_n28.zip` | không nộp | BM25 b1 + pyvi — dừng hướng BM25 (22/10: các biến thể đều ~0.09–0.11) | 67.7 |  |  |  |  |  | temp_submission/khong_nop | kaggle/outputs/vibiomir-bm25-b2/submission_b2_n28.zip |
| `b3_VI_n28.zip` | đã nộp | BM25 doc-level k1 0.9 b 0.4 (Pyserini), LZMA | 32.4 | 28.0 |  | 0.0874 | 0.1415 / 0.0908 | 0.1411 / 0.0953 | temp_submission | kaggle/outputs/vibiomir-build-v2/b3_VI_n28.zip |
| `b3_n28.zip` | không nộp | 124 MB deflate > trần; dùng b3_VI_n28 (LZMA) | 124.3 |  |  |  |  |  | temp_submission/khong_nop | kaggle/outputs/vibiomir-bm25-b3/submission_b3_n28.zip |
| `b4_n28.zip` | không nộp | BM25 chunk 512/100 — dừng hướng BM25 (22/10: các biến thể đều ~0.09–0.11) | 69.1 |  |  |  |  |  | temp_submission/khong_nop | kaggle/outputs/vibiomir-bm25-b4/submission_b4_n28.zip |
| `b5_n28.zip` | không nộp | BM25 chunk 4500 — dừng hướng BM25 (22/10: các biến thể đều ~0.09–0.11) | 76.0 |  |  |  |  |  | temp_submission/khong_nop | kaggle/outputs/vibiomir-bm25-b5/submission_b5_n28.zip |
| `c1_n28.zip` | không nộp | BM25 clean_v4 doc-level, VI — dừng hướng BM25 (22/10: các biến thể đều ~0.09–0.11) | 77.0 |  |  |  |  |  | temp_submission/khong_nop | kaggle/outputs/vibiomir-bm25-c1/submission_c1_n28.zip |
| `e1_m3_q3e_w122_lz_n70.zip` | không nộp | Qwen3 làm loãng (0.1948 @28) | 64.3 |  |  |  |  |  | temp_submission/khong_nop | runs/submissions/e1_m3_q3e_w122_lz_n70.zip |
| `e1v2_m3_w12_n28.zip` | đã nộp | RRF E1v2 + BGE-M3, trọng số **1:2** | 74.4 |  |  | 0.1997 | 0.3544 / 0.2029 | 0.3511 / 0.2101 | temp_submission | runs/submissions/e1v2_m3_w12_n28.zip |
| `e3v2_VI_n28.zip` | đã nộp | BM25 doc-level + pyvi | 75.8 | 28.0 |  | 0.1089 | 0.1792 / 0.1119 | 0.1784 / 0.1172 | temp_submission | kaggle/outputs/vibiomir-build-v2/e3v2_VI_n28.zip |
| `rerank_r1_n28.zip` | không nộp | nguồn gốc không rõ (phiên khác) — dùng rerank_r1_VI_n28 | 26.9 |  |  |  |  |  | temp_submission/khong_nop | runs/submissions/rerank_r1_n28.zip |
| `rerank_r1_n45.zip` | không nộp | nguồn gốc không rõ (phiên khác) | 40.3 |  |  |  |  |  | temp_submission/khong_nop | runs/submissions/rerank_r1_n45.zip |
| `rerank_r1_n70.zip` | không nộp | nguồn gốc không rõ (phiên khác) | 64.2 |  |  |  |  |  | temp_submission/khong_nop | runs/submissions/rerank_r1_n70.zip |
| `s1_n28.zip` | không nộp | BM25 clean_v4 + section 4500/1500 — dừng hướng BM25 (22/10: các biến thể đều ~0.09–0.11) | 72.7 |  |  |  |  |  | temp_submission/khong_nop | kaggle/outputs/vibiomir-bm25-s1/submission_s1_n28.zip |
| `s2_n28.zip` | không nộp | BM25 clean_v4 + section 4500/600 — dừng hướng BM25 (22/10: các biến thể đều ~0.09–0.11) | 74.8 |  |  |  |  |  | temp_submission/khong_nop | kaggle/outputs/vibiomir-bm25-s2/submission_s2_n28.zip |
| `w12VI_n100.zip` | không nộp | build cũ 109.3 MB > trần upload | 109.3 | 99.98 |  |  |  |  | temp_submission/khong_nop | kaggle/outputs/vibiomir-build-w12vi/w12VI_n100.zip |
| `w12VI_n120.zip` | không nộp | build cũ 129.0 MB > trần upload | 129.0 | 119.97 |  |  |  |  | temp_submission/khong_nop | kaggle/outputs/vibiomir-build-w12vi/w12VI_n120.zip |
| `w12_VI_d150c120.zip` | không nộp | 129 MB > trần upload | 129.2 | 149.94 | 119.97 |  |  |  | temp_submission/khong_nop | kaggle/outputs/vibiomir-build-depth-b/w12_VI_d150c120.zip |
| `w12_VI_d150c120r.zip` | không nộp | 116.7 MB > trần (xếp lại query giảm 10%) | 116.7 | 149.94 | 119.97 |  |  |  | temp_submission/khong_nop | kaggle/outputs/vibiomir-build-depth-b/w12_VI_d150c120r.zip |
