# REGISTRY — danh sách thí nghiệm

| ID | Ngày | Câu hỏi | Loại | Kết quả chính | Trạng thái |
|---|---|---|---|---|---|
| 2026-10-19_dense_m3_rerank_kaggle | 19/10 | BGE-M3 + rerank trên corpus 758k (nhiều PubMed) có đạt điểm cao không? | Kaggle + LB | FINAL 0.0648 (N=35) | xong |
| 2026-10-20_bm25_e1_e2_e3 | 20/10 | BM25 doc-level (E1), chunk-level (E2) hay pyvi/jieba (E3): cách nào tốt hơn? | Kaggle + LB | E1v2 FINAL 0.1076 > E2v2 0.1003; E3 chưa nộp LB ở 20/10 (xem 22/10) | xong |
| 2026-10-20_m3_dense_only | 20/10 | BGE-M3 đơn (không BM25) có tốt hơn fusion không? | Kaggle + LB | FINAL 0.1906; docs P 0.3449 so với BM25 E1v2 P 0.1781 | xong |
| 2026-10-20_rrf_weights_1to1_vs_1to2 | 20/10 | RRF E1v2 + BGE-M3: trọng số 1:1 hay 1:2 tốt hơn? | Kaggle + LB | 1:1 FINAL 0.1781; 1:2 FINAL 0.1997 (tốt nhất lúc đó) | xong |
| 2026-10-20_qwen3_emb_fusion | 20/10 | Thêm Qwen3-Emb vào fusion (1:2:2) có tăng FINAL không? | Kaggle + LB | 1:2:2 FINAL 0.1948 < 1:2 (0.1997) ⇒ Qwen3-Emb làm loãng | xong |
| 2026-10-20_en_zh_probe | 20/10 | Doc EN/ZH có bổ sung recall cho query VI không (14 EN + 14 ZH mỗi query)? | Kaggle + LB | FINAL 0.0089; R 0.0105 (~0.5 doc gold/query) ⇒ bỏ luồng EN/ZH | bỏ |
| 2026-10-20_upload_limit_lzma | 20/10 | Giới hạn upload là bao nhiêu? Nén LZMA có cứu được N=70 không? | LB | N=70 (159 MB) bị 413; LZMA 256 MiB được server chấm (b3_VI_n28, 22/10); trần ~102.6 MB (đội khác) | xong |
| 2026-10-20_recovery_cc_wayback_pubmed | 20/10 | Lấy lại được bao nhiêu doc thiếu từ Wayback, Common Crawl, PubMed? | Kaggle | Wayback LC +60,599; ask.39.net +31,789; PubMed +4; corpus sạch v2 1,183,751 (97.0%); CC cc-recover CANCEL | xong |
| 2026-10-20_corpus_processed_v3 | 20/10 | Xử lý v3 (chuẩn hóa, gỡ template theo host, khóa trùng) cho corpus sạch đến mức nào? | offline | 1,183,751 doc (vi 728,415 / zh 215,637 / en 239,699); gỡ 2.04% ký tự | xong |
| 2026-10-20_boilerplate_audit | 20/10 | Boilerplate theo host chiếm bao nhiêu corpus? Host nào bẩn nhất? | offline | 2.8% ký tự toàn corpus; suckhoeviet.org.vn 32%, 120ask 31%, familydoctor 17% | xong |
| 2026-10-21_rerank_r1 | 21/10 | Cross-encoder bge-reranker-v2-m3 trên E1v2 ∪ M3 (178 ứng viên/query) có nâng precision không? | Kaggle + LB | Chưa có điểm LB; top-28 rerank ∩ w12 = 14.2 doc/query; 94% VI; zip n28 25.69 MB | chờ LB |
| 2026-10-22_bm25_e3v2_pyvi_vi | 22/10 | BM25 doc-level + pyvi, VI-only (e3v2) có hơn E1v2 (0.1076) không? | Kaggle + LB | FINAL 0.1089 (E1v2 0.1076); docs P/R 0.1792 / 0.1119 | xong |
| 2026-10-22_bm25_b_c_s_variants | 22/10 | Chunk size, k1/b, tách section và VI-only có cải thiện BM25 không? | Kaggle + LB | Chỉ b3 có LB: 0.0874 (k1 0.9 b 0.4, LZMA); biến thể còn lại ~0.09–0.11 ⇒ dừng tinh chỉnh BM25 | bỏ |
| 2026-10-22_dense_m3_chunk_1000_vs_4500 | 22/10 | BGE-M3 chunk 1000/200 không tiêu đề (m3c) hay chunk 4500 có tiêu đề (m3L) tốt hơn? | Kaggle + LB | Chưa có LB; zip n28 m3c_VI 76.2 MB, m3L_VI 96.6 MB (trần ~102.6 MB) | chờ LB |
| 2026-10-22_overlap_top28 | 22/10 | Các hệ truy hồi có trùng top-28 nhiều không (để fuse có ích)? | offline | BM25↔dense 4.0–5.8 doc/query; dense↔dense 12.0–15.0; BM25↔BM25 7.5–20.7 | xong |
| 2026-10-22_icd_chapter_by_system | 22/10 | Chương ICD-10 của query và doc trùng nhau theo từng hệ truy hồi thế nào? | offline | BM25 e1v2 79.7% (3.80 doc lệch/q); M3 88.8%; reranker r1 90.0% (1.93) | xong |
| 2026-10-22_section_template_share | 22/10 | Bao nhiêu doc có khung mục chuẩn (nguyên nhân/triệu chứng/điều trị) để chunk theo mục? | offline | 3.9% doc vi (mẫu 18,255); Medlatec 30%, tiemchunglongchau 36%, báo chí ~0% | xong |
| 2026-10-22_body_span_chunk_text | 22/10 | chunk_text = đoạn thân bài liên tục có giữ đủ nội dung và còn chỗ nhét thêm chunk không? | offline | Giữ 98.8% ký tự toàn văn (clean_v4 rời rạc 98.2%) ⇒ không đủ chỗ nhét thêm chunk | xong |
| 2026-10-22_recovery_round2 | 22/10 | Vét doc còn thiếu (VI 12.183, ZH 24.363) qua Common Crawl, PubMed, tải trực tiếp, Wayback | Kaggle | CC CDX 1.096 · PubMed 178 · direct 451 → `data/raw/recovered/`; CC columnar ×5 + Wayback VI/ZH đang chạy | đang chạy |
| [2026-10-22_frame_probe](2026-10-22_frame_probe/README.md) | 22/10 | Đúng bệnh / đúng ý có đơn điệu theo P và hạng? | offline | ICD .553→.615→.640; ý .682→.717→.727; rác URL ~1,3% | xong |
| [2026-10-22_depth_probe](2026-10-22_depth_probe/README.md) | 22/10 | P(N) khi đi sâu; cắt ngưỡng reranker; reranker vào thứ tự? | Kaggle + LB | w12T2 (88 doc/75 chunk) **0.3108** (#6536); rdT2 0.3054 (#6537); 7 bài chờ | đang chạy |
| [2026-10-22_pool_ceiling](2026-10-22_pool_ceiling/README.md) | 22/10 | Tập ứng viên chứa bao nhiêu % gold (R@300/500)? BM25 top-2000 thêm gì? | Kaggle + LB | kernel build-pool + bm25-e1k2000 đã đẩy | đang chạy |

Nguồn số liệu: EXPERIMENTS.md (bảng A/B/C/E). Thí nghiệm từ 22/10 trở đi có thư mục riêng.
