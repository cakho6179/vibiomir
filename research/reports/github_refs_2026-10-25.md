# Repo GitHub liên quan (25/10/2026)

Repo được clone nông vào `research/refs/`, chỉ đọc, không chạy code. Bốn agent Haiku đọc song song; dưới đây là bản tổng hợp.

| Repo | Nội dung | Điểm LB | Đáng học |
|---|---|---|---|
| `Kyoyooo/ViBioMIR_RAG` | Chỉ EDA corpus (855.745 id, 28.480 id có nhiều nội dung, trùng SHA-256 sau NFKC) | — | Không |
| `minhtb51107/vibiomir_retrieval` | BM25 tự viết (k1 1,2, b 0,75) + BGE-M3 FlatIP, RRF, bge-reranker-v2-m3 top-100 ⇒ 10 doc / 20 chunk; chunk 512 tok/64 overlap; dịch opus-mt | FINAL ~0,05 | Không (cùng stack, nông hơn) |
| `tmanhococ/R2AI_MedicalRetrieval` | Chunk ≤ 320 tok + 64; bm25s/lucene; multilingual-e5-large; nhãn dev bằng pooling nhiều retriever | 0,0 (crawl 465 doc) | Chỉ ý tưởng nhãn dev bằng pooling |
| `TranVu2005/r2ai-stage3` (mạnh nhất) | bge-m3 hybrid (dense 0,7, max), chunk 256 tok, 3,9M chunk / 624k doc; K=100, rerank tầng 100/50 | 0,2218 (A1-115) | **Có**: (1) cắt doc theo logit reranker với số doc TB mục tiêu; (2) **mở rộng cụm gần trùng**; (3) **BM25 trên URL slug** cho doc không crawl được, nối vào relevant_docs |
| `hlong1006/Stage-3-guru-R2AI` | Rỗng (chỉ .git) | — | — |
| `aigurutinix/r2ai-stage-2` (tài khoản BTC) | Bài Stage 2 Text-to-Pandas tài chính (ViFinQA), 9 đội | — | Không liên quan ViBioMIR; không có schema MedKB hay script chấm |
| `AIVIETNAM-AIO-TuNguyen3107/onto-med-rag` | Pipeline NER lâm sàng tiếng Việt | — | Xem báo cáo NER |

Tìm thêm trên GitHub (API search, 25/10): không có repo ViBioMIR mới ngoài danh sách trên. Trang HF `AIGuruTinix/ViBioMIR` không link code.

**Không repo nào có:** NER query, viết lại/tách ý query bằng LLM, schema MedKB, script chấm của BTC.

**Áp dụng cho ta (phần dữ liệu):**
1. Cụm gần trùng (MinHash) ⇒ mở rộng id khi nộp. Hiện ta có 4.802 bản y hệt.
2. Doc VI thiếu text (9,9k link không crawl được): index slug URL làm pseudo-doc (bỏ dấu cả hai phía) để vẫn đưa được vào relevant_docs.
3. Cắt theo điểm với số doc trung bình mục tiêu: ta đã có (T2).
