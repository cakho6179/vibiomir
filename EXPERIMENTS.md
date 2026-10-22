# EXPERIMENTS — R2AI2026 ViBioMIR (sổ thí nghiệm, chỉ ghi số ĐÃ ĐO)

Quy ước: mỗi dòng là một thí nghiệm có kết quả thật (leaderboard hoặc thống kê kernel). Giả thuyết/kế hoạch ghi ở `WORKLOG.md`.
Thêm dòng mới ở cuối mỗi bảng; không sửa số cũ.

## A. Kết quả leaderboard (public, nộp tay bởi user)

Metric: FINAL = mean(DOCS_F2, CHUNKS_F2). Chunk = toàn văn doc (không cắt) ⇒ chunk F2 ≈ doc F2.

| Ngày | File (vibiomir/…) | Hệ thống | N | FINAL | Docs P / R | Chunks P / R |
|---|---|---|---|---|---|---|
| 19/10 | runs/submissions/kaggle_gpu_v1 | BGE-M3 rerank, corpus 758k, nhiều PubMed | 35 | 0.0648 | — | — |
| 20/10 | kaggle/outputs/vibiomir-bm25-e2v2/submission_e2_n28.zip | BM25 chunk 1200 ký tự, doc = max chunk, corpus v2 | 28 | 0.1003 | 0.1644 / 0.1026 | 0.1638 / 0.1069 |
| 20/10 | kaggle/outputs/vibiomir-bm25-e1v2/submission_e1_n28.zip | BM25 doc-level, âm tiết, giữ dấu, corpus v2 | 28 | 0.1076 | 0.1781 / 0.1094 | 0.1775 / 0.1144 |
| 20/10 | runs/submissions/e1v2_m3_rrf_n28.zip | RRF(k=60) E1v2 + BGE-M3, trọng số 1:1 | 28 | 0.1781 | 0.3077 / 0.1805 | 0.3058 / 0.1876 |
| 20/10 | runs/submissions/m3only_n28.zip | BGE-M3 đơn (đoạn 1500 ký tự + tiêu đề, doc = max) | 28 | 0.1906 | 0.3449 / 0.1929 | 0.3406 / 0.1996 |
| 20/10 | runs/submissions/e1v2_m3_w12_n28.zip | RRF E1v2 + BGE-M3, trọng số **1:2** | 28 | **0.1997** | 0.3544 / 0.2029 | 0.3511 / 0.2101 |
| 20/10 | runs/submissions/e1v2_m3_w12_n70.zip | như trên | 70 | **lỗi 413** | — | — |
| 20/10 | runs/submissions/e1_m3_q3e_w122_n28.zip | RRF E1v2 + M3 + Qwen3-Emb (8 shard), 1:2:2 | 28 | 0.1948 | 0.3454 / 0.1984 | 0.3419 / 0.2034 |
| 20/10 | runs/submissions/probe_xl_enzh_n28.zip | Probe: 14 doc EN (BM25 query EN) + 14 doc ZH (BM25 query ZH) | 28 | 0.0089 | 0.0170 / 0.0105 | 0.0167 / 0.0084 |
| 22/10 | submissions/e3v2_VI_n28.zip | BM25 doc-level + tách từ pyvi, chỉ VI | 28 | 0.1089 | 0.1792 / 0.1119 | 0.1784 / 0.1172 |
| 22/10 | submissions/b3_VI_n28.zip | BM25 doc-level k1 0.9 b 0.4 (kiểu Pyserini), VI, **LZMA được chấm** | 28 | 0.0874 | 0.1415 / 0.0908 | 0.1411 / 0.0953 |
| 22/10 | submissions/w12T2_VI_d150c90r.zip | **#6536, nộp 16:53.** w12 (E1v2+M3 1:2) giữ thứ tự, cắt ngưỡng reranker ≥ −2 (tối thiểu 20 doc), TB 88 doc / 75 chunk, xếp lại query, LZMA 71.3 MB — hạng 6/50 | 88/75 | **0.3108** | 0.2495 / 0.3823 | 0.2635 / 0.3698 |
| 22/10 | submissions/rdT2_VI_d150c90r.zip | **#6537, nộp 16:54.** RRF(E1v2, M3, reranker) 1:2:3 (reranker vào thứ tự), cùng ngưỡng ≥ −2, TB 103 doc / 78 chunk, LZMA 71.5 MB (user báo điểm, chưa có P/R) | 103/78 | 0.3054 | — | — |

Tham chiếu: admin BM25 FINAL 0.3764 (docs F2 0.5587, P/R 0.880/0.515, chunks F2 0.1941); Top 1 `toptire` 0.4071 (docs P/R 0.232/0.571).

### Kết luận rút ra
1. Gold ≈ 46 doc/query (P·N/R, nhất quán giữa các bản).
2. Dense >> BM25: M3 đơn P 0.345 so với BM25 0.178; BM25 chỉ thêm chút khi trọng số thấp (1:2 > M3 đơn > 1:1).
3. BM25 doc-level > chunk-level (E1 > E2).
4. **Giới hạn upload**: zip ~60–80 MB qua; 159 MB bị 413 (Request Entity Too Large) ⇒ giới hạn có lẽ ~100 MB.
   Nén LZMA từ điển 256 MB (ZIP method 14) giảm N=70 từ 164 MB → 64 MB (83,976 slot nhưng chỉ 60,908 doc khác nhau) — chưa nộp thử.
5b. (22/10) Biến thể BM25 đều ~0.09–0.11: pyvi ≈ âm tiết (0.1089 vs 0.1076), Pyserini k1 0.9 b 0.4 kém hơn (0.0874) ⇒ **dừng tinh chỉnh BM25**; ZIP LZMA 256 MiB của ta được server chấm.
5. **Gold ~99% tiếng Việt**: probe EN/ZH chỉ có R 0.0105 (~0.5 doc gold/query) ⇒ bỏ luồng EN/ZH.
6. **Qwen3-Emb làm loãng** (1:2:2 = 0.1948 < 1:2 = 0.1997) ⇒ BGE-M3 khớp hệ sinh gold nhất.

## E. Xử lý dữ liệu (đo trên docs_clean v2)

| Phép đo | Giá trị |
|---|---|
| Boilerplate template theo host (dòng lặp ≥5% trang cùng host), tính theo ký tự | **2.8%** toàn corpus (`runs/reports/boilerplate_audit.tsv`) |
| Host bẩn nhất | suckhoeviet.org.vn 32%, 120ask 31%, familydoctor 17%, baby.39.net 13%, Long Châu 6.3%, vinmec 4.9%, medlatec 4.7% |
| Doc trong crawl cũ chưa vào docs_clean | crawled_output ~1,150; recrawl 47 (bị bộ lọc chất lượng loại) |
| **Corpus processed v3** (`data/processed_v3/docs-00..07.parquet`, 2.2 GB) | 1,183,751 doc: vi 728,415 / zh 215,637 / en 239,699; template học cho 50 host (ngưỡng ≥20% trang) |
| Ký tự bị gỡ (template + dòng lặp + tiêu đề lặp) | 64.6M / 3.18G = **2.04%** |
| Gỡ nhiều nhất | Long Châu 15.7M ký tự (3.5%), 120ask 12.8M (31.6%), suckhoeviet 11.9M (32.1%), medlatec 6.4M (4.2%), vinmec 6.4M (4.7%) |
| Trùng nội dung chính xác (`dup_of` ≠ id) | 4,773 doc (vietnamnet 1,700, hellobacsi 426, phunusuckhoe 323, suckhoedoisong 304) |
| Rỗng sau xử lý (chỉ có tiêu đề) | 5 doc bachmai.gov.vn |

## B. Kernel Kaggle (thống kê đo)

| Kernel (account) | Nội dung | Kết quả |
|---|---|---|
| vibiomir-bm25-e1 / e2 / e3 (acc2) | BM25 corpus v1 1,052,703 doc | 860 s / 1271 s / 5271 s; vocab 782k |
| vibiomir-bm25-e1v2 / e2v2 / e3v2 (acc2) | BM25 corpus v2 1,183,760 doc | 980 s / 1254 s / 7063 s; vocab 946k (e3 1.34M) |
| vibiomir-dense-m3-s01,s23,s45,s67 (acc*) + m3-d1 (acc4) | BGE-M3 fp16 2×T4, max_len 512 | ~5.6–6.8 ks / 2 shard (~630k đoạn); delta 448,793 đoạn 4.4 ks |
| vibiomir-dense-q3e-s01,s23,s45,s67 | Qwen3-Embedding-0.6B + instruct prompt | 14.4–16.4 ks / 2 shard; q3e-d1 (delta) đang chạy |
| vibiomir-translate (acc4) | Qwen2.5-7B-Instruct dịch 1200 query VI→EN/ZH | `kaggle/outputs/vibiomir-translate/query_{en,zh}.parquet`, chất lượng tốt |
| vibiomir-bm25-e1en / e1zh (acc2) | BM25 E1 với query EN / ZH | top-28: EN 94.9% doc EN; ZH 100% doc ZH |
| vibiomir-rerank-r1p0,p1 (acc2), p2 (acc4) | bge-reranker-v2-m3 trên E1v2 top100 ∪ M3 top100 (178 doc/query) | Xong 3/3 shard (213,567 cặp/1,200 queries), đã tải về |
| vibiomir-recrawl-lc-final (acc1) | Thu thập nốt 3,317 URL Long Châu còn lại | Xong, đã tải và merge (+11 doc thật sạch, phần còn lại bị chặn 403/429) |
| vibiomir-cc-recover0..3 (main) | Common Crawl khôi phục 36,847 URL thiếu (lưu riêng, không gộp) | v1 lỗi header (đã sửa), v2 đang chạy |
| vibiomir-bm25-b1 / b2 (acc2), b3 / b4 / b5 (acc4) | BM25 VI-only: b1 chunk 1000/200; b2 = b1 + pyvi; b3 doc k1 0.9 b 0.4; b4 chunk 512/100; b5 chunk 4500 | 1395 / 8618 / 676 / 1883 / 876 s; units 4.04M / 4.04M / 728k / 8.16M / 987k; zip n28 deflate 68 / 68 / 124 / 69 / 76 MB |
| vibiomir-dense-m3c-s01..s67, d1 | BGE-M3 chunk 1000/200 không tiêu đề, VI-only | xong 5/5 |
| vibiomir-dense-m3L-a/b/c | BGE-M3 chunk 4500, max_len 1280, có tiêu đề, VI-only | xong 3/3 (8.1 ks cho part 0,1,d1) |
| vibiomir-bm25-c1 (acc2) / s1 (acc4) / s2 (acc2) | BM25 VI-only + clean_v4: c1 doc-level; s1 section 4500/1500; s2 section 4500/600 | 1632 / 1874 / 2084 s; units 728k / 1.37M / 1.95M; zip n28 deflate 77.0 / 72.7 / 74.8 MB |
| vibiomir-build-depth-a / -b (acc2) | w12 VI (RRF E1v2+M3 1:2), LZMA 256 MiB, [docs, chunks] | d100c28 36.6, d150c28 36.9, d200c28 37.1, d150c60 68.7, **d150c90 99.7**, d150c120 129.2, d150c120 xếp lại query 116.7 MB; 2388 s / 4258 s |
| vibiomir-build-v2 (acc2) | builder Kaggle: m3c_VI, m3L_VI, e3v2_VI (deflate), b3_VI (LZMA) n28 | 22 phút; 76.2 / **96.6** / 75.8 / **32.4** MB (LZMA b3: 124 → 32 MB) |

## C. Phân tích offline (không tốn lượt nộp)

| Phép đo | Giá trị |
|---|---|
| Overlap@28 E1–E2 / E1–E3 | 0.447 / 0.582 |
| Overlap@28 BM25–dense | ~0.13–0.14 (≈5.2 doc/query) |
| Overlap@28 M3–Q3E (đủ 8 shard) | 8.6 doc/query |
| Thành phần top-28 fusion 1:1 | 5.2 cả hai, 9.5 chỉ BM25, 9.6 chỉ M3, 3.6 từ hạng 28–200 |
| Ngôn ngữ theo id (`runs/reports/lang_bounds.txt`) | VI ≤ 766,441 < ZH ≤ 4,392,214 < EN |
| Ngôn ngữ top-28 M3 | VI 89.7%, ZH 7.4%, EN 2.9% |
| Corpus clean v2 | 1,183,751 / 1,220,598 (97.0%): EN 99.9%, VI ~98%, ZH ~90% |
| Overlap@28 (doc/query, 22/10) | BM25↔BM25 7.5–20.7 (e1v2–b5 20.7, e1v2–b3 17.4, b1–b4 16.6); dense↔dense 12.0–15.0; **BM25↔dense 4.0–5.8** |
| Tỷ lệ VI top-28 | mọi biến thể VI-only 100%; e1v2 99.6%; m3 89.7% |
| Host share top-28 (`runs/reports/host_share_top28.tsv`) | m3: Long Châu 22.7%, SKĐS 13.4%, Medlatec 11.2%, Vinmec 8.4%, tiemchunglongchau 6.9%, Thanh Niên 6.8%; suckhoecongdongonline 21.2% corpus nhưng 3.2% top-28 — gần phân bố nguồn trong probe MedKB |
| Chương ICD-10 trùng query–doc top-28 (22/10, `runs/reports/icd_systems.md`) | BM25 e1v2 79.7% (3.80 doc lệch/q), b3 72.8% (4.92), M3 88.8% (2.15), q3e 89.1%, m3L 89.4%, reranker r1 **90.0% (1.93)**; query có chương 68.6% |
| Khung mục chuẩn (≥3 mục nguyên nhân/triệu chứng/điều trị…) | 3.9% doc vi (mẫu 18,255); Medlatec 30%, tiemchunglongchau 36%, Vinmec 18%, HelloBacsi 17%, báo chí ~0% |
| Gold theo scorer (`details.jsonl`, file của đội khác) | public 600 query (593 có gold); |G| TB 70.97 doc, 74.7 chunk |
| chunk_text = đoạn thân bài liên tục (bỏ rác đầu/cuối theo clean_v4) | chỉ còn 98.8% ký tự so với toàn văn (clean_v4 rời rạc 98.2%), trọng số theo M3 top-90 ⇒ không đủ để nhét thêm chunk |
| Query chép nguyên văn từ corpus? | 3/40 query có ≥ 60% chuỗi liên tiếp trong top-10 (BM25+M3) ⇒ không |
| URL rác (xổ số/bóng đá/thời tiết/tử vi/giá…) (22/10) | 3.2% corpus VI; 1.2–1.4% trong top-28/150 của BM25, w12, rd; tỉ lệ host top-28 gần như giống hệt giữa BM25 (P .178) và w12 (P .354) ⇒ loại tài liệu không phải đòn bẩy |
| Khớp **mã ICD 3 ký tự** query–doc (tiêu đề+heading+600 ký tự đầu; chỉ 272/1200 query có mã) | top-28: BM25 .553 < w12 .615 < rd .640; theo hạng w12: 1–28 .615 > 29–90 .540 > 91–150 .477 ⇒ tín hiệu đơn điệu, nhưng độ phủ query thấp (cần LLM trích thực thể) |
| Khớp **ý hỏi ↔ heading** (8 ý, luật từ khoá; 1005/1200 query có ý) | top-28: BM25 .682 < w12 .717 < rd .727; theo hạng: .717 > .663 > .634 ⇒ đơn điệu, biên độ nhỏ (luật thô) |

## D. Bản đã dựng — từ 22/10 xem `submissions/MANIFEST.md` (bảng dưới là lịch sử, đường dẫn cũ)

| File (Desktop) | File gốc (runs/submissions/) | Hệ thống | MB | Dự kiến |
|---|---|---|---|---|
| `submission_rerank_r1_n70_sota.zip` | `rerank_r1_n70.zip` | BGE-Reranker-v2-m3 trên E1v2 ∪ M3 (LZMA, N=70) | 61.2 | **Ứng viên Top 1 mạnh nhất (kỳ vọng 0.40–0.50+)** |
| `submission_rerank_r1_n45_sota.zip` | `rerank_r1_n45.zip` | BGE-Reranker-v2-m3 trên E1v2 ∪ M3 (LZMA, N=45) | 38.4 | Thăm dò điểm cân bằng P–R của reranker |
| `submission_rerank_r1_n28_sota.zip` | `rerank_r1_n28.zip` | BGE-Reranker-v2-m3 trên E1v2 ∪ M3 (LZMA, N=28) | 25.7 | Đối chứng trực tiếp với bản 0.1997 để đo ΔP của reranker |
| `submission_fusion_e1v2_m3_n70.zip` | `e1v2_m3_w12_lz_n70.zip` | Fused E1v2 + BGE-M3 (1:2) (LZMA, N=70) | 64.5 | Đẩy recall từ bản 0.1997 hiện tại |
| — | `probe_xl_enzh_n28.zip` | Probe: 14 doc EN + 14 doc ZH / query | 45 | Đã đo xong (0.0089) |
| — | `q3eonly_n28.zip` | Qwen3-Emb đơn | 58 | Đã đo (0.1948 khi fusion) |
| — | `e1_m3_q3e_w122_lz_n70.zip` | RRF E1v2 + M3 + Q3E, 1:2:2 (LZMA, N=70) | 61.3 | Dự phòng |

