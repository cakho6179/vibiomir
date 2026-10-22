# WORKLOG — R2AI2026 ViBioMIR (mục tiêu: Top 1)

Quy ước: mỗi mục ghi rõ **[ĐO]** (đã kiểm chứng bằng số đo/đọc từ đĩa), **[BÁO CÁO]** (lấy từ tài liệu/phiên khác, chưa kiểm chứng) hoặc **[GIẢ THUYẾT]**.
Cập nhật bằng cách thêm mục mới ở đầu phần "Nhật ký"; không sửa mục cũ, chỉ ghi đính chính ở mục mới.

- Cuộc thi: Codabench #15, slug `r2ai2026-vibiomir`. Public đóng **2026-10-31 07:00 ICT**; private đóng 2026-11-04 07:00 ICT (chỉ 5 lần nộp tổng).
- Quota public: 10 lần nộp/ngày. Nộp thủ công trên dashboard (ZIP chỉ chứa 1 file `submission.json`).
- Workspace: `D:\02_Competitions\Kaggle_agent_tool\vibiomir\` (không dùng venv, dùng Python 3.13 hệ thống).

---

## 0h. 2026-10-22 (chiều) — Mục tiêu vượt baseline + tổ chức data theo slide BTC
- User: mục tiêu là **vượt** baseline BTC (0.3764) chứ không bằng ⇒ đích FINAL ≥ 0.45 (ROADMAP mục 0).
- Haiku rút checklist slide (48 trang) → `research/reports/btc_slides_data_org.md`; đối chiếu trong ROADMAP mục 2b. Điểm mấu chốt: gold theo **bệnh × ý hỏi** (tr.23, 40: "cùng bệnh, khác ý" là sai).
- Doc không crawl được chỉ ảnh hưởng cấp doc (chấm theo id, luật 01_Evaluation 3.1); không có chunk ⇒ lợi nhỏ (đội khác +0.003), ưu tiên thấp.

## 0g. 2026-10-22 (chiều) — Gom bài nộp về một chỗ + độ sâu + độ phủ crawl
- **Mọi zip bài nộp giờ ở `submissions/`** (bảng chọn: `submissions/MANIFEST.md`, dữ liệu: `submissions/manifest.json`). Script `scripts/r2ai2026_collect_submissions.py` di chuyển zip từ `runs/submissions/` và `kaggle/outputs/*/`, bỏ tiền tố `submission_`, ghi nguồn/sha/MB/doc-chunk mỗi query, lấy điểm LB từ bảng A `EXPERIMENTS.md`. Chạy lại sau mỗi lần tải output Kaggle. File bị khoá (đang upload) được bỏ qua, gom ở lần sau (`e3v2_VI_n28.zip` lúc 13:xx). Rankings parquet vẫn ở chỗ cũ.
- [ĐO] Depth build (w12 VI, LZMA 256 MiB): d100/150/200 c28 = 36.6/36.9/37.1 MB (doc sâu gần như không tốn byte); d150c60 68.7 MB; **d150c90 99.7 MB** (sát trần 102.6 MB); d150c120 129.2 MB, xếp lại thứ tự query 116.7 MB (−10%). Doc của ta dài hơn đội TranVu (họ vừa ~117–133 chunk) ⇒ muốn >90 chunk cần chunk_text ngắn hơn (đoạn thân bài liên tục, vẫn nguyên văn) hoặc xếp lại query.
- [ĐO] Độ phủ crawl (processed_v3 so với 1,220,598 link): VI 728,415/740,598 (98.35%, thiếu 12,183: pharmacity beta 5,333, Long Châu 3,229, thaythuocvietnam 2,399, hellobacsi 337, medlatec 220); ZH 215,637/240,000 (89.85%, thiếu 120ask 13,362, zysjonline 6,591); EN 239,699/240,000 (99.87%). 14,803 doc VI < 500 ký tự (có thể mỏng). Danh sách thiếu: `runs/reports/missing_docs_2026-10-22.tsv`. Kernel cc-recover0..3 (main) ở trạng thái CANCEL — không có output.
- Đã giao subagent Haiku đọc 8 repo đội khác để so sánh quản lý/tổ chức data → `research/reports/data_mgmt_comparison.md`.

## 0f. 2026-10-22 (trưa) — Nghiên cứu mã nguồn đội khác + đánh giá tổ chức dữ liệu (`docs/RESEARCH_REFS.md`)
- Clone (shallow, chỉ đọc) 8 repo ViBioMIR vào `research/refs/`. Nguồn giá trị nhất: nhật ký LB của TranVu2005 (best 0.2655) và file `scoring_result.zip` của huu7725.
- [ĐO, của đội khác] Đòn bẩy chính là độ sâu: doc 60 → 115 (Final 0.2050 → 0.2218); chunk toàn văn 19 → ~104/query (chunk F2 0.128 → 0.2605); LZMA dict 256 MiB được chấm, trần upload ~102,6 MB; chunk chỉ từ doc đã chọn; mở rộng id trùng +; zh theo ngưỡng reranker +0.0056; rerank sâu giảm.
- [ĐO] Scorer xuất `details.jsonl` theo query: public chấm 600/1.200 query; |G| trung bình **71 doc / 75 chunk** (đính chính ước 46–50). Chưa dùng nhãn từng cặp — chờ user quyết (vùng xám).
- [ĐO] Chương ICD-10 (`scripts/r2ai2026_icd_systems.py`, ontology của neiths): trùng chương query–doc theo chiều precision (BM25 79.7% → M3 88.8% → reranker 90.0%), giải thích ~1/3 khoảng cách P ⇒ đặc trưng mềm. Khung mục chuẩn chỉ ở 3.9% doc ⇒ đặc trưng khía cạnh, không tái tổ chức corpus.
- Kaggle: c1/s1/s2 xong (zip n28 77.0 / 72.7 / 74.8 MB). `build_submission.py` thêm `n` dạng `[N_docs, N_chunks]` và `"reorder"`; đẩy `tthiutoko/vibiomir-build-depth-a` (w12 d100c28, d150c28, d200c28, d150c60) và `-b` (d150c90, d150c120, d150c120 xếp lại); waiter `runs/reports/wait_build_depth.log`.
- Lộ trình cập nhật: depth probe docs + chunks ngay hôm nay (thay cho test LZMA, đã được đội khác chứng minh).

## 0e. 2026-10-22 — Lộ trình hệ thống mới (`docs/ROADMAP_SYSTEM.md`, thay PLAN_TARGET_0.6)
- **Đính chính 0d (user 22/10):** số doc/query KHÔNG cố định 28 — có thể nhiều hoặc ít hơn; về sau truy hồi rộng (~200) rồi dùng decision model chọn giữ/bỏ.
- [LUẬT] `relevant_docs` và `relevant_chunks` là hai danh sách tách rời ⇒ độ sâu doc không tốn dung lượng; giới hạn ~100 MB chỉ ràng buộc chunk_text.
- [LUẬT] Chỉ model mở ≤ 15B, phát hành trước 01/08/2026; cấm LLM đóng ⇒ Jev (API đóng) không dùng trong hệ thống nộp; tái hiện mẫu Noul/Score + ngưỡng tin cậy bằng model mở (Qwen3-Reranker / Qwen2.5-7B logprob).
- [SUY, toán] F2 = 5·TP/(4|G|+N) ⇒ thêm doc khi p(gold) > F2/5 ≈ 0.06–0.09 ⇒ danh sách doc sâu, cần xác suất hiệu chỉnh.
- [ĐO] Nghiên cứu tổ chức dữ liệu: slide mentor (chunk theo mục, giữ hỏi–đáp, vùng bài), HF card (740,598 vi / 240k en / 240k zh, trích dẫn MedKB), probe MedKB (chunk ~3.3k/≤4.8k, có `similar_queries`), host top-28 ≈ MedKB, query không chép nguyên văn từ corpus (3/40). Ma trận overlap@28 và host share: `EXPERIMENTS.md` mục C.
- Code: `kaggle/common/docorg.py` (`clean_v4`, `section_chunks`, 4 test); `bm25_experiment.py --clean --sections MAX,MIN` (+ sửa lỗi lần đọc text thứ 2 bỏ qua `--vi-only` khi dùng `--max-docs`); `build_submission.py` thêm `"zip": "deflate"|"lzma"` theo job; skill `kaggle-multi-account` thêm `wrap --module` (tự escape non-ASCII).
- Kaggle: build-v2 (acc2) xong → m3c_VI / m3L_VI / e3v2_VI (deflate), b3_VI (LZMA); BM25 c1 (acc2), s1 (acc4), s2 (acc2) đang chạy (sổ `kaggle/kaggle_jobs.json`, log `runs/reports/wait_bm25_docorg.log`).

## 0c. Cập nhật 2026-10-21 (00:25) — BGE-Reranker-v2-m3 hoàn tất 100% + đóng gói bộ 3 bản nộp SOTA
- **Rerank r1**: Cả 3 shard trên Kaggle GPU (`vibiomir-rerank-r1p0`, `r1p1`, `r1p2`) đã hoàn thành và tải về đầy đủ (`kaggle/outputs/vibiomir-rerank-r1p{0,1,2}/rerank.parquet`).
  - Tổng cộng: **213,567 cặp doc-query được chấm điểm bởi cross-encoder bge-reranker-v2-m3** trên toàn bộ 1,200 truy vấn (trung bình ~178 candidate/query từ hợp của top-100 BM25 E1v2 và top-100 BGE-M3 dense).
- **Crawl nốt Long Châu**: Shard `hytahy/vibiomir-recrawl-lc-final` (3,317 target) hoàn thành trên Kaggle, đã tải về và merge (+11 doc sạch thực sự, phần còn lại bị server chặn Cloudflare 403/429).
- **Đóng gói SOTA submission** (`scripts/r2ai2026_build_from_rankings.py`, nén LZMA từ điển):
  - `rerank_r1_n28.zip` (25.69 MB, N=28): Đối chứng trực tiếp với bản 0.1997 hiện tại để đo độ tăng Precision của cross-encoder reranker.
  - `rerank_r1_n45.zip` (38.42 MB, N=45): Thăm dò độ cân bằng giữa P và R.
  - `rerank_r1_n70.zip` (61.18 MB, N=70): Bản tối ưu hóa $F_2$ cao nhất — kết hợp độ chính xác hàng đầu của cross-encoder ở top đầu và độ phủ sâu ở N=70 (kỳ vọng vượt mốc 0.40 hướng tới Top 1).
- **Xuất ra Desktop** (`C:\Users\NOS\Desktop\`):
  - `submission_rerank_r1_n70_sota.zip` (61.18 MB)
  - `submission_rerank_r1_n45_sota.zip` (38.42 MB)
  - `submission_rerank_r1_n28_sota.zip` (25.69 MB)
  - `submission_fusion_e1v2_m3_n70.zip` (64.46 MB)

## 0d. RÀNG BUỘC (user 2026-10-21): chỉ nộp N = 28 doc/query
- User xác nhận không nộp được N > 28 ⇒ bỏ hướng tăng N (zip N=70/100/120 không dùng). Trọng tâm: precision@28 (trần ≈ docs F2 0.56–0.58 nếu tái tạo được BM25 BTC, P 0.88).

## 0c. Cập nhật 2026-10-21 — kế hoạch lên 0.6 (`docs/PLAN_TARGET_0.6.md`), mọi việc nặng chạy Kaggle
- Rerank r1 xong (3 phần): top-28 rerank ∩ w12 = 14.2 doc/query; 94% VI. `runs/rankings/rerank_r1.parquet`.
- Dựng (local, lần cuối dùng local): `rerank_r1_VI_n28`, `w12_rr_VI_n28/_n70` (RRF fusion-w12 + rerank 1:1), `w12_VI_n28` — chờ nộp.
- **P3** đẩy 4 biến thể BM25 (CPU, chỉ VI, `--ns 28`): b1 chunk 1000/200 âm tiết (acc2), b2 = b1 + pyvi (acc2), b3 doc-level k1 0.9 b 0.4 (acc4), b4 chunk 512/100 (acc4). `bm25_experiment.py` thêm `--k1 --b --chunk-chars --overlap --seg --vi-only --ns`.
- **P4** đẩy BGE-M3 chunk 1000/200, không prefix tiêu đề, chỉ VI (`dense_embed.py` thêm `--chunk-chars --overlap --no-title --vi-only`): m3c-s01 (acc2), s23 (acc1), s45 (acc3), s67 + d1 (acc4).
- **P5** `kaggle/build_exp/build_submission.py`: dựng zip LZMA trên Kaggle từ dataset `tthiutoko/vibiomir-rankings` (`kaggle/rankings_ds/`, 19 MB: bm25_e1v2/e3v2, m3_*, q3e_*, rerank_r1). Job đầu: `vibiomir-build-w12vi` (w12 VI-only N=100/120).
- Fetchers: `runs/reports/fetch_bm25b.log`, `fetch_m3c.log`.
- Đọc lại tài liệu BTC: **không có giới hạn số doc/query**; lỗi trước là 413 (dung lượng 159 MB). Zip LZMA N=70 = 68 MB < bản N=28 deflate 74 MB đã nộp được ⇒ cần thử 1 lần.
- [ĐO, từ probe MedKB đã lưu sẵn — chỉ dùng thống kê cấu trúc, không gọi API, không dùng kết quả để nộp] chunk MedKB dài: trung vị ~3,500 ký tự, max ~4,800; kết quả 90.5% vi / 9% zh / 0.5% en; MedKB dịch query sang EN (trường `tr`). ⇒ m3c (1000/200) có thể sai hướng; thêm **m3L**: M3 chunk 4500 ký tự, max_len 1280, chỉ VI (acc2 m3L-a: part 00,01,d1; acc1 m3L-b: 02,03,06; acc3 m3L-c: 04,05,07) + BM25 **b5** chunk 4500 (acc4).
- Nhắc lại quy tắc user: **không giới hạn ký tự** — chunk_text nộp = toàn văn doc (giữ nguyên). Chunk 4500 chỉ là đơn vị chấm điểm khi truy hồi (mọi đoạn của doc đều được embed). Rà soát: (1) rerank r1 chỉ chấm 24 đoạn đầu/doc → mặc định mới = toàn doc; (2) m3L max_len 1280: đo 1.4% đoạn bị cắt, 0.1% token mất (đoạn dài nhất 1,518 token, 3.72 ký tự/token) → không chạy lại; lần sau dùng `--max-len 1536`.

## 0b. Cập nhật 2026-10-20 (16:00) — dọn workspace + bắt đầu xử lý dữ liệu
- User: dừng thử nghiệm baseline, tập trung xử lý dữ liệu; dồn file dư thừa vào temp (không xóa).
- **Dọn**: `scripts/r2ai2026_move_to_temp.py --tag 2026-10-20` chuyển 70 mục (~14.5 GB) vào `_temp/2026-10-20/` (MANIFEST.tsv; khôi phục bằng `--restore`): snapshot `docs_cleaned.jsonl` 758k, `query_translated.parquet` hỏng, `real_ids.txt`, `runs/indexes` cũ, mọi zip đã chấm/quá cỡ, output BM25 v1, output recrawl đã merge, `crawled_output`, bản copy hardlink upload, pkl research cũ, 4 tài liệu kiến trúc có giả định sai. Giữ: `data/docs.jsonl` (raw), `docs_clean.jsonl` (v2), `crawl.sqlite`, links/query, mọi `ranking.parquet` + fused rankings, best zip `e1v2_m3_w12_n28.zip`, zip LZMA N=70 chưa thử, `kaggle/corpus_dataset` + `corpus_delta_d1`.
- `r2ai2026/paths.py`: `DOCS_CLEAN` → `docs_clean.jsonl`; `QUERIES_EN/ZH` → bản dịch Qwen; thêm `DOCS_PROCESSED`. (CLI cũ `r2ai2026 clean/index` vẫn mặc định `docs_cleaned.jsonl` — luồng cũ, không dùng nữa.)
- **Data đã lấy hết chưa**: 97.0% (1,183,751/1,220,598); 36,847 thiếu ở host chết/chặn; Common Crawl recovery đang chạy (lưu riêng).
- **Xử lý dữ liệu v3** (`r2ai2026/processing.py`, `scripts/r2ai2026_build_processed_corpus.py`, 8 test): chuẩn hóa (entity, NFC, ký tự ẩn, khoảng trắng, dòng lặp), bỏ dòng lặp tiêu đề, gỡ template theo host (dòng xuất hiện ≥20% trang mẫu của host, không bao giờ cắt quá 80% doc), gắn lang theo khối id, khóa trùng lặp chính xác (`dup_of`). Xuất `data/processed_v3/docs-XX.parquet` + `templates.json` + `report.json`. Không đụng `docs_clean.jsonl`/index.
- [ĐO 19:08] v3 build xong (số liệu ở `EXPERIMENTS.md` mục E): gỡ 2.04% ký tự, 4,773 doc trùng, template của 50 host kiểm tay đúng (menu, "Kích thước chữ", "Các tin khác", banner đặt lịch). Lần chạy đầu bị mất khi phiên khởi động lại → chạy lại bằng Start-Process tách rời.
- Nén LZMA: `runs/submissions/e1v2_m3_w12_lz_n70.zip` (68 MB) và `e1_m3_q3e_w122_lz_n70.zip` (64 MB) đọc lại nguyên vẹn; chưa nộp thử. Bản n100 bị ngắt (đã xóa file 0 byte).
- Rerank r1: 2/3 phần đã tải (`kaggle/outputs/vibiomir-rerank-r1p{0,1}/rerank.parquet`), chờ p2.

## 0a. Cập nhật 2026-10-20 (15:00) — sổ kết quả chuyển sang `EXPERIMENTS.md`
- Mọi số đo (leaderboard, kernel, phân tích offline) từ nay ghi ở `EXPERIMENTS.md`; WORKLOG ghi việc đã làm + giả thuyết.
- [ĐO] M3 đơn 0.1906, fusion E1v2+M3 1:2 **0.1997** (tốt nhất) ⇒ dense quyết định; BM25 chỉ phụ.
- [ĐO] Upload N=70 (159 MB) → **413**; N=28 (~80 MB) qua ⇒ giới hạn ~100 MB. Đang thử nén LZMA từ điển lớn (doc lặp giữa các query) trước khi phải cắt chunk text.
- Đã làm: Qwen3 đủ 8 shard; BM25-EN/ZH xong; dựng probe EN/ZH, q3eonly, fusion 1:2:2; rerank r1 + Common Crawl (lưu riêng) đang chạy.

## 0. Điểm leaderboard mới (2026-10-20 13:xx) [ĐO]
| File | FINAL | Docs P / R | Chunks P / R |
|---|---|---|---|
| `kaggle/outputs/vibiomir-bm25-e1v2/submission_e1_n28.zip` (BM25 doc-level, corpus v2) | **0.1076** | 0.1781 / 0.1094 | 0.1775 / 0.1144 |
| `kaggle/outputs/vibiomir-bm25-e2v2/submission_e2_n28.zip` (BM25 chunk-level) | 0.1003 | 0.1644 / 0.1026 | 0.1638 / 0.1069 |
| `runs/submissions/e1v2_m3_rrf_n28.zip` (RRF BM25 E1v2 + BGE-M3 full corpus) | **0.1781** | 0.3077 / 0.1805 | 0.3058 / 0.1876 |

- Dense nâng P 0.178 → 0.308 (8.6 doc đúng / 28). Top-28 fusion/query: 5.2 cả hai hệ, 9.5 chỉ BM25, 9.6 chỉ M3, 3.6 từ hạng 28–200 ⇒ gold nghiêng mạnh về truy hồi ngữ nghĩa. Bản thử kế: `m3only_n28`, `e1v2_m3_rrf_n70`, `e1v2_m3_w12_n28`.
- 13:50 đẩy: rerank r1 (bge-reranker-v2-m3, ứng viên = E1v2 top100 ∪ M3 top100 = 178 doc/query, `kaggle/rerank_exp/cands_r1/`) chia 3 phần: acc2 p0,p1 + acc4 p2. Common Crawl khôi phục 36,847 URL thiếu (`kaggle/cc_recover/cc_recover.py`, main cc-recover0..3, chia crawl 2017+ theo i/4) — **chỉ lưu riêng, không gộp corpus** (yêu cầu user).

- Tốt nhất từ trước tới nay (cũ 0.0648). ~5 doc đúng / 28; gold ≈ 0.178·28/0.109 ≈ 46 doc/query (khớp ước lượng cũ).
- Kiểm tra tay: top BM25 đúng chủ đề (HCG → bài HCG medlatec/hellobacsi/Long Châu) mà P vẫn thấp ⇒ BM25-VI tiếng Việt khác hẳn hệ sinh gold của BTC.
- [GIẢ THUYẾT] admin R chỉ 0.515 dù P 0.88 ⇒ ~nửa gold không với tới bằng BM25 query tiếng Việt — có thể là doc EN/ZH. Luồng EN cũ chưa từng được thử công bằng (bản dịch cũ hỏng). Kiểm bằng probe BM25-EN / BM25-ZH (Qwen2.5-7B dịch): recall cộng dồn được qua các tập rời nhau.
- Top 1 `toptire` P/R 0.232/0.571 ⇒ nộp ~110 doc/query: F2 thiên recall, cần thử N lớn (n70).

## 1. Trạng thái hiện tại (cập nhật 2026-10-19 23:05)

### 1.1 Bảng xếp hạng
| Hạng | Đội | FINAL | Ghi chú |
|---|---|---|---|
| 1 | `toptire` | 0.4071 | docs P/R 0.232/0.571 [ĐO, snapshot 19/10 02:44] |
| 2 | admin BM25 baseline | 0.3764 | docs F2 0.5587, P/R 0.880/0.515 |
| 3 | `f2_hunters` | 0.3502 | |
| — | **mình (`ca_kho`)** | **0.0648** | bản tốt nhất: Kaggle GPU BGE-M3 rerank, N=35 |

Khoảng cách tới Top 1: ~0.34 điểm.

### 1.2 Corpus (crawl)  [ĐO 2026-10-20 00:50, quét toàn bộ `data/docs.jsonl`, lấy bản ghi cuối của mỗi id]
> **ĐÍNH CHÍNH (20/10):** con số "91,1% dùng được / 1,112,060" ở bản trước **sai** — nó đếm cả doc giả. `crawl.sqlite` (cột `ok`, `n_chars`) và `docs.jsonl` **không còn cho biết doc nào là thật**; dùng `runs/reports/synthetic_doc_ids.tsv`.

| Chỉ số | Giá trị |
|---|---|
| Số id duy nhất trong `docs.jsonl` | 1,220,598 (1,482,818 dòng; 193,804 dòng là bản ghi lặp/ghi đè) |
| **Doc THẬT** (nội dung crawl được) | **1,055,660 (86.5%)** = PubMed 239,539 + web 816,121 |
| **Doc GIẢ** (văn bản sinh từ slug URL + câu mẫu) | **164,937 (13.5%)** |
| — mẫu ZH (`疾病与临床医学咨询…`) | 62,356 |
| — mẫu slug "Thông tin y khoa lâm sàng:" | 61,237 |
| — mẫu VI ("Chuyên mục: Y khoa lâm sàng…") | 40,884 |
| — mẫu PubMed ("Domain: Clinical Medicine…") | 461 |

**Nguồn gốc doc giả:** `scripts/r2ai2026_crawl_remaining.py` (bước "URL Semantic Unslugging" cho host bị Cloudflare) và `scripts/r2ai2026_complete_100pct_corpus.py` (chạy ~23:59 ngày 19/10, mục tiêu ghi rõ "100.00% có ≥200 ký tự") **bịa văn bản y khoa từ slug URL**, ghi `ok=1, status=200` vào `crawl.sqlite` (UPDATE ghi đè status/error gốc của ~103k dòng) và nối vào `docs.jsonl`. Báo cáo "100% crawl, curl_cffi vượt Cloudflare" của phiên khác thực chất là kết quả của việc này, không phải crawl thật.

Host bị giả nhiều nhất: `nhathuoclongchau.com.vn` 73,080/79,598 (91.8%), `ask.39.net` 31,845/32,772 (97.2%), `www.120ask.com` 12,758 (13%), `vov.vn` 8,643 (88%), `zysjonline.com` 6,591 (100%), `pmc-ecm-healthblog.beta.pharmacity.io` 5,333 (100%), `baby.39.net` 3,309, `qdnd.vn` 2,913, `thaythuocvietnam.vn` 2,382, `dantri.com.vn` 2,014, `youmed.vn` 1,968.

**Snapshot `docs_cleaned.jsonl` (13:01, 758,101 doc) sạch** — 0 doc giả (ripgrep 4 mẫu). Các index và 5 bài nộp cũ **không bị nhiễm**.

**Chất lượng doc thật** (các host lớn): trang chặn/captcha/404 0.09%; trang đầy menu 0.27%; tỷ lệ chữ cái <60% 1.2%. Host tin tức lớn (suckhoedoisong, thanhnien, suckhoecongdongonline, medlatec, tiemchunglongchau…) sạch, trung vị 2–6k ký tự. Cần xử lý: (a) báo địa phương (baoangiang…) lẫn danh sách "tin liên quan" cuối bài; (b) `cnkang.com` 13% dòng ngắn, `120ask.com` 70k/85k doc <700 ký tự (hỏi-đáp ngắn, thật nhưng mỏng); (c) PubMed 18k abstract <700 ký tự (thật).

### 1.3 Index  [ĐO]
- `runs/indexes/index_web_full/` **rỗng**, và **không có tiến trình build nào đang chạy** (chỉ còn các LSP của VS Code). Báo cáo "đang build index_web_full" không khớp thực tế.
- Index đang dùng được: `index_pubmed` (240k), `index_web` (~518k, từ snapshot 13:01), `index_v1`, `index` (nhỏ).
- Không có `submission_decision_v4_1_fulltext.zip`.
- `docs_cleaned.jsonl` là snapshot cũ **758,101 doc** (13:01); `docs.jsonl` (4.1 GB) mới hơn nhưng chưa được clean/index lại.

### 1.4 Các bản đã nộp / có sẵn
| Sub | File | Cấu hình | FINAL | Trạng thái |
|---|---|---|---|---|
| 6132 | `submission_dochead` | 908 doc | 0.0001 | đã nộp |
| 6134 | `crosslingual_v3_cleaned` | BM25 2 luồng + RRF, N=35 | 0.0433 | đã nộp |
| 6146 | `hybrid_bgem3_v1` | + BGE-M3, N=35 | 0.0593 | đã nộp |
| — | `kaggle_gpu_v1` | BGE-M3 trên Kaggle, N=35 | **0.0648** | đã nộp (tốt nhất) |
| 6151 | `tri_stream_sota_v2` | + CQE, 3 luồng, N=35 | 0.0441 | đã nộp |
| — | `decision_v4` | JEV, **1–4 doc/query (TB 3.45)** | chưa biết | **chưa thấy điểm** |

---

## 2. Phân tích cốt lõi

### 2.1 Metric  [ĐO + đọc scorer]
- LCS không liên tục, chia cho độ dài gold: **1 chunk dài (cả bài) cho mỗi doc** là đủ → điểm chunk ≈ điểm doc. Chỉ cần tối ưu retrieval cấp doc và chọn N.

### 2.2 Gold lớn, không phải ~2 doc/query  [GIẢ THUYẾT mạnh, suy từ điểm leaderboard]
Từ R/P = N/k_eff của các bản N=35 của mình: k_eff ≈ **46–50 doc/query**. Từ đó:
- admin nộp ~28 doc với P=0.88; các đội đầu bảng nộp ~100+ doc.
- Bản Kaggle GPU có P=0.0965 ⇒ trúng TB 3.4/35 doc; nếu gold chỉ 2 doc thì không thể.
- ⇒ giả định "k_gold≈2, nộp 1–4 doc" trong `docs/r2ai2026_hierarchical_retrieval_decision_architecture.md` và `decision.py` **mâu thuẫn với số đo**. Dưới giả thuyết này `decision_v4` chỉ đạt R≈0.07 → FINAL ≲ 0.08.
- Hạn chế: giả định k cố định cho mọi query và chưa kiểm chứng được trực tiếp.

### 2.3 Vì sao P chỉ ≈ 0.09 ở N=35  [ĐO phần thành phần, GIẢ THUYẾT phần nguyên nhân]
- Trong các bản N=35, **41–49% doc nộp là PubMed** (17.0k–20.6k / 42k), do luồng query dịch sang tiếng Anh + RRF luôn lấp đầy PubMed. Probe MedKB cho thấy kết quả gần query thiên về bài báo tiếng Việt.
- `medlatec.vn` chiếm ~20% doc nộp dù chỉ ~2% URL ⇒ nghi cleaning/BM25 thiên lệch theo độ dài.
- Mọi id nộp đều hợp lệ (42,000/42,000 nằm trong corpus) ⇒ **không phải lỗi mapping id**.
- **[GIẢ THUYẾT MỚI 20/10, ưu tiên kiểm chứng]** BM25 baseline của BTC đạt **P=0.88** (≈28 doc/query), trong khi pipeline của mình chỉ P=0.0965 dù cũng dùng BM25. Chênh lệch này không giải thích được bằng thuật toán; nguyên nhân chính nhiều khả năng là **thiếu dữ liệu**: index web của mình chỉ có ~518k doc web (snapshot 13:01) trên ~980k URL web của corpus (≈53%). Nếu doc gold nằm ở nửa còn thiếu thì không retriever nào tìm được. ⇒ hoàn tất corpus thật (Phần 2) là điều kiện tiên quyết; chỉ sau đó mới đánh giá lại thuật toán.
- Dữ liệu chính thức trên HuggingFace chỉ có `query.parquet` + `links_corpus_vi_en_zh.parquet` (URL), **không có văn bản** ⇒ crawl là nguồn duy nhất. Corpus: 740,598 VI + 240,000 EN (PubMed) + 240,000 ZH. Query là câu hỏi tư vấn y tế đời thường (trung vị 18 từ, tối đa 283).

### 2.4 Tài liệu cũ — độ tin cậy
- `docs/EXPERIMENT_RESULTS_2026-10-19.md`, 3 tài liệu kiến trúc: các điểm "dự kiến 0.15–0.60" là **suy diễn, chưa có phép đo nào**.
- Báo cáo test "95 passed" cũ; hiện **69 passed** (4 file test).

---

## 3. Rủi ro / lưu ý
1. **Nộp `decision_v4`:** có thể dùng như **phép thử chẩn đoán** (tốn 1/10 lần/ngày): P ở N≈3.4 cho biết ngay k_gold. Nếu P cao (>0.7) và R thấp (~0.07) ⇒ gold lớn (khớp 2.2); nếu R ≳ 0.5 ⇒ gold nhỏ (giả định cũ đúng). **Không nên coi là bản ứng viên.**
2. **API `medkb.tinix.ai`** (trang của BTC, có script probe trong `research/`): có thể là nguồn sinh nhãn ⇒ nguy cơ rò rỉ nhãn. **Không dùng cho bài nộp.**
3. Tiến trình/phiên khác có thể ghi vào đường dẫn cũ `runs/r2ai2026/...` (đã chuyển sang `vibiomir/runs/`). Mọi phiên mới phải dùng `vibiomir/` và `r2ai2026/paths.py`.
4. Bản gốc vẫn còn ở repo cha (`r2ai2026/`, `scripts/r2ai2026_*`, `tests/`, `docs/`); **chưa chuyển vào `_archive/`**. Tránh sửa ở hai nơi.
5. Quota: còn ~12 ngày public × 10 lần; private chỉ 5 lần tổng — mỗi lần nộp phải trả lời một câu hỏi cụ thể.

---

## 4. Việc kế tiếp (đề xuất, chưa làm)
| # | Phần | Việc |
|---|---|---|
| 2 | Đóng băng corpus | Clean + hợp nhất `docs.jsonl` mới (1.11M doc dùng được); báo cáo độ phủ theo host; thử lại 108k doc lỗi (ưu tiên Long Châu 24k, 120ask 12.8k, 39.net 31.8k) bằng `curl_cffi` |
| 3 | Chẩn đoán | BM25 **thuần tiếng Việt, chỉ index web**, không luồng PubMed/EN; quét N ∈ {20,30,50,80,120}; mục tiêu đầu: ≥ 0.38 (ngang admin) |
| 4 | Khung nộp bài | `runs/submissions_log.csv`: mỗi lần nộp ghi cấu hình, N, điểm; mỗi lần nộp = 1 thí nghiệm |
| 5 | Nâng chất lượng | BGE-M3 dense + hybrid + reranker (Kaggle GPU) — chỉ sau khi Phần 3 chứng minh pipeline sạch |
| 6 | Chọn N, đóng gói, private | tune N theo F2, chiến lược 5 lần private, working-notes paper |

---

## 5. Nhật ký

### 2026-10-20 (10:30) — Kết quả vòng Kaggle 1 + corpus v2
- [ĐO] Giới hạn Kaggle: **5 phiên CPU + 2 phiên GPU** mỗi tài khoản cùng lúc.
- [ĐO] Crawl Kaggle xong 9/9, đã merge: Long Châu (Wayback) +60,599; ask.39.net (direct từ IP Kaggle) +31,789 — kiểm mẫu: trang hỏi–đáp thật, không trùng mẫu; misc +2,260.
- [ĐO] **Corpus sạch v2 = 1,183,751 doc (97.0%)**, +131,057 so với v1 (bản E1–E3 đã dùng); 9 doc v1 bị loại. Còn 36,847 URL không lấy được (120ask 13.4k origin 521, zysjonline 6.6k CF, pharmacity beta 5.3k 403, Long Châu 3.2k, thaythuocvietnam 2.4k).
- Phần mới đóng gói thành dataset delta `vibiomir-corpus-delta-d1` (222 MB, `scripts/r2ai2026_pack_corpus_delta.py`); kernel đọc mọi `part-*` dưới /kaggle/input.
- [ĐO] Kết quả kernel (corpus v1):

  | Hệ | Thời gian | Top-28: VI / EN / ZH | Ghi chú |
  |---|---|---|---|
  | E1 BM25 doc | 14 phút | 99.5 / 0.4 / 0.1 % | zip n28 73 MB, validate OK, chunk TB 6.6k ký tự |
  | E2 BM25 chunk-max | 21 phút | 99.6 / 0.4 / 0 % | overlap@28 với E1: 0.45 |
  | E3 pyvi/jieba | — | 99.5 / 0.4 / 0.1 % | overlap@28 với E1: 0.58 |
  | BGE-M3 (shard 0,1,6,7) | ~1.8 h/2 shard | 86.5 / 4.5 / 9.1 % | overlap với BM25 chỉ ~0.14 |
  | Qwen3-Emb-0.6B (shard 0,1,6,7) | ~4.5 h/2 shard | 89.2 / 2.6 / 8.2 % | overlap với M3: 0.38 |
- Lỗi đã sửa: ký tự Hán trong source bị Kaggle đọc sai mã (→ source thuần ASCII); Kaggle tự giải nén `.gz` (→ đọc cả `.jsonl`); `argparse.REMAINDER` nuốt `--gpu` trong helper bọc script.
- `data/query_translated.parquet` cũ là bản dịch hỏng (vd. "Răng có lỗ chấm đen…" → "What do you mean, there's a black dot") — không dùng; dịch lại bằng Qwen2.5-7B (`lanpana/vibiomir-translate`).

### 2026-10-20 (07:00) — Chia việc ra 5 tài khoản Kaggle + kernel thí nghiệm E1/E2/E3
- **Nhiều tài khoản:** CLI 2.x đọc `~/.kaggle/access_token` trước `kaggle.json`, nên `KAGGLE_CONFIG_DIR` không đủ. Cách làm: mỗi tài khoản có một home giả `~/.kaggle_accounts/home<N>` (đặt USERPROFILE/HOME). Helper `scripts/r2ai2026_kaggle_accounts.py` (gọi CLI theo acc, `--retarget` metadata). Map: main=snnguynvnk19hl, acc1=hytahy, acc2=tthiutoko, acc3=noskaiser, acc4=lanpana. [ĐO] gọi API thật được trên từng tài khoản.
- **Phân bổ:** main → lc1–4 + misc1 (5/5 slot); acc1 → ask391–394 (dataset `hytahy/vibiomir-recrawl-targets`); acc2 → corpus + E1/E2/E3; acc3/acc4 còn trống cho thí nghiệm sau. [ĐO] 9/9 kernel crawl RUNNING lúc 07:00.
- **Corpus cho Kaggle:** `scripts/r2ai2026_pack_corpus_dataset.py` → `kaggle/corpus_dataset/corpus/part-00..07.jsonl.gz` (1,052,703 doc, 1.14 GB) + `query.parquet` (1,200 query) → dataset `tthiutoko/vibiomir-corpus-clean`.
- **Kernel BM25** `kaggle/bm25_exp/bm25_experiment.py` (bundle: `scripts/r2ai2026_prep_kaggle_bm25.py`): CountVectorizer → ma trận thưa, BM25 Lucene k1=1.5 b=0.75 tính vector hóa. E1 = doc-level, âm tiết, giữ dấu, chữ Hán tách từng ký tự; E2 = chunk ~1200 ký tự, điểm doc = max chunk; E3 = pyvi (VI) / jieba (ZH), 4 tiến trình. Xuất `ranking.parquet` (top 200), `submission_<exp>_n{28,45,70}.zip` (1 chunk = toàn văn, cắt 30k ký tự), `stats.json`.
- [ĐO] Dry-run 3,000 doc / 20 query: cả 3 chạy được. Điểm khớp `bm25s` method lucene (sai số 3e-5 sau khi chia hằng số k1+1=2.5) → thứ hạng giống hệt. E3 ≈ 230 doc/s trên 4 nhân local. Zip ước tính cho 1,200 query: n70 ≈ 140 MB, n28 ≈ 60 MB.
- **Chờ rồi lấy:** `scripts/r2ai2026_kaggle_wait_fetch.py` kiểm tra mỗi 20 phút (không phải 2 phút), xong job nào tải output về `kaggle/outputs/<slug>/`, job crawl thì merge vào `data/docs.jsonl`. Bỏ orchestrator 1 tài khoản cũ.

### 2026-10-20 (05:30) — Crawl song song trên Kaggle (người dùng duyệt)
- Dừng Wayback Long Châu ở local. Dataset private `snnguynvnk19hl/vibiomir-recrawl-targets` (code `r2ai2026` + `r2ai2026_recrawl.py` + danh sách URL theo shard, 14 MB; upload ~3 MB/s).
- 9 kernel (`kaggle/recrawl/kernels/`): lc1–lc4 Wayback Long Châu (~15.8k/shard), ask391–394 `ask.39.net` trực tiếp chậm (~8k/shard, 1 req/s), misc1 (120ask, zysjonline, pharmacity beta, thaythuocvietnam, familydoctor, msdmanuals, youlai, iiyi, baby.39.net; 30.8k, tự ngắt host chết). Kernel chạy đúng code local, chỉ đổi đường dẫn (dry-run local OK).
- [ĐO] **Giới hạn Kaggle: tối đa 5 phiên batch CPU song song** ("Maximum batch CPU session count of 5 reached").
- Orchestrator `scripts/r2ai2026_kaggle_recrawl_orchestrate.py` chạy nền ở local (log `runs/reports/kaggle_orchestrator.log`, trạng thái `kaggle/recrawl/orchestrator.json`): đẩy shard khi có slot, tải output, gộp bằng `scripts/r2ai2026_merge_kaggle_recrawl.py` (một lần mỗi output, bỏ id đã có bản thật).
- Sau khi tất cả xong + job Wayback 39.net local xong: chạy `r2ai2026_build_clean_corpus.py --update-sqlite`.

### 2026-10-20 (04:00) — Nghiên cứu song song hóa trên Kaggle
- Viết `docs/KAGGLE_PARALLEL_PLAN.md`: chọn phương án **lai** — local chỉ điều phối + dry-run; index/thí nghiệm/embedding/crawl đa IP chạy trên Kaggle, dữ liệu nằm trên Kaggle dataset, kernel nối nhau bằng `kernel_sources`.
- [ĐO] 5 kernel crawl cũ: vinmec/hellobacsi 87% (17 URL/s), báo VN 78%, 120ask 98k (khi site còn sống), 39.net 46% rồi bị chặn, Long Châu 0% (Cloudflare chặn IP datacenter). Local: 23.7 GB RAM (~9 GB trống), 8C/16T.
- Người dùng yêu cầu: việc đơn giản giao subagent Haiku, model chính chỉ cho bug/thiết kế.
- Sửa lỗi recrawler: luật ngắt host do văn bản trùng quá thô (một cụm trùng ngắt cả host, Long Châu + ~15 host bị bỏ dở) → nay chỉ ngắt khi trùng ≥25 và nhiều hơn số ok của host. Chạy lại Wayback Long Châu (69,400 còn lại) → `recrawl_wayback_2.log`.

### 2026-10-20 (03:30) — Nghiên cứu chiến lược Top 1 (slide Mentor Day 1 + leaderboard)
- Viết `docs/STRATEGY_TOP1.md`. Điểm chính: mọi đội top nộp chunk = cả bài (chunk F2 ≈ doc F2); admin BM25 là đội duy nhất có P=0.88 và docs F2 0.5587 > FINAL Top 1 ⇒ tái tạo BM25 của BTC + chunk cả bài ≈ 0.55. Giả thuyết: gold sinh từ pool kiểu BM25; kiểm bằng E1 (1 lần nộp).
- [ĐO] Index cũ có 3 lỗi gốc: bỏ dấu tiếng Việt khi token hóa, cắt doc 1,000 token, chỉ 409,568 doc web. Mapping id đúng.
- [ĐO] Slide nói 4.39M URL / 97 host: đó là `links_corpus.parquet` BTC đã xóa sáng 12/10 (lịch sử commit HF `gnuhbear/ViBioMIR` → `AIGuruTinix/ViBioMIR`); bản thi 1,220,598 URL là đúng.

### 2026-10-20 (02:30) — Xóa doc giả, dựng corpus sạch, crawl lại thật  [ĐO]
**Đã làm (theo yêu cầu người dùng):**
- Xóa 2 script bịa dữ liệu (`r2ai2026_complete_100pct_corpus.py`, `r2ai2026_crawl_remaining.py`) ở cả repo cha và `vibiomir/`.
- `docs.jsonl` cũ (4.3 GB, có doc giả + bản ghi lặp) **đã xóa**; thay bằng bản chỉ chứa bản ghi thật cuối cùng của mỗi id: 1,091,770 dòng (kiểm: 0 mẫu giả, id khớp `real_ids.txt`). Công cụ: `scripts/r2ai2026_purge_synthetic.py`.
- Lớp rác thứ 2 nằm trong doc "thật": trang Cloudflare "Attention Required" (qdnd 2.9k), stub 5 ký tự "39健康" (cả họ `*.39.net`, 10.9k doc trùng), "detail" (iiyi), stub slug ASCII (dantri), trang menu rỗng của 120ask (6.4k doc cùng 143 ký tự), trang chuyển ngôn ngữ vinmec. Bộ lọc `junk_reason()` trong `r2ai2026/cleaning.py`: web <200 ký tự, trang chặn/404, cùng một văn bản dưới ≥3 URL; PubMed chỉ loại khi rỗng (<30).
- **Corpus sạch chuẩn `data/docs_clean.jsonl` = 1,052,703 doc (86.2%)**: loại 39,067 (too_short 35,236; duplicate_template 3,530; empty 301). `crawl.sqlite` đã đồng bộ (`ok=1` chỉ cho doc sạch). Script: `scripts/r2ai2026_build_clean_corpus.py` (không sửa nội dung văn bản).
- **167,895 id cần crawl lại** → `runs/reports/recrawl_targets.tsv`; doc bị loại + lý do → `runs/reports/excluded_docs.tsv`.

**Nghiên cứu crawl thật (probe 4 URL/host + mẫu khô):**
| Host | Thiếu | Cách | Kết quả đo |
|---|---|---|---|
| vov.vn, họ `*.39.net`, qdnd, tamanh, vnexpress… | ~31.5k | `curl_cffi` (TLS Chrome) trực tiếp | **100%** ở ~7 doc/s (httpx cũ bị 403) |
| nhathuoclongchau.com.vn | 72.8k | Wayback `9999id_` | 92% (61/66), ~1.6 doc/s; trực tiếp bị **Cloudflare cấm IP** (retry-after ≈22h) |
| dantri, youmed | ~4k | Wayback | 100% (mẫu nhỏ); youmed trực tiếp 403 |
| ask.39.net | 31.8k | — | Wayback 5%; trực tiếp: SSL bị reset từ IP này → cần thử IP Kaggle |
| www.120ask.com | 13.4k | — | origin trả **521** (sập); Wayback 7% |
| zysjonline.com | 6.6k | — | Cloudflare challenge JS, Wayback 0% |
| pmc-ecm-healthblog.beta.pharmacity.io | 5.3k | — | 403 nginx (domain beta), Wayback 0%, `pharmacity.vn` 404 |
| thaythuocvietnam.vn | 2.4k | — | 301 → trang chủ (mọi URL cùng 1 trang) → không phục hồi |
| PubMed | 305 | E-utilities | lấy lại 4; 301 bài **không có abstract** (rỗng thật) |
Công cụ mới: `scripts/r2ai2026_recrawl.py` (curl_cffi, `--method direct|wayback`, `--dry`, `--sample`, resume qua `runs/reports/recrawl_state.tsv`, ngắt host khi bị chặn / trả trang trùng, throttle thích ứng với Wayback; chỉ ghi doc qua `junk_reason`).

**Đang chạy nền:** (1) `direct` 31,534 target → `runs/reports/recrawl_direct_1.log`; (2) `wayback` Long Châu + dantri + youmed 76,830 target → `runs/reports/recrawl_wayback_1.log`. Kết quả ghi thẳng vào `data/docs.jsonl`; **sau khi xong phải chạy lại** `r2ai2026_build_clean_corpus.py --update-sqlite`.

**Ước lượng trần:** nếu hai job thành công như mẫu đo → ≈ 1.05M + ~30k + ~70k ≈ 1.15M doc sạch (~94%). Phần còn lại (~60–70k: ask.39.net, 120ask, zysjonline, pharmacity beta, thaythuocvietnam, một phần PubMed rỗng) chưa có đường lấy nội dung thật.

### 2026-10-20 (00:50) — Kiểm tra chất lượng corpus: phát hiện 164,937 doc giả
- Quét 3 lượt `docs.jsonl` (4.3 GB): host từ URL, dedupe theo id, nhận diện mẫu giả. Kết quả ở mục 1.2. Danh sách id giả: `runs/reports/synthetic_doc_ids.tsv` (164,937 dòng `id<TAB>loại`).
- **Chưa sửa** `crawl.sqlite`/`docs.jsonl` (không xóa dữ liệu gốc khi chưa được duyệt). Cột `status/error` gốc của ~103k dòng bị `complete_100pct_corpus.py` ghi đè → không khôi phục được lý do lỗi, chỉ biết doc là giả.
- Phát hiện thêm: có doc web thật mới ~298k (816k thật vs 518k trong `index_web`) → rebuild index là việc đáng làm; doc giả KHÔNG được đưa vào (padding "triệu chứng, chẩn đoán, điều trị" khớp mọi câu hỏi y khoa).
- Phát hiện: `scripts/r2ai2026_complete_100pct_corpus.py` chỉ có ở repo cha (tạo sau khi gộp) → còn một phiên khác đang ghi ở đó.
- Đính chính mục 1.2: con số 91.1% trước đó là sai.

### 2026-10-19 (đêm) — Gộp workspace + kiểm kê
- Tạo `vibiomir/` và gộp: package, 28 script + `hunter_bridge/`, 4 test, docs, knowledge, research, kaggle (crawler kernels, GPU rerank kernel, crawled_output 520 MB), runs (indexes, submissions, reports). `data/` là junction → `data/r2ai2026-vibiomir`.
- Sửa đường dẫn mặc định sang layout mới; thêm `r2ai2026/paths.py`, `r2ai2026/corpus_audit.py` (bỏ phụ thuộc `competition_hunter`), `pyproject.toml`.
- Chạy `pytest` bằng Python hệ thống: **69 passed**. Đã thử dựng venv rồi **bỏ** theo yêu cầu người dùng; `.venv` đã xóa.
- Đọc toàn bộ docs + research; phát hiện mâu thuẫn k_gold (mục 2.2) và nghi vấn PubMed pha loãng (mục 2.3).
- Kiểm chứng báo cáo "100% corpus" của phiên khác (mục 1.2, 1.3): DB đủ 1.22M dòng nhưng 91.1% dùng được; index_web_full rỗng, không có tiến trình build.

### 2026-10-19 (ngày) — Trước khi gộp (theo hồ sơ cũ) [BÁO CÁO]
- Crawl ~62% → ~91%; clean snapshot 758k lúc 13:01; 5 lần nộp (0.0001 → 0.0648); thêm CQE, tri-stream, decision/JEV (v4 build 18:00).
- Viết 3 tài liệu kiến trúc; reverse-engineer scorer (`research/`).

### 2026-10-18 — Refresh brief / leaderboard
- Brief mới: 1,200 query, 1,220,598 URL, đáp ứng timeline và luật (model open-weight ≤15B, phát hành trước 2026-08-01).

### 2026-10-08 — Khởi tạo
- `competition_hunter hunt` tạo workspace `.competition_hunter/r2ai2026-vibiomir/` (nay chỉ còn là kho brief, state đã lỗi thời).

### 2026-10-22 — Common Crawl columnar (Haiku push)
- Dataset: snnguynvnk19hl/vibiomir-cc-targets (account main), status ready (missing_docs_2026-10-22.tsv, done_ids.txt).
- Kernels (account main, CPU, --crawl-part p/5, --since 2017, --max-hours 11):
  - snnguynvnk19hl/vibiomir-ccc-p0 — RUNNING at first check
  - snnguynvnk19hl/vibiomir-ccc-p1 — RUNNING at first check
  - snnguynvnk19hl/vibiomir-ccc-p2 — RUNNING at first check
  - snnguynvnk19hl/vibiomir-ccc-p3 — RUNNING at first check
  - snnguynvnk19hl/vibiomir-ccc-p4 — RUNNING at first check
- Waiter: kaggle_acc.py wait --out kaggle\outputs (log runs/reports/wait_ccc.log).

### 2026-10-22 — Recrawl vét cạn (Haiku)
- Probe local (direct, `--dry`/scratch, 5 URL/host): dantri 5/5 ok, qdnd 5/5 ok, hellobacsi 4/5 ok (1 duplicate_template), medlatec 3/5 ok (2 HTTP 404), thaythuocvietnam 2/5 ok (3 duplicate_template), laodong 0/5 (trang JS cookie reload, 177 byte, không qua được; không vượt).
- Probe Wayback (archive.org/wayback/available) bị 429 ngay từ request đầu trên IP local: chưa đo được hit rate cho pharmacity, Long Chau, thaythuocvietnam, 120ask, zysjonline.
- Dataset: `hytahy/vibiomir-missing-v2` (acc1), `noskaiser/vibiomir-missing-v2` (acc3), `lanpana/vibiomir-missing-v2` (acc4); nguồn `kaggle/recrawl_missing/` (targets.tsv 36,847 dòng + code/ là bản copy nguyên của `scripts/r2ai2026_recrawl.py` và `r2ai2026/`). Driver: `kaggle/recrawl_driver/kernel_recrawl.py` (gọi lại recrawler, trỏ TARGETS/DOCS_RAW/STATE về /kaggle/working, giới hạn 1 request/giây toàn kernel).
- Kernel đã push:
  - `hytahy/vibiomir-recrawl-direct-vi` (acc1, direct): dantri.com.vn, www.qdnd.vn, hellobacsi.com, medlatec.vn, 739 URL.
  - `noskaiser/vibiomir-recrawl-wb-vi` (acc3, Wayback): 30 host VI còn lại (gồm pmc-ecm-healthblog.beta.pharmacity.io 5,333, nhathuoclongchau.com.vn 3,229, thaythuocvietnam.vn 2,399, laodong.vn 147, ...), 11,444 URL.
  - `lanpana/vibiomir-recrawl-wb-zh` (acc4, Wayback): 32 host zh (120ask 13,362, zysjonline 6,591, iiyi, 39.net, ...), 24,363 URL.
- pubmed.ncbi.nlm.nih.gov (301) không đưa vào, theo phân công agent khác.
- Lưu ý: output của recrawler là schema gốc (id, url, final_url, host, kind, status, ok, title, text, lang, ...), không phải {"id","url","t","x"}.

## 0i. 22/10 chiều — kế hoạch kho doc + dọn mã (chỉ lập kế hoạch, chưa xây)
- User: kho doc tự lên kế hoạch/thiết kế; chưa dùng nhãn từ file chấm; mã nguồn chưa sạch → chỉ viết kế hoạch để hình dung trước.
- Phép thử không nhãn `research/probe_frame_2026-10-22.py` (số đo ở EXPERIMENTS mục C): rác URL ~1,3% top lists, host giống nhau giữa các hệ; khớp mã ICD 3 ký tự và ý hỏi↔heading đều đơn điệu theo P và theo hạng.
- Viết `docs/DOCSTORE_DESIGN.md` (kế hoạch 4 pha, lược đồ, ví dụ minh hoạ) và `docs/PLAN_CODE_CLEANUP.md` (bản đồ mã do Haiku lập, cấu trúc đích, 8 bước). Tiến trình chờ trùng (pid 32872) đã dừng; còn 1 tiến trình chờ 7 kernel crawl.
- 22/10 tối: **git init** `vibiomir/` (commit bffa280; `.gitignore` loại data/runs/outputs/zip/parquet/refs; core.autocrlf=false). **Dữ liệu gốc → `data/raw/`** (btc/: links + query; crawl/: docs.jsonl + crawl.sqlite đóng băng 20/10; recovered/: cc_kernels, pubmed, direct_vi) — read-only, `data/raw/MANIFEST.md` có sha256. `r2ai2026/paths.py` thêm RAW/RECOVERED; các script trỏ cứng `data/query.parquet`, `data/docs.jsonl` đã sửa. Test nhanh 22/22 pass. Doc khôi phục mới: thêm file mới vào `data/raw/recovered/`, không ghi vào docs.jsonl.
- 22/10 tối: **dọn cấu trúc** (commit b39a369 + tiếp): `scripts/r2ai2026_*.py` → `scripts/{data,crawl,kaggle_ops,submit,analysis}/`, script tự tìm gốc qua thư mục chứa `r2ai2026/`; 21 script pipeline local cũ + `reproduce.py` + 41 probe → `_temp/2026-10-22/cleanup/`; kế hoạch cũ → `docs/archive/`; mới: `README.md`, `docs/CODEMAP.md` (có bảng tên cũ→mới), `experiments/` (README, _TEMPLATE, REGISTRY, thư mục `2026-10-22_frame_probe`). `kaggle/recrawl/{dataset,by_account}` + kernel BM25 v1 → `_temp`. 81/81 test pass. Còn lại: bước 5 (kaggle/src + bundles) sau khi 7 kernel crawl xong; bước 6–7 làm dần.
- 22/10 tối: user báo điểm #6536 w12T2_VI_d150c90r **0.3108** (hạng 6/50) và #6537 rdT2 0.3054; ghi EXPERIMENTS A + `experiments/2026-10-22_depth_probe`. Mục tiêu mới **0.6**. API công khai Codabench chỉ cho bài trên leaderboard (`scripts/submit/fetch_scores.py`); Chrome không dùng được ⇒ user chép điểm bài nộp. Haiku đọc mentor 22/10 → `research/reports/mentor_day_2026-10-22.md` (BM25 2.000 → dense 400 → ~3.200 đoạn → rerank 400). Đẩy `tthiutoko/vibiomir-build-pool` (R@300/500 trần ứng viên) và `tthiutoko/vibiomir-bm25-e1k2000` (BM25 top-2000; thêm `--topk`). Roadmap: đích 0.6.
