# Audit hệ thống ViBioMIR và kế hoạch tối đa recall (25/10/2026)

Nhãn: **[ĐO]** số đo LB/thật · **[SUY]** suy luận cần kiểm. Recall = Docs R macro trên 600 query public, |G| ≈ 71 doc/query.

## 0. Yêu cầu (user 25/10)
- Sản phẩm là **hệ thống hoàn chỉnh chạy từ source code trên máy chủ BTC** (nộp code 04/11), không phải ghép các bài đã nộp.
  Union các bài đã có điểm chỉ dùng để **đo** (trần recall, mật độ từng tầng).
- Ưu tiên **tối đa recall bằng phương pháp truy vấn**; sau đó mới đến xếp hạng, cắt và chunk. Không tốn lượt nộp cho tinh chỉnh +0.01–0.03.
- Ràng buộc: chỉ model mở ≤ 15B (phát hành < 01/08/2026); không MedKB API; việc nặng chạy Kaggle.

## 1. Hiện trạng theo tầng
| Tầng | Đang có (file) | Trạng thái |
|---|---|---|
| L0 làm sạch | `scripts/data/build_clean_corpus.py`, `r2ai2026/cleaning.py`, `kaggle/common/docorg.py` (clean_v4, section ≤ 4500) | Có, nhưng 2 nguồn text song song: docstore v1 đã dựng, kernel chưa đọc |
| L1 xử lý query | Dịch VI→EN/ZH bằng Qwen2.5-7B-Instruct (`kaggle/translate_exp`) | Chạy thật nhưng vô dụng: gold EN/ZH ≈ 0 (probe P .017) [ĐO]. HyDE/tách ý/step-back có prompt trong `query_expansion.py`, **chưa từng chạy** |
| L1 truy hồi lexical | `kaggle/bm25_exp/bm25_experiment.py` (k1 1.5, b .75, top 200; 15 biến thể) | Kaggle, mỗi lần dựng lại index |
| L1 truy hồi dense | `kaggle/dense_exp/dense_embed.py`: bge-m3 (đoạn 1000/200, 1500, 4500), Qwen3-Embedding-0.6B | Kaggle; **embedding corpus không được lưu** ⇒ mỗi kiểu query mới phải embed lại ~1,18 triệu doc (~8 h GPU) |
| Fusion | `kaggle/build_exp/build_submission.py` RRF k=60, BM25:M3 = 1:2 | Production |
| L2 rerank | `kaggle/rerank_exp/rerank.py` bge-reranker-v2-m3, doc = max logit 24 đoạn | Chạy thật, nhưng **kém w12** ở cùng N (#6769) [ĐO] |
| L3 cắt | `cut` theo logit r1 ≥ −2 (min 20) ⇒ 88 doc/q | Production; `r2ai2026/decision.py` chưa nối |
| Chunk | cửa sổ 4500 quanh đoạn tốt nhất của r1 (fallback lexical) + chunk thêm từ doc ngoài danh sách | Production |
| Nộp/đo | `scripts/submit/submit.py`, `fetch_scores.py` | Xong 25/10 |
| **End-to-end** | Không có lệnh đơn từ corpus + query.parquet ra zip; `r2ai2026/cli.py` gãy (gọi hàm không tồn tại, sai schema `query_id`/`qid`) | **Thiếu: đây là việc bắt buộc trước 04/11** |

## 2. Recall đã đo [ĐO]
| Tập ứng viên | doc/q | Docs R | Gold/q (R×71) |
|---|---|---|---|
| BM25 e1v2 @28 | 28 | .109 | 7,8 |
| BGE-M3 @28 | 28 | .193 | 13,7 |
| RRF BM25:M3 1:2 @28 | 28 | .203 | 14,4 |
| w12T2 (cắt r1) | 88 | .382 | 27,1 |
| Pool RRF(E1v2:M3:M3L) @500 | 500 | .657 | 46,6 |
| w12T2 + sliceA (≥ 2 họ, hạng union ≤ 500) | 314 | .600 | 42,6 |
| Union mọi nguồn đã chấm (VI) | 1.570 | **chưa đo** | ? |

Mật độ gold theo tầng: union top-100 của sliceA ≈ 10%, phần sau ≈ 4%; tầng chỉ-m3 hạng 100–500 ≈ 6%.

**Độ phủ của union (VI, 1.570 doc/q):**
- Theo họ: lexical 411 doc/q (mỗi BM25 chỉ đến độ sâu **200**); m3 925, q3e 796 (dense đến ~800); EN/ZH 3.
- Chỉ một họ tìm thấy: lex 198, m3 491, q3e 377.
- Các nguồn chưa dùng (BM25 s1/s2/c1/b1/b2/b4/b5, m3c) thêm **0–1 doc mới ở top-28, 0–12 ở top-100** ⇒ union đã **bão hoà** với các cách truy vấn hiện có. Thêm biến thể BM25 hay embedder cùng kiểu không tăng recall.
- BM25 sâu (e1k2000) thêm doc ngoài union: 133 doc/q ở top-500, 489 ở top-1000 ⇒ phần đuôi lexical là vùng duy nhất chưa khai thác (mật độ chưa đo).

## 3. Cái gì đã chứng minh hiệu quả / không [ĐO]
**Hiệu quả**
- Dense ≫ BM25 (P@28 .345 so với .178).
- RRF BM25:M3 = 1:2.
- Cắt theo logit r1 (T2) thay cho N cố định.
- Danh sách sâu (88 → +100 doc tầng ≥ 2 họ: Docs F2 +.012).
- Chunk cửa sổ 4500 quanh đoạn tốt nhất (bão hoà từ 4500).
- Chunk thêm từ doc ngoài danh sách (+.010 chunk F2).
- Bỏ chunk nhị phân (hết bài Failed).

**Không hiệu quả**
- Tinh chỉnh BM25 (tách từ pyvi, k1/b, chunk-level).
- Qwen3-Emb trong fusion (làm loãng).
- Dịch EN/ZH.
- Thứ tự reranker r1 làm khoá chính.
- Thêm cả tầng mật độ thấp (F2 macro phạt query ít gold).

**Chưa thử: toàn bộ phía query.** Tách ý, HyDE, trích thực thể (bệnh/thuốc) + từ đồng nghĩa, truy vấn theo tiêu đề/heading. Query của ta dài (trung vị 18 từ, tới 283 từ, nhiều ý), mà mỗi query có ~71 doc gold [SUY: gold theo chủ đề/bệnh, không theo một câu trả lời] ⇒ một vector/một BM25 cho cả câu khó phủ hết.

## 4. Kế hoạch tối đa recall (phương pháp truy vấn)
Mục tiêu: union của **hệ thống** (không phải các bài đã nộp) đạt **R ≥ 0,85** ở ≤ 2.000 doc/q, rồi L2/L3 chọn.

**Hạ tầng trước (mở khoá thử nhanh), Kaggle:**
- I1. Embed corpus một lần với bge-m3 theo section (cấu hình m3L: ≤ 4500 ký tự, có tiêu đề) và **lưu vector** fp16 (~6 GB) làm kernel output/dataset. Sau đó mỗi kiểu query = embed 1.200 query + nhân ma trận (vài phút).
- I2. Lưu index BM25 (bm25s save) theo section + doc để chạy nhiều biến thể query trên CPU, không dựng lại (~30 phút mỗi lần hiện nay).

**Phương pháp truy vấn, theo thứ tự chi phí/kỳ vọng:**
| # | Phương pháp | Cách làm | Model |
|---|---|---|---|
| Q1 | Lexical sâu | BM25 top-1000 (đã có e1k2000) | — |
| Q2 | Trích thực thể + đồng nghĩa | LLM xuất JSON: bệnh/thuốc/triệu chứng chính, tên gọi khác, mã ICD nếu chắc; BM25 trên tiêu đề+heading với thực thể; dense với "thực thể + ý hỏi" | Qwen2.5-7B-Instruct (đã chạy trên Kaggle) |
| Q3 | Tách ý | 2–5 câu hỏi con mỗi query ⇒ BM25 + M3 từng câu ⇒ RRF | như Q2 (cùng một lần gọi LLM) |
| Q4 | HyDE | LLM viết đoạn trả lời ~150 từ tiếng Việt ⇒ embed bằng bge-m3 ⇒ dense | như Q2 |
| Q5 | Mở rộng theo doc (PRF) | top-10 doc ⇒ từ khoá/vector trung bình ⇒ truy hồi vòng 2 | — |

- Q2–Q4 chung **một kernel LLM trên 1.200 query** (~30–60 phút T4), chi phí nhỏ so với embed corpus.
- Mỗi phương pháp trả về top-K theo query gốc.

**Cách đo (1 lượt nộp cho mỗi phương pháp, chỉ thay doc):**
- docs = top-K của phương pháp **trừ** union hiện tại, chunk giữ nguyên #6751 ⇒ Docs R × 71 = số gold **mới** mà phương pháp mang lại, P = mật độ.
- Giữ phương pháp nếu số gold mới/query đáng kể (≥ 2–3).
- Một lượt đo trần union hiện tại (`unionS1_VI_d5000c88.zip` đã dựng) để biết phần còn thiếu.

## 5. Sau truy hồi: xếp hạng, cắt, chunk (hệ thống)
- **L2 điểm tổng hợp:** đặc trưng theo cặp (q, d) gồm:
  - hạng/điểm từng nguồn và số họ đồng thuận (đã thấy đơn điệu theo n_families);
  - khớp thực thể với tiêu đề/heading (Q2);
  - logit r1 dùng làm **đặc trưng**, không làm khoá.

  Trọng số chọn bằng các probe tầng (mật độ đo trên LB), không dùng nhãn từng cặp.
- **L3 cắt theo query:** ngưỡng p > F2_q/5 cần ước |G_q|. Proxy: số doc vượt ngưỡng điểm, độ đồng thuận giữa các nguồn. Bắt đầu bằng luật, rồi hiệu chỉnh bằng probe.
- **Chunk:** section chứa đoạn tốt nhất (≤ 4,8k ký tự) cho doc trong danh sách, cộng chunk thêm cho các doc kế tiếp, trong ngân sách 100 MB.

## 6. Hệ thống hoàn chỉnh (bắt buộc trước 04/11)
- Một entry point `python -m r2ai2026 run --corpus DIR --queries query.parquet --out submission.zip [--stages ...]`. Mỗi stage ghi cache parquet. Đường dẫn qua tham số, không hard-code `/kaggle/...`.
- Cấu hình bằng một file YAML.
- Các stage:
  1. clean;
  2. section-chunk;
  3. index BM25 + embed bge-m3 (lưu vector);
  4. LLM query understanding (Q2–Q4);
  5. truy hồi đa query;
  6. fusion + L2;
  7. L3 cắt;
  8. chunk;
  9. ghi zip.
- Dùng lại code đang chạy thật trong `kaggle/*` (BM25, dense, rerank, builder), không viết lại từ đầu. Sửa hoặc thay `cli.py` đang gãy.
- Kernel Kaggle chỉ là lớp gọi entry point này ⇒ chạy trên Kaggle hay máy BTC là cùng một code.
- **Cần biết từ BTC:** cấu hình máy (GPU, RAM, thời gian tối đa) để chọn model/độ sâu.

## 7. Thứ tự làm (đề xuất)
| Ngày | Việc |
|---|---|
| 25–26/10 | Khung entry point + cấu hình; I1 (embed lưu vector, 3 kernel GPU nếu có slot), I2 |
| 26/10 | Kernel LLM Q2–Q4 trên 1.200 query; đo trần union (1 lượt) |
| 27–28/10 | Truy hồi Q1–Q5 trên vector/index đã lưu; mỗi phương pháp 1 probe "gold mới" |
| 29–30/10 | L2 điểm tổng hợp + L3 cắt + chunk section trong hệ thống; bài nộp từ chính hệ thống |
| 31/10–03/11 | Chốt; chạy lại toàn bộ từ đầu bằng entry point (kiểm tái lập); private 5 lượt |
| 04/11 | Nộp code + working notes |

## 8. Dữ liệu: vấn đề còn lại và cách tổ chức lại (25/10) [ĐO trên `data/docstore/v1`]
**Số liệu kho v1** (dựng 24/10, local, **chưa lên Kaggle**):
- 1.185.389 doc, 3.582.922 đoạn.
- VI 730.691 / 740.598 link (98,66%; thiếu 9,9k). Thin (< 500 ký tự) 2,0%.
- Độ dài doc VI: trung vị 3.054 ký tự, P95 7.353.
- Đoạn v1: trung vị **844** ký tự (P95 1.466); 4,0 đoạn/doc.

| # | Vấn đề | Hệ quả | Cách sửa |
|---|---|---|---|
| D1 | Kernel vẫn đọc corpus cũ (docs_clean + delta), kho v1 chỉ ở local | 3 phiên bản text; 2.723 doc recovered chưa dùng; không tái lập | Đẩy `vibiomir-docstore-v1` (có MANIFEST) và chuyển mọi kernel sang đọc nó |
| D2 | Đơn vị đoạn ≠ đơn vị gold: đoạn v1 ~850 ký tự, gold chunk ước ~3,3k (proxy MedKB, max 4,8k; cửa sổ bão hoà ở 4,5k) | Chunk lệch chỗ (chunk R = 0,73 × doc R) | Thêm tầng **section** = gộp đoạn theo heading tới ≤ 4,8k, có offset ⇒ chunk = span section |
| D3 | EN/ZH = 38% corpus nhưng gold ≈ 0 (probe P .017) | Tốn index/GPU | Hệ thống chỉ index VI; EN/ZH chỉ để báo cáo |
| D4 | Không có metadata ngữ nghĩa (bệnh/thực thể chính, nhóm ICD, loại tài liệu, loại mục) | Không truy hồi theo chủ đề được, trong khi gold ~71 doc/q mang tính chủ đề | Gắn nhãn bằng LLM trên tiêu đề + heading (730k doc, ngắn) ⇒ index thực thể và loại mục |
| D5 | Trùng lặp: 4.802 bản y hệt, chưa có cụm gần trùng | Phí chỗ trong danh sách; bỏ sót bản sao là gold | Cụm MinHash ⇒ chấm một lần, mở rộng id khi nộp |
| D6 | Không lưu embedding | Mỗi kiểu query = embed lại 8 h | Lưu vector theo section/đoạn |
| D7 | Query chưa có cấu trúc | Một vector cho query dài nhiều ý | Khung query: thực thể, ý hỏi, câu con, HyDE |
