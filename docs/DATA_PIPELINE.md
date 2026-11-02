# Pipeline dữ liệu: từ `data/raw` đến bài nộp (hiện trạng + thiết kế mới) — 22/10

Nhãn: **[ĐO]** số đo thật · **[SUY]** suy luận · **[ĐX]** đề xuất, chưa xây.
Điểm xuất phát: crawl đã xong và đóng băng trong `data/raw/` (chỉ đọc). Mọi thứ phía sau là **dẫn xuất**: xoá đi phải dựng lại được
bằng script trong git.

---

## 1. Dữ liệu gốc [ĐO]

| File (`data/raw/`) | Nội dung | Số lượng |
|---|---|---|
| `btc/links_corpus_vi_en_zh.parquet` | (id, url) do BTC phát hành; id chia khối theo ngôn ngữ | 1.220.598 = VI 740.598 · ZH 240.000 · EN 240.000 |
| `btc/query.parquet` | (id, query) tiếng Việt, trung vị 81 ký tự | 1.200 (public 600 + private 600, đã phát hành hết) |
| `crawl/docs.jsonl` + `crawl.sqlite` | đầu ra crawler của ta, đóng băng 20/10 23:53 | 5,26 GB |
| `recovered/cc_kernels_2026-10-22.jsonl` | Common Crawl (CDX) | 1.096 doc |
| `recovered/direct_vi_2026-10-22.jsonl` | recrawl trực tiếp VI | 451 doc |
| `recovered/pubmed_2026-10-22.jsonl` | PubMed E-utilities | 178 doc |
| *(đang chạy)* | Wayback VI/ZH, Common Crawl columnar p0–p4 | sẽ thành file mới trong `recovered/` |

### Luật chấm chunk — **đổi ngày 23/10** (email BTC; trang Evaluation trên web chưa cập nhật) [ĐO]
- Hai phía được chuẩn hoá: NFKC, giải mã HTML entity, chữ thường, gộp khoảng trắng.
- **Luật mới:** c đúng với gold chunk g cùng doc_id khi **cả hai** LCS_token(c, g) / |g| ≥ 0,4 **và** LCS_token(c, g) / |c| ≥ 0,4
  (LCS = dãy con chung, không cần liên tiếp). Luật cũ (đến 22/10) chỉ có điều kiện thứ nhất.
- P = số chunk nộp đúng / số chunk nộp. R = số gold chunk được tìm thấy / tổng gold chunk.
- ~~Toàn văn là tối ưu; furniture trong chunk không mất gì~~ — **sai theo luật mới** (đã rút lại 24/10). Hệ quả mới:
  1. Chunk dài bị phạt: nếu c chứa trọn g thì cần |c| ≤ 2,5·|g|. Cửa sổ cỡ s khớp được gold cỡ ~0,4s–2,5s nếu đặt đúng chỗ.
     Gold chưa biết cỡ; proxy MedKB: trung vị ~3,3–3,5k ký tự, max ~4,8k. Chunk toàn văn của #6536: trung vị 5,6k, 92% ≤ 8,75k.
  2. Furniture **làm mất điểm** (tăng |c|, giảm LCS/|c|) ⇒ cửa sổ nên tránh furniture ở hai mép; không ghép các dòng rời nhau.
  3. Một doc có thể cần nhiều chunk (mỗi gold chunk phải có một c riêng khớp); khởi đầu 1 chunk/doc.
  4. Thử nghiệm cỡ cửa sổ: kernel `tthiutoko/vibiomir-build-win` (1,5k / 3k / 4,5k quanh best passage của reranker r1, danh sách doc giữ như #6536).
  4. Chuẩn hoá Unicode/khoảng trắng phía ta là an toàn vì bộ chấm cũng chuẩn hoá.

Độ phủ hiện tại (so với links): VI 98,35% (thiếu 12.183), ZH 89,85% (thiếu 24.363), EN 99,87%. 14.803 doc VI < 500 ký tự.

---

## 2. Pipeline hiện tại (as-is)

```
data/raw/crawl/docs.jsonl
   │  scripts/data/build_clean_corpus.py   lọc rác (cleaning.junk_reason), KHÔNG sửa text
   ▼
data/docs_clean.jsonl  (1.183.751 doc, 97,0%)
   ├──► scripts/data/pack_corpus_dataset.py ─► Kaggle dataset vibiomir-corpus-clean (1.052.703 doc, bản cũ)
   │    scripts/data/pack_corpus_delta.py   ─► + vibiomir-corpus-delta-d1 (phần còn lại)  {"id","t","x"}
   │                                            │
   │                                            ▼  MỖI KERNEL TỰ CẮT TEXT THEO KIỂU RIÊNG
   │       BM25 e1v2 ........ cả doc, âm tiết, giữ dấu        (tuỳ chọn clean_v4 / section_chunks của docorg.py)
   │       BGE-M3 ........... đoạn ~1500 ký tự (m3c 1000/200, m3L 4500), doc = max đoạn
   │       reranker ......... đoạn ~1500 ký tự, doc = max logit đoạn
   │       build_submission . chunk = TOÀN VĂN doc (title + text)
   │                                            ▼
   │                    ranking.parquet mỗi nguồn → RRF w12 → cắt τ → zip LZMA
   │
   └──► scripts/data/build_processed_corpus.py ─► data/processed_v3/ (chuẩn hoá, gỡ template 2,04% ký tự,
                                                   lang, content_key, dup_of 4.773)   ← CHỈ Ở LOCAL, KERNEL KHÔNG DÙNG
```

### Vấn đề của hiện trạng
| # | Vấn đề | Hệ quả |
|---|---|---|
| P1 | Có **ba phiên bản text**: docs_clean thô (kernel dùng), processed_v3 (chỉ local), clean_v4 (cắt trong kernel BM25) | không biết nguồn nào thấy text nào; công gỡ template của v3 bị bỏ phí |
| P2 | **Mỗi giai đoạn tự chia đoạn** (1000, 1200, 1500, 4500 ký tự) và không có mã đoạn chung | không ghép được điểm cấp đoạn giữa BM25/M3/reranker; không chọn được "cửa sổ khớp nhất" để làm chunk |
| P3 | ~1.725 doc trong `recovered/` **chưa vào** dataset Kaggle | mất recall với đúng các doc khó lấy |
| P4 | Thông tin trùng (`content_key`, `dup_of`) không lên Kaggle | không gom cụm trùng khi gộp nguồn |
| P5 | Embedding corpus không lưu thành tài sản dùng lại (các output dense chỉ có `q_emb.npy`) [ĐO] | mỗi biến thể query hay model mới lại phải embed lại ~1,18 triệu doc |
| P6 | Dataset Kaggle = bản gốc + delta vá thêm, không có phiên bản và manifest; Kaggle có 1.183.760 doc còn docs_clean 1.183.751 (≈9 doc lệch, có thể là doc giả đã gỡ) | khó tái lập cho hạn nộp hệ thống 04/11 |

---

## 3. Thiết kế mới: một kho doc chuẩn dùng chung cho mọi giai đoạn [ĐX]

Nguyên tắc: **dựng một lần từ raw → mọi kernel đọc cùng một bảng doc và một bảng đoạn**. Text dùng cho `chunk_text` luôn là
lát cắt nguyên văn của doc (đã chuẩn hoá Unicode/khoảng trắng; bộ chấm cũng chuẩn hoá trước khi so), không viết lại.

```
data/raw/{crawl, recovered/*}
  S0 HỢP NHẤT  ── một bản ghi / id; nhiều nguồn cùng id ⇒ chọn theo ưu tiên + độ dài hợp lệ; ghi rõ source
  S1 LÀM SẠCH  ── lọc rác (như nay) + normalize_text (v3); template/clean_v4 KHÔNG xoá chữ mà ĐÁNH DẤU span "furniture"
  S2 METADATA  ── lang, host, doc_type, n_chars, thin, content_key, cụm MinHash (gần trùng)
  S3 PHÂN ĐOẠN ── MỘT cách chia duy nhất: theo heading (docorg.section_chunks), đích ~1.500 ký tự, bỏ qua span furniture
                  mỗi đoạn có (doc_id, pid, start, end, heading, sec_type); cửa sổ lớn = ghép 3 đoạn liền nhau
  S4 ĐÓNG GÓI  ── một dataset Kaggle có phiên bản: vibiomir-docstore-vN (docs + passages + queries + MANIFEST sha)
  S5 TÀI SẢN INDEX (Kaggle, tính một lần, lưu lại)
                  BM25 cấp doc + cấp đoạn (ma trận thưa) · BGE-M3 dense (fp16) + sparse theo pid · e5-large-instruct theo pid
  S6 PHÍA QUERY  ── query.parquet → khung query (luật + Qwen2.5-7B) + biến thể q0..q5 → queries_xl.parquet
           ▼
  L1 truy hồi rộng (chỉ cần encode query + tìm trên tài sản S5) → L2 giảm → L3 quyết định → L4 chunk = cửa sổ đoạn
```

### Lược đồ (`data/docstore/vN/`) — v1 đã xây (`scripts/data/build_docstore.py`, `r2ai2026/docstore.py`)
Text chỉ có **một bản**, đúng định dạng kernel đang đọc ⇒ BM25 / dense / build chạy trên v1 không cần sửa kernel.
Offset `start`/`end` = vị trí ký tự trong `x` (thân bài); tiêu đề `t` để riêng.
| File | Khoá | Cột |
|---|---|---|
| `corpus/part-00..07.jsonl.gz` | id | `{"id","t","x"}`: x = thân bài chuẩn (prepare_body), **giữ furniture** |
| `docs_meta.parquet` | id | url, host, lang, source (crawl/direct/wayback/cc/pubmed), n_chars, thin (<500), content_key, dup_of, furniture_chars, n_passages |
| `passages.parquet` | (doc_id, pid) | start, end, heading, content_chars (ký tự ngoài furniture) — các đoạn **phủ kín** x, không hở, không chồng |
| `furniture.parquet` | (doc_id, start) | end, kind (template / references / bio / signature / related / url / cta) |
| `templates.json`, `MANIFEST.json` | — | dòng template theo host; sha256 đầu vào raw, commit git, tham số, số đếm, lý do loại rác |

Quy tắc S0: crawl = bản ghi ok cuối cùng mỗi id + lọc rác với số trùng đếm **trong crawl** (tái lập đúng docs_clean,
kiểm bằng số 1.183.751). Recovered = chỉ id crawl không có; lọc rác với số trùng đếm trên **crawl + mọi file recovered**
(bắt trang chủ/404 mà Wayback/CC trả về cho nhiều URL chết); nhiều nguồn cùng id ⇒ direct > wayback > cc > pubmed, rồi dài nhất.
Để sau v1 (không cần cho cổng 4a): `doc_type`, cụm MinHash gần trùng, `sec_type`, `queries_xl`.

### Vì sao giải quyết được P1–P6
- **P1, P2:** mọi kernel đọc cùng `docs` + `passages`; điểm BM25/M3/reranker gắn với cùng pid nên cộng được ở cấp đoạn.
  L4 lấy chunk = span [start, end) của đoạn hoặc cửa sổ tốt nhất, đúng là thứ thử nghiệm (a) cần.
- **P3:** S0 đọc mọi file trong `recovered/`; crawl mới về ⇒ chạy lại S0–S4 ra `vN+1`, không vá delta.
- **P4:** `cluster_id` đi cùng doc ⇒ gộp nguồn, rerank một lần mỗi cụm, mở rộng bản sao khi nộp.
- **P5:** S5 lưu embedding theo pid ⇒ thêm biến thể query chỉ tốn encode 1.200 × k query (vài phút GPU).
- **P6:** dataset có phiên bản + manifest ⇒ tái lập được cho hạn 04/11 và bài báo mô tả phương pháp.

---

## 4. Thứ tự làm và cổng kiểm tra (không dùng nhãn)

| Bước | Việc | Chạy ở | Cổng qua |
|---|---|---|---|
| 1 | S0–S2 trên local (CPU, đọc raw 5,3 GB) | local | số doc = docs_clean + recovered hợp lệ; 0 id trùng; mỗi id có source |
| 2 | S3 phân đoạn + thống kê độ dài | local | ≥ 99% doc có ≥ 1 đoạn; trung vị đoạn 1.000–2.000 ký tự; span furniture < 5% ký tự |
| 3 | S4 đẩy dataset `vibiomir-docstore-v1` | Kaggle | kernel đọc thử được, đếm khớp manifest |
| 4a | **Kiểm hồi quy đường ống:** BM25 e1 trên v1, index **giữ** furniture | Kaggle CPU | top-28 trùng ≥ 95% với E1v2 cũ (chỉ khác ở doc recovered) |
| 4b | A/B: index **bỏ** furniture | Kaggle CPU + 1 lượt LB | không có cổng trùng; chỉ so điểm LB |
| 5 | S5 BGE-M3 dense+sparse theo pid, lưu embedding (`--save-emb`) | Kaggle GPU: lần trước 5 kernel × 1,2–1,9 h ≈ 8 h kernel cho ~2,97 triệu đoạn; embedding fp16 ≈ 6 GB | ranking M3 trên v1 trùng top-28 ≥ 90% với M3 cũ |
| 6 | S6 + L1 nhiều nguồn | Kaggle | — |

Bước 1–4 không cần GPU và không phụ thuộc kết quả crawl đang chạy (crawl xong chỉ cần dựng lại v2).

### Lịch (lùi từ hạn public 31/10 23:59; còn ~9 ngày × 10 lượt nộp)
| Ngày | Kho doc (đường dài) | Thử nghiệm LB trên corpus CŨ (chạy song song, không chờ kho) |
|---|---|---|
| 22–23/10 | S0–S3 local, đẩy v1 | (a) chunk cửa sổ trong `build_submission.py` trên danh sách doc w12T2; `pool_VI_d500`; (b) rerank top-400 bằng reranker VI |
| 24/10 | 4a/4b BM25 trên v1; S5 M3 bắt đầu | slice probe nguồn BM25-2000 |
| 25–26/10 | S5 xong; S6 biến thể query | slice probe theo nguồn × dải hạng |
| 27–29/10 | L1 nhiều nguồn trên v1 → L2/L3 | bảng mật độ → bộ quyết định |
| 30–31/10 | đóng băng, chỉ bài chính | chọn bài cho private (5 lượt) |
Nếu v1 trễ quá 26/10: bài chính vẫn dựng từ corpus cũ, kho doc chỉ phục vụ bài báo và hệ thống nộp 04/11.

---

## 5. Quyết định còn mở
1. **Ưu tiên khi một id có nhiều nguồn** (crawl gốc vs Wayback vs Common Crawl): đề xuất lấy bản crawl gốc nếu qua lọc rác, nếu không
   thì bản dài nhất qua lọc.
2. **Furniture:** theo luật mới (23/10) furniture trong chunk làm giảm LCS/|c| ⇒ cửa sổ chunk tránh furniture ở mép. Text index bỏ hay giữ quyết định bằng bước 4b.
3. **Đoạn ~1.500 ký tự hay theo mục:** chọn theo mục, cận 1.500; mentor BTC dùng ~3.200 đoạn/400 doc ≈ 8 đoạn/doc — khớp cỡ này.
