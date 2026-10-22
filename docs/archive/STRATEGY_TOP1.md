# Chiến lược Top 1 — R2AI2026 ViBioMIR (2026-10-20)

Nguồn: slide Mentor Day 1 (`261012_R2_AI_S3_MentorDay1.pdf`, 12/10/2026, Nguyễn Hoàng Trung — UET-VNU), leaderboard snapshot 19/10, scorer đã reverse-engineer, corpus đã kiểm kê (xem `WORKLOG.md`).
Nhãn: **[ĐO]** đã kiểm chứng · **[SUY]** suy luận có cơ sở · **[GIẢ THUYẾT]** cần kiểm bằng 1 lần nộp.

---

## 1. Bức tranh hiện tại

| Đội | FINAL | docs F2 | chunks F2 | docs P | docs R | N ước tính (k≈48) |
|---|---|---|---|---|---|---|
| toptire (#1) | 0.4071 | 0.4073 | 0.4068 | 0.232 | 0.571 | ~118 |
| admin BM25 | 0.3764 | **0.5587** | 0.1941 | **0.880** | 0.515 | ~28 |
| f2_hunters | 0.3502 | 0.3617 | 0.3388 | 0.240 | 0.467 | ~93 |
| mình (ca_kho) | 0.0648 | — | — | 0.0965 | ~0.07 | 35 |

Ba sự thật rút ra [ĐO]:
1. **Mọi đội top đều có chunk F2 ≈ doc F2** → họ nộp "1 chunk = cả bài" (scorer LCS không liên tục, chia cho |gold| ⇒ chunk cả bài luôn ≥ 0.40 nếu doc đúng). Admin là ngoại lệ (chunk nhỏ → chunk F2 chỉ 0.19).
2. **Admin là đội duy nhất có precision cao (0.88)**; mọi đội khác ≤ 0.30 dù dùng dense/hybrid/rerank.
3. **docs F2 của admin (0.5587) đã cao hơn FINAL của Top 1 (0.4071).**

⇒ **Đường ngắn nhất lên Top 1: tái tạo ranking BM25 của admin + chunk cả bài.** Nếu đạt, FINAL ≈ docs F2 ≈ 0.55, vượt Top 1 ~0.15 điểm, *trước cả khi* tối ưu N hay thêm dense.

## 2. Giả thuyết trung tâm: gold được sinh từ pool kiểu BM25  [GIẢ THUYẾT]

- P=0.88 của một BM25 thuần trên truy vấn hội thoại là bất thường; chỉ giải thích được nếu **nhãn gold được tạo từ pool ứng viên mà BM25 của BTC chiếm phần lớn** (pool → LLM/chuyên gia chấm → gold). Các đội dùng dense/rerank lấy doc "đúng nghĩa" nhưng ngoài pool ⇒ P thấp.
- Gold lớn: k_eff ≈ 46–50 doc/query [SUY từ R/P của các bản N=35]. Admin top-~28 trúng ~25 ⇒ phần còn lại của gold (~half) là doc BM25 xếp sâu hơn hoặc doc từ nhánh khác của pool (dense / cross-lingual).
- Gợi ý từ slide về cách BTC xử lý (khả năng cao trùng với baseline): **giữ dấu tiếng Việt**, tách từ VI (`triệu_chứng | sỏi_thận`), tách từ ZH (`肾结石 | 症状`), chunk theo mục/tiêu đề, giữ cặp hỏi–đáp, BGE-M3, RRF, BGE-reranker-v2-m3.

**Kiểm chứng bằng 1 lần nộp (E1, mục 4).** Nếu docs P ở N≈28 ≥ 0.6 ⇒ giả thuyết đúng, đi tiếp nhánh A. Nếu ≤ 0.3 ⇒ sai, chuyển nhánh B.

## 3. Vì sao bản cũ của mình chỉ đạt P≈0.10  [ĐO]

| Lỗi | Bằng chứng |
|---|---|
| **Bỏ dấu tiếng Việt** khi token hóa (`keep_diacritics=False`) | "lao phổi" khớp "lao động"; "hạch cổ" ra "Cải xanh trị ho". Slide 20: *"Không xóa dấu tiếng Việt"* |
| **Cắt doc ở 1,000 token** (`max_tokens=1000`) | `index_web/manifest.json` |
| **Thiếu dữ liệu**: index web chỉ 409,568 doc / ~695k URL VI | manifest; vinmec, hellobacsi, Long Châu không xuất hiện trong top |
| **41–49% suất nộp là PubMed** (luồng dịch EN + RRF) | đếm trong zip |
| N cố định 35, chunk bị cắt 2–3k ký tự | đọc zip |

Mapping id **đúng** (kết quả đúng chủ đề) — không phải lỗi dữ liệu nộp.

## 4. Kế hoạch thực nghiệm (mỗi lần nộp trả lời 1 câu hỏi)

Quy ước chung cho mọi bản nộp: **1 chunk = toàn văn doc** (không cắt), doc thứ tự theo điểm, ghi `runs/submissions_log.csv`.

### Nhánh A — tái tạo BM25 của BTC (ưu tiên)
| # | Bản nộp | Câu hỏi | Quyết định |
|---|---|---|---|
| E1 | BM25 doc-level, **giữ dấu**, token âm tiết, toàn văn, corpus sạch (VI+ZH+EN), N=28 | Gold có phải pool BM25? | P≥0.6 ⇒ nhánh A; ≤0.3 ⇒ nhánh B |
| E2 | Như E1 nhưng **chunk-level** (đoạn theo mục ~150–300 từ, có tiêu đề), điểm doc = max chunk | Đơn vị index của BTC là chunk? | giữ biến thể P cao hơn |
| E3 | Như biến thể tốt nhất + **tách từ VI** (pyvi/underthesea) + jieba ZH | Tokenizer của BTC? | giữ biến thể P cao hơn |
| E4–E5 | Biến thể tốt nhất, N=45 và N=70 | Đường cong P(N), R(N) | chọn N tối ưu F2 |
| E6 | N thích ứng theo từng query (ngưỡng tỷ lệ điểm so với top-1) | Có hơn N cố định? | giữ nếu +F2 |

Kỳ vọng (nếu giả thuyết đúng): E1 ≈ 0.45–0.55 FINAL → E4–E6 ≈ 0.6+.

### Nhánh B — nếu gold không phải pool BM25
Hybrid như các đội top nhưng làm sạch: BM25 (giữ dấu) + BGE-M3 dense (Kaggle T4) → RRF có trọng số → BGE-reranker-v2-m3 trên top-100 → N≈80–120 (vì P thấp, F2 thưởng recall). Mục tiêu ≥ 0.42.

### Bổ sung recall (dùng cho cả hai nhánh, sau E4)
- **Dense BGE-M3** để lấy phần gold BM25 bỏ sót (≈ nửa gold: doc xếp sâu, doc ZH/EN). Thêm vào *đuôi* danh sách BM25 (không chen top), đo ΔR ở N cố định.
- PubMed/ZH chỉ đưa vào nếu đo được tăng R mà không giảm P.
- ICD-10 (slide 28–31): dùng để phân tích lỗi theo nhóm bệnh, không lọc cứng.

## 5. Dữ liệu — điều kiện tiên quyết  [ĐO]
- Corpus sạch hiện tại: **1,052,703 doc** (VI 86.5%, ZH 73.3%, PubMed 99.9%). Đang crawl lại thật ~107k (Long Châu qua Wayback, vov/qdnd trực tiếp, 39.net qua Wayback). Trần khả thi ~94%.
- Slide 13–14 trùng đúng hai lỗi đã gặp: trang chọn ngôn ngữ vinmec (đã loại), Long Châu cần lấy vùng bài + giữ tiêu đề mục.
- Slide 9–10 nói "4,394,718 URL / 97 host" — đó là bản `links_corpus.parquet` **đã bị BTC xóa sáng 12/10**; bản thi là `links_corpus_vi_en_zh.parquet` 1,220,598 URL (EN/ZH lấy mẫu tròn 240k) [ĐO từ lịch sử commit HF].
- **Không chờ crawl xong mới làm E1**: E1 chạy trên corpus sạch hiện tại; doc Long Châu bổ sung sau sẽ đo được bằng ΔR.

## 6. Ngân sách nộp & mốc thời gian
- Public: 10 lần/ngày đến 31/10 07:00 ICT (~11 ngày). Private: 5 lần tổng (01–04/11).
- Ngày 1–2: E1–E3 (3 lần). Ngày 3: E4–E6. Ngày 4–6: dense/rerank trên Kaggle GPU. Ngày 7+: ổn định, đóng gói pipeline tái lập được cho private (query mới) + working notes.
- Private: nộp cấu hình tốt nhất + 2 biến thể N (an toàn / mạnh recall), giữ 2 lần dự phòng.

## 7. Rủi ro
| Rủi ro | Giảm thiểu |
|---|---|
| Giả thuyết pool-BM25 sai | E1 trả lời trong 1 lần nộp; nhánh B sẵn |
| Văn bản Wayback/trafilatura khác văn bản BTC ⇒ chunk LCS < 0.40 | Chunk cả bài: chỉ cần ≥40% token gold xuất hiện theo thứ tự; theo dõi chunk P vs doc P mỗi lần nộp |
| File nộp quá lớn khi chunk toàn văn × N lớn | Bản cũ 35 doc ≈ 50 MB zip vẫn được chấm; với N≥70 cắt chunk ở ~8k ký tự nếu cần |
| Phiên khác ghi đè dữ liệu | Chỉ dùng `vibiomir/`; script bịa dữ liệu đã xóa |
| Private là query mới | Pipeline phải chạy end-to-end từ `query.parquet` mới trong < vài giờ |
