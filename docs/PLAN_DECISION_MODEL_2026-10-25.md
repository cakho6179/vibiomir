# Kế hoạch huấn luyện decision model (backbone Qwen), 25/10/2026

Nhãn: **[LUẬT]** luật cuộc thi · **[ĐO]** số đo · **[SUY]** suy luận cần kiểm.
Nối tiếp `docs/RESEARCH_DECISION_FINETUNE_2026-10-25.md` (giữ nguyên phần loss và hiệu chỉnh), nhưng sửa 3 điểm:
1. Có cổng kiểm trước khi tốn GPU.
2. Backbone là họ Qwen3-Reranker.
3. Đặc trưng "pool BTC" là đầu vào chính.

## 1. Sự thật quyết định thiết kế
| # | Sự thật | Hệ quả |
|---|---|---|
| F1 | [ĐO #6769] Reranker "liên quan đúng nghĩa" (r1, bge-reranker-v2-m3) xếp **kém** w12 ở cùng N_q | Nhãn liên quan ≠ nhãn gold. Distill thuần văn bản có thể lặp lại thất bại r1 ⇒ phải kiểm trước (G0) |
| F2 | [ĐO] Admin BM25: P .88 / R .515 ở ~28–41 doc; BM25 của ta P .18. **Đo theo luật chấm cũ (30/09)**, chưa biết theo luật mới | Gold nhiều khả năng đi theo pool truy hồi của BTC. Decision model cần đặc trưng pool (hạng BM25 theo section, số họ đồng thuận, khớp thực thể), không chỉ văn bản |
| F3 | [ĐO] 1.200 query trong `query.parquet`; public chấm 600, private chấm **600 còn lại của cùng file** (`docs/ROADMAP_SYSTEM.md`) | Gán nhãn teacher trên cả 1.200 query × ứng viên = huấn luyện đúng phân phối test (transductive, không dùng nhãn BTC). Query tổng hợp (InPars) chỉ là tuỳ chọn |
| F4 | [LUẬT] Model mở, ≤ 15B, ra trước 01/08/2026. [ĐO HF] Qwen3-Reranker 0.6B/4B/8B (05–06/2025), Qwen3-8B/14B, Qwen3.5-9B/4B (27/02/2026) hợp lệ | Teacher và student đều thuộc họ Qwen. Teacher ra sau 01/08 chỉ cân nhắc nếu BTC trả lời là được |
| F5 | [ĐO] Không có nhãn từng cặp; chỉ có số tổng hợp từ LB (P/R theo lượt nộp) | Hiệu chỉnh và trọng số fusion phải học qua probe tầng (bin), mỗi probe là 1 lượt nộp |

## 2. Kiến trúc
```
ứng viên (union/fusion, top-200/q)
   │  chọn đơn vị: section ≤ 4,8k ký tự khớp query nhất (bge-m3 hoặc BM25), cắt ≤ 1.024 token
   ▼
[Student Qwen3-Reranker, đầu yes/no]  ──►  logit_text(q, section)       (+ logit theo từng câu hỏi con → max/mean)
   ▼
[L2 fusion, ít tham số]  logit_text + hạng từng nguồn + n_families + hạng BM25-section + khớp thực thể + cỡ cụm gần trùng
   ▼
[Isotonic theo probe bin]  ⇒  P(gold | q, d)
   ▼
[L3 cắt theo query]  giữ d nếu P > F2*_q / 5;  |G_q| ≈ Σ P   ⇒ relevant_docs
   ▼
[Chunk]  student chấm các section của doc đã chọn ⇒ section tốt nhất (+ chunk thêm trong ngân sách 100 MB)
```
- **Backbone student:**
  - **Qwen3-Reranker-0.6B**: bản chạy trên máy BTC, nhanh, full fine-tune.
  - **Qwen3-Reranker-4B** (QLoRA): bản chính xác hơn, dùng nếu máy BTC đủ GPU.
  - Cả hai là Qwen3 dense, attention chuẩn, nên chạy fp16 sạch trên T4. Sẵn đầu quyết định yes/no: P(yes) = softmax(logit "yes", "no") ở token cuối.
- **Teacher:** Qwen3-Reranker-8B (fp16 chia 2 × T4). Qwen3.5-9B chỉ được làm teacher nếu qua kiểm "đầu vào dài": tốc độ, không NaN ở fp16, đồng thuận với Reranker-8B trên ~300 cặp. Smoke kernel hiện tại mới đo JSON trên query ngắn, chưa nói gì về vai trò teacher.
- **Instruction của reranker** (Qwen3-Reranker nhận instruction tuỳ biến):
  - Viết theo đúng tiêu chí gold: doc thuộc nhóm tài liệu cùng bệnh/chủ đề và đề cập ít nhất một ý được hỏi.
  - Một instruction cố định cho cả teacher và student.

## 3. Các cổng (không vượt cổng thì không tốn GPU bước sau)
| Cổng | Việc | Lượt nộp | Đạt khi | Không đạt thì |
|---|---|---|---|---|
| **G0** | Qwen3-Reranker-4B zero-shot chấm top-200 fusion × 1.200 query; ghi tokens/s | 2 | (a) **Thay thế:** top-N_q theo P(yes), cùng N_q như #6751 ⇒ Docs P > .2495. (b) **Phần lề:** chỉ doc judge thêm mà top-N_q của w12 không có ⇒ mật độ > ~6,7% | Cả hai trượt ⇒ không distill văn bản thuần; dồn sức vào đặc trưng pool (G0b) và recall |
| G0b | Khi kernel BM25-section (phiên khác) xong: đo P của BM25-section ở ~41 doc/q | 1 | P tiến gần .88 | Gold không theo BM25-section ⇒ tìm đơn vị/tách từ khác của BTC |
| G1 | Teacher 8B gán nhãn → student 0.6B; offline trên 10% query giữ lại (nDCG@50, đồng thuận với teacher) và 1 probe thay thế | 1 | Student ≥ judge zero-shot ở G0 | Dùng 4B QLoRA, hoặc giữ judge zero-shot làm đặc trưng |
| G2 | Hiệu chỉnh: 5 probe bin theo điểm L2 (chỉ đổi doc, giữ chunk) ⇒ isotonic | 5 | Mật độ đơn điệu theo bin | Gộp bin, dùng luật cắt cũ (logit ≥ ngưỡng) |
| G3 | Bài nộp đầy đủ từ entry point `python -m r2ai2026 run` | 1–2 | FINAL > .2805 (#6935) | Lùi về cấu hình tốt nhất có từng stage |

Tổng khoảng 10–12 lượt trong 26–31/10, mỗi lượt trả lời một câu hỏi.

## 4. Dữ liệu huấn luyện
- **Cặp:** 1.200 query × top-200 ứng viên fusion ≈ 240k cặp. Đơn vị là section khớp nhất của doc (≤ 1.024 token). Doc trùng cụm thì chấm một lần.
- **Nhãn teacher:** P(yes) mềm của Qwen3-Reranker-8B, chấm theo query gốc và theo từng câu hỏi con của khung query (max làm nhãn chính).
- **Hard negative:** sẵn trong top-200 (doc teacher chấm thấp). Thêm 20 doc ngẫu nhiên cùng host/chủ đề cho mỗi query để student thấy cả vùng dễ.
- **Chia:** theo query, 90% train / 10% giữ lại để đo offline. Không dùng nhãn BTC từng cặp [LUẬT nội bộ].
- **Tuỳ chọn:** query tổng hợp (Qwen3.5 viết câu hỏi kiểu bệnh nhân từ section) nếu cần thêm dữ liệu cho bước fine-tune embedding (recall), không bắt buộc cho decision model (F3).
- **Ghi nguồn trong working notes:** nhãn do model hợp lệ sinh, transductive trên query không nhãn.

## 5. Công thức huấn luyện (Kaggle T4, không có bf16)
| Model | Cách | Cấu hình | Loss |
|---|---|---|---|
| Qwen3-Reranker-0.6B | full fine-tune, AMP fp16 + loss scaling | max_len 1.024, lr 2e-5, warmup 5%, 1–2 epoch, batch 8 × grad-acc 4, gradient checkpointing | BCE(nhãn mềm teacher) + KL listwise trên nhóm 16 ứng viên cùng query (λ = 1) |
| Qwen3-Reranker-4B | QLoRA NF4 (bitsandbytes), LoRA r 16, α 32, dropout .05, target q/k/v/o/gate/up/down; adapter fp32 | max_len 1.024, lr 1e-4, batch 2 × grad-acc 16, gradient checkpointing | như trên |
- Chỉ lấy logit 2 token "yes"/"no" ở vị trí cuối. Không thêm head mới, nên giữ được tri thức reranker gốc.
- Chi phí: **đo trong kernel G0** (tokens/s) rồi mới chia việc cho các tài khoản. Ước sơ bộ [SUY]: 240k × ~1,2k token ≈ 290M token prefill.
- Thư viện: transformers ≥ 5 (Kaggle), peft, bitsandbytes. Kiểm phiên bản trong log smoke trước.

## 6. L2 fusion và L3 cắt (đặc trưng pool là đầu vào chính, F2)
- **Điểm L2:** `z = a·logit_student + b·rrf_fusion + c·n_families + d·bm25_section_rank_feat + e·entity_match`, với ≤ 5 tham số.
- **Trọng số:** chọn trên lưới nhỏ bằng 2–3 probe "phần lề" (mật độ gold của phần mỗi cấu hình thêm vào). Không fit trên nhãn teacher, vì nhãn teacher không biết pool.
- **Isotonic:** z → P(gold) từ 5 probe bin (G2).
- **Cắt theo query:** sắp theo P, tính F2 kỳ vọng với |G_q| ≈ Σ P, giữ tiền tố có F2 kỳ vọng lớn nhất. Đây là bản tổng quát của luật P > F2/5.
- **Chunk:** section có điểm student cao nhất trong doc (đúng đơn vị ≤ 4,8k như gold, D2 trong audit).

## 7. Lịch (còn 6 ngày public)
| Ngày | Việc | GPU |
|---|---|---|
| 26/10 | Kernel G0: Qwen3-Reranker-4B zero-shot, 240k cặp chia 2–3 tài khoản. Đọc log smoke (Qwen3.5 fp16, fla, phiên bản transformers). Nộp 2 probe G0 | ~6–10 h T4 |
| 26–27/10 | Nếu G0 đạt: teacher 8B gán nhãn (song song nhiều tài khoản). G0b khi BM25-section có | ~15–20 h T4 |
| 27–28/10 | Train student 0.6B (và 4B QLoRA nếu còn slot). Probe G1 | ~4–8 h |
| 29/10 | 5 probe bin (G2) + 2–3 probe trọng số L2 | CPU |
| 30–31/10 | Gói stage `judge` (tuỳ chọn) + `decide` vào entry point; nộp G3 từ entry point | ~2 h |
| 01–04/11 | Private 5 lượt; kiểm tái lập; working notes | — |

## 8. Chạy trên máy BTC
- Stage `decide` mặc định dùng student 0.6B trên top-300/q (≈ 360k cặp). Tốc độ thật lấy từ kernel G0/G1. Có tuỳ chọn 4B nếu máy BTC đủ GPU.
- Teacher không chạy trên máy BTC: chỉ dùng offline để tạo dữ liệu train, có script tái tạo đi kèm.
- Cần hỏi BTC cấu hình máy để chọn mặc định.

## 9. Rủi ro
- F1 lặp lại (nhãn liên quan ≠ gold) ⇒ G0 chặn trước. L2 dùng đặc trưng pool để bù.
- Slot GPU bị dự án khác và phiên khác chiếm ⇒ chia nhỏ kernel theo shard query, đẩy vào tài khoản có slot trống.
- Lượt nộp có hạn: 10/ngày public, 5 tổng private ⇒ mỗi probe phải trả lời một câu hỏi.
- fp16 trên T4: Qwen3 dense thì ổn [SUY]; Qwen3.5 chưa kiểm ⇒ không dùng làm student.

## 10. Điều chỉnh theo paper arXiv 2610.27099 (System One cho hồ sơ lâm sàng, MGB, 19/10/2026)
Tóm tắt: `research/reports/paper_2610.27099_system1_medical.md`.
- Student CS1-4B: full fine-tune từ Decision-2.0-Nox-4B (nền Qwen3.5-4B) trên 12,4M cặp.
- Nhãn mềm do GLM-5.3-Flash (320B) sinh, loss soft cross-entropy T = 1.
- Trial screening: xếp hạng bằng **tích xác suất của từng tiêu chí** ⇒ nDCG@20 .64 so với BM25 .11.
- Model của paper không dùng được (ra sau 01/08); ta chỉ lấy phương pháp.

| Ý của paper | Áp vào hệ thống | Thay đổi kế hoạch |
|---|---|---|
| Nhãn mềm, soft CE, T = 1, không làm sắc nhãn | Đã có (BCE nhãn mềm). Giữ T = 1, không ngưỡng hoá nhãn teacher | Không đổi |
| Tách quyết định theo tiêu chí, kết hợp xác suất | Hỏi 2 câu yes/no cho mỗi (q, section) thay cho 1 câu "liên quan": (A) "nói về đúng bệnh/chủ đề chính của câu hỏi?" và (B) "đề cập ý được hỏi?" (theo từng câu hỏi con từ khung query). Điểm = P(A) · max_k P(B_k), đồng thời đưa P(A) và P(B) làm đặc trưng riêng vào L2 | Teacher và student học cả hai loại câu hỏi (instruction khác nhau, cùng model). Probe G0 so cả "1 câu" và "A·B" (vẫn 2 lượt: dùng biến thể tốt hơn theo offline cho probe thay thế) |
| Lựa chọn "không có thông tin" (abstain) | Ở L3: query mà max P thấp ⇒ chỉ nộp phần lõi (top theo fusion), không mở rộng | Thêm luật này vào L3 |
| Coverage tại precision cố định | Ngưỡng theo F2 kỳ vọng từng query (đã có), không dùng ngưỡng cứng 0,95 | Không đổi |
| Hiệu chỉnh lại theo nhóm (dịch log-odds) | Nhóm query theo loại ý hỏi/độ dài; khi probe bin đủ, cho mỗi nhóm một độ dịch log-odds riêng | Chỉ làm nếu G2 còn lượt |
| Teacher ≫ student (320B → 4B) | Teacher ta bị chặn ≤ 15B ⇒ dùng **ensemble teacher** (Qwen3-Reranker-8B + Qwen3-14B hoặc Qwen3.5-9B làm judge, trung bình log-odds) để bù chất lượng nhãn | Thêm 1 teacher nếu G0 đạt và còn GPU |
| Backbone Qwen3.5-4B full fine-tune | Ứng viên student thứ ba, chỉ khi smoke + kiểm đầu vào dài cho thấy fp16 ổn trên T4 | Giữ Qwen3-Reranker-0.6B/4B làm mặc định |
| Train nhiều giai đoạn, re-anneal LR | Giai đoạn 1: cặp top-200 thật; giai đoạn 2: cặp khó (teacher P trong .3–.7) | Tuỳ chọn |
