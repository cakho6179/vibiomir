# Decision model kiểu Jev bằng model hợp lệ: nghiên cứu và cách fine-tune (25/10/2026)

Nhãn: **[LUẬT]** luật cuộc thi · **[ĐO]** số đo · **[SUY]** suy luận cần kiểm. Tra web lỗi tạm thời trong phiên này ⇒ ngày phát hành
lấy từ `research/reports/vi_retrieval_models_2026-10-22.md`; dòng nào chưa xác minh có ghi rõ.

## 1. Ràng buộc
- [LUẬT] Model mở, phát hành trước 01/08/2026 (ICT), ≤ 15B. Được dùng dữ liệu ngoài nếu ghi nguồn. BTC có quyền dùng code và dữ liệu sinh ra.
- ⇒ Các model trong bài tham khảo ra sau 08/2026 (Clef-flash, Laya, Kev, AnyJev/Tacit) **không được nằm trong hệ thống**. Chỉ học **phương pháp** của chúng.
- Fine-tune một base hợp lệ là hợp lệ: model sau fine-tune là tài sản của ta. Cần công bố dữ liệu train và cách sinh trong working notes.

## 2. Decision model làm gì trong hệ thống
| Vị trí | Câu hỏi quyết định | Đầu ra |
|---|---|---|
| L3 chọn doc | "Doc/section này có thuộc gold của query không?" | P(yes) đã hiệu chỉnh ⇒ thêm doc nếu P > F2_q/5 |
| L3 độ sâu | "Query này có nhiều gold không?" (ước \|G_q\|) | Score ⇒ ngưỡng cắt theo query |
| Chunk | "Section nào của doc là đoạn gold?" | Choice giữa các section |
| L1 query | Thực thể chính, ý hỏi (8 loại), các câu hỏi con | JSON có cấu trúc |

Điểm mấu chốt (bài học #6769): reranker "đúng nghĩa liên quan" (r1) **xếp kém hơn** w12 [ĐO]. Gold được sinh bởi pipeline của BTC
(pool từ hệ truy hồi rồi lọc), không phải bởi một khái niệm liên quan chung chung [SUY]. ⇒ Decision model phải được
**hiệu chỉnh theo gold thật** (qua probe LB tổng hợp), không chỉ theo nhãn tự sinh.

## 3. Base model hợp lệ (ứng viên)
| Model | Cỡ | Phát hành | Vai trò | Ghi chú |
|---|---|---|---|---|
| BAAI/bge-reranker-v2-m3 | 0,57B | 2024 | Student cross-encoder (P(yes) qua sigmoid) | Đang dùng (r1). Full fine-tune fp16 vừa T4 |
| AITeamVN/Vietnamese_Reranker | ~0,57B | ~05/2025 (chưa xác minh ngày chính xác) | Như trên, đã thích nghi tiếng Việt | Kiểm zero-shot trước |
| Qwen/Qwen3-Reranker-0.6B / 4B | 0,6B / 4B | 06/2025 | Decision yes/no theo logit token "yes" (đúng kiểu Noul/Score) | 4B: LoRA/QLoRA trên T4; 8B fp16 không vừa T4 |
| Qwen2.5-7B-Instruct / Qwen2.5-14B-Instruct | 7B / 14B | 09/2024 | Teacher (LLM judge), query understanding | 7B đã chạy trên Kaggle; 14B cần 4-bit trên T4 |
| Qwen3-4B / 8B / 14B | ≤ 14B | 04/2025 | Teacher hoặc decision head + LoRA | Có chế độ non-thinking cho đầu ra ngắn |
| BAAI/bge-m3, AITeamVN/Vietnamese_Embedding | 0,57B | 2024 / ~05/2025 | Fine-tune embedding để **tăng recall L1** | Hợp với ưu tiên recall |
| XLM-R, mDeBERTa-v3 | ≤ 0,6B | 2019–2021 | Classifier phụ (ý hỏi, \|G_q\|) | Rẻ, dùng nếu cần |

(Qwen3.5: chưa xác minh ngày phát hành ⇒ chưa dùng cho tới khi kiểm được.)

## 4. Dữ liệu huấn luyện (không có nhãn từng cặp từ BTC)
1. **LLM-as-judge distillation** (chính):
   - Teacher Qwen2.5-14B hoặc Qwen3-14B (4-bit) chấm (query, section) trên top-K ứng viên của **1.200 query thật**, theo thang 0–3, kèm P(yes) từ logit.
   - Đây là nhãn do model sinh, không phải gold. Cần nêu trong working notes: transductive, không dùng nhãn BTC.
2. **Sinh query tổng hợp (InPars / Promptagator)**:
   - Qwen2.5-7B viết câu hỏi kiểu bệnh nhân (dài, nhiều ý, như query thật) từ các section của corpus.
   - Positive = section nguồn cộng các section được teacher chấm ≥ 2.
   - Hard negative = top BM25/M3 bị teacher chấm 0.
   - Dùng để fine-tune **embedding** (recall) và reranker.
3. **Hiệu chỉnh theo gold (aggregate)**:
   - Chia điểm student thành các bin, rồi nộp probe chỉ-doc cho từng bin (docs = các cặp thuộc bin, chunk giữ nguyên) ⇒ Docs P của bin = mật độ gold thật.
   - Fit isotonic điểm → P(gold). Khoảng 5–6 lượt nộp, chỉ dùng số tổng hợp.

## 5. Cách fine-tune (chạy trên Kaggle T4 16 GB)
| Việc | Phương pháp | Cấu hình gợi ý | Chi phí ước [SUY] |
|---|---|---|---|
| Student reranker | bge-reranker-v2-m3 full fine-tune, loss BCE với nhãn mềm của teacher + listwise (softmax trên 1 pos + n neg) | fp16, max_len 512, lr 1e-5–2e-5, 1 epoch, batch 16 × grad-acc 4 | ~200k cặp ≈ 2–3 h T4 |
| Decision LLM | Qwen3-Reranker-4B QLoRA (4-bit NF4, LoRA r 16, alpha 32, target q/k/v/o), loss CE trên token yes/no | max_len 1024, gradient checkpointing | ~50k cặp ≈ 4–6 h T4 |
| Embedding (recall) | bge-m3 contrastive (InfoNCE, in-batch + 3–7 hard neg), query tổng hợp | fp16, max_len 512, lr 1e-5, 1 epoch | ~100k cặp ≈ 3–4 h; phải embed lại corpus (~8 h) |
| Hiệu chỉnh | temperature/Platt trên tập teacher giữ lại, rồi isotonic theo probe LB | — | CPU |

Thư viện: FlagEmbedding / sentence-transformers (CrossEncoder, SentenceTransformer trainer), peft + bitsandbytes cho QLoRA, vLLM nếu cài được trên Kaggle (teacher nhanh hơn).

**Chỉ số đánh giá:**
- Offline trên tập teacher giữ lại: nDCG@50, ECE/Brier, selective accuracy theo ngưỡng tin cậy.
- Online (quyết định cuối): probe LB chỉ-doc ở cùng N_q với base.

## 6. Thứ tự đề xuất (thí nghiệm trước, đầu tư sau)
1. **Kiểm độ khớp với gold (zero-shot, 1 lượt nộp).** Qwen3-Reranker-4B (hoặc Qwen2.5-14B judge) chấm top-200 của w12 cho 1.200 query. Probe: top-N_q theo P(yes) (cùng N_q như #6751).
   - Docs P > .2495 ⇒ judge kiểu LLM khớp gold hơn w12 ⇒ đi tiếp bước 2.
   - Ngược lại ⇒ không fine-tune reranker; dồn sức vào recall (L1) và tái tạo pool BTC.
2. **Distill** teacher → student 0,57B (để hệ thống chạy nhanh trên máy BTC), rồi hiệu chỉnh isotonic bằng probe bin.
3. **Fine-tune embedding** bằng query tổng hợp, nếu probe "gold mới" của các phương pháp truy vấn cho thấy dense còn thiếu recall.
4. Gói vào entry point (`docs/SYSTEM_AUDIT_2026-10-25.md` mục 6): stage `judge` (teacher, tuỳ chọn) và `decide` (student + isotonic + ngưỡng F2).

**Rủi ro:**
- GPU: các dự án khác đang giữ slot. Qwen3-Reranker-4B trên 240k cặp ≈ 6 h ⇒ chia nhiều tài khoản.
- Thời gian: còn 6 ngày tới hết public.
- Lệch gold: đã thấy ở r1 ⇒ bước 1 là cổng quyết định, chưa fine-tune gì trước khi có số.
