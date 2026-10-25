# Các model Qwen mới hơn Qwen2.5 (open-weight, <= 15B, phát hành trước 2026-08-01)

Ngày nghiên cứu: 2026-10-25. Nguồn chính: HF API (`/api/models`, `/api/models/<id>`, `config.json` raw), đọc qua WebFetch. Không tải weights, không chạy code từ repo nào.

Quy tắc cuộc thi: open-weight, tổng tham số <= 15B, phát hành trước 2026-08-01.

## 1. Tóm tắt nhanh

- Các model Qwen3.5 dense nhỏ (0.8B, 2B, 4B, 9B và bản Base) đều là model multimodal (`Qwen3_5ForConditionalGeneration`, có `vision_config`), Apache-2.0, tạo trên HF ngày 2026-02-27 đến 2026-02-28. Đây là các ứng viên mạnh nhất còn hợp lệ.
- Qwen3.6: chỉ có 27B dense và 35B-A3B MoE, cả hai vượt 15B tổng tham số, loại.
- Qwen3.7: không có model nào trên HF (API trả về rỗng).
- Qwen3.8: tất cả checkpoint đều tạo từ 2026-08-05 trở đi, vi phạm quy tắc 2026-08-01, KHÔNG ĐƯỢC DÙNG.
- Qwen3 (2025) dense 0.6B/1.7B/4B/8B/14B, Qwen3-2507 (chỉ có 4B được tìm thấy ở kích thước <= 15B), Qwen3-Embedding và Qwen3-Reranker 0.6B/4B/8B: đều hợp lệ, text-only, chạy được với transformers thông thường.

## 2. Bảng ứng viên (<= 15B, hợp lệ)

Cột `createdAt` là ngày tạo repo trên HF (lấy từ API). Ngày phát hành chính thức có thể khác, xem mục 5.

| HF id | Loại | Tổng tham số (safetensors) | createdAt | License | Architecture / model_type / transformers_version | Ghi chú |
|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-9B | Instruct/thinking hybrid, multimodal (image-text-to-text) | 9,653,104,368 (~9.65B) | 2026-02-27 | apache-2.0 | `Qwen3_5ForConditionalGeneration`; model_type `qwen3_5` (text_config `qwen3_5_text`); transformers_version `4.57.0.dev0`; dtype `bfloat16` | Có vision_config. Đây là bản mạnh nhất <= 15B. |
| Qwen/Qwen3.5-9B-Base | Base, multimodal | chưa kiểm chứng (cùng cỡ ~9B) | 2026-02-26 | chưa kiểm chứng | chưa kiểm chứng | Base, không dùng cho chat/JSON trực tiếp. |
| Qwen/Qwen3.5-4B | Instruct, multimodal | 4,659,865,088 (~4.66B) | 2026-02-27 | apache-2.0 | `Qwen3_5ForConditionalGeneration`; `qwen3_5`; 4.57.0.dev0; bfloat16 | Vừa một T4 16GB ở fp16 (~9.3GB weights). Dự phòng tốt. |
| Qwen/Qwen3.5-4B-Base | Base, multimodal | chưa kiểm chứng (~4B) | 2026-02-27 | chưa kiểm chứng | chưa kiểm chứng | Base. |
| Qwen/Qwen3.5-2B | Instruct, multimodal | 2,274,069,824 (~2.27B) | 2026-02-28 | apache-2.0 | `Qwen3_5ForConditionalGeneration`; `qwen3_5`; bfloat16 | Quá yếu cho teacher labels, chỉ phù hợp kiểm thử pipeline. |
| Qwen/Qwen3.5-2B-Base | Base | chưa kiểm chứng | 2026-02-28 | chưa kiểm chứng | chưa kiểm chứng | Base. |
| Qwen/Qwen3.5-0.8B | Instruct, multimodal | 873,438,784 (~0.87B) | 2026-02-28 | apache-2.0 | `Qwen3_5ForConditionalGeneration`; `qwen3_5`; bfloat16 | Quá nhỏ cho JSON tiếng Việt chất lượng. |
| Qwen/Qwen3.5-0.8B-Base | Base | chưa kiểm chứng | 2026-02-28 | chưa kiểm chứng | chưa kiểm chứng | Base. |
| Qwen/Qwen3-14B | Thinking/non-thinking hybrid, text-only | ~14.8B (chưa kiểm chứng qua API; tên gọi 14B) | 2025-04-27 | chưa kiểm chứng (Qwen3 thường là Apache-2.0) | `Qwen3ForCausalLM` (chưa kiểm chứng qua config) | Text-only, không cần `Qwen3_5` support. Fp16 ~29.6GB nên phải chia hai T4 hoặc dùng 4-bit (~9GB). |
| Qwen/Qwen3-8B | Hybrid thinking, text-only | ~8.2B (chưa kiểm chứng qua API) | 2025-04-27 | chưa kiểm chứng | chưa kiểm chứng | Fp16 ~16.4GB, không vừa một T4. Hai T4 hoặc 4-bit. |
| Qwen/Qwen3-4B-Instruct-2507 | Instruct (non-thinking), text-only | 4,022,468,096 (~4.02B) | 2025-08-05 | apache-2.0 | `Qwen3ForCausalLM`; library transformers | Qwen3-2507. Fp16 ~8GB, vừa một T4. |
| Qwen/Qwen3-4B-Thinking-2507 | Thinking, text-only | chưa kiểm chứng (~4B) | 2025-08-05 | chưa kiểm chứng | chưa kiểm chứng | Qwen3-2507. Output dài, chậm cho batch teacher labels. |
| Qwen/Qwen3-4B | Hybrid thinking, text-only | chưa kiểm chứng (~4B) | 2025-04-27 | chưa kiểm chứng | chưa kiểm chứng | Bản gốc, đã có 2507. |
| Qwen/Qwen3-1.7B, Qwen3-0.6B | Hybrid thinking, text-only | chưa kiểm chứng | 2025-04-27 | chưa kiểm chứng | chưa kiểm chứng | Quá nhỏ cho bài toán này. |
| Qwen/Qwen3-Embedding-0.6B | Embedding (feature-extraction, sentence-transformers) | 595,776,512 (~0.6B) | 2025-06-03 | apache-2.0 | `Qwen3ForCausalLM`; library sentence-transformers | Embedding, không sinh text. |
| Qwen/Qwen3-Embedding-4B | Embedding | chưa kiểm chứng (~4B) | 2025-06-03 | chưa kiểm chứng | chưa kiểm chứng | Embedding. |
| Qwen/Qwen3-Embedding-8B | Embedding | chưa kiểm chứng (~8B) | 2025-06-03 | chưa kiểm chứng | chưa kiểm chứng | Embedding. |
| Qwen/Qwen3-Reranker-0.6B | Reranker (text-ranking) | 595,776,512 (~0.6B) | 2025-05-29 | apache-2.0 | `Qwen3ForCausalLM`; library transformers | Reranker, chấm (query, doc) bằng logit yes/no. |
| Qwen/Qwen3-Reranker-4B | Reranker | chưa kiểm chứng (~4B) | 2025-06-03 | chưa kiểm chứng | chưa kiểm chứng | Reranker. |
| Qwen/Qwen3-Reranker-8B | Reranker | chưa kiểm chứng (~8B) | 2025-05-29 | chưa kiểm chứng | chưa kiểm chứng | Reranker, phù hợp làm teacher relevance. |
| Qwen/Qwen3-VL-Embedding-2B | Embedding multimodal | 2,127,532,032 (~2.1B) | 2026-01-07 | apache-2.0 | `Qwen3VLForConditionalGeneration`; sentence-transformers | Multimodal, không cần cho text. |
| Qwen/Qwen3-VL-Embedding-8B | Embedding multimodal | chưa kiểm chứng (~8B) | 2026-01-07 | chưa kiểm chứng | chưa kiểm chứng | Multimodal. |
| Qwen/Qwen3-VL-Reranker-2B / -8B | Reranker multimodal | chưa kiểm chứng | 2026-01-07 | chưa kiểm chứng | chưa kiểm chứng | Multimodal, không cần cho text. |
| Qwen/Qwen3Guard-Gen-4B / -8B | Guard (classifier an toàn) | chưa kiểm chứng | 2025-09-23 | chưa kiểm chứng | chưa kiểm chứng | Không phải LLM tác vụ chung. |
| Qwen/Qwen3-ASR-0.6B / -1.7B, Qwen3-TTS-* | ASR / TTS | chưa kiểm chứng | 2026 | chưa kiểm chứng | chưa kiểm chứng | Không liên quan. |

## 3. Bị loại (vi phạm quy tắc)

**Vi phạm ngày 2026-08-01, KHÔNG ĐƯỢC DÙNG:**

| HF id | createdAt | Lý do |
|---|---|---|
| Qwen/Qwen3.8-27B (và -FP8) | 2026-08-05 / 2026-08-13 | Tạo sau 2026-08-01 |
| Qwen/Qwen3.8-2.4T-A95B (và -FP8) | 2026-08-08 | Tạo sau cutoff, và 2.4T >> 15B |
| Qwen/Qwen3.8-Flash-Next (và -FP8) | 2026-08-24 | Tạo sau cutoff |

**Vượt 15B tổng tham số:**

| HF id | Tổng tham số | createdAt |
|---|---|---|
| Qwen/Qwen3.6-27B (và -FP8) | ~27B | 2026-04-21 |
| Qwen/Qwen3.6-35B-A3B (và -FP8) | 35,951,822,704 (~36B, 3B active) | 2026-04-15 |
| Qwen/Qwen3.5-27B, -35B-A3B, -122B-A10B, -397B-A17B (và FP8, GPTQ-Int4) | 27B đến 397B | 2026-02-16 đến 2026-03-03 |
| Qwen3-Next-80B-A3B, Qwen3-Coder-Next, Qwen3-VL-30B/32B/235B, Qwen3-30B-A3B, Qwen3-32B, Qwen3-235B-A22B | >15B | 2025 đến 2026 |

**Không tìm thấy:** Qwen3.7 (API trả về rỗng).

## 4. Phiên bản transformers

- Config của Qwen3.5 ghi `transformers_version: 4.57.0.dev0`, đó là bản lúc lưu, không phải yêu cầu tối thiểu. Yêu cầu tối thiểu để load `qwen3_5` chưa kiểm chứng.
- Kaggle docker-python, file `kaggle_requirements.txt` trên nhánh main (GitHub raw): `transformers>=5.0.0`. Đây là ràng buộc dưới, phiên bản thực tế đang cài trong image chưa kiểm chứng (cần chạy `import transformers; print(transformers.__version__)` trong kernel).
- Kết luận: image Kaggle hiện tại nhiều khả năng đã đủ mới để load Qwen3.5 và Qwen3-2507 (cần xác minh bằng một kernel test).

## 5. Lưu ý về ngày

- HF createdAt của Qwen3.5 là 2026-02-27/28. Một nguồn tin (Artificial Analysis, Gigazine) ghi ngày công bố chính thức là 2026-03-02/03. Dù lấy ngày nào, cả hai đều trước 2026-08-01, nên không ảnh hưởng quyết định.
- Ngày phát hành chính thức của Qwen3-2507 (2025-08-05) lấy theo createdAt, chưa kiểm chứng với blog.

## 6. Khuyến nghị

**Lưu ý chung trên T4 (không hỗ trợ bf16):** Qwen3.5 khai báo `dtype: bfloat16`. Chạy fp16 trên T4 có rủi ro tràn số (overflow) ở một số lớp. Phải kiểm tra thực nghiệm trước khi chốt, chưa kiểm chứng.

**(a) Phân tích query JSON tiếng Việt (entities, aliases, sub-questions, HyDE):**
- Ứng viên chính: **Qwen/Qwen3.5-9B**, dùng 4-bit (khoảng 5-6GB, vừa một T4) hoặc fp16 chia hai T4 (~19.3GB weights).
- Dự phòng: **Qwen/Qwen3.5-4B** ở fp16 (~9.3GB, vừa một T4, dễ chạy và nhanh hơn).
- Tắt thinking mode để giảm độ dài output khi sinh JSON.

**(b) Relevance judging (teacher labels cho cặp query, doc):**
- Ứng viên chính: **Qwen/Qwen3-Reranker-8B** nếu chỉ cần điểm relevance, vì đây là model chuyên cho việc chấm cặp (query, doc), tổng ~8B, fp16 ~16GB chia hai T4 hoặc 4-bit.
- Lựa chọn thay thế (LLM sinh nhãn): **Qwen/Qwen3.5-9B** đã nêu ở (a), dùng cùng pipeline.
- Qwen3-14B (text-only, ổn định hơn về tương thích) là phương án dự phòng nếu Qwen3.5 gặp vấn đề fp16 hoặc transformers.

**Fit trên 2x T4 16GB:**
- fp16: 9B cần chia hai GPU (~19.3GB weights + KV cache). 4B vừa một GPU. 14B và 8B cần chia hai GPU hoặc dùng 4-bit.
- 4-bit (bitsandbytes): 9B và 14B đều vừa.

**Kết luận:** Dùng Qwen3.5-9B cho (a) và (b) nếu thử fp16 ổn; nếu không, dùng Qwen3.5-4B fp16 cho (a) và Qwen3-Reranker-8B cho (b).

## 7. Việc chưa kiểm chứng

- License và tham số chính xác của nhiều biến thể Qwen3 2025 (14B, 8B, 4B-Thinking-2507, Embedding-4B/8B, Reranker-4B/8B, Base variants của Qwen3.5).
- Yêu cầu transformers tối thiểu để load `qwen3_5`.
- Phiên bản transformers đang cài trong image Kaggle (chỉ biết ràng buộc `>=5.0.0` trong file requirements).
- Chất lượng tiếng Việt của từng model (chưa đánh giá, không có benchmark trong nghiên cứu này).
- Hành vi fp16 trên T4 của Qwen3.5 (có tràn số không).

Nguồn đã dùng: HF API (`huggingface.co/api/models?...`, `huggingface.co/api/models/Qwen/...`, `huggingface.co/Qwen/.../raw/main/config.json`), `raw.githubusercontent.com/Kaggle/docker-python/main/kaggle_requirements.txt`, WebSearch về Qwen3.5 small (Artificial Analysis, Gigazine, rits.shanghai.nyu.edu).
