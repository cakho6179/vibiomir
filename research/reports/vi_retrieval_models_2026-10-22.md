# Nghiên cứu mô hình retrieval tiếng Việt và đa ngôn ngữ (2026-10-22)

Phạm vi: embedding dense, reranker, sparse/lexical và bộ tách từ tiếng Việt cho bài toán truy xuất câu hỏi bệnh nhân tiếng Việt (1.200 câu hỏi) trên ~740k trang web y tế tiếng Việt, cùng ~240k tài liệu tiếng Trung và ~240k tài liệu tiếng Anh. Ràng buộc: mô hình mở trọng số, tối đa 15B tham số, phát hành trước 2026-08-01, chạy trên Kaggle T4 16GB.

Quy ước: "không thấy" nghĩa là không tìm được số liệu trong nguồn đã mở. Mọi con số đều lấy từ model card hoặc paper nêu trong mục Nguồn. Các số ước lượng về throughput trên T4 là ước lượng của tôi, không có nguồn, và được ghi rõ trong phần khuyến nghị.

Lưu ý quan trọng khi đọc bảng: các con số không so sánh trực tiếp được với nhau. Legal Zalo 2021 (MRR@10 trên tập dev, tự báo cáo trong model card của AITeamVN), MMARCO-VI (dev set, tự báo cáo trong model card ViRanker), và VN-MTEB (bảng retrieval trung bình của paper VN-MTEB) dùng protocol khác nhau. Không có nguồn nào được tìm thấy đo trực tiếp trên dữ liệu y tế tiếng Việt cho các mô hình trong bảng.

---

## Nhóm 1: Dense embedding

Chỉ có một bảng so sánh chung trên tiếng Việt: VN-MTEB (arXiv 2507.21500, Bảng 3, cột "Retr." = điểm retrieval trung bình, metric chính của bảng). Thứ hạng dưới đây lấy theo cột này.

| Hạng | Mô hình (HF id) | Tham số | Max token | Phát hành | License | Điểm VN-MTEB retrieval (Retr. avg, %) | Điểm khác (tiếng Việt) | Prompt | T4 fp16 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | gte-Qwen2-7B-instruct (Alibaba-NLP) | 7B | không thấy | không thấy | không thấy (chưa xác minh) | 46.05 | không thấy | có (instruct) | 7B fp16 ≈ 15GB, sát giới hạn, cần 4-bit hoặc không chạy được ổn định |
| 2 | gte-Qwen2-1.5B-instruct (Alibaba-NLP) | 1.5B | không thấy | không thấy | không thấy (chưa xác minh) | 42.01 | không thấy | có (instruct) | ~3GB fp16, khả thi |
| 3 | e5-Mistral-7B-instruct | 7B | không thấy | không thấy | không thấy (chưa xác minh) | 41.73 | không thấy | có (instruct) | ~14-15GB fp16, sát giới hạn |
| 4 | multilingual-e5-large-instruct (intfloat) | không thấy trên card (~0.56B, cùng họ XLM-R large) | 512 | 2024 (arXiv 2402.05672) | MIT | 40.88 | Card không có số VN | `Instruct: {task}\nQuery: {query}`, doc không cần prefix | khả thi |
| 5 | **bge-m3 (BAAI/bge-m3)** | 0.57B (suy ra từ Safetensors/F32 ~2.27GB, chưa xác nhận trực tiếp trên card) | 8192 | Feb 2024 (arXiv 2402.03216) | MIT | 39.84 | Zalo legal MRR@10 0.6822 (AITeamVN card, số tự báo cáo); MRR@10 0.6822 | không cần prompt | khả thi, ~2-3h embed 740k doc (ước lượng, xem phần khuyến nghị) |
| 6 | gte-multilingual-base (Alibaba-NLP) | 0.305B | 8192 | không thấy trên trang | Apache-2.0 | 38.38 | Card: "hơn 70 ngôn ngữ", không xác nhận được tiếng Việt trong danh sách | không cần prompt | rất nhẹ, khả thi |
| 7 | m-e5-large (intfloat) | không thấy | 512 | 2024 | MIT | 37.65 | không thấy | không cần prefix ("query:"/"passage:" theo chuẩn e5, chưa đối chiếu) | khả thi |
| 8 | halong_embedding (hiieu) | 0.3B (từ base multilingual-e5-base) | 512 | 2024 | Apache-2.0 | 34.45 | Zalo legal 20% sample: NDCG@10 0.8976, MRR@10 0.8799 (số của card, tập đánh giá tự chọn) | không cần prompt | khả thi |
| 9 | AITeamVN/Vietnamese_Embedding | 0.6B (F32, Safetensors) | 2048 | Collection cập nhật 06/05/2025 (ngày phát hành chính xác không thấy) | Apache-2.0 | **34.18** | Zalo legal MRR@10 0.8181 (tự báo cáo, cùng bộ dữ liệu train Zalo); BGE-M3 cùng bộ 0.6822 | không cần prompt; dot-product, dim 1024 | khả thi |
| 10 | bkai-foundation-models/vietnamese-bi-encoder | 0.1B (PhoBERT-base-v2) | 256 | arXiv 2403.01616, 03/2024 | Apache-2.0 | 25.37 | Zalo legal 20% MRR@10 0.7951 (bảng của paper VN; card gốc ghi 0.7951 trong bảng so sánh AITeamVN) | **bắt buộc word-segment (pyvi/underthesea/RDRSegmenter)** trước khi encode | rất nhẹ |
| - | dangvantuan/vietnamese-embedding (PhoBERT) | 0.1B (card), 135M (bảng card) | 512 | không thấy | Apache-2.0 | không có trong VN-MTEB | Chỉ có STS (Pearson mean 84.21, Spearman mean 82.45) trên STS vi, không phải retrieval | pyvi `ViTokenizer.tokenize()` trước khi encode | rất nhẹ, nhưng yếu về retrieval |

Các mô hình Qwen3-Embedding và jina-v3 không được VN-MTEB đánh giá:

| Mô hình (HF id) | Tham số | Max token | Phát hành | License | Số liệu đã xác minh | Số liệu tiếng Việt | Prompt | T4 fp16 |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | 0.6B | 32K | paper 05/06/2025 | Apache-2.0 | MTEB multilingual mean (task) 64.33 | không thấy (Vietnamese không được nêu trong 100+ ngôn ngữ) | Query: `Instruct: {task}\nQuery:{query}`, doc không cần | khả thi |
| Qwen/Qwen3-Embedding-4B | 4B (suy ra từ họ; trang 4B chưa mở) | 32K | 06/2025 | Apache-2.0 (suy ra từ họ, chưa mở trang) | không thấy | không thấy | cùng format | fp16 ≈ 8GB, khả thi nhưng chậm (~7x BGE-M3) |
| Qwen/Qwen3-Embedding-8B | 8B | 32K | 06/2025 | Apache-2.0 | MTEB multilingual mean (task) 70.58; Retrieval 70.88 | không thấy | cùng format | fp16 ≈ 16GB, KHÔNG vừa T4; cần 4-bit/8-bit |
| jinaai/jina-embeddings-v3 | 0.57B | 8192 | 16/09/2024 | **CC BY-NC-SA 4.0** (theo arXiv 2409.10173) | không thấy số MTEB trong abstract | không thấy | task LoRA adapter (retrieval.query/passage) | khả thi, nhưng license cấm thương mại |

Nhận xét nhóm 1:
- Trên VN-MTEB, mô hình lớn hơn (gte-Qwen2-7B, gte-Qwen2-1.5B, e5-Mistral-7B) dẫn đầu rõ rệt. Nhưng các mô hình này không được kiểm tra trên y tế, và không tìm thấy license/ngày phát hành xác nhận.
- AITeamVN/Vietnamese_Embedding có số Zalo legal cao hơn bge-m3 rõ rệt (0.8181 so với 0.6822), nhưng trên VN-MTEB retrieval lại thấp hơn bge-m3 (34.18 so với 39.84). Hai bộ đánh giá mâu thuẫn nhau. Số Zalo do chính tác giả báo cáo, trên tập train Zalo, nên dễ bị thiên vị. Đây là điểm cần kiểm chứng trên dữ liệu thật trước khi dùng.

---

## Nhóm 2: Reranker

| Hạng | Mô hình (HF id) | Tham số | Max token | Phát hành | License | Số liệu tiếng Việt (nguồn, protocol) | Số liệu khác | Prompt | T4 fp16 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **namdp-ptit/ViRanker** | 0.6B (F32 Safetensors) | 512 (ví dụ trên card; giới hạn mô hình không thấy) | arXiv 2509.09131, nộp 11/09/2025 | Apache-2.0 | MMARCO-VI dev: **MRR@10 0.7107, NDCG@10 0.7302**, MRR@3 0.6641, NDCG@3 0.6815 | So với bge-reranker-v2-m3 MRR@10 0.6209 / NDCG@10 0.6872; bge-reranker-v2-gemma 0.6249 / 0.6785; PhoRanker 0.6830 / 0.7422 (bảng trên card ViRanker, tự báo cáo) | FlagReranker hoặc AutoModelForSequenceClassification, không cần prompt | khả thi, nhẹ |
| 2 | **AITeamVN/Vietnamese_Reranker** | ~0.57B (suy ra từ file 2.27GB F32; card không ghi tham số) | 2304 (256 query + 2048 passage) | không thấy (card, ~2025) | Apache-2.0 | Legal Zalo 2021 (train, không dùng để train): **MRR@10 0.8672, Acc@1 0.7944, Acc@10 0.9740** (tự báo cáo, trên dữ liệu Zalo legal) | Fine-tune từ bge-reranker-v2-m3 trên ~1,1M bộ triplet | cross-encoder, không cần prompt | khả thi, nhẹ |
| 3 | BAAI/bge-reranker-v2-m3 | 0.6B (Safetensors F32) | không thấy trên card (ví dụ dùng 512) | papers 12/2023 và 02/2024 | Apache-2.0 | Tiếng Việt không được xác nhận trên card. MRR@10 MMARCO-VI 0.6209 (số của card ViRanker, không phải card BGE) | MIRACL chỉ có ảnh, không có số text | cross-encoder, không cần prompt | khả thi |
| 4 | BAAI/bge-reranker-v2-gemma | 3B (Safetensors F32), base gemma-2b | không thấy | papers 12/2023 và 02/2024 | Apache-2.0 | không xác nhận tiếng Việt. MRR@10 MMARCO-VI 0.6249 (số của card ViRanker) | BEIR, MIRACL chỉ có ảnh | Prompt: `determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'`, input `A: {query}` / `B: {passage}`, điểm = logit "Yes" | 3B fp16 ≈ 6GB, khả thi nhưng chậm hơn cross-encoder |
| 5 | Qwen/Qwen3-Reranker-4B | 4B (36 layer) | 32K | 06/2025 (paper) | Apache-2.0 | không thấy | MTEB-R 69.76; CMTEB-R 75.94; MMTEB-R 72.74 | Bắt buộc system prompt "yes/no" + template chat; mặc định `Given a web search query, retrieve relevant passages that answer the query` | fp16 ≈ 8GB, khả thi nhưng chậm |
| 6 | Qwen/Qwen3-Reranker-0.6B | 0.6B | 32K | 06/2025 | Apache-2.0 | không thấy | MTEB-R 65.80; CMTEB-R 71.31; MMTEB-R 66.36 (top-100 từ Qwen3-Embedding-0.6B) | cùng template với 4B | rất nhẹ |
| - | Qwen/Qwen3-Reranker-8B | 8B | 32K | 06/2025 | Apache-2.0 | không thấy | không mở trang | cùng template | fp16 ≈ 16GB, KHÔNG vừa T4 |
| 7 | jinaai/jina-reranker-v3 | 0.6B (BF16) | 131K, tối đa 64 doc/lần | paper 29/09/2025 | **CC BY-NC 4.0** (không thương mại) | không thấy (tiếng Việt không được nêu trong ngôn ngữ) | BEIR 61.94; MIRACL 66.83; MKQA 67.92; CoIR 70.64 | `trust_remote_code=True` | khả thi, nhưng license cấm thương mại |
| 8 | mixedbread-ai/mxbai-rerank-large-v2 | 2B (Safetensors; tiêu đề ghi 1.5B, mâu thuẫn) | không thấy | arXiv 2506.03487, 04/06/2025 | Apache-2.0 | không thấy (tiếng Việt không được nêu tên) | BEIR avg 57.49; Multilingual 29.79 (metric không ghi rõ trên card) | không cần prompt | khả thi, chậm hơn cross-encoder cùng cỡ |

Nhận xét nhóm 2:
- Đây là nhóm có số tiếng Việt đáng tin nhất. ViRanker (MMARCO-VI) và AITeamVN (Zalo legal) đều là fine-tune của BGE (ViRanker từ BGE-M3, AITeamVN từ bge-reranker-v2-m3). Cả hai đều chạy được trên T4.
- Qwen3-Reranker-4B là lựa chọn mạnh về đa ngôn ngữ, nhưng không có số tiếng Việt và tốn gấp ~7 lần cross-encoder 0,6B.
- Các số MMARCO-VI của ViRanker là số tự báo cáo trên card của chính tác giả ViRanker, chưa được tái lập bởi bên thứ ba trong nguồn đã mở.

---

## Nhóm 3: Sparse / lexical và bộ tách từ tiếng Việt

### 3a. Sparse / learned lexical

| Mô hình | Tham số | License | Số liệu tiếng Việt | Ghi chú cho T4 | Prompt |
|---|---|---|---|---|---|
| BAAI/bge-m3, sparse head (lexical_weights) | cùng model bge-m3 (0.57B) | MIT | không thấy (không có số riêng cho sparse tiếng Việt) | Cùng chi phí forward với dense, không cần model thứ hai; lưu trọng số token | không cần |
| SPLADE đa ngôn ngữ | chưa tra cứu | chưa tra cứu | chưa tra cứu | chưa tra cứu | chưa tra cứu |

Ghi chú: trang BGE-M3 chỉ mô tả API dense/sparse/colbert và không có số MIRACL/MKQA dạng text (chỉ có ảnh).

### 3b. Bộ tách từ tiếng Việt cho BM25

| Công cụ | License | Phương pháp | Tốc độ | Ghi chú |
|---|---|---|---|---|
| underthesea (undertheseanlp/underthesea) | Apache-2.0 | Toolkit tổng hợp, từ điển + thống kê | không thấy | Có `word_tokenize`; số F1/tốc độ không thấy trong nguồn đã mở |
| pyvi (trungtv/pyvi) | MIT | CRF | không thấy | Được dùng trong dangvantuan/vietnamese-embedding và bkai (tiền xử lý). Một paper ghi pyvi tách âm tiết quá nhỏ trong thí nghiệm NER |
| VnCoreNLP (vncorenlp/VnCoreNLP) | không xác định (trang LICENSE.md không hiện tên license) | Pipeline Java: tách từ, POS, NER, dependency | không có số trên trang | Hệ thống 2026 (arXiv 2603.05690, xác nhận qua kết quả tìm kiếm): VnCoreNLP + BPE F1 98.1% ở 4.900 câu/s; VnCoreNLP đơn lẻ F1 97.8% (số trong snippet, chưa mở trực tiếp paper) |
| RDRSegmenter | chưa xác minh license | Rule-based (RDR) | ~62k từ/giây single-thread trên i7 2.2GHz (paper, arXiv 1801.01331, qua kết quả tìm kiếm) | F1 97.90% trên Vietnamese treebank (kết quả tìm kiếm) |

Nhận xét nhóm 3: với BM25 trên 740k trang, RDRSegmenter hoặc pyvi là lựa chọn nhanh và nhẹ nhất. VnCoreNLP chính xác hơn nhưng chạy Java và cần kiểm tra license trước khi dùng. Chưa có số tốc độ đo trực tiếp của pyvi và underthesea.

---

## Không tìm thấy

- Embedding y tế tiếng Việt với số liệu retrieval: không thấy. Tìm kiếm trên HF chỉ ra các mô hình NER y tế (VietMed-NER, umt5-base-vimedner) và sinh văn bản (ViHealthcare), không có embedding y tế nào có số liệu.
- ViHealthQA: không thấy trong bất kỳ benchmark retrieval nào đã mở. Chỉ tìm thấy paper SPBERTQA (2022) đề cập đến bộ dữ liệu này, 10.015 cặp QA, không có số trên bộ retrieval hiện đại.
- ViMedRetrieve (arXiv 2503.07470): đây là benchmark retrieval y tế tiếng Việt duy nhất tìm được có số. Nhưng số liệu rất thấp (xem bên dưới) và dùng mô hình riêng ~20M tham số, không phải các mô hình trong danh sách này.
- Số VN→ZH hoặc VN→EN cross-lingual: không thấy cho bất kỳ mô hình nào trong bảng.
- Điểm MTEB-vie / VN-MTEB của Qwen3-Embedding, jina-v3, jina-reranker-v3, bge-reranker-v2-m3 và Qwen3-Reranker: không thấy.

Số ViMedRetrieve (Bảng 6 của paper, metric accuracy@k, bộ test >2.000 mẫu, dữ liệu từ ViMedAQA):

| Mô hình | k@5 | k@10 | k@20 |
|---|---|---|---|
| SimCSE | 0.09 | 0.11 | 0.12 |
| Bi-encoder | 0.25 | 0.45 | 0.73 |
| SBERT | 0.18 | 0.26 | 0.32 |
| Mô hình của paper (~20M, MiniLM-L6) | 0.30 | 0.44 | 0.50 |

Ghi chú: Bảng 6 và Bảng 3 của paper không khớp nhau ở mục "Proposed loss, HN" và "Paper's model", nên tôi không chắc chắn các số chính xác. Số trên chỉ dùng để tham khảo.

---

## Khuyến nghị cho đề bài

Ước lượng compute (của tôi, không có nguồn, dựa trên ~0,6 GFLOP/token cho mô hình 0,57B và ~25 TFLOPS hiệu dụng trên T4 fp16; cần đo thực tế trước khi lập kế hoạch):
- 740k doc × ~1.500 ký tự tiếng Việt ≈ 400 token/doc với tokenizer XLM-R (ước lượng). Tổng khoảng 3×10^8 token.
- Embed 740k doc bằng mô hình 0,57B (bge-m3, Vietnamese_Embedding): khoảng 2-3 giờ trên 1 T4 ở 512 token. Nếu dùng max 2048 token, chi phí tăng mạnh hơn nhiều.
- Embed 740k doc bằng mô hình 4B (Qwen3-Embedding-4B): khoảng 15 giờ trên 1 T4 (~7x). Vượt quá một phiên Kaggle, phải chia nhiều lượt hoặc nhiều tài khoản.
- Embed 740k doc bằng mô hình 8B: không vừa fp16 trên T4, cần lượng tử hóa; không khuyến nghị cho khối lượng này.
- Rerank top-100 cho 1.200 câu hỏi = 120k cặp. Với cross-encoder 0,6B: ~25 phút trên T4 (ước lượng). Với Qwen3-Reranker-4B: ~3 giờ (ước lượng).

### Nhóm 1 (dense): 2 lựa chọn

1. **bge-m3** (BAAI/bge-m3). Lý do: đã là baseline hiện tại, có VN-MTEB retrieval 39.84 (cao hơn AITeamVN 34.18 và bkai 25.37), MIT, 8192 token, không cần prompt, và có sẵn sparse + colbert để dùng lại. Chi phí embed đã biết.
2. **AITeamVN/Vietnamese_Embedding**. Lý do: fine-tune của bge-m3 trên dữ liệu tiếng Việt, cùng kiến trúc và chi phí với bge-m3, Apache-2.0. Số Zalo legal (MRR@10 0.8181 so với 0.6822 của bge-m3) cao hơn đáng kể, nhưng VN-MTEB lại cho thấy kém hơn bge-m3. Cần đánh giá trên một tập dev từ 1.200 câu hỏi, không tin vào số của chính tác giả.

Không khuyến nghị: multilingual-e5-large-instruct (40.88, MIT, chi phí tương đương bge-m3; có thể cân nhắc như ứng viên thứ 3 để đối chiếu, chưa có số tiếng Việt riêng); jina-v3 (CC BY-NC-SA, cấm thương mại); Qwen3-Embedding-8B (không vừa T4 fp16).

Nếu muốn thử mô hình lớn hơn: gte-Qwen2-1.5B-instruct (42.01 VN-MTEB, ~3GB fp16) là ứng viên tốt về điểm/chi phí, nhưng cần xác minh license và ngày phát hành trước khi dùng.

### Nhóm 2 (reranker): 2-3 lựa chọn

1. **namdp-ptit/ViRanker** (0.6B, Apache-2.0). Lý do: có số MMARCO-VI tốt nhất trong các mô hình đã so sánh (MRR@10 0.7107 so với 0.6209 của bge-reranker-v2-m3), nhẹ, chạy T4 thoải mái. Được xây dựng trên BGE-M3, nên tương thích kiến trúc với pipeline hiện tại.
2. **AITeamVN/Vietnamese_Reranker** (~0.57B, Apache-2.0). Lý do: fine-tune trên 1,1M triplet tiếng Việt từ bge-reranker-v2-m3, số Zalo legal MRR@10 0.8672 (tự báo cáo). Nên chạy thử như một đối chứng hoặc kết hợp với ViRanker.
3. **Qwen/Qwen3-Reranker-4B** (4B, Apache-2.0). Lý do: mạnh về đa ngôn ngữ (MMTEB-R 72.74) và có thể đóng vai trò reranker đa ngôn ngữ cho tài liệu tiếng Trung và tiếng Anh. Chi phí ~3 giờ cho 120k cặp trên T4 (ước lượng). Cần kiểm tra thêm trên tiếng Việt.

### Nhóm 3 (sparse + tách từ): 2 lựa chọn

1. **BGE-M3 sparse (lexical_weights)**. Lý do: dùng cùng forward pass với dense, không tốn thêm model. Có thể kết hợp với BM25 để cải thiện recall. Không có số tiếng Việt riêng, cần đo.
2. **RDRSegmenter** hoặc **pyvi** cho tách từ BM25. Lý do: nhanh và nhẹ. RDRSegmenter có số tốc độ đã đo (~62k từ/giây single-thread), license chưa xác minh. Pyvi là MIT, dùng được ngay. Cần chọn một và kiểm tra trên dev set.

---

## Nguồn

Model card và trang HF (đã mở trực tiếp):
- AITeamVN/Vietnamese_Embedding: https://huggingface.co/AITeamVN/Vietnamese_Embedding
- AITeamVN/Vietnamese_Reranker: https://huggingface.co/AITeamVN/Vietnamese_Reranker
- namdp-ptit/ViRanker: https://huggingface.co/namdp-ptit/ViRanker
- dangvantuan/vietnamese-embedding: https://huggingface.co/dangvantuan/vietnamese-embedding
- bkai-foundation-models/vietnamese-bi-encoder: https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder
- hiieu/halong_embedding: https://huggingface.co/hiieu/halong_embedding
- BAAI/bge-m3: https://huggingface.co/BAAI/bge-m3
- BAAI/bge-reranker-v2-m3: https://huggingface.co/BAAI/bge-reranker-v2-m3
- BAAI/bge-reranker-v2-gemma: https://huggingface.co/BAAI/bge-reranker-v2-gemma
- Alibaba-NLP/gte-multilingual-base: https://huggingface.co/Alibaba-NLP/gte-multilingual-base
- intfloat/multilingual-e5-large-instruct: https://huggingface.co/intfloat/multilingual-e5-large-instruct
- Qwen/Qwen3-Embedding-0.6B: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Qwen/Qwen3-Embedding-8B: https://huggingface.co/Qwen/Qwen3-Embedding-8B
- Qwen/Qwen3-Reranker-0.6B: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
- Qwen/Qwen3-Reranker-4B: https://huggingface.co/Qwen/Qwen3-Reranker-4B
- jinaai/jina-reranker-v3: https://huggingface.co/jinaai/jina-reranker-v3
- mixedbread-ai/mxbai-rerank-large-v2: https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2
- jinaai/jina-embeddings-v3 (trang HF quá lớn, không mở được; dùng arXiv): https://huggingface.co/jinaai/jina-embeddings-v3

Paper (đã mở trực tiếp):
- VN-MTEB, arXiv 2507.21500 (Findings EACL 2026): https://arxiv.org/html/2507.21500v1
- ViRanker, arXiv 2509.09131 (chỉ abstract, không có bảng số): https://arxiv.org/abs/2509.09131
- Advancing Vietnamese IR (ViMedRetrieve), arXiv 2503.07470: https://arxiv.org/html/2503.07470v1
- jina-embeddings-v3, arXiv 2409.10173 (abstract): https://arxiv.org/abs/2409.10173
- Qwen3-Embedding paper (ngày 05/06/2025, theo card Qwen3-Reranker-4B; không mở trực tiếp)

Công cụ tách từ (GitHub, đã mở):
- underthesea: https://github.com/undertheseanlp/underthesea (Apache-2.0)
- pyvi: https://github.com/trungtv/pyvi (MIT, CRF)
- VnCoreNLP: https://github.com/vncorenlp/VnCoreNLP (license không xác định từ trang)

Nguồn chỉ qua kết quả tìm kiếm (chưa mở trực tiếp):
- RDRSegmenter paper, arXiv 1801.01331: https://export.arxiv.org/pdf/1801.01331
- FreeTxt-Vi (benchmark 2026, VnCoreNLP + BPE), arXiv 2603.05690: https://arxiv.org/pdf/2603.05690
- ViHealthQA / SPBERTQA (2022): không mở được trực tiếp, chỉ thấy trong kết quả tìm kiếm.
