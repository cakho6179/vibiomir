# Các phương pháp tái tạo truy vấn (query formulation) cho bài toán retrieval y tế tiếng Việt

Ngày: 2026-10-22
Phạm vi: nghiên cứu web các phương pháp mở rộng/viết lại truy vấn, PRF, tách truy vấn, làm sạch truy vấn dài, dịch truy vấn đa ngôn ngữ, và hợp nhất nhiều biến thể truy vấn. Trọng tâm là các con số đo được và các bằng chứng trên miền y tế.

## Ghi chú về mức độ kiểm chứng

- Các lần WebFetch vào trang abs của arXiv (Query2doc 2303.07678, CSQE 2402.18031) chỉ trả về abstract, không có bảng kết quả. Vì vậy hầu hết con số bên dưới lấy từ đoạn tóm tắt của công cụ tìm kiếm hoặc từ bài tổng hợp thứ cấp, không phải từ bảng gốc.
- Những mục đánh dấu **chưa kiểm chứng** nghĩa là chưa đối chiếu được với bảng gốc của bài báo.
- Không có nguồn nào trong nghiên cứu này đo trực tiếp trên tiếng Việt. Mọi suy luận sang tiếng Việt là ngoại suy, cần thí nghiệm nội bộ.
- Nghiên cứu về "query variations" của Benham/Culpepper/Bailey không lấy được nội dung; phần này để trống có đánh dấu.

## 1. Bảng phương pháp

Ký hiệu: nDCG@10 (N@10), Recall@k (R@k), MAP. "Retriever" là retriever được hưởng lợi đã được đo. "Chi phí" là chi phí sinh văn bản LLM hoặc tính toán mỗi truy vấn.

| # | Phương pháp | Cách hoạt động | Kết quả đo được (dataset, metric, baseline -> kết quả) | Retriever | Chi phí mỗi truy vấn | Điểm yếu / rủi ro | Nguồn |
|---|---|---|---|---|---|---|---|
| 1 | Query2doc | LLM viết một đoạn văn trả lời/giả định, nối vào truy vấn gốc, rồi BM25 | Abstract: BM25 tăng 3% đến 15% trên MS MARCO và TREC DL, không fine-tune. Tóm tắt thứ cấp: tăng tới +15.2% N@10 trên DL'20 (**chưa kiểm chứng**). Các con số Table 1 không lấy được. RM3 chỉ tăng nhẹ R@1k theo bài gốc | Chủ yếu BM25 | 1 lần sinh (vài trăm token) | Ảo giác; độ trễ sinh; lợi ích lớn với truy vấn thực thể ngắn, nhỏ hơn với truy vấn dài | [arXiv 2303.07678](https://arxiv.org/pdf/2303.07678), [tổng hợp 2025](https://arxiv.org/pdf/2509.07794) |
| 2 | HyDE | LLM sinh tài liệu giả định, mã hóa bằng Contriever, tìm bằng dense | DL19 N@10 61.3 so với Contriever-ft 62.1; DL20 57.9 so với 63.2 (từ bài tổng hợp, **đối chiếu bảng gốc chưa làm**). Reproduce với Mistral-7B-Instruct: TREC-COVID 56.9, NFCorpus 27.7 (**chưa kiểm chứng** trên backbone khác) | Dense (Contriever) | 1 lần sinh | Phụ thuộc mạnh vào encoder; không dùng sẵn cho BM25 | [Gao et al. 2022 (arXiv 2212.10496)](https://arxiv.org/abs/2212.10496), [tổng hợp](https://arxiv.org/pdf/2410.21242) |
| 3 | LameR | LLM sinh câu trả lời có điều kiện trên các tài liệu BM25 vòng đầu, rồi truy vấn lại | Tóm tắt: DL19 N@10 69.1, DL20 64.8; BM25 50.6 / 48.0; HyDE 61.3 / 57.9 (**chưa kiểm chứng**, lấy từ tóm tắt Table 2) | BM25 là retriever vòng đầu | Hai lượt: BM25 + sinh có ngữ cảnh | Bài báo lập luận: retriever yếu thì augmentation kém hiệu quả | [Shen et al. (arXiv 2304.14233)](https://arxiv.org/pdf/2304.14233) |
| 4 | CSQE (Corpus-Steered QE) | LLM sinh mở rộng + chọn câu quan trọng từ tài liệu vòng đầu để bổ sung vào mở rộng | Vượt RM3 và KEQE về mức tăng so với BM25 trên DL19/20; vượt Q2D/PRF ở mọi metric; BM25+CSQE vượt Contriever-ft ở mọi metric (định tính). NovelEval (bảng bị lỗi định dạng): BM25 68.4 -> KEQE 62.0 -> CSQE 82.6 N@10 (**chưa kiểm chứng**) | BM25 | LLM sinh + 1 lượt BM25 | Phụ thuộc chất lượng tài liệu vòng đầu; thông tin ít liên quan làm trôi chủ đề | [Lei et al., EACL 2024 (arXiv 2402.18031)](https://arxiv.org/pdf/2402.18031) |
| 5 | Query2Expand (Jagerman et al. 2023) | LLM mở rộng truy vấn bằng prompt (có chain-of-thought), so sánh với Bo1/Bo2/KL | Tuyên bố: LLM mở rộng vượt PRF cổ điển trên MS MARCO và BEIR, với BM25 Terrier. Chi tiết số **chưa kiểm chứng** | BM25 | 1 lần sinh | Prompt CoT giúp nhưng tăng độ dài sinh | [arXiv 2305.03653](https://arxiv.org/pdf/2305.03653) |
| 6 | Generative Relevance Feedback (GRF) | LLM sinh nhiều dạng văn bản (truy vấn, thực thể, sự kiện, bài báo, tiểu luận), xây mô hình phản hồi xác suất | Tóm tắt abstract: N@10 tăng 17-24% so với RM3; MAP tăng 5-19%; recall SOTA. Bộ dữ liệu cụ thể **chưa kiểm chứng** | BM25 (mô hình phản hồi kiểu RM3) | Nhiều lần sinh tùy kiểu | Bị ảnh hưởng bởi kiểu văn bản sinh ra; so với RM3 nên baseline yếu | [Mackie et al., SIGIR 2023 (arXiv 2304.13157)](https://arxiv.org/abs/2304.13157) |
| 7 | RM3 / Rocchio / Bo1 (PRF cổ điển) | Lấy thuật ngữ quan trọng từ top-k tài liệu vòng đầu, mở rộng truy vấn | Làm gốc so sánh; Query2doc ghi RM3 chỉ tăng nhẹ R@1k. Số liệu cụ thể trên y tế **chưa kiểm chứng** | BM25 | Rất rẻ (không cần LLM) | Dễ trôi chủ đề khi top-k nhiễu; cần k và số thuật ngữ được tinh chỉnh | (tham chiếu trong các nguồn trên) |
| 8 | ColBERT-PRF | PRF ở mức biểu diễn đa vector: đưa vector từ tài liệu phản hồi vào truy vấn ColBERT | MAP tăng tới 26% trên TREC DL19 và 10% trên DL20 (MS MARCO passage); N@10 chỉ định tính. Cross-corpus (MS MARCO làm corpus ngoài): N@10 tăng tới 28% (sparse) và 12% (dense) | ColBERT (dense đa vector) | Chỉ tính toán, không cần LLM | Chỉ áp dụng cho ColBERT | [Wang et al. (arXiv 2106.11251)](https://arxiv.org/pdf/2106.11251) |
| 9 | ANCE-PRF | PRF ở mức dense: nối vector tài liệu phản hồi vào truy vấn, huấn luyện lại | Bảng reproducibility (OCR lỗi): DL20 N@10 ANCE ~0.646 -> ANCE-PRF ~0.695; MS MARCO dev ~0.388 -> 0.401 (**chưa kiểm chứng**). Cross-corpus tới +30% (sparse), +29% (dense) | ANCE (dense) | Chỉ tính toán | Lợi ích giảm khi retriever mạnh hơn | [Li et al. reproducibility (arXiv 2112.06400)](https://arxiv.org/pdf/2112.06400), [ANCE-PRF (arXiv 2108.13454)](https://arxiv.org/pdf/2108.13454) |
| 10 | PromptPRF (2025) | LLM trích đặc trưng không phụ thuộc truy vấn (thực thể, tóm tắt, từ khóa) từ tài liệu top, tính trước offline, đưa vào biểu diễn truy vấn dense zero-shot | Tuyên bố: cải thiện nhất quán trên TREC DL và BEIR. Con số **chưa kiểm chứng** | Dense (PromptReps, decoder LLM) | Offline cho corpus; truy vấn gần như không tăng chi phí | Cần chạy LLM trên top tài liệu của corpus; phương pháp mới | [arXiv 2503.14887](https://arxiv.org/pdf/2503.14887) |
| 11 | RAG-Fusion / multi-query + RRF | Sinh nhiều biến thể truy vấn, truy xuất từng biến thể, hợp bằng RRF (k=60 thường dùng; Cormack 2009) | Không có số retrieval đáng tin. Đánh giá 2026 trên sản xuất: tăng recall thô nhưng phần lớn biến mất sau rerank và cắt top (**chưa kiểm chứng** chi tiết). RAGElo 2024: so sánh định tính | Cả BM25 và dense | N biến thể * retrieval | Biến thể lạc đề kéo theo nhiễu; RRF làm loãng nếu biến thể yếu | [arXiv 2402.03367](https://arxiv.org/abs/2402.03367), [RRF, Cormack 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) |
| 12 | Query variations (Benham/Culpepper/Bailey) | Nhiều cách diễn đạt của cùng một chủ đề, hợp bằng fusion | **Không lấy được** số liệu. Chỉ có tham chiếu: fusion có thể đạt đánh đổi rủi ro-lợi ích tốt hơn (DESIRES 2018, chưa kiểm chứng số) | BM25 | N lần truy xuất | Chi phí tăng tuyến tính với số biến thể | [Benham & Culpepper, ADCS 2017 (DBLP)](https://dblp.dagstuhl.de/pid/210/8217.html) |
| 13 | Rewrite-Retrieve-Read (RaR) | LLM viết lại truy vấn cho retriever; có thể dùng rewriter nhỏ được huấn luyện bằng RL | Tóm tắt: rewriter huấn luyện cải thiện accuracy QA và hit rate, nhất là với LLM đọc yếu. Số BM25 recall **chưa kiểm chứng** | BM25 và dense | 1 lần sinh | Miền QA mở, không phải truy vấn hội thoại y tế | [Ma et al., EMNLP 2023 (arXiv 2305.14283)](https://arxiv.org/abs/2305.14283) |
| 14 | Key-concept identification / verbose query reduction | Chọn các khái niệm quan trọng trong truy vấn dài, bỏ phần thừa | Bendersky & Croft SIGIR 2008: nhận diện khái niệm chính xác hơn IDF; cải thiện hiệu quả có ý nghĩa thống kê trên tập newswire và web (số liệu **chưa kiểm chứng**). Nghiên cứu máy tìm kiếm thương mại: bỏ cấu trúc từ dừng cải thiện; đặt trích dẫn cho khái niệm làm giảm mạnh | BM25 (và dense với 2026 paper, chưa kiểm chứng) | Rất rẻ với heuristic; LLM thì một lần sinh | Bỏ nhầm từ quan trọng; trích dẫn cứng làm hỏng BM25 | [Bendersky & Croft 2008](https://maroo.cs.umass.edu/pub/web/getpdf.php?id=810) |
| 15 | Medical: LLM + ontology (BMQExpander) | Mở rộng truy vấn bằng định nghĩa và quan hệ ontology, kết hợp LLM | TREC-COVID: BM25 N@10 0.656 -> 0.801 (ablation đầy đủ). So với BM25: +22.1% TREC-COVID, +11.7% NFCorpus, +5.9% SciFact (tương đối). Lưu ý: không phải LLM thuần | BM25 | LLM + tra cứu ontology | Phụ thuộc ontology; không có ontology tiếng Việt | [arXiv 2508.11784](https://arxiv.org/pdf/2508.11784) |
| 16 | Medical: doc expansion (tham chiếu, không phải query) | LLM mở rộng tài liệu phía corpus | BM25 TREC-COVID N@10 65.6, NFCorpus 32.5 (baseline bảng); không áp dụng trực tiếp cho truy vấn | Dense | Offline, tốn cho 740k tài liệu | Chi phí cao với corpus lớn | [arXiv 2308.08285](https://arxiv.org/pdf/2308.08285) |
| 17 | Dịch truy vấn sang tiếng Anh/Trung (CLIR) | Dịch truy vấn rồi truy xuất corpus đa ngôn ngữ | Medical CLIR (ACL 2020): dịch truy vấn nhìn chung tốt hơn dịch tài liệu. Nghiên cứu 2025: dịch tài liệu chủ yếu giúp truy vấn tiếng Anh; embedding ít bị ảnh hưởng. **Không có số liệu VI->EN** | BM25 và dense đa ngữ | 1 lần dịch truy vấn ngắn | Dịch thuật ngữ y khoa sai; truy vấn là câu không đầy đủ, dịch khó | [ACL 2020.acl-main.613](https://aclanthology.org/2020.acl-main.613), [arXiv 2511.19324](https://arxiv.org/pdf/2511.19324), [arXiv 2304.01019](https://arxiv.org/pdf/2304.01019) |
| 18 | Tách truy vấn (decomposition) | Tách truy vấn thành các câu hỏi con | **Chưa tìm được số liệu** kiểm chứng trong nghiên cứu này | — | — | — | — |
| 19 | Step-back / keyword extraction (LLM) | Hỏi câu tổng quát hơn, hoặc trích từ khóa | **Chưa tìm được số liệu** kiểm chứng; chỉ có tham chiếu gián tiếp qua RaR và Query2Expand | — | — | — | — |
| 20 | Tách từ tiếng Việt (underthesea, CRF) | Phân đoạn từ để BM25 có token đúng | Không có số liệu BM25 tiếng Việt. Nghiên cứu VSMRC 2025 nói phân đoạn cải thiện RAG (định tính) | BM25 | Rất rẻ | Cần kiểm tra trên corpus | [underthesea](https://underthesea.readthedocs.io/en/1.1.16/readme.html), [arXiv 2506.15978](https://arxiv.org/html/2506.15978v1) |

## 2. Nhận định chung

1. **Mở rộng bằng LLM có bằng chứng tốt nhất với BM25 trên truy vấn ngắn và trung bình** (Query2doc, LameR, CSQE, Query2Expand). Con số đáng tin nhất là tăng 3-15% của Query2doc trên MS MARCO và TREC DL, theo abstract.
2. **PRF theo corpus (CSQE, PRF dense) hữu ích hơn PRF cổ điển** ở các báo cáo được tìm thấy. Với bài toán này, phần corpus là 740k trang web, nên kết quả có thể khác.
3. **Multi-query + RRF có bằng chứng yếu.** Phần lớn kết quả là định tính, và một đánh giá 2026 cho thấy lợi ích recall thường biến mất sau rerank.
4. **Không có bằng chứng trực tiếp nào về truy vấn hội thoại y tế dài.** TREC-COVID, NFCorpus và các bài về medical CLIR cho thấy LLM expansion và ontology có thể tăng N@10 đáng kể, nhưng truy vấn trong bài này khác biệt lớn (chào hỏi, tiền sử, kết quả xét nghiệm).
5. **Cần thí nghiệm nội bộ.** Các con số ở trên phần lớn đến từ corpus tiếng Anh; chưa có số liệu cho tiếng Việt.

## 3. Khuyến nghị cho đề bài

Nguyên tắc: mọi biến thể đều được thêm như nguồn truy xuất bổ sung và hợp bằng RRF (k=60) với truy vấn gốc. Không thay thế truy vấn gốc.

Ưu tiên theo khả năng tăng recall pool:

### Biến thể 1: Làm sạch và cô đọng thành một câu hỏi y khoa (cho BGE-M3 dense và BM25)
Mục tiêu: loại bỏ chào hỏi, thông tin cá nhân và giữ các triệu chứng, xét nghiệm, giai đoạn thai, thuốc. Đây là key-concept reduction bằng LLM.

Prompt mẫu (Qwen2.5-7B-Instruct hoặc Qwen3-8B, bật chế độ không suy luận dài):
```
Bạn là chuyên gia y khoa. Từ câu hỏi bệnh nhân dưới đây, hãy viết lại thành MỘT câu hỏi y khoa ngắn gọn, bằng tiếng Việt. Bỏ lời chào, tên, ngày tháng cá nhân không cần thiết. Giữ lại triệu chứng, chỉ số xét nghiệm (kèm tên xét nghiệm), tình trạng, thuốc và bệnh được nhắc đến. Chỉ trả lời câu hỏi đã viết lại, không giải thích.

Câu hỏi: {query}
```
Retriever: BGE-M3 dense (nguồn bổ sung) và BM25 đã tách từ.

### Biến thể 2: Đoạn trả lời giả định kiểu Query2doc bằng tiếng Việt (cho BM25 và BGE-M3)
Mục tiêu: sinh một đoạn 100-150 từ chứa thuật ngữ y khoa, cơ sở của Query2doc và HyDE.

Prompt mẫu:
```
Hãy viết một đoạn văn ngắn (khoảng 120 từ), bằng tiếng Việt, trích từ một bài báo sức khỏe phổ thông, trả lời trực tiếp câu hỏi sau. Dùng đúng thuật ngữ y khoa tiếng Việt và tên xét nghiệm, kể cả tên gọi khác thường gặp. Không chào hỏi, không khuyên cá nhân hóa.

Câu hỏi: {query}
```
Retriever: BM25 (nối với truy vấn gốc, như Query2doc) và BGE-M3 dense (embedding riêng, không nối). Rủi ro: ảo giác về số liệu; cân nhắc giới hạn độ dài.

### Biến thể 3: Từ khóa và thực thể y khoa (cho BM25)
Mục tiêu: danh sách từ khóa ngắn để BM25 có token khớp cao. Đây là biến thể rẻ nhất.

Prompt mẫu:
```
Trích xuất từ khóa tìm kiếm y khoa từ câu hỏi sau. Trả về một dòng, các cụm từ phân cách bằng dấu phẩy, gồm: triệu chứng, bệnh, xét nghiệm (tên và giá trị), thuốc, thủ thuật, tình trạng sinh lý. Có thể thêm tên tiếng Anh cho thuật ngữ chuyên môn. Không thêm giải thích.

Câu hỏi: {query}
```
Retriever: BM25 (một truy vấn ngắn riêng).

### Biến thể 4: Dịch thuật ngữ y khoa sang tiếng Anh và tiếng Trung (cho BGE-M3 đa ngữ và BM25 trên phần EN/ZH)
Mục tiêu: phủ corpus tiếng Anh/Trung. Chỉ dịch thuật ngữ và tên xét nghiệm, không dịch toàn bộ câu.

Prompt mẫu:
```
Dịch các thuật ngữ y khoa trong câu hỏi sau sang tiếng Anh, sau đó sang tiếng Trung giản thể. Chỉ trả về hai dòng: "EN: ..." và "ZH: ...", mỗi dòng là các cụm thuật ngữ phân cách bằng dấu phẩy. Giữ nguyên tên xét nghiệm dạng viết tắt (ví dụ beta HCG, AMH).

Câu hỏi: {query}
```
Retriever: BGE-M3 dense đa ngữ (truy vấn EN/ZH riêng) và BM25 trên tài liệu tiếng Anh/Trung. Chưa có số liệu VI->EN; cần kiểm tra.

### Biến thể 5: Mở rộng theo corpus (PRF kiểu CSQE) (cho BM25, chạy sau vòng 1)
Mục tiêu: lấy top-20 tài liệu từ biến thể gốc, để LLM chọn câu quan trọng, rồi sinh một truy vấn mở rộng. Đây là biến thể tốn kém nhất, nhưng có bằng chứng tốt hơn RM3 trên TREC DL.

Prompt mẫu (bước chọn câu):
```
Câu hỏi: {query}
Đoạn văn bản sau được lấy từ kho dữ liệu. Liệt kê tối đa 3 câu trong đoạn có liên quan trực tiếp đến câu hỏi, nguyên văn. Nếu không có câu nào liên quan, trả về "KHONG".

Đoạn: {doc}
```
Sau đó dùng các câu chọn được nối với biến thể 2 để tạo truy vấn BM25 thứ hai. Retriever: BM25. Cần giới hạn số tài liệu để giữ chi phí.

### Biến thể 6 (tùy chọn, tăng chi phí): nhiều paraphrase hợp bằng RRF
Mục tiêu: tăng độ phủ, nhưng bằng chứng yếu. Chỉ thử với 2-3 biến thể và đo lợi ích sau khi thêm vào pool, không đo trên rerank. Cách làm: hợp bằng RRF với k=60, giữ các biến thể 1-4 làm nguồn riêng.

Lưu ý thứ tự thử: biến thể 3 (rẻ, BM25), rồi 1, 2, 4, 5. Mỗi biến thể cần đo Recall@1000 trên tập validation có nhãn, và so sánh với baseline truy vấn gốc trước khi thêm vào pool.

## 4. Nguồn đã dùng

- Query2doc: https://arxiv.org/pdf/2303.07678 ; tổng hợp 2025: https://arxiv.org/pdf/2509.07794
- HyDE/Gao 2022: https://arxiv.org/abs/2212.10496 ; tổng hợp: https://arxiv.org/pdf/2410.21242
- LameR: https://arxiv.org/pdf/2304.14233
- CSQE: https://arxiv.org/pdf/2402.18031
- Query2Expand (Jagerman): https://arxiv.org/pdf/2305.03653
- GRF: https://arxiv.org/abs/2304.13157
- ColBERT-PRF: https://arxiv.org/pdf/2106.11251 ; ANCE-PRF: https://arxiv.org/pdf/2108.13454 ; reproducibility: https://arxiv.org/pdf/2112.06400
- PromptPRF: https://arxiv.org/pdf/2503.14887
- RAG-Fusion: https://arxiv.org/abs/2402.03367 ; RRF: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- Benham/Culpepper: https://dblp.dagstuhl.de/pid/210/8217.html
- RaR: https://arxiv.org/abs/2305.14283
- Verbose query: https://maroo.cs.umass.edu/pub/web/getpdf.php?id=810
- BMQExpander: https://arxiv.org/pdf/2508.11784 ; doc expansion: https://arxiv.org/pdf/2308.08285
- CLIR medical (ACL 2020): https://aclanthology.org/2020.acl-main.613 ; https://arxiv.org/pdf/2511.19324 ; https://arxiv.org/pdf/2304.01019
- Tách từ tiếng Việt: https://underthesea.readthedocs.io/en/1.1.16/readme.html ; https://arxiv.org/html/2506.15978v1

## 5. Việc còn lại để kiểm chứng

- Lấy bảng Table 1 của Query2doc và bảng kết quả CSQE từ bản PDF/HTML đầy đủ để thay các con số tóm tắt.
- Tìm nghiên cứu query variations của Benham/Culpepper/Bailey 2017 và xác nhận số liệu.
- Tìm số liệu về decomposition và step-back trên BEIR/TREC-COVID.
- Thí nghiệm nội bộ: Recall@1000 trên tập có nhãn cho từng biến thể, với BM25 (pyvi/underthesea) và BGE-M3.
