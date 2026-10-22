# Mentor day 2026-10-22: ViBioMIR, retrieval đa ngôn ngữ và indexing

Nguồn: `_AI_Guru__R2AI_Stage3_MentorDay_Oct_22th.pdf` (Downloads), **29 trang** (slide ghi "Slide n/29"). Diễn giả: Nguyễn Tấn Minh, CECS VinUni. Ngày 22 tháng 10 năm 2026.

Ghi chú về cách đọc:
- Công cụ Read với tham số `pages` không dùng được (thiếu `pdftoppm`). Nội dung được trích xuất bằng PyMuPDF theo từng trang. Lớp text của PDF làm mất một số khoảng trắng tiếng Việt (ví dụ "Mởđầu", "ngữnghĩa"). Các trích dẫn trong báo cáo đã được khôi phục khoảng trắng. Hai trang có hình (24, 25) được render và kiểm tra bằng mắt; các hình là sơ đồ minh họa, không có số liệu mới.
- Mỗi trang có 3 hình logo/nền mẫu (AI Guru, tinix). Không có biểu đồ kết quả thực nghiệm nào trong deck.
- Deck **không có** phần quy định cuộc thi, ngưỡng điểm, metric FINAL, quy tắc nộp bài, lịch thi, hay thông tin về private test. Mọi mục liên quan trong báo cáo đều ghi rõ là "không có trong deck".

---

## 1. Tóm tắt 10 ý quan trọng nhất

1. **Pipeline hai tầng document → passage.** Document stage A: BM25 (lexical). Document stage B: dense reranking trên shortlist khoảng 400 tài liệu. Passage stage: chấm điểm passage trong các tài liệu đã chọn, rồi rerank (slide 6, 7, 9). Số lượng ứng viên được nêu: 2,000 document → 400 document → khoảng 3,200 passage → rerank 400 passage (slide 9).
2. **Giữ query tiếng Việt gốc làm cơ sở cho dense scoring và reranking.** Mỗi query VI có thêm một bản dịch EN và ZH sát nghĩa. Rà soát thuật ngữ y khoa bằng mẫu song ngữ và nguồn chính thống (slide 13).
3. **Embedding model được gợi ý: BGE-M3** (568M, đa ngữ, general-purpose retrieval). Các lựa chọn khác trong bảng: multilingual-e5-large (560M), Jina Embeddings v3 (570M), mGTE Base (~305M), MMed-Embed (568M, medical retrieval, 2026), Qwen3 Embedding (0.6/4/8B). Slide 8 dẫn tới bảng xếp hạng MTEB-Medical để chọn thêm. Deck không chọn một model cụ thể cho cuộc thi.
4. **Reranker nhỏ gọn để bắt đầu: BGE-reranker-v2-m3** (~0.6B, đa ngữ). Slide 10 liệt kê thêm BGE-reranker-v2-gemma (~2B, LLM-based), Jina Reranker v3.5 (0.6B), mGTE Multilingual Reranker Base (~0.3B), Qwen3-Reranker (0.6/4/8B). Slide 9 nói "thử LLM trên tập nhỏ hơn".
5. **Indexing: exact search trên tập ứng viên, ANN cho dense toàn cục khi cần.** Faiss IndexFlat làm chuẩn so sánh. HNSW khi đủ RAM. IVF-PQ khi bộ nhớ hạn chế, đánh đổi bằng sai số nén (slide 21, 22).
6. **Ước lượng dung lượng vector (slide 18, nguồn Faiss):** 1M document vector, 1,024-d, FP16 = 2.05 GB. 8M passage vector, 1,024-d, FP16 = 16.38 GB. 8M passage vector, FP32 = 32.77 GB. 8M passage code, 64-byte PQ = 0.512 GB.
7. **Stack tối giản do BTC gợi ý (slide 22):** Pyserini/Lucene cho BM25 (trên title, headings, body), fusion bằng RRF. NumPy memmap cho candidate vectors. NumPy/PyTorch cho exact dense scoring. Faiss cho global dense index. OpenSearch là lựa chọn thay thế khi cần BM25, vector search và metadata filter trong một hệ thống.
8. **Dịch query: thứ tự ưu tiên.** Baseline là direct translation. Glossary-augmented prompting có ưu tiên "Cao". Retrieved few-shot có ưu tiên "Cao, sau baseline". Domain-specific fine-tuning chỉ "Khi có dữ liệu". MedEV có khoảng 360K cặp câu VI–EN (slide 14, 15).
9. **Late chunking và late interaction là hướng nâng cao, không phải baseline.** Late chunking: encode cửa sổ tài liệu dài rồi pooling theo ranh giới chunk, để vector chunk có ngữ cảnh toàn văn bản (slide 24). Late interaction: s(q, p) = Σ_i max_j ⟨q_i, p_j⟩, có trong BGE-M3 multi-vector hoặc Jina-ColBERT-v2 (slide 25).
10. **Giữ ánh xạ về tài liệu cha.** Slide 9: "chọn passage gốc cùng ID tài liệu cha" và "Cập nhật thứ hạng document từ passage score". Slide 17 nhấn mạnh cần đánh giá số lượng chunk trước khi chọn phương pháp indexing, và cấu trúc hai tầng document–chunk khó tổ chức dữ liệu.

---

## 2. Theo từng trang

**Trang 1. Tiêu đề.**
- "ViBioMIR: Retrieval đa ngôn ngữ trên corpus hơn một triệu tài liệu".
- Phụ đề: "Tìm kiếm đa ngôn ngữ, corpus indexing và retrieval nâng cao".
- Diễn giả: Nguyễn Tấn Minh, College of Engineering & Computer Science (CECS), VinUni. Ngày 22 tháng 10 năm 2026.

**Trang 2. Mục lục.**
- Mở đầu; 1. Kiến trúc tổng thể; 2. Truy hồi thông tin đa ngôn ngữ; 3. Indexing cho dữ liệu lớn; 4. Các phương pháp truy hồi nâng cao; Tổng kết và thảo luận.

**Trang 3. Truy hồi hai cấp độ.**
- "Đầu vào: query tiếng Việt; nguồn bằng chứng: tiếng Việt, Anh và Trung."
- "Quy mô: khoảng 1200 câu hỏi, nguồn thông tin từ 1.22M URLs."
- "Đầu ra cấp document xác định nguồn liên quan; cấp passage xác định bằng chứng trong nguồn đó."

**Trang 4. Trang chia phần.** "1. Kiến trúc tổng thể". Không có nội dung kỹ thuật.

**Trang 5. Kiến trúc tổng thể hệ thống truy hồi.**
- Luồng: Câu hỏi tiếng Việt → Tìm theo từ khóa (Lexical retrieval) và Tìm theo ngữ nghĩa (Dense retrieval) → Hợp các tài liệu và đoạn ứng viên → Xếp hạng lại theo mức độ liên quan → Trả về tài liệu và đoạn bằng chứng gốc.
- Đây là sơ đồ tổng quát. Trang 6 mô tả pipeline cụ thể theo từng stage, khác ở chỗ dense không chạy song song với BM25 mà chạy như reranker trên shortlist.

**Trang 6. Pipeline ví dụ: từ document đến passage.**
- Query tiếng Việt → query gốc cùng bản dịch sát nghĩa sang tiếng Anh và tiếng Trung.
- Document stage A: lexical retrieval bằng BM25; kết hợp thứ hạng giữa các ngôn ngữ.
- Document stage B: multilingual dense reranking trên tập ứng viên lexical.
- Passage stage: chấm điểm passage trong tài liệu đã chọn, rồi rerank hoặc dùng "LLM hợp lệ" (cụm từ này được ghi nguyên văn trên slide; [không đọc rõ] về ý nghĩa cụ thể).

**Trang 7. Document stage B: multilingual dense reranking.**
- Encode query tiếng Việt gốc bằng BGE-M3 hoặc multilingual encoder. Chấm điểm document vector đã tính sẵn trong shortlist, "giữ khoảng 400 tài liệu".
- Bắt đầu với tiêu đề, đề mục và phần mở đầu. Encode theo phần đối với tài liệu dài, đa chủ đề.
- So sánh xếp hạng chỉ dùng dense với kết hợp lexical/dense trên cùng shortlist.
- Nguồn: BGE-M3 model card; Qwen3 Embedding and Reranking.

**Trang 8. Bảng dense embedding models.** Cột: Model, Năm, Size, Ngôn ngữ, Domain.
- SapBERT (PubMedBERT), 2021, ~110M, EN, biomedical concept/entity. BioLORD-2023, 2023, ~110M, EN, biomedical concept, semantic similarity.
- BGE-M3, 2024, 568M, đa ngôn ngữ, general-purpose retrieval. multilingual-e5-large, 2023, 560M, đa ngôn ngữ, general-purpose retrieval.
- Jina Embeddings v3, 2024, 570M, đa ngôn ngữ. mGTE Base, 2024, ~305M, đa ngôn ngữ. MMed-Embed, 2026, 568M, đa ngôn ngữ, medical retrieval.
- Qwen3 Embedding, 2025, 0.6/4/8B, đa ngôn ngữ, LLM-based. NV-Embed-v2, 2024, ~7.8B, EN. SFR-Embedding-Mistral, 2024, ~7B, "Thiên về EN".
- Tham khảo bảng xếp hạng cho dữ liệu y sinh: MTEB-Medical.

**Trang 9. Passage stage: tìm và chọn bằng chứng.**
- Lấy passage của 400 document ứng viên: "nếu mỗi tài liệu có tám passage, tổng là khoảng 3,200" (giả định 8 passage/tài liệu nằm trong câu slide).
- Chấm điểm passage bằng vector và lexical match; giữ khoảng 400 passage để rerank.
- Dùng multilingual reranker với toàn bộ câu hỏi tiếng Việt; "thử LLM trên tập nhỏ hơn".
- Điểm chính: "Số ứng viên: 2,000 document → 400 document → khoảng 3,200 passage → rerank 400 passage." Nguồn: Qwen3 Embedding and Reranking; "Bài toán và đánh giá R2AI".

**Trang 10. Bảng reranker models.** Cột: Model, Năm, Size, Ngôn ngữ, Domain/đặc điểm.
- MedCPT-Cross-Encoder, 2023, ~110M, EN, biomedical.
- BGE-reranker-v2-m3, 2024, ~0.6B, đa ngôn ngữ, "General-purpose; baseline nhỏ gọn".
- BGE-reranker-v2-gemma, 2024, ~2B, đa ngôn ngữ, "General-purpose; LLM-based reranker".
- Jina Reranker v3.5, 2026, 0.6B, đa ngôn ngữ, listwise, "tăng cường medical, legal, financial và structured data".
- mGTE Multilingual Reranker Base, 2024, ~0.3B, đa ngôn ngữ, "phù hợp nhánh VI/EN/ZH".
- GTE Reranker ModernBERT Base, 2025, 149M, EN, "bản mới hơn". Ô năm và size của dòng này được đọc theo thứ tự trên slide, không có cột năm riêng rõ ràng ([không đọc rõ] nếu cần xác minh).
- Qwen3-Reranker, 2025, 0.6/4/8B, đa ngôn ngữ, hỗ trợ instruction. Qwen3-VL-Reranker, 2026, 2/8B, multimodal (text, image, video).

**Trang 11. Trang chia phần.** "2. Truy hồi thông tin đa ngôn ngữ".

**Trang 12. Giải pháp cho medical retrieval đa ngôn ngữ.**
- "Dịch query và đối chiếu thuật ngữ giúp tăng hiệu quả khi truy hồi thông tin."
- "Multilingual embedding so khớp câu hỏi tiếng Việt với tài liệu EN/ZH."
- Điểm chính: "Giữ đúng ngữ nghĩa, đặc biệt là thuật ngữ y khoa khi mở rộng phạm vi truy hồi."

**Trang 13. Những lưu ý khi dịch query.**
- Lưu một bản dịch tiếng Anh và tiếng Trung sát nghĩa cho mỗi query tiếng Việt.
- Giữ nguyên thuật ngữ, tình trạng bệnh được mô tả trong query gốc.
- Rà soát thuật ngữ y khoa bằng mẫu song ngữ và nguồn chính thống. Dùng query tiếng Việt gốc làm cơ sở cho dense scoring và reranking.
- Điểm chính: "Bản dịch trôi chảy vẫn có thể bị sai lệch thông tin y khoa cần tìm kiếm."

**Trang 14. Query translation: các hướng tiếp cận.**
- 1. Direct translation: dùng mô hình dịch hoặc LLM dịch trực tiếp query VI sang EN/ZH. Trích dẫn: Vilar et al., ACL 2023.
- 2. Glossary-augmented prompting: đưa thuật ngữ và bản dịch đã xác minh vào prompt. Trích dẫn: Bogoychev & Chen, WMT 2023.
- 3. Retrieved few-shot examples: lấy các cặp câu song ngữ tương tự từ translation memory làm ví dụ trong prompt. Trích dẫn: Mu et al., Findings ACL 2023.
- 4. Domain-specific fine-tuning: fine-tune trên medical parallel data; "MedEV cung cấp khoảng 360K cặp câu VI–EN". Trích dẫn: Vo et al., LREC-COLING 2024.

**Trang 15. Query translation: so sánh chi tiết.** Bảng: Kỹ thuật, Cách áp dụng, Ưu điểm/giới hạn, Ưu tiên.
- Direct translation: dễ triển khai nhưng có thể sai thuật ngữ hoặc mất ngữ nghĩa. Ưu tiên: "Baseline".
- Glossary-augmented prompting: kiểm soát thuật ngữ, cần chọn đúng nghĩa theo ngữ cảnh. Ưu tiên: "Cao".
- Retrieved few-shot examples: bổ sung ngữ cảnh, cần dữ liệu sạch, tránh sao chép chi tiết ngoài query. Ưu tiên: "Cao, sau baseline".
- Domain-specific fine-tuning (ví dụ MedEV cho VI–EN): học cách dùng từ trong miền, tốn training và cần dữ liệu đúng chiều dịch. Ưu tiên: "Khi có dữ liệu".

**Trang 16. Trang chia phần.** "3. Indexing cho dữ liệu lớn".

**Trang 17. Indexing 1M documents: các thách thức.**
- Số lượng chỉ mục rất lớn. Một document cần tách thành nhiều đoạn hoặc vectors; multi-vector representation tốn thêm dung lượng lưu trữ.
- Cấu trúc hai tầng document–chunk khó cấu trúc hóa dữ liệu. "cần đánh giá số lượng chunk trước khi chọn phương pháp indexing".
- Chi phí preprocessing và encoding: tài liệu dài ngắn khác nhau, boilerplate, trùng lặp và lỗi extraction làm tăng thời gian xử lý và giảm chất lượng index.
- Multilingual lexical indexing (VI/EN/ZH) cần cách lưu trữ phù hợp. Quản lý bộ nhớ và disk I/O cho embeddings, graph, IDs, metadata và bộ nhớ tạm.

**Trang 18. Ước tính dung lượng cho embedding vectors.** Bảng: Representation, Kiểu lưu trữ, Dung lượng. Nguồn: Tài liệu Faiss index.
- 1M document vector, 1,024-d FP16: 2.05 GB.
- 8M passage vector, 1,024-d FP16: 16.38 GB.
- 8M passage vector, 1,024-d FP32: 32.77 GB.
- 8M passage code, 64-byte PQ: 0.512 GB.

**Trang 19. Exact search và ANN: chính xác hay tốc độ?** Bảng so sánh Exact search (Flat) với ANN.
- Exact: chính xác 100% nhưng rất chậm; tính khoảng cách với toàn bộ N vector, độ phức tạp O(Nd); không cần xây đồ thị hay học phân cụm.
- ANN: tìm kiếm gần đúng với độ trễ thấp; dùng đồ thị hoặc phân cụm để xét ít vector hơn; có thể kết hợp nén vector; tốn chi phí xây chỉ mục và bộ nhớ phụ thuộc cấu trúc và mức nén.
- Nguồn: Faiss.

**Trang 20. Một số cấu trúc chỉ mục ANN.**
- HNSW: đồ thị nhiều tầng; tăng efSearch thường tăng recall và độ trễ; tốn thêm RAM cho đồ thị.
- IVF-Flat: phân cụm thành nlist nhóm, chỉ quét trong nprobe nhóm gần query; tăng nprobe giảm bỏ sót nhưng chậm hơn; cần học bộ phân cụm.
- IVF-PQ: kết hợp IVF với PQ, nén từng phần bằng codebook; giảm bộ nhớ nhưng có sai số nén; cần học cả cụm và codebook.

**Trang 21. Các phương pháp indexing.** Bảng: Method, Vai trò trong pipeline, Trade-off/lưu ý.
- Inverted index (BM25): lấy candidate documents từ query VI và các bản dịch EN/ZH. Hiệu quả với thuật ngữ chính xác; phụ thuộc analyzer và lexical overlap.
- Exact scoring: rerank document candidates, tìm passages trong các documents đã chọn; không có sai số ANN, chi phí tăng theo số candidate vectors.
- HNSW: global dense retrieval khi có đủ RAM; graph tăng memory, điều chỉnh efSearch.
- IVF-Flat: global dense retrieval bằng cách tìm trong một số clusters; cần train index; nprobe thấp có thể bỏ sót.
- IVF-PQ: global dense retrieval khi memory hạn chế; sai số do quantization.

**Trang 22. Recommended packages: một stack tối giản.** Bảng: Thành phần, Package, Cách sử dụng.
- Lexical document index: Pyserini / Lucene. BM25 trên title, headings và body; kiểm tra analyzer cho VI/EN/ZH; fusion bằng RRF.
- Candidate vector store: NumPy memmap; sắp passage vectors liên tiếp theo document.
- Exact dense scoring: NumPy / PyTorch; normalize nếu dùng cosine similarity.
- Global dense index: Faiss; IndexFlat làm exact reference; HNSW, IVF-Flat hoặc IVF-PQ khi cần giảm chi phí.
- Search service: OpenSearch, lựa chọn thay thế khi cần BM25, vector search, metadata filtering và API trong một hệ thống.

**Trang 23. Trang chia phần.** "4. Các phương pháp truy hồi nâng cao".

**Trang 24. Late chunking: thêm ngữ cảnh khi chunking.**
- Encode một cửa sổ tài liệu dài thành contextual token representation.
- Sau encoding, pooling token vector theo ranh giới từng chunk.
- Điểm chính: "Vector biểu diễn mỗi chunk có thêm ngữ cảnh của toàn bộ văn bản." Slide có hình so sánh naive chunking với late chunking. Nguồn: Late Chunking.

**Trang 25. Late interaction: matching chi tiết ở cấp token.**
- Giữ token vector của query và doc thay vì một pooled vector.
- Được tích hợp trong model BGE-M3 multi-vector scoring hoặc Jina-ColBERT-v2.
- Công thức: s(q, p) = Σ_i max_j ⟨q_i, p_j⟩. Nguồn: BGE-M3 model card; Jina-ColBERT-v2.

**Trang 26. Tổng kết và Q&A.**
- Pipeline: lexical document retrieval → multilingual dense reranking → chọn passage bằng chứng.
- Truy hồi tài liệu y khoa đa ngôn ngữ: kết hợp dịch câu hỏi, multilingual embedding, LLM-based re-rankers.
- Lưu trữ và tìm kiếm đối với 1M văn bản: sử dụng ANN, Faiss. Nâng cao: late chunking, late interaction.

**Trang 27. Trang cảm ơn.** "Cảm ơn và trao đổi".

**Trang 28. Tham khảo 1/2.**
- 1. Cuộc thi R2AI 2026: "Tổng quan, đánh giá, dữ liệu và hướng dẫn nộp bài". Đây là tài liệu khác, không có trong deck này.
- 2. AI Guru: ViBioMIR dataset card (thống kê corpus và data schema).
- 3. Chen et al., BGE M3-Embedding, 2024, arXiv:2402.03216.
- 4. Zhang et al., Qwen3 Embedding, 2025, arXiv:2506.05176. 5. Akram et al., Jina-embeddings-v5-text, 2026, arXiv:2602.15547.

**Trang 29. Tham khảo 2/2.**
- 6. Günther et al., Late Chunking, 2024 (cập nhật 2025), arXiv:2409.04701.
- 7. Jina-ColBERT-v2 model card (late interaction đa ngôn ngữ).
- 8. Scheerer et al., WARP, SIGIR 2025, arXiv:2501.17788. 9. Jin et al., MedCPT, 2023 (mã nguồn NCBI).
- 10. Faiss documentation: chọn index, compression và tham số tìm kiếm.

---

## 3. Quy định / ràng buộc

Deck này **không chứa** quy định cuộc thi. Các mục sau đều không có trong deck:

- **Model được phép và kích thước:** không có quy định. Deck chỉ liệt kê các model với kích thước tham khảo (slide 8, 10), không nói model nào được dùng trong cuộc thi.
- **Giới hạn ngày phát hành model hoặc dữ liệu:** không có. Deck có năm phát hành của từng model, nhưng không nói mốc thời gian nào được chấp nhận.
- **Dữ liệu ngoài và API:** không có quy định. Slide 22 gợi ý dùng API/OpenSearch như công cụ, không phải là quy định.
- **Định dạng submission, dung lượng file, số lần nộp mỗi ngày:** không có.
- **Deadline cho public và private phase:** không có.
- **Quy tắc chọn kết quả cuối cùng:** không có.
- **Yêu cầu tái lập hoặc nộp code, yêu cầu báo cáo:** không có.
- Duy nhất có tham chiếu: slide 28 ghi "Cuộc thi R2AI 2026. Tổng quan, đánh giá, dữ liệu và hướng dẫn nộp bài". Đây là nguồn quy định thật sự, cần đọc riêng.

---

## 4. Gợi ý kỹ thuật của BTC

**Retrieval:**
- Baseline lexical là BM25 trên query tiếng Việt cộng bản dịch EN/ZH. Dense (BGE-M3) đóng vai trò rerank trên shortlist, không phải first stage.
- Query tiếng Việt gốc luôn được dùng cho dense scoring và reranking (slide 13).
- Shortlist: khoảng 400 tài liệu ở document stage B; khoảng 400 passage để rerank ở passage stage (slide 7, 9).

**Reranking:**
- BGE-reranker-v2-m3 là điểm bắt đầu gọn nhẹ. Reranker LLM (BGE-reranker-v2-gemma, Qwen3-Reranker) thử trên tập nhỏ hơn (slide 9).
- Deck không có số liệu so sánh giữa các reranker.

**Chunking:**
- Deck dùng khái niệm "passage" cho bằng chứng và "chunk" cho đoạn cắt tài liệu, không có hướng dẫn cụ thể về kích thước hay overlap.
- Late chunking là hướng nâng cao để chunk có ngữ cảnh (slide 24). Late interaction ở cấp token là hướng nâng cao khác (slide 25).
- Slide 17 yêu cầu đánh giá số lượng chunk trước khi chọn indexing.

**Tổ chức dữ liệu:**
- Giữ ánh xạ passage → tài liệu cha bằng ID (slide 9).
- Lọc boilerplate, trùng lặp và lỗi extraction trước khi index (slide 17).
- Analyzer riêng cho VI/EN/ZH cho BM25 (slide 22).
- Dung lượng vector: xem slide 18 để ước tính RAM/disk.

**Cách tạo gold label:** deck không mô tả cách xây dựng nhãn tham chiếu (gold). Không có thông tin về số tài liệu hoặc chunk cần trả về.

**Đánh giá:** deck không có công thức metric (Precision, Recall, F2) và không có thông tin về tính điểm FINAL. Công thức FINAL trong brief của team không có trong deck này.

**Gợi ý về private test:** không có. Deck không nói gì về private test, public test hay phân chia dữ liệu.

**Thứ tự ưu tiên triển khai dịch query (slide 15):** direct translation → glossary-augmented → retrieved few-shot → fine-tuning (khi có dữ liệu song ngữ y khoa).

---

## 5. Khác với buổi mentor trước

So sánh với `D:\02_Competitions\Kaggle_agent_tool\vibiomir\research\reports\btc_slides_data_org.md` (checklist từ `261012_R2_AI_S3_MentorDay1.pdf`, 48 trang, buổi 12/10).

**Mới trong deck này (22/10):**
- Trọng tâm chuyển sang truy hồi nâng cao: dịch query, bảng embedding và reranker, indexing ANN (HNSW, IVF, PQ), late chunking, late interaction.
- Ước tính dung lượng vector (slide 18), so sánh exact vs ANN (slide 19), các cấu trúc ANN (slide 20).
- Stack gợi ý: Pyserini/Lucene, NumPy memmap, Faiss, OpenSearch (slide 22).
- Pipeline cụ thể: 2,000 → 400 document → ~3,200 passage → rerank 400 passage (slide 9).

**Không còn trong deck này (có trong buổi 12/10):**
- Thu thập dữ liệu: danh sách host, các cách crawl (HTML, API, trình duyệt, sitemap), cảnh báo trang lỗi và trang chọn ngôn ngữ.
- Metadata schema, chuẩn hóa tiêu đề và ngày tháng, làm sạch HTML và phủ định, tách từ.
- Phân nhóm ICD-10 và chương ICD-10 I–XXII.
- Khử trùng (dedup), chunking theo độ dài/cấu trúc/ngữ nghĩa, Q&A chunking.
- Công thức metric: P, R, F2 cấp document và chunk; LCS overlap ≥ 0.40; ví dụ tính F2 (0.714 và 0.882).

**Khác biệt về số liệu và thuật ngữ:**
- Số URL: buổi 12/10 ghi "4.394.718 URL từ 97 host" (trang 9) và "Tổng 1.194.469 URL" (danh sách mới, bản md). Deck 22/10 ghi "nguồn thông tin từ 1.22M URLs" (slide 3) và "1M documents" (slide 17). Hai con số 1.194.469 và "1.22M" không khớp đúng; deck mới chỉ làm tròn (không rõ là làm tròn từ con số nào) [không chắc].
- Thuật ngữ: buổi 12/10 dùng "chunk" làm đơn vị truy hồi. Deck 22/10 dùng "passage" làm đơn vị bằng chứng và "chunk" cho đoạn cắt tài liệu.
- Luồng first-stage: buổi 12/10 mô tả hybrid song song (lexical và dense → hợp thứ hạng bằng RRF → rerank). Deck 22/10 mô tả BM25 là first stage, dense là rerank trên shortlist; RRF chỉ xuất hiện trong stack gợi ý để fusion lexical.
- Mô hình: buổi 12/10 nêu mBERT là ví dụ dense; deck 22/10 không nhắc mBERT và chỉ dùng BGE-M3 làm baseline (BGE-M3 vẫn có ở cả hai buổi).
- Reranker: cả hai buổi đều nêu BGE-reranker-v2-m3; deck 22/10 mở rộng thêm nhiều lựa chọn.

**Không xung đột trực tiếp:** không có mâu thuẫn trực tiếp về quy tắc giữa hai buổi, vì deck 22/10 không có quy định nào. Các số liệu trong buổi 12/10 (metric, LCS) không bị phủ định, chỉ không được nhắc lại.
