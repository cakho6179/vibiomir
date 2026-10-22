# Checklist: quy trình dữ liệu và truy hồi trong slide mentor ViBioMIR (R2AI2026)

Nguồn: `261012_R2_AI_S3_MentorDay1.pdf` (48 trang), đối chiếu với bản markdown `261012_R2_AI_S3_MentorDay1.md`. Số `[tr.N]` là số trang PDF. Mục đánh dấu `[md]` nằm trong phần mở đầu của bản markdown, không có trên slide.

Các trang không có nội dung kỹ thuật: 1 (tiêu đề), 2 (mục lục), 3, 8, 27 (trang chia phần), 44 (cảm ơn).

## Tổng quan đầu vào và pipeline (nhóm bổ sung)

1. [tr.4] Đầu vào: "một truy vấn y khoa bằng tiếng Việt". Kho dữ liệu: "tài liệu từ các nguồn Việt – Anh – Trung". "Hệ thống cần tìm đúng tài liệu và định vị đúng đoạn nội dung gốc."
2. [tr.9] "4.394.718 URL từ 97 host." "Mỗi tài liệu có một ID và một URL." "Danh sách URL là điểm bắt đầu. Cần thu thập và xử lý nội dung trước khi có thể tìm kiếm."
3. [tr.9] Tập truy vấn: "1.200 câu hỏi tiếng Việt." "Độ dài trung vị: 18 từ." "Có câu hỏi ngắn và câu hỏi chứa nhiều ý."
4. [tr.16] Pipeline: "Danh sách URL → Thu thập nội dung → Trích xuất metadata và văn bản → Làm sạch và chuẩn hóa → Chia đoạn nội dung → Lập chỉ mục và truy hồi."
5. [tr.26] "Không thu thập được bài liên quan thì mô hình không thể tìm thấy bài đó." "Trích xuất thiếu đoạn quan trọng thì bằng chứng bị mất." "Nội dung nhiễu hoặc trùng làm kết quả tìm kiếm khó chính xác." "Chia đoạn không phù hợp làm mất ngữ cảnh của câu trả lời."

## Thu thập

1. [md] Danh sách URL mới `links_corpus_vi_en_zh.parquet` tại "AIGuruTinix/ViBioMIR · Datasets at Hugging Face". Tổng "1.194.469 URL": Tiếng Việt "Giữ nguyên 714.469 URL"; Tiếng Trung "Rút gọn còn 240.000 URL"; Tiếng Anh "Bổ sung 240.000 URL".
2. [md] "Mã định danh (ID) của các liên kết tài liệu không thay đổi. Các đội đã thu thập dữ liệu chỉ cần loại bỏ những tài liệu tiếng Trung đã bị rút khỏi danh sách và bổ sung tài liệu tiếng Anh vào kho dữ liệu truy hồi."
3. [tr.10] Bảng host lớn: "www.cnkang.com 963.438; www.120ask.com 918.479; www.familydoctor.com.cn 447.453; ask.39.net 337.375; zysjonline.com 243.347." "Hai host lớn nhất chiếm khoảng 42,8% số URL. Cần hiểu cấu trúc nguồn lớn và vẫn quan sát các nguồn nhỏ để tránh bỏ sót loại tài liệu."
4. [tr.12] "Một trang web gồm bài viết, menu, quảng cáo và nội dung liên quan." "Có trang chỉ trả về thông báo hoặc màn hình chọn ngôn ngữ." "Tiêu đề, tác giả và ngày đăng có thể thiếu hoặc không đúng với bài."
5. [tr.12] "Cần nhận diện phần nội dung chính trước khi lập chỉ mục." Câu hỏi kiểm tra: "Văn bản đang thu được có phải là bài viết cần tìm và còn giữ đầy đủ cấu trúc quan trọng không?"
6. [tr.13] Ví dụ 1 (vinmec, "Bản thu thập ngày 11/10/2026"): "Tiêu đề chỉ còn tên website." "Trang thông báo chưa có nội dung tiếng Anh." "Yêu cầu tiếp tục để đọc tiếng Việt." Nếu đưa vào chỉ mục ngay: "Không có kiến thức về chóng mặt." "Kết quả tìm kiếm có thể trả về thông báo thay cho bằng chứng."
7. [tr.15] "Đọc HTML trực tiếp: nhanh và đơn giản với trang tĩnh. Khó xử lý nội dung được tải bằng JavaScript."
8. [tr.15] "API công khai: dữ liệu thường có cấu trúc rõ. Không phải nguồn nào cũng cung cấp API."
9. [tr.15] "Trình duyệt tự động: lấy được trang cần render. Tốn thời gian và tài nguyên hơn."
10. [tr.15] "Sitemap: hỗ trợ tìm danh sách trang. Thường không cung cấp đầy đủ nội dung bài viết."
11. [tr.15] Cách lựa chọn: "Khảo sát từng nguồn để chọn cách thu thập phù hợp trước khi mở rộng quy mô."

## Trích xuất vùng bài

1. [tr.14] Ví dụ 2 (Nhà thuốc Long Châu): "Lấy vùng bài viết thay vì toàn bộ trang." "Giữ tiêu đề mục cùng đoạn bên dưới." "Chia đoạn theo từng mục để tìm đúng nội dung."
2. [tr.14] Ví dụ các mục trong bài: "Thành phần dinh dưỡng trong 1 bánh Trung thu", "Bánh nướng và bánh dẻo", "Lượng calo trung bình", "Các câu hỏi thường gặp."
3. [tr.19] "Bỏ phần giao diện, giữ phần bài viết." Loại khỏi nội dung chính: "Trang chủ / Tìm kiếm" (VI), "Home / Search" (EN), "首页 / 搜索 / 相关文章" (ZH). Giữ tiêu đề bài: "Sỏi thận", "Kidney stones", "肾结石".
4. [tr.11] "Mỗi loại tài liệu cần một cách trích xuất và chia đoạn phù hợp."

## Metadata

1. [tr.17] "Thông tin mô tả: tiêu đề, tóm tắt, chuyên mục và ngôn ngữ."
2. [tr.17] "Thông tin nguồn: URL, tác giả, ngày đăng và ngày cập nhật."
3. [tr.17] "Thông tin tổ chức: loại tài liệu và nhóm chủ đề."
4. [tr.17] "Đưa các cách ghi khác nhau về một cấu trúc thống nhất."
5. [tr.17] "Metadata giúp hiểu tài liệu, phân nhóm nội dung và hỗ trợ tìm kiếm trước khi đọc toàn văn."
6. [tr.18] Ví dụ cùng chủ đề, giữ nguyên tiêu đề và ngôn ngữ: VI "Sỏi thận: triệu chứng và chẩn đoán"; EN "Kidney stones: symptoms and diagnosis"; ZH "肾结石：症状与诊断".
7. [tr.18] Ngày đăng "khác cách viết, cùng một định dạng": "15/09/2026 | September 15, 2026 | 2026 年 9 月 15 日 → 2026-09-15".
8. [tr.18] "Giữ riêng: ID, URL, tiêu đề gốc, ngôn ngữ và ngày đăng." "Thiếu thông tin thì để trống, không tự suy tác giả hay ngày đăng."

## Làm sạch / chuẩn hoá

1. [tr.19] "Làm sạch cách biểu diễn": "<p>Sỏi thận</p> → Sỏi thận"; "<p>Kidney stones</p> → Kidney stones"; "<p> 肾结石 </p> → 肾结石".
2. [tr.19] "Giữ cấu trúc: tiêu đề mục, đoạn văn, danh sách và bảng."
3. [tr.20] Ví dụ ba câu phủ định: VI "Không ghi nhận sốt.", EN "No fever reported.", ZH "未报告发热". "Bỏ từ phủ định sẽ làm thay đổi thông tin của câu."
4. [tr.20] "Không xóa dấu tiếng Việt hoặc tự bỏ các từ được coi là ít quan trọng."
5. [tr.20] "Giữ nguyên số và đơn vị: 5 mg khác 50 mg."
6. [tr.20] Nguyên tắc: "Làm sạch định dạng, không viết lại ý nghĩa. Lưu văn bản gốc để đối chiếu."
7. [tr.21] Tách từ cho tìm kiếm từ khóa: VI "triệu chứng sỏi thận → triệu_chứng | sỏi_thận"; EN "kidney stones symptoms → kidney | stones | symptoms"; ZH "肾结石症状 → 肾结石 | 症状".
8. [tr.21] "Tiếng Việt: khoảng trắng có thể nằm giữa các tiếng của một từ." "Tiếng Trung: nhiều từ không được ngăn cách bằng khoảng trắng." "Dense retrieval dùng tokenizer đi kèm mô hình."
9. [tr.21] "Cùng khái niệm: sỏi thận ↔ kidney stones ↔ 肾结石."
10. [tr.7] Khi đối chiếu chunk với tham chiếu: "Chuẩn hóa NFKC, HTML entity, lowercase và khoảng trắng."

## Chunking

1. [tr.22] Các cách chia phổ biến: "Theo độ dài: đơn giản và dễ kiểm soát." "Theo cấu trúc: dựa trên câu, đoạn và tiêu đề mục." "Theo ngữ nghĩa: gom các câu cùng chủ đề."
2. [tr.22] Đánh đổi: "Đoạn quá ngắn dễ mất ngữ cảnh." "Đoạn quá dài dễ chứa nhiều nội dung không liên quan." "Overlap giữ ngữ cảnh nhưng tạo nội dung trùng."
3. [tr.22] "Truy hồi đoạn giúp định vị bằng chứng cụ thể thay vì trả về cả tài liệu dài."
4. [tr.23] Ví dụ một bài, hai mục (Triệu chứng, Chẩn đoán) ở cả ba ngôn ngữ: "Chunk A: mục Triệu chứng và các câu thuộc mục này." "Chunk B: mục Chẩn đoán và các câu thuộc mục này."
5. [tr.23] "Giữ ID và vị trí gốc để truy ngược mỗi chunk về tài liệu."
6. [tr.24] Hỏi đáp: "Cách chia: giữ cặp hỏi đáp cùng đoạn nếu độ dài cho phép." (ví dụ VI/EN/ZH)

## Khử trùng

1. [tr.25] Ví dụ hai URL chứa cùng một bài tiếng Việt (A và B giống hệt): "→ Nhận diện bản trùng, vẫn giữ ánh xạ về các ID và URL gốc."
2. [tr.25] "Cùng chủ đề chưa chắc là bản trùng" (ví dụ "Sỏi thận | Kidney stones | 肾结石"). "→ Không xóa bài chỉ vì tiêu đề gần nghĩa hoặc khác ngôn ngữ."
3. [tr.25] "Kiểm tra trang lỗi, nội dung rỗng và đoạn mất ngữ cảnh."
4. [tr.25] "Giữ nguồn của từng chunk để trả về đúng bằng chứng."
5. [tr.41] "Loại các đoạn bị lặp do chia đoạn chồng lấn." "Gộp các đoạn thuộc cùng một tài liệu."

## Phân nhóm ICD-10 / loại tài liệu

1. [tr.11] Các loại nội dung trong corpus: "Hỏi đáp y khoa: câu hỏi, câu trả lời và thông tin người trả lời." "Bài kiến thức: bệnh, triệu chứng, chẩn đoán và điều trị." "Sách và cẩm nang: chương, mục, bảng và tài liệu tham khảo." "Thuốc, dược liệu và tin tức: cấu trúc và mục đích sử dụng khác nhau."
2. [tr.28] "ICD-10 tổ chức bệnh và vấn đề sức khỏe theo hệ phân cấp." "Có thể bắt đầu bằng nhóm lớn như tuần hoàn, hô hấp, nội tiết và tiêu hóa."
3. [tr.28] "Dựa trên tiêu đề, tóm tắt và chuyên mục để gán nhóm chủ đề." "Một tài liệu có thể thuộc nhiều nhóm."
4. [tr.28] "Có một hệ quy chiếu thống nhất để tổ chức dữ liệu và phân tích kết quả giữa các ngôn ngữ."
5. [tr.29] Dựa trên từ điển và quy tắc: "Khớp tên bệnh và từ đồng nghĩa." "Dễ giải thích và không cần nhiều dữ liệu gán nhãn." "Dễ bỏ sót cách diễn đạt mới hoặc từ nhập nhằng."
6. [tr.29] Dựa trên mô hình ngôn ngữ: "So khớp ngữ nghĩa hoặc phân loại văn bản." "Nhận diện được nhiều cách diễn đạt." "Cần kiểm tra nhãn sai và khả năng phù hợp y sinh."
7. [tr.29] Hướng tiếp cận: "Bắt đầu bằng nhóm lớn, kết hợp nhiều tín hiệu và giữ trạng thái chưa xác định khi thiếu bằng chứng."
8. [tr.30] Ví dụ "Đái tháo đường type 2" (VI) / "Type 2 diabetes" (EN) / "2 型糖尿病" (ZH) → "Nhóm E00–E90: bệnh nội tiết, dinh dưỡng và chuyển hóa." "Từ điển đa ngôn ngữ giúp nối các tên gọi về cùng nhóm."
9. [tr.30] "Đọc thêm tóm tắt hoặc nội dung nếu tiêu đề chưa đủ rõ." "Tài liệu có nhiều chủ đề có thể nhận nhiều nhóm."
10. [tr.30] "Không suy chẩn đoán: tiêu đề 'đau ngực' chưa đủ để kết luận bệnh mạch vành."
11. [tr.31] Dùng nhóm ICD-10 để: "Khám phá dữ liệu" (nhóm nào nhiều hoặc ít tài liệu); "Phân tích kết quả" (làm tốt và chưa tốt ở nhóm nào); "Hỗ trợ tìm kiếm" (ưu tiên tài liệu thuộc nhóm liên quan); "Hỗ trợ dữ liệu đa ngôn ngữ" (kết nối các cách gọi về cùng một nhóm).
12. [tr.31] "Tránh lọc quá chặt: Câu hỏi có thể liên quan nhiều bệnh. Phân nhóm sai không nên làm mất toàn bộ tài liệu hữu ích."
13. [tr.45] Chương ICD-10 I–XI: I (A00–B99), II (C00–D48), III (D50–D89), IV (E00–E90), V (F00–F99), VI (G00–G99), VII (H00–H59), VIII (H60–H95), IX (I00–I99), X (J00–J99), XI (K00–K93).
14. [tr.46] Chương ICD-10 XII–XXII: XII (L00–L99), XIII (M00–M99), XIV (N00–N99), XV (O00–O99), XVI (P00–P96), XVII (Q00–Q99), XVIII (R00–R99), XIX (S00–T98), XX (V01–Y98), XXI (Z00–Z99), XXII (U00–U85).

## Index / truy hồi baseline

1. [tr.32] Bảng nhóm phương pháp: Lexical retrieval "Khớp từ khóa" (TF-IDF, BM25); Dense retrieval "So khớp vector ngữ nghĩa" (mBERT, BGE-M3); Hybrid retrieval "Kết hợp lexical và dense" (Hợp thứ hạng, RRF); Reranking "Đọc kỹ tập ứng viên" (Cross-encoder).
2. [tr.32] "Bắt đầu bằng phương pháp đơn giản để hiểu dữ liệu, sau đó bổ sung thành phần khi giải quyết được một hạn chế cụ thể."
3. [tr.33] Kiến trúc: "Tìm theo từ khóa (Lexical retrieval)" và "Tìm theo ngữ nghĩa (Dense retrieval)" → "Hợp các tài liệu và đoạn ứng viên" → "Xếp hạng lại theo mức độ liên quan" → "Trả về tài liệu và đoạn bằng chứng gốc".
4. [tr.34] TF-IDF: "Ưu tiên từ xuất hiện nhiều trong bài nhưng hiếm trong corpus." "Chưa mô hình hóa tốt ảnh hưởng của độ dài và lặp từ."
5. [tr.34] BM25: "Bổ sung bão hòa tần suất từ và điều chỉnh độ dài." "Phù hợp với tài liệu có độ dài khác nhau." "Vẫn phụ thuộc từ khóa xuất hiện trong văn bản."
6. [tr.35] Lexical: "Nhanh và tương đối ít tốn tài nguyên." "Mạnh với thuật ngữ hoặc thực thể chính xác." "Hạn chế khi query và tài liệu khác ngôn ngữ." Lý do vẫn dùng: "Là mốc so sánh đơn giản và bổ sung khả năng khớp chính xác mà tìm kiếm ngữ nghĩa có thể bỏ sót."
7. [tr.36] Dense: "Mã hóa câu hỏi và đoạn văn thành vector." "Tìm các đoạn có vector gần với vector của câu hỏi." "Ví dụ mô hình: mBERT và BGE-M3."
8. [tr.37] Dense: "Có thể hỗ trợ truy hồi xuyên ngôn ngữ." "Dễ nhầm đoạn gần nghĩa nhưng sai chi tiết." "mBERT cần cách tạo vector và huấn luyện phù hợp. BGE-M3 được huấn luyện cho retrieval."
9. [tr.38] Hybrid: "Lấy ứng viên từ lexical và dense." "Có thể hợp thứ hạng bằng RRF hoặc kết hợp điểm sau chuẩn hóa." "Phải vận hành nhiều chỉ mục." "Cần chọn cách hợp kết quả phù hợp."
10. [tr.39] Reranking: "Đọc đồng thời câu hỏi và từng đoạn ứng viên." "Chỉ xử lý một tập ứng viên nhỏ." "Retrieval tìm rộng, reranking chọn kỹ. Ví dụ mô hình đa ngôn ngữ: BGE-reranker-v2-m3."
11. [tr.40] Ví dụ "Triệu chứng của sỏi thận là gì?": A (VI) "Chẩn đoán sỏi thận" = "Cùng bệnh, khác ý"; B (EN) "Symptoms of kidney stones" = "Đúng ý triệu chứng"; C (ZH) "肾结石 : 症状" = "Đúng ý triệu chứng". Cột "Mức liên quan kỳ vọng".
12. [tr.40] "Lexical dễ khớp từ tiếng Việt ở A nhưng có thể bỏ sót B, C." "Dense đa ngôn ngữ có thể đưa cả ba đoạn vào tập ứng viên." "Reranker cần ưu tiên nội dung trả lời đúng ý, dù khác ngôn ngữ."
13. [tr.41] "Chọn các đoạn liên quan đến từng ý trong câu hỏi." "Trả về nội dung gốc và tài liệu nguồn tương ứng."
14. [tr.47] Phụ lục TF-IDF: "wt,d = tf(t, d) idf(t)." "TF: mức xuất hiện trong tài liệu." "IDF: mức phân biệt trong corpus." "So sánh bằng cosine."
15. [tr.47] Phụ lục BM25: "Sử dụng độ hiếm của từ." "Giảm lợi ích của việc lặp từ quá nhiều." "Điều chỉnh theo độ dài tài liệu." "BM25 bổ sung bão hòa tần suất từ và chuẩn hóa độ dài thay vì chỉ dựa trên trọng số TF-IDF."

## Đánh giá

1. [tr.5] "D: tài liệu dự đoán. G: tài liệu tham chiếu." P = |D ∩ G| / |D|, R = |D ∩ G| / |G| (cấp tài liệu).
2. [tr.5] Chunk retrieval: "Precision: tỷ lệ chunk nộp được tính đúng. Recall: tỷ lệ chunk tham chiếu tìm thấy."
3. [tr.5] F2(q) = 5·P(q)·R(q) / (4·P(q) + R(q)). F2 macro = (1/|Q|) Σ F2(q). "Tính từng truy vấn rồi lấy trung bình, riêng cho tài liệu và đoạn."
4. [tr.6] Ví dụ G = {d2, d5, d9}. Trả 2 tài liệu D_A = {d2, d5}: P = 1, R = 2/3, F2 ≈ 0,714. Trả 5 tài liệu D_B = {d2, d5, d9, d11, d12}: P = 0,6, R = 1, F2 ≈ 0,882.
5. [tr.6] Bài học: "Ví dụ B tìm đủ tài liệu liên quan nên có điểm cao hơn. Trả thêm tài liệu không liên quan vẫn làm precision giảm."
6. [tr.7] Khớp chunk với tham chiếu: "Đối chiếu trong cùng truy vấn và cùng doc_id."
7. [tr.7] "LCS giữ thứ tự token nhưng không bắt buộc liên tiếp." overlap(c, g) = LCS_tokens(c, g) / |g| ≥ 0,40.
8. [tr.7] Ví dụ: "Đoạn tham chiếu 100 token, LCS 45 token: overlap = 45%, đạt ngưỡng."
9. [tr.42] "So sánh trên cùng corpus, cùng bộ câu hỏi và cùng cách chia đoạn." "Đo Precision, Recall và F2 ở cấp tài liệu và đoạn."
10. [tr.42] "Xem thêm thời gian tìm kiếm và tài nguyên sử dụng." "Phân tích theo ngôn ngữ, nhóm bệnh và loại câu hỏi."
11. [tr.42] Câu hỏi cần trả lời: "Phương pháp mới cải thiện ở đâu, còn sai ở đâu và có đáng với chi phí bổ sung không?"
12. [tr.23] Nhãn gợi ý cho câu hỏi triệu chứng: "Câu hỏi về triệu chứng nên tìm đến chunk A."
13. [tr.40] Mức liên quan kỳ vọng theo ứng viên: "Cùng bệnh, khác ý" và "Đúng ý triệu chứng" (xem Index mục 11).

Ghi chú: slide không mô tả cách xây dựng nhãn tham chiếu (gold) ngoài các ý nêu ở mục 12–13 phần Đánh giá. Slide cũng không mô tả cách ghi ID cho bản trùng ngoài "ánh xạ về các ID và URL gốc" [tr.25].
