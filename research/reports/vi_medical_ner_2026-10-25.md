# NER y tế tiếng Việt cho kho doc và query (25/10/2026, agent Haiku tra web + HF API)

Điều kiện: model mở, tạo trước 01/08/2026, ≤ 15B. Không nguồn nào phủ đủ 5 loại (bệnh, triệu chứng, thuốc, bộ phận, xét nghiệm/thủ thuật). **Không tìm thấy danh mục ICD-10 tiếng Việt mở đã kiểm chứng.**

| Nguồn | HF id / URL | Ngày | Cỡ | Nhãn | License | F1 công bố | Hợp |
|---|---|---|---|---|---|---|---|
| **BamiBERT-ViMedNER** | `cbc-528a/BamiBERT-ViMedNER` | tạo 05/07/2026 (base arXiv 02/07/2026) | 0,1B | ten_benh, trieu_chung_benh, bien_phap_chan_doan, bien_phap_dieu_tri, nguyen_nhan_benh | bsd-3-clause-clear + RAIL (nghiên cứu) | micro 0,686; bệnh 0,81, triệu chứng 0,62 | Văn bản sức khoẻ trên web ⇒ hợp tiêu đề/heading |
| ViMQ (dataset) | github.com/tadeephuy/ViMQ | 2021 | 9.000 câu hỏi bệnh nhân (Vinmec) | SYMPTOM&DISEASE, MEDICAL_PROCEDURE, MEDICINE | không có file license | PhoBERT micro 70,8 | Hợp nhất với **query**; dùng làm dữ liệu train |
| Qwen2.5-7B-Vietnamese-Medical-NER | `PeterPaker123/...` | 30/05/2026 | 7,6B | bệnh/triệu chứng, thủ thuật, thuốc, bộ phận, liều, nhân khẩu | Apache-2.0 base; dữ liệu train research-only ⇒ không rõ | không công bố | Đủ loại nhưng card không đáng tin |
| ViMedNER | github.com/tdtrinh11/ViMedNer | 2024 | ? | bệnh, triệu chứng, nguyên nhân, chẩn đoán, điều trị | ? | XLM-R tốt nhất | Web, nhưng nhãn mâu thuẫn giữa các nguồn |
| PhoNER_COVID19, VietBioNER, VietMed-NER, VnCoreNLP/underthesea | — | 2018–2024 | — | COVID / lao / ASR / NER chung | — | — | Không hợp |

`research/refs/AIVIETNAM-AIO-TuNguyen3107_onto-med-rag` chỉ là khung rỗng (chưa có model, chưa tải ICD-10).

## Đề xuất cho hệ thống
1. **Doc (730k VI):** BamiBERT-ViMedNER trên tiêu đề + heading + ~600 ký tự đầu ⇒ thực thể bệnh / triệu chứng / chẩn đoán / điều trị. Model nhỏ nên chạy GPU nhanh.
2. **Query (1.200):** Qwen2.5-7B-Instruct (09/2024, Apache-2.0) với JSON có schema:
   - thực thể chính + tên gọi khác;
   - ý hỏi;
   - câu hỏi con.

   Bỏ thực thể nào không xuất hiện nguyên văn trong query; tên gọi khác thì để riêng một trường.
3. **Chuẩn hoá thực thể:** bỏ dấu + chữ thường + gộp biến thể. Sau đó dùng LLM gom các chuỗi thực thể duy nhất (đã khử trùng, ít hơn nhiều so với số doc) về tên chuẩn ⇒ index "thực thể → doc".
4. Thuốc/bộ phận không có trong BamiBERT ⇒ lấy từ LLM hoặc gazetteer tự dựng từ tiêu đề Long Châu (trang thuốc/hoạt chất).
5. Kiểm license khi viết working notes (BamiBERT: dùng nghiên cứu).
