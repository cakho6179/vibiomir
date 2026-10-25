# Báo cáo: arXiv 2610.27099 — "A general-purpose System One decision model for clinical records"

Nguồn: https://arxiv.org/abs/2610.27099 và https://arxiv.org/html/2610.27099 (đọc qua WebFetch, hai trang đều được tóm tắt bởi model nhỏ; PDF https://arxiv.org/pdf/2610.27099 không đọc được phần thân, chỉ lấy được metadata). Không có repo GitHub/HF nào được liên kết trong trang abs, nên không có README để đọc.

Lưu ý độ tin cậy: các con số dưới đây lấy từ bản tóm tắt của WebFetch, không phải văn bản gốc được copy nguyên. Mọi số liệu nên đối chiếu lại với PDF gốc trước khi dùng cho quyết định. Các mục đánh dấu "chưa kiểm chứng" là chưa xác minh được.

## 1. Thông tin chung

- Tiêu đề: "A general-purpose System One decision model for clinical records".
- Tác giả: 17 người. Tác giả đầu: Andrew Zhang, Trinh T.L. Vuong, Sophia J. Wagner. Long Phi Le và Faisal Mahmood là đồng tác giả cấp cao (co-senior); Faisal Mahmood là tác giả liên hệ.
- Tổ chức: Mass General Brigham AI Institute (Boston), Harvard Medical School, MIT.
- Ngày nộp v1: 19/10/2026 (18:00:51 UTC), arXiv cs.AI.
- "System One" theo cách dùng của bài: theo thuật ngữ của Kahneman, là phán đoán nhanh, có kiểu (typed). Đối lập với System 2 (suy luận chậm, như chẩn đoán và lập kế hoạch điều trị). Mô hình System One nhận một trạng thái bệnh nhân cùng một tập câu hỏi có kiểu và trả về phân phối xác suất cho từng câu hỏi trong một forward pass.
- Mô hình chính: CS1-4B (Clinical System One, 4B tham số).

## 2. Định dạng bài toán

- Loại câu hỏi: Choice (2–11 lựa chọn có tên), yes/no, và Score (mức có thứ tự 1..N). Mỗi câu có lựa chọn "not_documented" khi tài liệu có thể im lặng (Methods 0.11).
- Đầu vào: một tài liệu (ghi chú lâm sàng, báo cáo chẩn đoán, hồ sơ nhiều ghi chú, hoặc timeline có cấu trúc được render thành text) cộng câu hỏi có kiểu viết tại thời điểm truy vấn. Mỗi lựa chọn có tiêu chí cho ca biên, nên thêm tiêu chí mới không cần train lại (Results 0.1).
- Đầu ra: candidate head trả một logit cho mỗi lựa chọn, softmax theo từng câu hỏi cho ra phân phối đáp án (Methods 0.11).
- Đọc xác suất: báo cáo kết quả dùng xác suất thô. Một temperature theo loại câu hỏi "có thể được fit" trên tập validation nhưng không áp dụng cho kết quả báo cáo (Methods 0.11).
- Định tuyến: đáp án có P ≥ 0.95 được chấp nhận tự động, phần còn lại chuyển cho reviewer (Fig. 1f). Không thấy cơ chế abstention được huấn luyện riêng.

## 3. Backbone

- CS1-4B: full fine-tune từ Decision 2.0 Nox-4B, là kiến trúc hybrid Qwen3.5 với 4.2B tham số (24 lớp Gated DeltaNet và 8 lớp gated full-attention) (Methods 0.11). Abstract và Fig. 1 gọi là "4-billion" và "4.2 billion".
- Ngày phát hành: Decision 2.0 được trích là năm 2026 (ref. 27), ngày cụ thể không nêu, chưa kiểm chứng. Qwen3.5 có blog 2026 (ref. 103), ngày cụ thể không nêu. GLM-5.3-Flash: không nêu ngày.
- License: Decision 2.0 được ghi là Apache-2.0 (ref. 27). Trang arXiv ghi CC BY-NC-ND 4.0 cho bài báo.
- Teacher gán nhãn: GLM-5.3-Flash, MoE 320B tổng và 18B active (Methods 0.10).
- Trọng số CS1-4B: "credentialed release upon institutional approval", chưa phát hành. Code (inference, training, benchmark) cũng "will be released upon institutional approval".

## 4. Dữ liệu huấn luyện và nhãn

- Nguồn: Klaros-12M của Mass General Brigham. 12,416,448 cặp document–question, 3,643,798 bệnh nhân, 7.47M document, 24 cơ sở bệnh viện (Results 0.1; Fig. 1a). Không dùng dữ liệu công khai hay dữ liệu từ viện khác để fine-tune.
- Phân rã (Extended Data Table 1):
  - Ghi chú lâm sàng (nhãn teacher readout): 4,511,638 cặp.
  - Báo cáo chẩn đoán (teacher readout): 2,460,078 cặp.
  - Cấu trúc, dự báo outcome: 433,430 cặp.
  - Cấu trúc, tiêu chí theo rule (nhãn chính xác): 5,011,302 cặp.
- Cách tạo nhãn: LLM teacher (GLM-5.3-Flash) lấy xác suất next-token, trung bình qua các hoán vị thứ tự lựa chọn để triệt bias vị trí. Temperature T=1 (Methods 0.10). Không thấy distillation từ mô hình System 2 riêng, cũng không thấy dữ liệu tổng hợp độc lập.
- Kiểm tra nhãn: GLM-5.3 đồng ý với key người ở 94.0% (Apixaban), 97.9% (EHRNoteQA), 94.4% (LongHealth), 90.6% (MIMIC-III phenotypes) (Methods 0.10).
- Lọc: nhãn theo rule chỉ giữ khi rule áp dụng lên text đã render cho cùng đáp án như trên hồ sơ đầy đủ (Methods 0.8).
- Tiếng Việt hoặc đa ngôn ngữ: không nêu. Hard negatives: không nêu thành mục riêng (một số bộ đánh giá có cân bằng đáp án theo template).

## 5. Công thức huấn luyện

- Full fine-tune, không dùng LoRA. Loss là soft cross-entropy trung bình trên các câu hỏi (Methods 0.12).
- Optimizer: fused AdamW, β1=0.9, β2=0.98, ε=1e-6, weight decay 0.01, gradient clip 1.0.
- Ba giai đoạn:
  1. 13,591 steps, peak LR 7e-6, ưu tiên timeline có cấu trúc.
  2. 11,709 steps, anneal lại từ 3.6e-6 xuống 7e-7, ưu tiên ghi chú và báo cáo.
  3. 1,141 steps, optimizer mới, LR 3e-6 xuống 3e-7, trên tài liệu nhiều ghi chú.
- Tổng: 26,441 steps, khoảng 13.8B token, 26.4 giờ trên 8 GPU NVIDIA B300 (211 GPU-giờ).
- Độ dài: document tối đa 15,356 token, segment tối đa 1,024, tổng tối đa 20,480.
- Batch size và số epoch: không nêu.
- Không thấy RL, DPO, hay bước calibration riêng trong quá trình train.

## 6. Calibration, ngưỡng, abstention

- Calibration thô: top-label ECE trên bộ public 0.012–0.052 tùy loại task (Nox-4B 0.033–0.139). ECE gộp: CS1-4B 0.029, Nox-4B 0.058, JEV-27B 0.010, Clef 0.024 (Extended Data Table 2b; Results 0.4).
- Recalibration theo outcome: dịch log-odds theo từng outcome để khớp tỉ lệ mắc, ước lượng trên nửa bệnh nhân còn lại. ECE giảm còn 0.014, Brier từ 0.182 xuống 0.110 (Nox-4B sau cùng hiệu chỉnh là 0.118) (Results 0.4).
- Ngưỡng: P ≥ 0.95 tự động chấp nhận (Fig. 1f). Không có quy trình chọn ngưỡng hình thức. Table 2c báo coverage tại precision 99%, tức sắp xếp phán đoán theo confidence.

## 7. Đánh giá

- Held-out MGB (MGBQA-1M): 963,408 bệnh nhân, 28 task, metric balanced accuracy. CS1-4B trung bình 89.7 trên 6 loại task, Nox-4B 73.3, Vega-27B 81.7. Thắng 13/14 task theo rule, outcome, hoặc LLM mở (Results 0.2).
- 21 bộ public bên ngoài: CS1-4B trung bình 86.7, Nox-4B 81.7, Vega-27B 88.6. CS1-4B nhanh hơn 6.4 lần. Thời gian: 1.80 s so với 11.54 s mỗi document 40 câu hỏi (Results 0.2–0.3).
- Bảng con (Extended Data Table 3, balanced accuracy CS1 / JEV / Clef / Vega / Nox):
  - Care process (rule): 95.4 / 72.1 / 66.2 / 64.2 / 49.2.
  - Conditions (rule): 98.4 / 93.1 / 90.9 / 94.3 / 85.3.
  - Measurements (rule): 98.7 / 90.1 / 89.2 / 90.5 / 79.5.
  - Medications (rule): 92.9 / 76.8 / 72.5 / 80.1 / 58.8.
  - Coding review (rule): 89.3 / 67.9 / 73.5 / 76.7 / 61.9.
  - Imaging (teacher): 91.6 / 90.1 / 88.5 / 88.6 / 85.3.
  - Pathology/cytology (teacher): 91.7 / 89.1 / 88.9 / 89.0 / 85.0.
- Bảng public (Extended Data Table 4): Apixaban trial criteria CS1 84.3, JEV 86.7, Clef 85.3, Nox 86.8, Vega 88.1. MIMIC-Timelines CS1 92.7, JEV 92.4, Clef 93.4, Nox 85.9, Vega 93.0.
- Prognosis (mean AUROC, JEV / Clef / Nox / Vega / CS1): MIMIC 0.766 / 0.753 / 0.702 / 0.769 / 0.721; eICU 0.741 / 0.729 / 0.702 / 0.745 / 0.725; EHRSHOT 0.652 / 0.681 / 0.595 / 0.672 / 0.766.
- Attribution: xóa 3 câu hàng đầu làm đổi đáp án trong 41% (necessity) và 59% (sufficiency) trường hợp, so với 1% với câu ngẫu nhiên (Results 0.5).
- Ablation: không thấy bảng ablation đầy đủ về loss, LR, hay data mix trong phần đọc được (chưa kiểm chứng). Ablation duy nhất có số liệu là so sánh serializer (Extended Data Table 5, CS1-4B, timeline so với encounter):
  - Timelines: 87.5 so với 93.3 (+5.8 [+4.4, +7.3]).
  - Composite: 83.1 so với 85.4 (+2.3 [−0.4, +5.0]).
  - Prognosis AUROC: 0.697 so với 0.721 (+0.024 [−0.002, +0.047]).
  - Kết luận: layout encounter tốt hơn layout timeline ở hai/ba nhóm, và trong nhóm composite khoảng tin cậy chạm 0.
- Extended Data Fig. 3c: đọc document một lần rồi chạy một pass mỗi câu hỏi cho cùng top answer.
- Trial screening (Results 0.7): criterion-level balanced accuracy 0.884 so với 0.869 của LLM thô (tiêu chí có cấu trúc); 0.766 so với 0.796 (tiêu chí chỉ có ghi chú). nDCG@20 0.64 so với 0.11 của BM25. Số bệnh nhân đủ điều kiện trong top 20: 3.5 so với 0.8. Ít token LLM hơn 690 lần so với đọc toàn bộ hồ sơ.
- Whole-record QA: chỉ định tính, ba ví dụ, không có điểm tổng hợp.

## 8. Retrieval, reranking, RAG (liên quan trực tiếp)

- Trial screening như ranking: xếp bệnh nhân theo P(eligible) = tích xác suất inclusion và phần bù của xác suất exclusion. Baseline là BM25 trên toàn bộ hồ sơ, dùng tiêu chí inclusion làm query (Methods 0.14).
- Whole-record QA là kiểu agentic, không phải RAG cổ điển: LLM lập kế hoạch các câu hỏi con từ manifest, CS1 trả lời trên từng document, code tổng hợp, trích dẫn được kiểm tra với evidence list.
- Không có model rerank riêng. Evidence sentence được chọn bằng sufficiency attribution của CS1 (Extended Data Fig. 5), không phải bước retrieval.

## 9. Hạn chế do tác giả nêu

- Phụ thuộc thiết kế curation; mã thanh toán có thể thiếu timing và mức độ nặng. Outcome có thể không quan sát được nếu chăm sóc diễn ra ở nơi khác.
- Một số key do model viết, có thể lệch hệ thống so với annotation lâm sàng.
- Không dành cho chẩn đoán phân biệt từ mô tả ca; ở đó LLM suy luận tốt hơn.
- Phụ thuộc vào context: thiếu thông tin có thể làm sai lệch đáp án.
- Lợi thế open model có thể đến từ dữ liệu huấn luyện không kiểm toán được.
- Đánh giá bằng cùng một LLM viết câu hỏi có nguy cơ self-preference; κ=0.84 đo tính nhất quán, không đo đồng thuận với bác sĩ.
- Chưa validate tiền cứu. Tác giả nhấn mạnh đây là công cụ nghiên cứu, vận hành và rà soát có người trong vòng lặp.

## 10. Đối chiếu với bài toán của chúng ta

Bài toán: retrieval y tế tiếng Việt, 1,200 query, ~730k doc web sức khỏe, F2 trên doc và chunk. Mô hình: teacher Qwen3-Reranker-8B gán nhãn cặp (query, section), distill vào Qwen3-Reranker-0.6B/4B bằng logit yes/no, isotonic calibration, cutoff theo query.

Ràng buộc: chỉ dùng open-weight ≤15B phát hành trước 2026-08-01.

Kiểm tra quy tắc với các model của bài:
- CS1-4B: trọng số chưa phát hành, chỉ credentialed. Không đủ điều kiện (không phải open-weight phát hành công khai).
- Decision 2.0 Nox-4B (nền tảng, 4.2B): license Apache-2.0 theo ref. 27 (trang tham chiếu không đọc được trong văn bản WebFetch). Ngày phát hành: chỉ biết là 2026; không có tháng. Không xác định được có trước 2026-08-01 hay không. Chưa kiểm chứng. Thử truy cập model card trên Hugging Face bị 401, không đọc được. Ngoài ra, bài không công bố trọng số Nox-4B thành open-weight đã kiểm chứng.
- Qwen3.5 (nền tảng của Nox-4B): ref. 103 là blog "Qwen3.5: Towards native multimodal agents" (2026), không có ngày cụ thể. Qwen3.5 thuộc họ có kích thước khác nhau; nếu dùng đúng biến thể ≤15B thì có thể đủ điều kiện về kích thước, nhưng ngày phát hành cụ thể chưa kiểm chứng.
- GLM-5.3-Flash (teacher của bài): 320B tổng, 18B active, vượt 15B, không đủ điều kiện theo quy tắc của chúng ta.
- JEV-27B, Vega-27B, Clef-27B (so sánh trong bảng): đều lớn hơn 15B, không đủ điều kiện. Ref không có trong văn bản đọc được.
- Kết luận: không có model nào của bài có thể dùng trực tiếp; chỉ Nox-4B/Qwen3.5 là có thể cân nhắc, và phải chờ xác minh ngày.

Ý tưởng chuyển giao được, xếp theo mức độ hữu ích:
1. (Quan trọng nhất) Distill bằng soft label: bài train trên xác suất next-token của teacher bằng soft cross-entropy, T=1, không sharpen, không dùng hard label (Methods 0.10, 0.12; Extended Data Fig. 1). Áp trực tiếp cho việc distill P(yes) của Qwen3-Reranker-8B vào 0.6B/4B: lưu xác suất (hoặc logit) của teacher làm target, không làm nhị phân.
2. Trung bình hóa theo hoán vị lựa chọn để triệt bias vị trí. Lưu ý: bài hoán vị thứ tự các lựa chọn trong prompt, không hoán vị thứ tự input (query, section). Với template yes/no cố định của Qwen3-Reranker thì phương pháp này chỉ áp dụng được nếu prompt liệt kê nhiều lựa chọn; nếu không, gần như không chuyển được.
3. Chấm "not_documented" như một lựa chọn riêng để cho phép thiếu thông tin, thay vì ép đúng/sai. Có thể dùng cho cutoff: query không có doc chắc chắn thì không trả về gì.
4. Đo coverage tại precision cố định (Table 2c) và dùng ngưỡng vận hành cao (ở bài là 0.95) để chọn cutoff. Với F2, nên đo lại trên validation của ta, không sao chép 0.95.
5. Recalibration theo nhóm (per outcome/per loại task): dịch log-odds theo nhóm trên nửa dữ liệu rồi đo trên nửa còn lại. Tương ứng với isotonic theo loại query, cần giữ split tách bạch theo query.
Ý tưởng phụ (một dòng mỗi cái):
- Huấn luyện nhiều giai đoạn với LR re-anneal, giai đoạn cuối nhắm vào loại tài liệu khó nhất (Methods 0.12).
- Xếp hạng theo tích xác suất từng tiêu chí (inclusion × phần bù exclusion): nDCG@20 0.64 so với 0.11 của BM25 (Results 0.7). Tương tự relevance đa khía cạnh.

Điểm không chuyển được: bài dùng full fine-tune trên 12M cặp EHR tiếng Anh; không có dữ liệu tiếng Việt, không có hard negative được mô tả, không có ablation loss/hyperparameter đầy đủ trong phần đọc được. Benchmark của bài (balanced accuracy trên hồ sơ bệnh án) khác với F2 trên web doc.

## 11. Chưa kiểm chứng

- Toàn bộ số liệu lấy từ WebFetch (model tóm tắt), chưa đối chiếu với PDF gốc. Phần PDF bị nén, không đọc được thân bài.
- Ngày phát hành Decision 2.0, Qwen3.5, GLM-5.3-Flash: chưa kiểm chứng.
- Batch size, số epoch, hard negative, dữ liệu đa ngôn ngữ, ablation loss/LR/data mix: không nêu hoặc chưa kiểm chứng.
- Không tìm thấy repo code hay weights công khai để đọc README.

## Bổ sung (kiểm HF API, 25/10)
- `vllm-sr/Decision-1.0-Nox-4B` tạo 21/09/2026; `vllm-sr/Decision-2.0-Nox-4B` tạo 29/09/2026 ⇒ **cả hai ra sau 01/08/2026 ⇒ KHÔNG được dùng**. CS1-4B chưa phát hành.
- Base của Nox là Qwen3.5-4B (27/02/2026, hợp lệ) ⇒ ta có thể đi lại đúng con đường đó từ chính Qwen3.5-4B / Qwen3-4B.
