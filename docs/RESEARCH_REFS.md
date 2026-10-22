# Nghiên cứu mã nguồn các đội khác + đánh giá cách tổ chức dữ liệu (22/10)

Nhãn: **[ĐO]** số đo thật (của ta hoặc nhật ký LB của đội khác) · **[SUY]** suy luận.
Mã nguồn đã clone (shallow, chỉ đọc, không chạy): `research/refs/` (không đưa vào hệ thống nộp).

## 1. Các repo đã clone
| Repo | Có gì đáng học |
|---|---|
| `TranVu2005_r2ai-stage3` | **Nhật ký ~60 lượt nộp có P/R** (`docs/PROGRESS.md`), best 0.2655 (15/10). Quan trọng nhất — mục 2. |
| `huu7725_dataa` | File kết quả chấm tải từ Codabench (`results/scoring_result.zip`, `details.jsonl`) — lộ cấu trúc output của scorer — mục 3. |
| `neiths_med-doc-retrieval` | Ontology ICD-10 tam ngữ VI/EN/ZH 10.002 thực thể (nguồn Bộ Y tế icd.kcb.vn + WHO + ICD-10-CN), từ điển viết tắt y khoa. Ta dùng ontology này để gán chương ICD. |
| `quyen244_medial-retrieval` | Bảng thí nghiệm LB: augment query/tách ý bằng LLM **làm giảm** điểm; hybrid dense+sparse không đổi. Prompt sinh query/judge. |
| `Platypus27-coder_VietMedBridge` | Chunk cha/con, Qwen3-Reranker-8B, cấu hình cascade (không có điểm LB). |
| `chimsedibui_RAG-for-Medical` | Chunk cha–con + simhash dedup, consensus/diversity trong fusion. |
| `phongdaohuy0905CSE_Cross-Lingual-Medical-Retrieval`, `PromptSmithX_vibiomir-rag` | Baseline/crawler, ít giá trị thêm. |

## 2. Bài học đã đo của đội 0.2655 (TranVu2005)
Hệ của họ: BGE-M3 hybrid trên chunk 256 token (3,9 triệu chunk / 624k doc vi) + bge-reranker-v2-m3 + RRF; chunk nộp = toàn văn doc.
| Thay đổi | Kết quả LB [ĐO, của họ] |
|---|---|
| Số doc: 60 → 100 → 115 → 130 | Final 0.2050 → 0.2198 → **0.2218** → 0.2205 (doc P 0.183 → 0.153 → 0.144; R 0.265 → 0.345 → 0.358) |
| Số chunk toàn văn: 19 → 32 → 40 → 50 | Chunk F2 0.128 → 0.164 → 0.179 → 0.188 |
| Chunk K80 / K100 / K117 / pool 150 (~104 chunk/q) | Chunk F2 0.237 / 0.247 / 0.250 / **0.2605** |
| Chunk chỉ lấy từ doc đã có trong `relevant_docs` | tốt hơn chunk "ngoài danh sách" (+0.0004) |
| **LZMA dict 256 MiB** | server chấm bình thường; K≈117–133 chunk/q vừa **≤ 102,6 MB** (trần đã chứng minh 102.648.895 B) |
| Mở rộng id trùng nội dung (MinHash, Jaccard ≥ 0.8) | Doc R +0.0013 (gold chứa một phần bản sao) |
| Doc zh theo ngưỡng logit reranker, trần 30/q | +0.0056 Final (S6-f 0.2599 → C1 0.2655); en: không lợi |
| Rerank sâu 150–300 doc | **giảm** (R150 −0.0031, P300 −0.0004) |
| Doc vi không crawl được, đoán bằng BM25 trên slug URL | +0.0029 |

⇒ Hai đòn bẩy lớn nhất là **độ sâu** (doc ~115–150, chunk ~100–130 toàn văn nhờ LZMA 256 MiB) chứ không phải mô hình.
Ta đang nộp 28/28 ⇒ còn rất nhiều điểm "miễn phí". Precision của ta ở top-28 (0.354) cao hơn họ nhiều (≈0.18 ở top-60),
nên cùng độ sâu ta kỳ vọng hơn 0.2655 [SUY — đang đo bằng depth probe].

## 3. Output của scorer (Codabench `scoring_result.zip`)
`evaluation.json` + `scores.txt` + **`details.jsonl` (mỗi query một dòng)**: `document.{precision, recall, f2, predicted,
relevant_predictions, gold}`, `chunk.{gold, found_gold, predictions[{doc_id, relevant, best_overlap, best_gold_chunk_id}]}`.
Từ file của một bài nộp toàn sai (độc lập với dự đoán) [ĐO]:
- Public phase chấm **600/1.200 query** (593 có gold; 7 query gold rỗng bị loại khỏi trung bình).
- **|G| trung bình 70,97 doc và 74,7 chunk/query** (ta từng ước 46–50 từ P/R — sai do trung bình macro). Ví dụ query 1: 114 doc gold.
- ⇒ Danh sách doc tối ưu còn sâu hơn ta nghĩ; ngưỡng F2/5 vẫn đúng.

**Cần user quyết định:** file chấm của *chính bài mình nộp* cho biết doc/chunk nào đúng ở từng query public. Dùng nó làm
nhãn huấn luyện decision model là vùng xám (học trên nhãn test public; private là 600 query khác). Đề xuất: chỉ dùng
thống kê tổng hợp (|G|, đường precision theo hạng) cho hiệu chỉnh; không dùng nhãn từng cặp trừ khi BTC cho phép.

## 4. Cách tổ chức dữ liệu của ta đã hiệu quả chưa?
### 4a. Phân nhóm theo hệ cơ quan (chương ICD-10, slide mentor tr.28–31, 45–46)
`scripts/r2ai2026_icd_systems.py`: so khớp dài nhất trên ontology ICD-10 (5.452 cụm) + từ ngữ đời thường theo hệ.
Query có chương: 823/1.200 (68,6%); nhiều nhất tiêu hoá 155, tiết niệu–sinh dục 147, thai sản 111, nhiễm trùng 92.
Trên top-28 (tiêu đề doc) [ĐO, `runs/reports/icd_systems.md`]:
| Hệ thống | P LB @28 | Trùng chương (cặp đều có chương) | Doc khác chương / query |
|---|---|---|---|
| BM25 E1v2 | 0.178 | 79.7% | 3.80 |
| BM25 b3 (doc-level Pyserini) | ? | 72.8% | 4.92 |
| BGE-M3 | 0.345 | 88.8% | 2.15 |
| Qwen3-Emb | (fusion kém hơn) | 89.1% | 2.13 |
| bge-reranker r1 | ? | **90.0%** | **1.93** |
Kết luận: trùng chương **đi cùng chiều** precision (BM25 thấp nhất, reranker cao nhất), nhưng chênh doc lệch chương
(3.8 → 2.15 ≈ 1.7 doc) chỉ giải thích ~1/3 khoảng cách P (≈ 4.7 doc/28). ⇒ Chương ICD là **đặc trưng mềm** tốt cho
decision model (đúng lời mentor: "ưu tiên, tránh lọc quá chặt"), không phải bộ lọc cứng; phần lớn sai nằm *trong cùng hệ*
(đúng bệnh nhưng sai ý hỏi / sai đối tượng) ⇒ cần câu hỏi nguyên tử "đúng ý hỏi", "đúng đối tượng" ở L2.

### 4b. "Cùng hệ thì chung kiến trúc bài" (khung mục: định nghĩa/nguyên nhân/triệu chứng/chẩn đoán/điều trị/phòng ngừa)
Mẫu 18.255 doc vi rải đều 8 shard [ĐO]: chỉ **3,9%** doc có ≥ 3 mục thuộc khung chuẩn, 21,7% có ≥ 1 mục. Khung này tập
trung ở trang bệnh viện/nhà thuốc: Medlatec 30%, tiêm chủng Long Châu 36%, Vinmec 18%, HelloBacsi 17%; báo chí ≈ 0%.
⇒ Phần lớn corpus là tin/bài báo không theo khung; chunk theo mục (s1/s2) chỉ đổi cách cắt cho ~1/5 corpus. Dùng khung
mục như **đặc trưng khía cạnh** (ý hỏi của query ↔ loại mục trong doc), không tái tổ chức toàn corpus quanh nó.

### 4c. Các biến thể tổ chức đang chờ LB
`kaggle/outputs/vibiomir-bm25-{c1,s1,s2}/submission_*_n28.zip` (clean_v4 doc-level / section 4500-1500 / 4500-600).

## 5. Hệ quả cho lộ trình (đã cập nhật `ROADMAP_SYSTEM.md`)
1. **Ngay:** depth probe cả docs lẫn chunks (đã đẩy `tthiutoko/vibiomir-build-depth-{a,b}`: d100/150/200 c28, d150 c60/90/120, c120 xếp lại thứ tự query để LZMA nén tốt hơn).
2. Decision model mở: đặc trưng = hạng/điểm các retriever + reranker + chương ICD trùng + khía cạnh + host + cụm trùng; nhãn bạc từ LLM mở ≤ 15B; hiệu chỉnh bằng đường P(N) từ depth probe.
3. Mở rộng id trùng nội dung (ta có sẵn `content_key`/`dup_of` trong processed_v3).
4. Không đầu tư augment/tách ý query bằng LLM (đội quyen244 đo: giảm) — chỉ thử nếu làm đặc trưng, không thay query.
