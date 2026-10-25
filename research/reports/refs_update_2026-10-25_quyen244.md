# Review repo quyen244/medial-retrieval: 22 commit mới (2026-10-25)

Phạm vi: `git log 45e8813..HEAD` trong `D:\02_Competitions\Kaggle_agent_tool\vibiomir\research\refs\quyen244_medial-retrieval`. Chỉ đọc, không chạy code, không tải dữ liệu lớn. 77 file đổi, +47.676 / -101 dòng.

Lưu ý về diff lớn: phần lớn 47k dòng là `data/vocab/trilingual_terms.tsv` (41.205 dòng, bảng thuật ngữ en/vi/zh từ ICD-11, cột `nguon` = icd11) và docs/prompts. Các file `res/*.zip` chỉ 364-372 byte, là zip chứa `scores.txt` (bảng chấm, không phải dữ liệu lớn).

## 1. Thay đổi trong 22 commit

Thứ tự thời gian (cũ -> mới):
- `405c2b1`, `5047994`, `9994600`, `da307b2`, `02af46d`: Phase 0 (khai thác cách chấm, FINAL 0.0805 -> 0.1112 không GPU), kế hoạch Phase 1-2, khảo sát dịch thuật.
- `fd55f7f`: doc-level pipeline sẵn sàng server, ICD-11 en-zh.
- `77d2a35`: làm sạch `doc_texts`, đổi reranker sang BGE.
- `901b5d5`, `c6e2c45`: nhánh query: sửa chính tả bằng LLM, dịch en/zh, rule-table tiền xử lý, script backup/restore `medical_docs`.
- `353576a`: Phase 1: pool sâu top300, biến thể truy xuất (orig/corr/en/zh/aspects/hyde), rescore theo ngôn ngữ.
- `fb3e085`, `c88a0c1`: `fuse_submission` và `assemble_doc_submission`: lắp bài nộp offline từ pool đã rerank, không gọi lại retriever.
- `e484653`: merge PR #2.

Các thành phần chính:
- **Query augmentation** (`src/rag/augment/`): `query_normalize.py` là bước tất định, không LLM, áp bảng `query_normalize.tsv` (64 dòng, mỗi dòng có `require`/`forbid` regex trong cửa sổ 40 ký tự). Ví dụ: "ko"->"không" nhưng cấm "500k", "k+", "kg". Chạy trước `scripts/correct_queries.py` (LLM sửa chính tả, prompt `query_spelling_correction`).
- **Dịch query**: `scripts/translate_queries.py`, prompt `prompts/query_translation.txt` (JSON `query_en`, `query_zh`).
- **HyDE**: `scripts/gen_hyde.py`, prompt `prompts/hyde_doc.txt` (JSON `hyde_document`).
- **Doc-level index**: `scripts/build_doc_texts.py`, `denoise_doc_texts.py`, `md_strip.py`, `src/rag/...ingest_doc_index.py`. Mỗi tài liệu có một điểm, text = title | description | heading | phần đầu ≤1.500 ký tự, dense BGE-M3, không lọc ngôn ngữ.
- **Retrieval/rerank**: `scripts/phase1_doc_retrieve.py`, `phase1_deep_retrieve.py`, `rescore_pools.py`. Reranker đổi sang `BAAI/bge-reranker-v2-m3` (`src/rag/pipeline/components/reranking.py`). Cấu hình `branch_top_k` = {vi: 35, en: 20, zh: 10} trong `src/rag/config.py`.
- **Lắp bài nộp**: `scripts/assemble_doc_submission.py`, `scripts/fuse_submission.py`, `scripts/append_phase0_results.py`.
- **Submission configs**: `scripts/sub1200/` chỉ có một file, `scripts/sub1200/14_A_flat_hyb_k100to50.sh`: retriever `qdrant-hybrid-flat`, `--flat-retrieve-k 100 --final-k 50`, không augment, không phân rã, không chia ngôn ngữ (một nhánh "all"). Chưa có điểm được báo cáo cho file này.
- **Trạng thái**: `TODO.md` ghi FINAL tốt nhất 0.1112 (N3_d35_c44), còn `ket_qua_thi_nghiem.txt` ghi 0.1117 (`chunk_d35_c44_full8k_plain`) là "TỐT NHẤT". Hai con số mâu thuẫn nhau nhẹ. Điểm cao nhất thật sự là 0.2316 (mục 4).

## 2. Labeling: có LLM-label cặp (query, doc) không?

Có, nhưng theo cách khác với ý nghĩa thường gặp. `src/rag/labeling/` là pipeline sinh dữ liệu train cho reranker, gồm 5 bước:
1. Sinh query từ chunk: `prompts/gen_query_short.txt` (và `gen_query_narrative.txt`), mỗi chunk sinh 2 câu hỏi tiếng Việt 8-25 từ. Query là tổng hợp từ corpus, không phải query BTC.
2. Lấy ứng viên: hybrid dense+sparse, 50 ứng viên/query, quota ngôn ngữ vi 26 / en 16 / zh 8 (`labeling/config.py`).
3. Tầng 1 (5a): `BAAI/bge-reranker-v2-m3` gốc chấm toàn bộ 50 ứng viên -> `t1_score`. Không phải LLM.
4. Tầng 2 (5b): LLM chấm 15/50 (top 10 theo t1 + 5 quanh trung vị), gán `t2_score = 0` cho 35 còn lại mà không gọi LLM. Batch 3, 5 lời gọi/query. Model: `Qwen/Qwen3-8B-AWQ` qua vLLM, `judge_temperature = 0.0`, `judge_max_tokens = 320`.
5. Build train: `src/rag/labeling/build_train_set.py` chỉ dùng ứng viên đã được LLM chấm. Ứng viên chưa chấm KHÔNG được coi là negative (code ghi: ~23% ứng viên t1<0.01 vẫn được LLM chấm >=2; vùng t1>=0.9 chưa chấm có ~93% positive).

Thang nhãn và trọng số (`src/rag/labeling/judge.py`):
- `SCORE_WEIGHT = {3: 1.0, 2: 0.4, 1: None (bỏ khỏi train), 0: 0.0}`.
- Lỗi parse JSON sau retry: `JUDGE_FAIL_SCORE = -1` (không dùng 0, để không nhiễm negative).
- Format train: `{"query", "passage", "label"}` cho `BinaryCrossEntropyLoss` của sentence-transformers.

Prompt chấm `prompts/judge_tier2.txt` (copy phần cốt lõi, tiếng Việt giữ nguyên):

> Bạn là bác sĩ thẩm định tài liệu. Với một câu hỏi y tế và vài đoạn văn (chunk), chấm từng chunk theo một câu hỏi duy nhất: **bác sĩ có thể dùng chính nội dung trong chunk này để trả lời câu hỏi của người bệnh không?**
>
> Cổng A, đúng đối tượng? Chunk nói về CHÍNH đối tượng đó (hoặc tên gọi khác, bản dịch của nó)...
> Cổng B, đúng khía cạnh? Chunk có câu chữ cụ thể trả lời khía cạnh được hỏi...
>
> | Điểm | Điều kiện |
> | 3 | Qua cổng A và cổng B, và chunk trả lời đủ để bác sĩ dùng ngay. |
> | 2 | Qua cổng A, và trả lời được MỘT PHẦN khía cạnh được hỏi (có ít nhất một câu dùng được). |
> | 1 | Cùng lĩnh vực hoặc nhắc tới đối tượng nhưng không trả lời được khía cạnh được hỏi; HOẶC nói về đối tượng khác gần giống. |
> | 0 | Không liên quan tới đối tượng lẫn lĩnh vực. |
>
> **Chỉ được chấm 2 hoặc 3 khi chỉ ra được câu cụ thể trong chunk dùng để trả lời. Phân vân giữa 1 và 2 thì chọn 1.**
>
> Output: `[{"chunk_id": int, "same_subject": bool, "answers": "full|partial|no", "score": int}]`, chấm từng chunk độc lập, không suy diễn.

Bộ bẫy trong prompt (đáng chép lại): trùng từ khóa khác đối tượng -> 0; cùng cơ quan khác bệnh -> 1; lời khuyên chung chung -> 1; đúng bệnh sai khía cạnh -> 1; tin tức/ca lẻ -> 0 hoặc 1; danh sách bác sĩ/bệnh viện -> 1; tóm tắt nghiên cứu chấm theo kết luận.

Ý nghĩa với ta: đây là đúng kiểu nhãn "LLM-judge có gate hai cổng" để distill cho Qwen3-Reranker. Các điểm đáng học: gate `same_subject` trước khi cho điểm, temperature 0, cờ -1 cho lỗi, không coi unjudged là negative, weight 0.4 cho điểm 2. Chưa kiểm chứng: Repo không có dữ liệu nhãn đã chấm (`dev_set.jsonl` có `human_score` toàn None, `data/` gitignored), nên không biết chất lượng nhãn và không có số đo độ tin cậy.

## 3. Nguồn tài liệu: restore/unzip scripts

- `scripts/restore_medical_docs.sh` và `scripts/restore_medical_documents.sh` (trùng logic, bản sau hardcode `data/backups/medical_documents-20261022-011147`) nạp lại collection Qdrant từ bundle backup. Tải từ HF nếu có `HF_REPO=quyenpro/vibiomir-medical-docs`. Bundle ghi trong comment: collection 3.927.800 point, snapshot 27GB. Đây là index của chính team, không phải dữ liệu BTC.
- `scripts/unzip_raw_documents.sh` giải nén `zip_folder/*.zip` tải từ HF dataset `quyenpro/vibiomir-raw-documents` vào `data/raw_documents/{domain}/{doc_id}.json`.
- Gọi HF API: `huggingface.co/api/datasets/quyenpro/vibiomir-medical-docs` và `.../vibiomir-raw-documents` đều trả HTTP 401 (private hoặc gated). Không lấy được createdAt, size, license. Chưa kiểm chứng.
- Nguồn gốc văn bản: không phải dataset ngoài. `src/ingestion/` crawl trực tiếp các URL của BTC (`crawler.py`, `sites/*.py`: baotayninh, laodong, vnnews, generic, pubmed qua E-utilities). `excluded.py` liệt kê domain bỏ qua (ví dụ familydoctor.com.cn cần IP Trung Quốc). Pubmed chỉ dùng E-utilities, không scrape web.
- Có text của BTC corpus hay gold label không: không thấy trong repo. Chỉ có text do crawler lấy. Không có file gold. `dev_set.jsonl` được ghi là chưa có human score.

## 4. Điểm được báo cáo (FINAL = trung bình docs F2 và chunks F2)

| Bài nộp | Cấu hình | DOCS_F2 | CHUNKS_F2 | FINAL | Nguồn |
|---|---|---:|---:|---:|---|
| D_ft / N3_d35_c44 (Phase 0) | chunk index, fine-tune reranker | 0.1280 | 0.0944 | 0.1112 | ket_qua, TODO |
| chunk_d35_c44_full8k_plain | chunk, cửa sổ 8.000, bóc markdown | 0.1280 | 0.0955 | 0.1117 | ket_qua |
| doc_d35_c44_top60_bge | doc-level, top60, BGE rerank | 0.1972 | 0.1815 | 0.1894 | phat-hien mục 13 |
| doc_d50_c55_top60_bge | lắp offline từ cùng pool | 0.2251 | 0.1930 | 0.2090 | mục 14 |
| doc_d80_c55_top300_bge | pool top300 | 0.2574 | 0.1887 | 0.2230 | mục 15 |
| B 80/55 (vcorr_enzh) | pool orig+corr+en+zh | 0.2520 | 0.1913 | 0.2216 | mục 17 |
| C 80/55 (aspects) | thêm aspects | 0.2276 | 0.1749 | 0.2013 | mục 18 |
| B 35/44 (nộp nhầm) | pool top300 orig+corr+en+zh | 0.1914 | 0.1758 | 0.1836 | mục 16 |
| doc_d110_c55_top300_bge | chỉ đổi m_doc 80->110 | 0.2745 | 0.1887 | **0.2316** | mục 19 (điểm cao nhất) |

Các cấu hình "k100to50", "hybrid" (`14_A_flat_hyb_k100to50.sh`) chưa có điểm trong repo. Chưa kiểm chứng: `hybrid` đã bật ở đâu. `docs/chan-doan-he-retrieval.md` nói hybrid chưa bật ở vòng 1. Các điểm trên là tự báo cáo. Chưa kiểm chứng: đây có phải điểm LB công khai hay đo offline. Repo ghi "leaderboard là thước đo duy nhất" nên có thể là LB.

Hai ràng buộc kỹ thuật họ ghi nhận (mục 14): submission bị giới hạn 100 MB, trần chunk 8.000 ký tự cho ~1,58 MB/chunk nên m_chunk tối đa ~58-60. Hai tầng doc và chunk có thể chọn m riêng.

## 5. Model dùng (tên chính xác) và kiểm tra luật

| Model | Vai trò | Luật (mở, <=15B, phát hành trước 2026-08-01) |
|---|---|---|
| `BAAI/bge-m3` | dense + sparse embedding (`src/rag/embedding.py`) | mở, ~568M, phát hành 2024: đạt. License MIT (chưa kiểm chứng) |
| `BAAI/bge-reranker-v2-m3` | reranker (`reranking.py`, labeling tầng 1) | mở, ~568M, 2024: đạt |
| `Qwen/Qwen3-8B-AWQ` | `scripts/serve_vllm.sh`: vLLM, sinh query, sửa chính tả, dịch, HyDE, judge tầng 2 | 8B, Qwen3 phát hành 2025: đạt. Bản AWQ lượng tử hoá, license kế thừa (chưa kiểm chứng) |
| `Qwen/Qwen3-1.7B` | chỉ trong `src/playground/` (thử nghiệm) | đạt |
| `quyenpro/vibiomir-medical-reranker` | reranker fine-tune, HF (`fetch_reranker.py`, `upload_reranker.py`) | fine-tune của team, base là BGE. Repo không có số đo. Chưa kiểm chứng |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | `models/medical-reranker` bị loại vì train trên 3 mẫu | đạt nhưng bị bỏ |

`scripts/serve_vllm.sh` dùng `Qwen/Qwen3-8B-AWQ` (`MODEL` mặc định), port 8000, `--quantization awq`, `--max-model-len 8192`, `GPU_MEM_UTIL` 0.6.

## 6. Ý tưởng đáng mượn (đánh dấu NEW so với ta)

Ta đã có: BM25 trên section <=4.8k ký tự, bge-m3 dense, bge-reranker-v2-m3, RRF, window chunk, NER, near-dup cluster, kế hoạch distill Qwen3-Reranker.

NEW, xếp theo ưu tiên:
1. **Index cấp document (một điểm mỗi tài liệu)** với text = title | description | heading | phần đầu ≤1.500 ký tự, dense BGE-M3, lấy top-60/300 doc, rồi nộp cửa sổ markdown gốc. Đây là bước nhảy lớn nhất của team (0.1117 -> 0.1894). Ta nên thử nếu hiện tại chỉ index chunk. Chưa kiểm chứng trên dữ liệu của ta.
2. **Pool khác nhau cho hai tầng**: doc lấy từ pool sâu (top300), chunk lấy từ pool nông (top60). Họ đo chiều sâu giúp doc nhưng hại nhẹ chunk (mục 15).
3. **Tối ưu m riêng cho từng tầng** bằng ngưỡng biên `h/(4g+m)`, với h = P x m. Công thức dẫn từ F2, họ tự cảnh báo `h = P x m` chứ không suy từ F2 gộp (sai ~12%). Chưa kiểm chứng.
4. **Lắp bài nộp offline từ pool đã rerank** (`assemble_doc_submission.py`): đổi số nộp không cần chạy lại retriever. Tiết kiệm lượt nộp.
5. **Bài học negative, nên tránh**: (a) thêm bản dịch en/zh vào pool không lợi khi rerank bằng câu tiếng Việt (mục 16-17); (b) aspects với 17,8 list RRF làm tụt cả hai tầng (mục 18); (c) cửa sổ chunk dài hơn ~2.500 ký tự quanh chunk tốt nhất giúp tầng chunk nhiều hơn độ dài tuyệt đối (mục 2-3 của ket_qua).
6. **Tiền xử lý query tất định bằng bảng regex có require/forbid ngữ cảnh** (`query_normalize.tsv`) chạy trước LLM sửa chính tả. Ví dụ "cu nhóc" không phải bộ phận sinh dục. Dễ tái dùng, không tốn GPU.
7. **Quy trình nhãn hai tầng** (mục 2 ở trên): CE lọc trước, LLM chấm 0-3 với gate `same_subject`, temperature 0, cờ -1 cho lỗi, unjudged không phải negative. Áp dụng trực tiếp cho kế hoạch distill Qwen3-Reranker.
8. **Chất lượng bằng chứng**: họ có tài liệu "chẩn đoán" và "phát hiện đã đo" tách bằng chứng khỏi phán quyết, và kết luận "21 bài nộp đổi nhiều biến cùng lúc không rút ra gì". Đáng học về cách thiết kế thí nghiệm, mỗi bài chỉ đổi một biến.

Không mới, chỉ là xác nhận: hybrid BGE-M3 sparse (chưa đo sạch), reranker BGE v2-m3 là lựa chọn chung của hai bên.

## Mục chưa kiểm chứng (tóm tắt)
- Nội dung nhãn thật của pipeline labeling (chưa có dữ liệu đã chấm trong repo).
- Metadata HF `quyenpro/vibiomir-*` (HTTP 401).
- Điểm có phải LB công khai hay đo offline.
- License các model (bge-m3, bge-reranker-v2-m3, Qwen3-8B-AWQ, medical-reranker).
- Công thức ngưỡng biên `h/(4g+m)` và tính toán h = P x m.
- Trạng thái hybrid trong `14_A_flat_hyb_k100to50` (chưa có điểm).
