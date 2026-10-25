# Audit nguồn dữ liệu ngoài và asset mới (25/10/2026)

Mục tiêu: xem trực tiếp các resource đang có, **bê về thứ dùng được**, đo chất lượng trên 1.200 query, và loại bỏ thứ không đáng dùng. Tất cả số liệu dưới đây được đo trên repo này, không phải ước lượng.

## 1. Kết luận ngắn

| Resource | Có gì | Đo trên dữ liệu của mình | Quyết định |
|---|---|---|---|
| `neiths_med-doc-retrieval/data/lexicon/icd10_ontology.json` | 10.002 thực thể ICD-10 VI-EN-ZH; nguồn icd.kcb.vn, WHO, ICD-10-CN | 9.365 cặp VI-EN, 9.076 cặp VI-ZH; phủ **411/1.200 query (34,3%)** bằng khớp cụm chính xác | **Bê về**, dùng cho chuẩn hoá bệnh/entity linking, không phải recall chính |
| `medical_acronyms.json` | 42 acronym y khoa 3 ngữ | Chất lượng cao, đúng dạng query hay gặp (COPD, STEMI, DKA...) | **Bê về**, dùng mở rộng acronym |
| `medical_terms.json` | 9.440 khoá | Khoá là tên ICD, rất dài (tới 39 token); **không có** "đau bụng", "sốt", "ho", "mất ngủ" | Chỉ dùng như từ điển bệnh, không dùng làm symptom lexicon |
| `minhtb.../m4_vi_zh_medical_lexicon.json` | 26 entity + 21 relation VI→ZH | Nhỏ nhưng có symptom/relation đúng kiểu query | **Bê về làm seed**, không đủ làm lexicon chính |
| `ViHealthQA` | 10.015 cặp hỏi-đáp (7.009 train / 993 val / 2.013 test) | 9.990 câu unique; **0/10.015 link** nằm trong `links_corpus_vi_en_zh.parquet` | Giữ để khai thác ngôn ngữ query, **không phải qrels**, không dùng làm candidate |
| `hungnm/vietnamese-medical-qa` | 9.335 cặp hỏi-đáp, Apache-2.0 | 9.329 câu unique, không có link | Giữ để khai thác ngôn ngữ query; không phải qrels |
| `PromptSmithX.../corpus` | 384.125 doc (49.393 VI, 334.728 ZH) | Chỉ 79.416 URL (6,5%) nằm trong link list; chỉ cứu được **365 doc thiếu** (328 ZH, 37 VI), median 764 ký tự | **Loại**, không merge |
| `huu7725.../corpus_manifest.json` | Manifest corpus 1.041.242 doc | Không có shard text trong repo; HF dataset chỉ có link/query | **Loại**, không thể phục hồi |
| `huu7725.../training/pairs.jsonl` | 1.536 cặp weak same-document | Nhãn tự ghi là weak, không phải BTC relevance | **Loại khỏi qrels**; có thể dùng weak-training sau này |
| `neiths.../queries_enriched.jsonl` | 1.200 query "enriched" | Dịch máy rất kém ("I've had a third attack under my esophagus..."), `zh_keywords` rỗng, acronym rỗng | **Loại** |
| `minhtb.../human_review.csv` | 60 query review | `human_judgment` toàn NaN | **Loại** |
| `cbc-528a/BamiBERT-ViMedNER` | 102M params, 5 loại entity | Giấy phép BSD-3-Clause-Clear/Qualcomm; phát hành 05/07/2026 | **Dùng** cho query-side NER |
| Qrels/gold doc IDs | Không có nguồn nào | Codabench chỉ trả aggregate; `scoring_result.zip` chỉ có metadata + scores | **Chưa thể train supervised reranker** |

## 2. Lexicon ICD-10: dùng được nhưng không phải "thuốc chữa recall"

Đã copy vào:

- `data/external/lexicons/icd10_ontology.json`
- `data/external/lexicons/icd10_vi_en.json`
- `data/external/lexicons/icd10_vi_zh.json`
- `data/external/lexicons/medical_terms.json`
- `data/external/lexicons/medical_acronyms.json`
- `data/external/lexicons/m4_vi_zh_medical_lexicon.json`
- `data/external/lexicons/m41_query_terms.json`

Nguồn gốc:

- VI: Cục Quản lý Khám, chữa bệnh - Bộ Y tế (icd.kcb.vn)
- EN: WHO ICD-10
- ZH: ICD-10-CN (`chaseliu/ICD-10-CN`)
- Repo trung gian: `research/refs/neiths_med-doc-retrieval` (không thấy file LICENSE riêng; cần xác nhận trước khi đóng gói vào bài nộp)

Điểm mạnh:

- 10.002 thực thể, có `icd10_code`, `name_vn`, `name_en`, `name_cn`, `chapter`, `level`.
- 9.365 cặp VI-EN và 9.076 cặp VI-ZH.
- 42 acronym lâm sàng 3 ngữ, ví dụ `COPD`, `STEMI`, `NSTEMI`, `HFrEF`, `DKA`, `CURB-65`.
- Rất phù hợp để chuẩn hoá tên bệnh, mở rộng acronym, và mapping VI→EN/ZH khi query có tên bệnh rõ.

Điểm yếu đã đo:

- Khớp cụm chính xác chỉ phủ **411/1.200 query (34,3%)**, trung bình 0,45 match/query; 789 query không match gì.
- Đây là **danh mục bệnh ICD**, không phải từ điển triệu chứng. Các từ rất phổ biến trong query như `đau bụng`, `sốt`, `ho`, `mất ngủ`, `buồn nôn`, `chóng mặt` **không có** trong `medical_terms.json`.
- Nhiều khoá dài tới 39 token (tên ICD đầy đủ), gần như không bao giờ xuất hiện nguyên văn trong query.

Kết luận: dùng lexicon để **normalize/expand entity đã được NER bắt**, không kỳ vọng nó tự tăng recall trên query triệu chứng tự do.

## 3. ViHealthQA và hungnm: data tốt nhưng không phải qrels

Đã tải về `data/external/hf/`:

- `vihealthqa_train.csv` (7.009 dòng)
- `vihealthqa_val.csv` (993 dòng)
- `vihealthqa_test.csv` (2.013 dòng)
- `vihealthqa_all.parquet` (10.015 dòng, 9.990 câu unique)
- `vietnamese_medical_qa_train.parquet` (9.335 dòng, 9.329 câu unique)

Kiểm tra trực tiếp:

- `ViHealthQA`: cột `id, question, answer, link`; câu hỏi median 63 ký tự, mean 104. Link chủ yếu `www.vinmec.com` (7.205) và `vnexpress.net` (2.810).
- **Không có link nào khớp với 1.220.598 URL trong `links_corpus_vi_en_zh.parquet`**. Vì vậy không thể coi cặp question→link là qrels cho corpus BTC, và các answer không thể là candidate hợp lệ.
- `hungnm`: 9.335 dòng, câu hỏi median 228 ký tự; không có cột link; license Apache-2.0.
- `ViHealthQA` không khai báo license rõ trên HF card; cần thận trọng nếu đóng gói.

Giá trị thực tế:

- Dùng làm **query-language corpus** để khai thác triệu chứng/cách diễn đạt, hoặc weak-training query encoder/reranker.
- Không dùng để đánh giá retrieval, không dùng làm positive doc.
- Có thể chạy BamiBERT NER trên 19k câu này để xây lexicon triệu chứng có tần suất; đây là việc nên đẩy Kaggle, không chạy local.

## 4. Các resource đã kiểm tra và loại

### PromptSmithX corpus

- 384.125 doc, 48 part parquet; `language`: VI 49.393, ZH 334.728, EN 1, unknown 3.
- Chỉ **79.416 URL** khớp link list (6,5%).
- Trong số URL khớp, chỉ **365 doc** là doc đang thiếu khỏi docstore v1: 328 ZH + 37 VI; median 764 ký tự.
- Text có markdown heading, không đồng nhất với body docstore.
- Kết luận: không đáng merge; lợi ích 37 doc VI quá nhỏ so với rủi ro format/lệch chunk.

### huu7725 corpus

- `corpus_audit.json`: 1.041.242 doc saved, 2.585.592.455 ký tự, 35 doc <100 ký tự.
- `corpus_manifest.json`: 5.937 retryable + 147.113 unvisited; shard text không có trong repo.
- HF dataset `AIGuruTinix/ViBioMIR` chỉ có link/query, không có corpus text.
- Kết luận: không thể phục hồi doc thiếu từ resource này.

### huu7725 training pairs

- 1.536 dòng, weak same-document context; file tự ghi không phải BTC relevance label.
- Kết luận: không dùng làm qrels; nếu sau này cần weak-training reranker thì dùng có kiểm soát.

### neiths queries_enriched

- 1.200 dòng, nhưng bản dịch EN sai nghĩa y khoa; `zh_keywords` rỗng; `acronym_expansion` chỉ có 3 khoá `vi/en/zh` đều rỗng.
- Kết luận: loại.

### minhtb human_review

- 60 dòng, `human_judgment` toàn NaN.
- Kết luận: loại.

## 4b. Query-side NER: đã có kết quả thật

Kernel `noskaiser/vibiomir-ner-query` (BamiBERT-ViMedNER, GPU) đã chạy xong 1.200 query trong **9 giây**:

- 2.392 entity trên **1.096/1.200 query**.
- `trieu_chung_benh` (triệu chứng): 1.154
- `ten_benh` (tên bệnh): 510
- `bien_phap_dieu_tri` (điều trị): 343
- `bien_phap_chan_doan` (chẩn đoán): 292
- `nguyen_nhan_benh` (nguyên nhân): 93

Ví dụ query 1 được bắt đúng:

- bệnh: `viêm xung huyết 1/3 dưới thực quản`, `viêm dạ dày trợt xung huyết mạn tính`, `sẹo loét hành tá tràng`
- triệu chứng: `bị nghẹn ở cổ`, `hơi buồn nôn`

Lỗi cần postprocess:

- Entity quá ngắn/không có nghĩa: `rối`, `th`, `đau`, `hạch`.
- Truncation: `iểu sản động mạch thông sau hai bên` (mất chữ `d`).
- Một số entity là xét nghiệm/thuốc bị gán nhầm type (ví dụ `Beta-HCG` thành `bien_phap_dieu_tri`).

Đối chiếu ICD-10 trên chính entity:

- 510 entity `ten_benh`: chỉ **35 exact VI→EN**, **76 exact VI→ZH**.
- Khớp substring (rất nhiễu) đạt 239/510 entity, nhưng các match như `rối` (655 khoá), `th` (3.940 khoá) cho thấy không thể dùng substring trần.
- 1.154 entity triệu chứng: 64 exact EN, 65 exact ZH; phần lớn là cụm tự do, không có trong ICD.

Kết luận: NER **dùng được** để hiểu query/rerank theo entity, nhưng lexicon ICD chỉ là lớp chuẩn hoá high-precision cho một phần nhỏ bệnh có tên chuẩn; không phải recall lever chính. Cần lọc entity ngắn, sửa truncation, và ưu tiên exact match.

## 5. Tình trạng asset mới

### BM25 section index — đã xong

Kernel `tthiutoko/vibiomir-bm25-sections2` đã `COMPLETE`.

Output đã tải về `kaggle/outputs/bm25_sections2/{working}/`:

- `weights.npz`: CSR 1.643.339 section × 2^20 feature, 351.409.903 nnz, ~1,38 GB.
- `ids.npy`: 1.643.339 section id.
- `meta.json`: 1.643.339 section, 1.185.389 doc, top-300 doc/query.
- `ranking.parquet`: 360.000 dòng (1.200 query × 300 doc), 163.955 doc unique.

So với submission tốt nhất hiện tại (`mixA100_w12T2_VI_win4500_n150`, 186,9 doc/query):

- Overlap trung bình: **97,4 doc/query**.
- Section BM25 thêm trung bình **202,6 doc/query** mới.
- 1.195/1.200 query có ≥100 doc mới.
- Union trung bình ~400 doc/query.

Đây là candidate source đáng thử, nhưng **chưa có bằng chứng vàng** rằng doc mới là relevant; cần một probe LB union trước khi đưa vào hệ thống.

### Query-only NER — đang chạy lại

Kernel `noskaiser/vibiomir-ner-query` lần 1 lỗi vì thiếu `--queries/--out`; đã sửa default path và push version 2. Kernel này chỉ chạy 1.200 query, không đụng doc-side NER, nên rất nhanh.

### Các job còn chạy

- `tthiutoko/vibiomir-embed-sections`: RUNNING
- `tthiutoko/vibiomir-ner-tag`: RUNNING (doc-side NER, có thể lâu)
- `hytahy/vibiomir-query-frame`: RUNNING
- `hytahy/vibiomir-recrawl-misc1`: RUNNING

## 6. Quyết định kỹ thuật

1. **Không merge PromptSmithX/huu7725 corpus**; lợi ích quá nhỏ hoặc không có text.
2. **Dùng ICD ontology + acronyms** ở tầng query normalization/expansion, sau khi NER bắt entity.
3. **Dùng BamiBERT query-side NER**; doc-side NER chạy một lần và ship `doc_entities.parquet`.
4. **ViHealthQA/hungnm chỉ là ngữ liệu query**, không phải qrels/candidate.
5. **Không train supervised reranker** cho tới khi có qrels hoặc một zero-shot gate thắng w12 rõ ràng.
6. **Thử union section BM25 top-300 với w12** bằng một submission probe; nếu LB không tăng, giữ w12 làm baseline.
7. Trước khi đóng gói final: xác nhận license của lexicon neiths và ViHealthQA; nếu không rõ, chỉ dùng nội bộ nghiên cứu hoặc thay bằng pipeline regenerate từ nguồn công khai.

## 7. File đã thay đổi/ thêm

- `data/external/lexicons/*` (7 file, ~8,8 MB)
- `data/external/hf/*` (5 file, ~15 MB)
- `kaggle/ner_query/ner_query.py` + `kernel-metadata.json` (query-only NER)
- `kaggle/datasets/query_only/*` (dataset query.parquet)
- `kaggle/outputs/bm25_sections2/*` (BM25 section output)
- `runs/reports/promptsmith_missing_overlap.parquet` (365 doc overlap)

Các file này chưa commit; sẽ gom vào commit P0 cùng docstore reader/CLI.
