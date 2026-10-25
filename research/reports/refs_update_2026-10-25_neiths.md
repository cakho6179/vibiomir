# Review cập nhật: neiths/med-doc-retrieval (2026-10-25)

Phạm vi: repo `D:\02_Competitions\Kaggle_agent_tool\vibiomir\research\refs\neiths_med-doc-retrieval`, commit cũ `bc1fb16` -> mới `af726bd` (27 commit, 30 file đổi, +8116/-1715). Chỉ đọc, không chạy code, không pip install. Chỉ tải metadata HF API và vài trăm byte/1 file nhỏ (`scraped_ids.parquet` 1.34 MB). Không tải shard dữ liệu nào.

Các mục đánh dấu "chưa kiểm chứng" là suy luận từ code/metadata, chưa đo được.

---

## 0. Tóm tắt nhanh

- Repo của đội khác (thienhb / neiths). Pipeline cuối: BM25s (VI/EN regex word, ZH unigram+bigram) + FAISS SQ8 bge-m3, RRF (0.6/0.4, k=60, top 350), rerank bge-reranker-v2-m3 trên `contextual_text`, rồi chọn top-20 doc, 1 chunk/doc, không gọi PubMed live khi inference (dùng cache).
- Không có điểm LB của chính đội. README chỉ dẫn điểm của đội khác (ongthomiu 0.4871, toptire 0.4274). Số liệu "Top-1, k=4-5" là suy từ các con số đó, không phải validation của họ.
- DataByHien = HF `tmh1008/ViBioMIR-crawled-corpus` (shard `train_shard_*`). Đây là crawl text cho chính các URL BTC, khớp đúng `id` và `url` với `links_corpus_vi_en_zh.parquet`. Không có nhãn/qrels.
- Bucket `nieths/ViBioMIR` (40.7 GB, 1214 file) có: corpus parquet (1.12 GB), DataByHien (1176 shard, 1.91 GB), indices, `scraped_ids.parquet`, `pubmed_cache.jsonl`, `queries_enriched.jsonl`.
- Đối chiếu với corpus của ta (`vibiomir/data/docs_clean.jsonl`): trong 391,209 registry id ta chưa có text OK, có 256,814 id nằm trong `scraped_ids` của đội khác (cận trên, chưa kiểm chứng là có text).
- Không có qrels/gold thật ở đâu trong repo. `data/mock/ground_truth.json` là dữ liệu tổng hợp do `scripts/generate_mock_dataset.py` tạo ra (id dạng `vi_urology_001`).
- Compliance model: bge-m3, bge-reranker-v2-m3, opus-mt-vi-en đều hợp lệ. `ndhieu1101/medical-bidirectional-...` tạo ngày 2026-09-23, sau mốc 2026-08-01, nên vi phạm nếu dùng. Config mặc định không dùng nó.

---

## 1. Thay đổi trong 27 commit (bc1fb16..af726bd)

Thứ tự thời gian từ cũ đến mới. Tất cả ngày 2026-10-22 đến 2026-10-25.

| Nhóm | Commit | Nội dung |
|---|---|---|
| Kiến trúc indexing | 166a0d0, d637ead, 454c086, a4d1d20 | Tách 2 notebook Kaggle: `01_kaggle_build_bm25_cpu` (CPU, chunk + BM25s) và `02_kaggle_build_dense_gpu` (GPU, bge-m3 FP16 -> FAISS). Bỏ BM25 khỏi pipeline dense. `09_kaggle_quantize_and_slim_indices`: FAISS SQ8 + SQLite "slim". |
| Chunking | 9c62b3c, 0f98396 | Chunk 1500/250 -> 750/120 (commit ghi "LCS >= 0.40 bi-directional"). `DocumentChunker.chunk_document` nhận `title` và kwargs. |
| Dense/Multi-query | c762d56, 1381776, 1046601 | Multi-query dense: VI query (w=1.0), EN dịch (w=0.85), ZH keywords (w=0.75), gộp bằng RRF k=60. Chỉ chạy trong nhánh `elif self.dense_index` (xem mục 3.3). `encode_dense` alias. |
| Reranker | c762d56, src/reranker/bge_reranker.py | Đổi từ bge-reranker-large sang bge-reranker-v2-m3. Rerank trên `contextual_text` (có tiêu đề). Chọn doc theo chunk score cao nhất, `max_chunks_per_doc`, `score_threshold=-8`. |
| Sparse | 763a38b, 31e4e9e, d323f0d | Sửa `sparse_index.py`: map `chunk_id` kiểu chuỗi `docid__cN` của bm25s về metadata; nạp nhiều đường dẫn. |
| Inference | 49b78a6, 4aeafed, af726bd | `run_inference_only.py`: tách `--dense-dir` / `--sparse-dir`, tự tải SQ8 index từ HF bucket, nạp `queries_enriched.jsonl`. Notebook cuối `inference-queries.ipynb`: BM25s + pubmed_cache từ bucket, k=20, 1 chunk/doc. |
| Corpus/crawl | scripts/integrate_databyhien.py, scripts/sync_indices_from_hf.py, scripts/audit_hf_and_corpus.py, scripts/crawl_missing_urls.py, scripts/append_new_shards.py | Xem mục 2. |
| UI/tooling | 54b9208, e4cb4eb, 36790b7, 2ed25af, 6550b51, f3ecc8d, 8e3152c, apps/doc_chunk_viewer.py (576 dòng), src/retrieval/doc_reader.py (781 dòng), 10_interactive notebook | Streamlit viewer, notebook tương tác. Không ảnh hưởng điểm số. |
| Docs | ca98071, README.md | Ghi lại kiến trúc. README có vài số liệu (xem mục 3). |

Tổng: không có mô hình LLM nào trong code (grep `qwen|llama|openai|gemini|gpt|mistral` trong src/scripts/configs: 0 kết quả). Pipeline chỉ dùng encoder và cross-encoder.

Ghi chú về lệch cấu hình (quan trọng khi đọc code):
- Docstring `run_inference_only.py` vẫn nói "BGE-Reranker-Large", nhưng config là v2-m3.
- `scripts/generate_kaggle_indexing_notebooks.py` (generator) vẫn ghi `max_chunk_size=1500, chunk_overlap=250`. Notebook 01 đã sửa thành 750/120 nhưng metadata `bm25s_metadata.json` vẫn ghi 1500/250 (xem 3.4).
- `notebooks/08_crosslingual...` chunk theo đoạn 1200/200 ký tự, ghi ra `chunks_meta.parquet`. Đây là pipeline song song, không phải pipeline đang dùng.
- `docs/pipeline_flow.md` ghi "paragraph ~400-500 ký tự". Không khớp với config 750.

---

## 2. Dữ liệu có thể tái sử dụng

### 2.1 DataByHien (`scripts/integrate_databyhien.py`)

Script đọc `data/raw/DataByHien/train_shard_*.parquet`, các cột: `id, url, domain, success, status_code, title, content`. Lọc `success==True and status_code==200 and content != ''`, rồi chuẩn hoá bằng `normalize_text`. Ghi ra `corpus_part_NNN.parquet` (schema: doc_id, url, title, text, lang, domain, char_count, status). Cập nhật `scraped_ids.parquet/txt`.

Nguồn đúng (bằng chứng, đều là metadata):

1. HF dataset `tmh1008/ViBioMIR-crawled-corpus` (tạo 2026-10-20, sửa 2026-10-22, public, không gated, license: không khai báo, 162 downloads). Có 1176 file `data/train_shard_*.parquet` (1.913 GB) và 234 file `data/fixed_shard_*.parquet` (0.085 GB). Datasets-server báo 1,174,600 rows, 8 cột, 1.913 GB. Đây là tổng đúng của `train_shard`.
2. Bucket `nieths/ViBioMIR` có thư mục `corpus/DataByHien/` gồm đúng 1176 file `train_shard_*.parquet`, tổng 1.91 GB, upload 2026-10-22. Số file và dung lượng khớp với HF dataset trên.
3. Không có chỗ nào khác trong repo nêu tên nguồn. `DataByHien` là tên thư mục, có thể là tên người đóng góp.

Kết luận: DataByHien = `tmh1008/ViBioMIR-crawled-corpus` (train_shard). Mức chắc chắn: cao (khớp số file và dung lượng). Vẫn đánh "chưa kiểm chứng" vì chưa đọc hết file.

Schema và nội dung (đọc qua datasets-server `/info` và `/rows`, không tải file):
- `id` int64, `url`, `domain`, `status_code` int64, `success` bool, `title`, `content` (large_string), `content_length` int64.
- Mẫu 500 row trải đều trên 1.17M row: 500/500 id nằm trong `links_corpus_vi_en_zh.parquet` của ta (đọc từ `vibiomir/_temp/.../links_corpus_vi_en_zh.parquet`), và 500/500 URL khớp chính xác với registry.
- `success=True` ở 341/500 (68%), `status_code=200` ở 353/500. 159/500 có `content` rỗng (lỗi crawl).
- `content_length` median 1264 ký tự trong mẫu.
- Vậy: CÓ text cho URL BTC, không chỉ URL. Không có nhãn, không có qrels.

fixed_shard: chưa kiểm chứng schema. Datasets-server đếm đúng 1,174,600 row chỉ từ train_shard, nên fixed_shard có thể nằm ngoài view mặc định. Không tải để kiểm tra.

Ước lượng: 1.17M row x ~68% thành công -> khoảng 0.8M doc có text (sai số lớn vì mẫu chỉ 500 row; chưa kiểm chứng).

### 2.2 Bucket `nieths/ViBioMIR` (`scripts/sync_indices_from_hf.py`, `scripts/audit_hf_and_corpus.py`)

Bucket (không phải dataset, không có datasets API). Liệt kê qua `https://huggingface.co/api/buckets/nieths/ViBioMIR/tree` (2 trang, 1214 file, 40.7 GB). Các file chính:

| Đường dẫn | Dung lượng | Ghi chú |
|---|---|---|
| corpus/corpus_part_000..013.parquet | 1.12 GB (14 file) | Text corpus theo doc_id BTC (upload 10-06 đến 10-22). Nội dung cụ thể chưa kiểm chứng. |
| corpus/DataByHien/train_shard_*.parquet | 1.91 GB (1176 file) | Xem 2.1 |
| scraped_ids.parquet | 1.34 MB | 1,025,647 doc_id (đã tải, xem 2.4) |
| scraped_ids.txt | 7.41 MB | Cùng nội dung dạng txt |
| processed/pubmed_cache.jsonl | 43.55 MB | Cache PubMed (README: ~21,377 bài). Chưa tải. |
| processed/queries_enriched.jsonl | 1.14 MB | 1200 query đã dịch/enrich (xem 2.5) |
| processed/chunks.jsonl | 4.36 GB | Chunk (chưa kiểm chứng nội dung) |
| raw/crawled_shard_1..4.jsonl | 1.65 GB | Crawl thô (chưa kiểm chứng) |
| indices/dense_index_sq8.faiss | 2.37 GB | SQ8 bge-m3, 2,318,540 vector, dim 1024 (dense_metadata.json) |
| indices/dense_index.faiss | 9.50 GB | FP32 |
| indices/chunks_meta_slim.sqlite | 5.42 GB | Metadata chunk, upload 10-23 |
| indices/chunks_meta.sqlite | 10.73 GB | Metadata chunk đầy đủ, upload 10-23 |
| indices/bm25s_index/* | 3.58 GB | Upload 10-05 (cũ hơn chunk meta 18 ngày) |

Các bucket khác đã thử: `datasets/nieths/ViBioMIR` và `models/nieths/ViBioMIR` trả về "Invalid username or password" (không tồn tại hoặc private). `datasets/nieths/Crawled-VibioMIR` (MIT, 2 file, chỉ có README, không có dữ liệu).

Đường dẫn Hub công khai đã kiểm: `https://huggingface.co/buckets/nieths/ViBioMIR/resolve/<path>` tải được (có redirect sang CDN). Trong code, `hf sync` và `api.download_bucket_files` dùng cùng bucket.

### 2.3 Crawl bù (`scripts/crawl_missing_urls.py`)

Cách đội khác khôi phục doc còn thiếu:
- Đọc `missing_urls.parquet` (id, url, host). Bỏ PubMed (đã lấy qua E-utilities).
- `normalize_target_url`: URL `pmc-ecm-healthblog.beta.pharmacity.io/<slug>/` được đổi thành `https://www.pharmacity.vn/<slug>.htm`, theo "BTC Announcement" trong docstring. Notebook 08 ghi "5,318 bài Pharmacity vừa giải cứu" (chưa kiểm chứng số).
- Adaptive: round-robin theo host (`interleave_by_host`); mỗi domain tối đa 3 kết nối, cách nhau 0.25 s; HTTP 429/403 -> cooldown domain 45 s, tăng interval x1.5 (trần 3 s); 5xx -> retry sau 15 s; 404/410 -> ghi `failed_urls.jsonl`, không retry; tối đa 3 lần retry.
- Lao Dong (laodong.vn): xử lý JS cookie challenge bằng regex `document.cookie="..."` rồi retry.
- Trang tiếng Trung: decode `gb18030` khi `content-type` hoặc 1000 ký tự đầu có `gbk/gb2312`.
- Ghi JSONL 5000 bài/shard. Bỏ qua doc đã có trong shard (resume).

Đây là cách họ lấy text cho doc thiếu: crawl trực tiếp URL BTC, không lấy từ nguồn ngoài. DataByHien và bucket corpus là nguồn đã crawl sẵn (xem 2.1, 2.2).

### 2.4 Đối chiếu với corpus của ta (đo thật, local, chỉ đọc)

Nguồn: `vibiomir/data/docs_clean.jsonl` (4.76 GB, 1,183,674 dòng, 1,178,235 id duy nhất). Lấy `id` và `ok` bằng stream. Registry: 1,220,598 id.

| Chỉ số | Số lượng |
|---|---|
| Registry id có record OK ở ta | 829,389 (67.9%) |
| Registry id thiếu OK ở ta | 391,209 |
| Trong đó, có trong `scraped_ids.parquet` của đội khác | 256,814 |
| `scraped_ids` của đội khác có trong OK của ta | 550,768 |
| `scraped_ids` tổng số | 1,025,647 (807,582 nằm trong registry, 218,065 không nằm) |

Cách hiểu:
- 256,814 là cận trên số doc ta đang thiếu mà đội khác có record. Chưa kiểm chứng có text hay không, vì `scraped_ids` có thể gồm cả record thất bại (xem 2.5).
- Chưa tách được PubMed ra khỏi 391,209 doc thiếu của ta (không có host trong `docs_clean.jsonl` đã đếm). Theo audit của đội khác, host `pubmed.ncbi.nlm.nih.gov` là nhóm thiếu lớn nhất của họ (184,266 URL). PubMed được lấy qua API, không qua DataByHien.
- Với `scraped_ids`: không có cách nào biết từ metadata nó là "chỉ thành công". Code `integrate_databyhien.py` chỉ thêm id thành công vào `new_seen_ids`, nhưng các script khác (crawl, PubMed, append_new_shards) có thể thêm id khác. Chưa kiểm chứng.

Hành động đề xuất cho ta (không làm ở đây vì ngoài phạm vi read-only): trên Kaggle, tải DataByHien (1.91 GB) và `corpus/corpus_part_*` (1.12 GB), lọc `success==True and status_code==200`, rồi so với 391,209 id thiếu. Đó mới là số doc ta được thêm text thật.

### 2.5 `scraped_ids.parquet` (đã tải, 1.34 MB)

Đọc bằng pyarrow ở thư mục tạm. Kết quả: 1,025,647 doc_id, không trùng. Đã nêu ở 2.4.

### 2.6 Gold labels / qrels

Không có qrels thật trong repo, không có trên bucket, không có trong dataset HF nào đã kiểm:
- `data/mock/ground_truth.json` (cả `queries_val.jsonl`, `articles_*.jsonl`): do `scripts/generate_mock_dataset.py` tạo ra. Doc id dạng `vi_urology_001`, `zh_urology_001`, PMID thật lẫn lộn. Đây là dữ liệu tổng hợp, không phải nhãn BTC.
- `notebooks/01_baseline_exploration.ipynb` in P=R=F2=0.6667 trên mock. Không có ý nghĩa cho LB.
- Các HF dataset khác (`AIGuruTinix/ViBioMIR`, `manhnt7/ViBioMIR`, `gnuhbear/ViBioMIR`, `522H0134-.../ViBioMIR-Corpus`): đều có `query.parquet` (1200 query, cột id, query) và `links_corpus_vi_en_zh.parquet` (id, url). Không có cột relevance. Chưa kiểm chứng: không tải file query để đếm, nhưng README chỉ liệt kê `id, query` và `id, url`.

Kết luận: không có nhãn.

### 2.7 Các HF dataset còn lại (metadata qua API)

| Dataset | Tạo | Nội dung | License | Ghi chú |
|---|---|---|---|---|
| `AIGuruTinix/ViBioMIR` | 2026-10-12 | query.parquet, links_corpus_vi_en_zh.parquet (1,220,598 link) | CC BY-NC 4.0 (theo README) | Bản chính thức (mirror của BTC). 3481 downloads. |
| `manhnt7/ViBioMIR` | 2026-09-28 | query.parquet, links_corpus.parquet, pubmed_url.parquet | không khai báo | Có tách pubmed_url (danh sách URL EN). |
| `gnuhbear/ViBioMIR` | 2026-10-12 | query.parquet, links_corpus_vi_en_zh.parquet | không khai báo | Bản trùng. |
| `522H0134-NguyenNhatHuy/ViBioMIR-Corpus` | 2026-10-02 | corpus.parquet, demo_corpus.parquet | MIT | Gated "manual", 133 downloads. Không tải được metadata size (datasets-server trả về lỗi "not accessible"). Chưa kiểm chứng có text. |
| `nieths/Crawled-VibioMIR` | 2026-10-01 | chỉ README | MIT | Trống. |

Điều kiện chung của cuộc thi (`competition_guide.md`, mục 35-40): dữ liệu bên ngoài được phép, phải trích dẫn nguồn. Nếu ta dùng DataByHien/bucket, cần ghi nguồn `tmh1008/ViBioMIR-crawled-corpus` và `nieths/ViBioMIR`.

---

## 3. Điểm số đã báo và config

### 3.1 Điểm của chính đội

Không có điểm LB của chính đội trong repo (không có log, không có file submission, không có kết quả cuối). Kiểm tra: grep "LB", "leaderboard", "Macro F2", "public" trong README, notebook outputs, docs. Chỉ thấy:
- README (dòng 84): "Bí quyết từ Top Leaderboard (`ongthomiu` 0.4871, `toptire` 0.4274)". Đây là điểm của đội khác.
- README (dòng 84): các đội đó đạt DOCS_PRECISION ~ CHUNKS_PRECISION ~ 0.24-0.27 và DOCS_RECALL ~ CHUNKS_RECALL ~ 0.60-0.66. Chưa kiểm chứng nguồn gốc các con số này.
- Notebook 01: F2 0.6667 trên mock (synthetic).

### 3.2 Config cuối (tái tạo từ `notebooks/inference-queries.ipynb`, cell 6, đọc ở repo)

```
run_inference_only.py
  --dense-dir /kaggle/input/duong-dan-dense-sq8   (dense_index_sq8.faiss + chunks_meta_slim.sqlite)
  --sparse-dir data/indices                        (bm25s_index từ bucket)
  --hybrid-top-k 350
  --top-k-chunks 20
  --top-k-docs 20
  --max-chunks-per-doc 1
  --batch-size 16
  --disable-pubmed-network
```

Config mặc định (`configs/config.yaml`): dense_top_k=sparse_top_k=hybrid_top_k=350; rrf_k=60; dense_weight=0.6; sparse_weight=0.4; reranker top_k_chunks=120, top_k_docs=60, max_chunks_per_doc=3, score_threshold=-8.0; reranker bge-reranker-v2-m3; embedding bge-m3 max_length 512, fp16. Notebook 10 dùng k=4 (cũ hơn). Commit cuối "Top-1 (k=20)" khớp với notebook cuối.

Nguồn điểm: không có. Config cuối là lựa chọn của đội, không có bằng chứng validation.

### 3.3 Đường đi thực tế của query (quan trọng)

Code: `src/pipeline.py::search_query`.

1. Query text để embed và BM25 = `search_query_text` (đã enrich trong `queries_enriched.jsonl`: VI gốc + mở rộng viết tắt + từ khoá ZH). Không có embedding EN riêng.
2. Khi có sparse index (chạy notebook cuối: có BM25s): `self.hybrid_retriever` được khởi tạo, nên nhánh `if self.hybrid_retriever is not None` chạy: dense (SQ8) + BM25s, gộp RRF 0.6/0.4.
3. Nhánh multi-query VI/EN/ZH dense (weights 1.0/0.85/0.75) nằm trong `elif self.dense_index is not None`. Với config cuối, nhánh này KHÔNG chạy. Đây là code không được dùng trong cấu hình cuối.
4. PubMed: `enriched["pubmed_candidate_ids"]` lấy từ `_memory_cache` (đọc từ `pubmed_cache.jsonl`). Với `--disable-pubmed-network`, không gọi API live. Với query không có cache, `search_candidate_articles` vẫn có thể gọi Europe PMC (không kiểm chứng trong run cuối vì đã tắt mạng).
5. Rerank: `reranker.rerank(query=query_VI_gốc, candidates)`. Lưu ý: cross-encoder nhận query VI gốc, không nhận query đã enrich. Cặp (query, contextual_text) với contextual_text = `[title] + chunk`.
6. Chọn doc: lấy doc theo thứ tự chunk score cao nhất, tối đa top_k_docs=20, `max_chunks_per_doc=1`, chunk đầu tiên của mỗi doc.

### 3.4 Dấu hiệu chỉ mục không đồng bộ (chưa kiểm chứng)

- README: dense 2,318,540 vector; BM25s 1,629,691 chunk. Hai chỉ mục có số lượng khác nhau, nên không cùng tập chunk. Khớp với việc BM25s upload 10-05 (trước commit 9c62b3c đổi chunk 750/120 ngày 10-23).
- `bm25s_metadata.json` (đọc 1 đoạn đầu qua redirect) có `chunk_ids` dạng `docid__cN`. Dense dùng chunk_id số nguyên trong `chunks_meta*.sqlite`. Hai hệ id khác nhau; `sparse_index.py` phải map qua `docid__cN` -> chunk. Commit 31e4e9e sửa đúng lỗi này.
- Nếu BM25s được build trên chunk 1500/250, còn dense trên 750/120, thì cùng một `docid` có thể trả về đoạn text khác nhau. Điều này làm RRF gộp sai. Chưa kiểm chứng (cần so `bm25s_metadata.json` đầy đủ với `chunks_meta_slim.sqlite`).

### 3.5 Lỗi/rủi ro đáng lưu ý khi đọc code

- `validate_submission_data(stitch_adjacent=False)` (formatter.py, đổi từ True). Stitching đã tắt, không có bằng chứng về tác dụng.
- Chunk ghi vào SQLite có `min_chunk_size` khác nhau: 50 (notebook 02), 80 (notebook 01 và generator). Có thể dẫn đến số chunk khác nhau giữa BM25 và dense.
- Dedup: DataByHien chỉ lọc theo `doc_id` trùng với `scraped_ids`. Không kiểm tra trùng URL/nội dung.

---

## 4. Mô hình và kiểm tra quy tắc (open, <=14B, phát hành trước 2026-08-01)

Quy tắc (`competition_guide.md`, mục 35-40 và 245-254): mô hình phải mở trọng số, phát hành trước 2026-08-01 (giờ VN), <=14B tham số riêng từng mô hình (dung sai đến 15B). Không dùng LLM đóng.

| Mô hình | Vai trò | Tạo (HF API) | License | Tham số | Kết quả |
|---|---|---|---|---|---|
| BAAI/bge-m3 | Dense + (sparse không dùng) | 2024-01-27 | MIT | Chưa kiểm chứng (API `safetensors` không trả số); model card ~568M | OK |
| BAAI/bge-reranker-v2-m3 | Cross-encoder | 2024-03-15 | apache-2.0 | 567,755,777 (API) | OK |
| Helsinki-NLP/opus-mt-vi-en | Dịch VI->EN (mặc định config, `src/config.py`, `query_translator.py`) | 2022-03-02 | apache-2.0 | Chưa kiểm chứng (Marian ~77M) | OK |
| ndhieu1101/medical-bidirectional-machine-translation-checkpoints-511042 | Dịch thay thế (chỉ có trong `config.yaml` comment và `preload_models.py`) | 2026-09-23 | không khai báo | 275,102,976 (API) | VI PHẠM mốc 2026-08-01 nếu được dùng. Mặc định config không dùng. Chưa kiểm chứng: `precompute_queries.py` gọi `QueryTranslator(device=...)` với model mặc định, nên `queries_enriched.jsonl` được tạo bằng opus-mt. |

Không có LLM nào trong pipeline. Không có mô hình nào khác (không NER, không ICD tagger trong code chạy thật).

Lưu ý: pipeline cũng gọi PubMed qua API (Europe PMC / NCBI). BTC cho phép (thông báo 26/09: tiếng Anh tìm qua NCBI Entrez, Europe PMC, PubTator 3.0). Không phải mô hình nên không thuộc quy tắc mô hình.

---

## 5. Ý tưởng đáng mượn cho ta

Danh sách hiện có của ta: BM25 trên section <=4.8k ký tự, bge-m3 dense, bge-reranker-v2-m3, RRF, cửa sổ chunk quanh đoạn tốt nhất, NER, cụm near-dup. Mục dưới chỉ ghi cái MỚI hoặc khác.

### 5.1 Thu thập text (mới)

1. Đổi URL Pharmacity: `pmc-ecm-healthblog.beta.pharmacity.io/<slug>/` -> `https://www.pharmacity.vn/<slug>.htm`. Theo BTC (đã nêu trong docstring). Ảnh hưởng: nhiều doc. Cần kiểm tra ta đã xử lý chưa.
2. Lao Dong JS cookie: regex `document.cookie="k=v;..."` rồi retry. Mới với ta (chưa kiểm chứng ta có gặp).
3. GB18030 fallback và `fix_mojibake` cho trang ZH (commit đề cập: `鐧界櫆` -> `白癜风`). Nếu ta đang dùng UTF-8 mặc định, đây là lỗi âm thầm.
4. Round-robin theo host (`interleave_by_host`) và cooldown per-domain (45 s khi 429/403, tăng interval, tối đa 3 s). Tách retry khỏi URL đã fail để không tốn lượt retry khi domain đang cooldown.
5. Nguồn có sẵn để không phải crawl lại: DataByHien và `corpus/corpus_part_*` (xem 2.1, 2.2). Chỉ cần lọc `success` và so với danh sách thiếu của ta (256,814 candidate, chưa kiểm chứng).

### 5.2 Truy hồi và xếp hạng

6. Query enrichment offline: gom VI gốc + mở rộng viết tắt + từ khoá ZH thành một `search_query_text` duy nhất cho cả dense và BM25. Đây là cách làm đơn giản, dễ thử. Mới so với danh sách của ta nếu ta chưa làm.
7. Tách rõ: query dùng để embed/BM25 (đã enrich) và query dùng để rerank (VI gốc). Cross-encoder không thấy từ khoá thêm.
8. PubMed precompute: lưu danh sách PMID ứng viên cho từng query (`pubmed_candidate_ids`) và cache abstract (`pubmed_cache.jsonl`). Inference chỉ đọc cache, không gọi API. Mới với ta nếu ta gọi API lúc chạy.
9. Chọn 1 chunk tốt nhất cho mỗi doc (`max_chunks_per_doc=1`), k=20 doc. Đội khác dùng k nhỏ dựa trên điểm đối thủ. Với ta: đây là một cấu hình đáng so sánh, không phải kết luận đã chứng minh.
10. Multi-query dense (VI + EN dịch + ZH keywords) với trọng số 1.0/0.85/0.75 gộp RRF. Code tồn tại nhưng không chạy ở config cuối. Là thí nghiệm ablation mới cho ta.
11. BM25 cho ZH: unigram + bigram ký tự (`lang=="zh"`). Cho VI: regex `\w+` (không tách từ). `pyvi` được import nhưng không dùng ở query. Ý tưởng mới: thử tách từ VI bằng pyvi (chưa kiểm chứng gain).
12. Chunk key `docid__cN` và chunk_id số nguyên có thể lệch (xem 3.4). Ý tưởng: dùng một khoá chunk duy nhất cho cả BM25 và dense.
13. FAISS SQ8 `IndexScalarQuantizer`, metric IP, 2.37 GB thay cho 9.5 GB FP32, mmap. Đội nói ">98% độ chính xác cosine" (chưa kiểm chứng). Nếu RAM/disk là ràng buộc (notebook 09 nêu giới hạn 20 GB), đây là đường đi thực tế.
14. Metadata SQLite slim (`chunks_meta_slim.sqlite`, 5.4 GB) thay JSON. Mới về tổ chức lưu trữ.

### 5.3 Chunking

15. Heading-aware: gắn tiêu đề bài + tên mục H2 vào mỗi chunk (notebook 07, mục 7). Chưa kiểm chứng gain. Khác với "cửa sổ chunk quanh đoạn tốt nhất" của ta.
16. Q&A: ghép cặp câu hỏi và trả lời bác sĩ vào cùng một chunk (notebook 07). Mới với ta nếu corpus có dạng hỏi đáp.
17. Chunk 750/120 đã chọn để phù hợp với rule LCS >= 0.40 (xem 5.5).

### 5.4 Phân loại chủ đề

18. ICD-10 category routing: phân loại query và doc theo 22 chương ICD-10 để lọc chủ đề (notebook 07 đề xuất, chưa triển khai trong code chạy thật). Khác với NER của ta ở chỗ dùng mã chương để lọc, không chỉ gán nhãn. Chưa kiểm chứng.

### 5.5 Quy tắc chấm (mới, cần xác nhận)

19. Mentor Day (`docs/261012_R2_AI_S3_MentorDay1.md`, dòng 83-94): chunk khớp gold nếu `LCS_tokens(c, g) / |g| >= 0.40`, LCS giữ thứ tự nhưng không liên tiếp. Ví dụ: gold 100 token, LCS 45 token -> đạt. Chuẩn hoá NFKC, HTML entity, lowercase, khoảng trắng. So trong cùng query và cùng doc_id.
20. Commit 9c62b3c nói thêm: "bi-directional" = `LCS/|g| >= 0.40` VÀ `LCS/|c| >= 0.40`. Phần `LCS/|c|` không có trong slide mà tôi đọc được. Chưa kiểm chứng. Nếu đúng, chunk quá dài bị phạt precision, nên 750 ký tự (~150 token) là lựa chọn của họ.
21. Hệ quả cho ta: nếu rule có `LCS/|c|`, chunk nên có độ dài gần với đoạn gold. Đây là điều kiện cần kiểm tra trước khi chốt kích thước chunk.

### 5.6 Những điều không nên mượn

- Metadata bm25s 1500/250 và generator cũ: không đồng bộ với code chạy (3.4).
- Docstring và notebook lệch số liệu (reranker large vs v2-m3; k=4 vs k=20). Đọc cấu hình cuối, không đọc docstring.
- Dùng `ndhieu1101/...` (vi phạm mốc ngày).
- Gọi PubMed API trực tiếp trong `search_query` khi thiếu cache (`pipeline.py`, nhánh `if not articles`), vì không kiểm soát được rate limit.

---

## 6. Phần chưa kiểm chứng (tổng hợp)

- Số chính xác docs có text trong DataByHien (ước 0.8M, chỉ dựa trên mẫu 500 row).
- fixed_shard có schema giống train_shard không.
- Ý nghĩa `scraped_ids` (chỉ thành công hay gồm cả thất bại).
- Nội dung `corpus/corpus_part_*.parquet` (nguồn crawl nào).
- Bucket `bm25s_index` có được build trên chunk 1500/250 không (3.4).
- Số tham số bge-m3 và opus-mt-vi-en (API không trả số).
- Mô hình đã dùng để tạo `queries_enriched.jsonl` (có khả năng là opus-mt, suy từ config).
- Điều khoản "bi-directional" (LCS/|c|) có trong BTC hay không.
- Hiệu quả của SQ8, heading-aware chunk, pyvi, multi-query dense: chưa đo.
- Điểm của đội khác (0.4871, 0.4274) và các con số P/R: chưa kiểm chứng.

---

## 7. Phương pháp đã dùng

- `git log --oneline bc1fb16..HEAD`, `git diff --stat bc1fb16 HEAD`, `git show 9c62b3c`.
- Đọc: `scripts/integrate_databyhien.py`, `scripts/sync_indices_from_hf.py`, `scripts/audit_hf_and_corpus.py`, `scripts/crawl_missing_urls.py`, `scripts/append_new_shards.py` (diff), `src/pipeline.py` (search_query, load_indices), `src/retrieval/hybrid.py`, `src/reranker/bge_reranker.py`, `src/retrieval/sparse_index.py` (tokenizer), `scripts/run_inference_only.py`, `configs/config.yaml`, `README.md` (grep + đoạn chính), `competition_guide.md` (mục rule), `docs/261012_R2_AI_S3_MentorDay1.md` (metric), notebook 01, 07, 08, 09, 10, `inference-queries.ipynb` (source và output có số).
- HF API (metadata): `api/datasets/<id>`, `api/models/<id>`, `api/buckets/nieths/ViBioMIR/tree` (phân trang), `datasets-server.huggingface.co/{size,info,rows}`, redirect `resolve` cho 2 file JSON nhỏ, và `scraped_ids.parquet` (1.34 MB) để đếm.
- Đọc local (chỉ đọc): `vibiomir/data/docs_clean.jsonl` (stream, đếm id/ok), `vibiomir/_temp/.../links_corpus_vi_en_zh.parquet` (id, url).
- Không chạy code từ repo, không pip install, không tải shard lớn.
