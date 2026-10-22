# So sánh quản lý dữ liệu crawl và chỉ mục: 8 repo tham chiếu R2AI2026 ViBioMIR

Ngày rà soát: 2026-10-22. Phạm vi: chỉ đọc README, docs, config và đoạn code liên quan trong 8 repo shallow-clone. Không chạy code, không cài đặt, không sửa file nào ngoài file này.

Quy ước:
- Mọi đường dẫn tương đối với `D:\02_Competitions\Kaggle_agent_tool\vibiomir\research\refs\`.
- Các con số lấy nguyên từ file ghi chú của repo, không kiểm chứng lại. Khi file không nói thì ghi "không rõ".
- Chữ "ĐO" trong ngoặc vuông nghĩa là repo có ghi điểm LB hoặc điểm chấm đo được gắn với lựa chọn đó. "SUY" nghĩa là chỉ là suy luận, không có số đo.

Tên viết tắt: VMB = Platypus27-coder_VietMedBridge; PSX = PromptSmithX_vibiomir-rag; TV = TranVu2005_r2ai-stage3; CS = chimsedibui_RAG-for-Medical; HU = huu7725_dataa; NE = neiths_med-doc-retrieval; PD = phongdaohuy0905CSE_Cross-Lingual-Medical-Retrieval; QU = quyen244_medial-retrieval.

---

## (a) Từng repo

### 1. VMB (Platypus27-coder_VietMedBridge)

**Quản lý dữ liệu**
- Checkpoint khóa theo "crawl signature" (hash input, hash corpus, khoảng ID, config, runtime, code). Shard attempt bất biến, chỉ con trỏ "latest complete" được đổi. Có marker `.done.json` cho shard đã commit. `docs/DATA_PIPELINE.md` (mục Checkpoint và commit).
- Raw lưu dạng JSONL nén gzip, nhiều nguồn một shard, giữ body hash, size, original/final URL, content type, status, latency. `docs/DATA_PIPELINE.md` (mục Raw shard).
- Ledger: mỗi input có đúng một outcome trong documents hoặc failures, kèm một record ledger. `docs/DATA_PIPELINE.md`.
- Freeze: validator DuckDB (kiểm uniqueness, ID/URL, source hash, slice, foreign key) rồi FROZEN_CANDIDATE. Build có `contract.json`, `frozen.json`, `complete.json`, SQLite spool. `docs/DATA_PIPELINE.md` (mục Hard validation), `docs/FREEZE_NEW_DATA.md`.
- Nơi lưu: Google Drive hoặc HF Private snapshot. `README.md`, `docs/HF_STORAGE.md`.
- Nộp bài: có validator trước khi ghi bài nộp. `docs/RETRIEVAL_BASELINE.md` (dòng về validator), `docs/SUBMIT_AVAILABLE_DATA.md`.

**Tổ chức dữ liệu**
- Trích xuất: HTML dùng trafilatura, fallback BeautifulSoup; PDF dùng pypdf; không OCR. `docs/DATA_PIPELINE.md` (mục Extraction).
- Làm sạch: `src/vietmedbridge/site_cleanup.py` chỉ bỏ UI theo chữ ký neo riêng từng domain (youlai, familydoctor, medlatec, phutho, nghean), chạy trước chunking.
- Dedup: `content_hash` (SHA-256 sau NFC và chuẩn hóa khoảng trắng) để gom nhóm; mỗi official doc_id giữ đúng một alias canonical; không xóa vật lý. `docs/DATA_PIPELINE.md` (mục Child, parent và representation). Dedup chunk: token LCS/union >= 0.8 trong cùng doc. `docs/DECISIONS.md` (dòng 62).
- Chunk: tokenizer BGE-M3 pin revision; child 180 token, overlap 40; parent 512 token; ranh giới theo heading neo chính xác vào source_text; offset là ký tự Unicode. `docs/DATA_PIPELINE.md`; `src/vietmedbridge/chunks.py`.
- Metadata: quality tier HIGH/MEDIUM/LOW (heuristic), language kèm confidence, section spans, flag NO_SECTIONS. Submission evidence dùng `text` (slice chunk). `docs/DATA_PIPELINE.md`.
- Tách index theo ngôn ngữ: không rõ từ phần đã đọc.

**Độ phủ**
- Corpus-wide: không có con số đầy đủ. Có: 798.948 ID giữ được nội dung cũ; 97.758 ID có raw metadata thành công; 2.304 ID mới có text. `docs/MAXIMIZE_COLLECTED_DATA.md` (dòng 11, 41).
- Baseline BGE + BM25 trên 90.669 documents đã nộp. `README.md` (khoảng dòng 106).

**Site bị chặn**
- Robots guard, giới hạn tốc độ, kiểm từng chặng redirect, handler cookie challenge cho laodong. Có trạng thái `laodong_cookie_challenge_unresolved`, `robots_blocked`, `body_too_large`. `src/vietmedbridge/crawl.py` (dòng 94-160).
- Nhánh Crawl4AI/Scrapling để thu lại Stage A, có Cloudflare solver của Scrapling. `docs/CRAWL4AI_RECOVERY.md`.
- Không thấy Common Crawl hay Wayback trong phần đã đọc.

**Số đo LB**
- FINAL 0.0081; doc F2 0.0115; chunk F2 0.0048, baseline BGE + BM25 whole-doc trên 90.669 docs. Không có so sánh ablation. `README.md` (khoảng dòng 106-107).

### 2. PSX (PromptSmithX_vibiomir-rag)

**Quản lý dữ liệu**
- SQLite là nguồn sự thật cho trạng thái crawl: `crawl_manifest.sqlite` với bảng `shards` và `shard_items`; shard ở trạng thái WRITING/COMMITTED; startup đối chiếu Parquet với staging. `README.md`; `crawler/manifest.py`; `crawler/writer.py`.
- Parquet ZSTD, 10.000 dòng mỗi shard, atomic write qua file tạm. `config/crawler.yaml` (mục writer).
- Raw response bị bỏ sau khi trích xuất; chỉ giữ raw khi bật `debug_raw_limit` (tối đa 100). `README.md`.
- Pilot có `manifest init` idempotent, run-name tách manifest/corpus/log theo run. `README.md`.
- Kaggle: mỗi worker có partition và checksum; merge chọn một record mỗi doc_id theo thứ tự SUCCESS > EMPTY_CONTENT > NEEDS_OCR > NEEDS_JS > TOO_LARGE > UNSUPPORTED > FAILED > ROBOTS_DENIED; xung đột ghi vào `merge_conflicts.jsonl`. `KAGGLE_CRAWL_GUIDE.md` (khoảng dòng 126, 173).
- Có `audit_crawl.py` tạo mẫu QA và `domain_summary.csv`, không sửa corpus. `README.md`.
- Không thấy hash manifest cho submission. Vị trí submission: không rõ ngoài `retrieval/submission/exporter.py`.

**Tổ chức dữ liệu**
- Trích xuất: tự viết theo domain (`crawler/extractors/`), trafilatura là fallback (`crawler/extractors/html_generic.py`, dòng 170-182). PDF dùng pypdf với ngưỡng ký tự mỗi trang (`crawler/extractors/pdf.py`).
- Làm sạch: lọc theo danh sách từ khóa share/sharing và khử block trùng lặp liền kề (`crawler/extractors/html_generic.py`, dòng 44-121). `curation/rules/` có quy tắc riêng từng site (baidu, hanoimoi, longchau, vnnews...).
- Dedup: `content_hash = sha256(normalize_for_hash(text))` (`crawler/extractors/__init__.py`, dòng 57-68). Không thấy near-dup.
- Chunking: sentence-aware, mục tiêu 400 token, max 512, overlap 48 token, min 30 ký tự. Passage expansion thử 768/1024 token. `config/baseline_a1.yaml`; `config/experiments/*.yaml`; `retrieval/dense/chunker.py`.
- Chọn chunk: dedup theo sim 0.85 trên text chuẩn hóa; giới hạn chunk mỗi doc (max 2-3); boilerplate filter. `retrieval/submission/chunk_selector.py` (dòng 406-475); `config/baseline_a1.yaml`.
- Metadata: doc_id, url, domain, title, lang, content_hash, crawl_status, error_type. `crawler/writer.py` (schema).
- Tách index theo ngôn ngữ: không rõ.

**Độ phủ**
- Không có số tổng toàn corpus. Có pilot 10.000 URL (`README.md`). Không rõ thêm.

**Site bị chặn**
- Kiểm robots (`crawler/robots.py`), giới hạn theo domain, "domain profile" đo thời gian fetch để chia tải giữa worker (`README.md`; `build_domain_profile.py`).
- Trạng thái ROBOTS_DENIED, NEEDS_JS, NEEDS_OCR được ghi thành outcome riêng, không tính là success (`KAGGLE_CRAWL_GUIDE.md`).
- Không thấy Common Crawl, Wayback hay cách giải Cloudflare.

**Số đo LB**
- Không tìm thấy điểm LB trong các file đã đọc. Chỉ có comment trong config. Không rõ.

### 3. TV (TranVu2005_r2ai-stage3)

**Quản lý dữ liệu**
- Trạng thái crawl trong `state/crawl.db` (SQLite WAL), bảng `urls` với url_norm, doc_ids, domain, status, http_status, attempts, shard; bảng `domains` lưu rate đã học. `docs/crawl.md` (dòng 24).
- Raw shard `data/raw_vi`, doc Parquet theo domain `data/docs_vi/<domain>__<shard>.parquet`, có url_norm, status, extractor, n_tokens. `docs/crawl.md` (dòng 26).
- Chuyển state từ bản cũ sang mới có kiểm hash/inventory/SQL, không ghi đè. `README.md` (mục Path đọc và ghi); `docs/data-transition.md`.
- Snapshot corpus v2: kiểm SHA256 file parquet (`1.220.598` URL, file SHA256 ghi lại), ID và URL duy nhất. `reports/s9_corpus_v2.md`.
- Submission: có `submissions/LOG.md`, SHA256 của ZIP, kiểm validator, replay kiểm byte diff = 0. `docs/PROGRESS.md` (dòng 44, mục M8).
- Nguồn text có provenance: Common Crawl ghi corpus id, URL, crawl ID, WARC filename/offset/length, SHA256 text. `docs/COMPLIANCE.md`.

**Tổ chức dữ liệu**
- Extract: `scripts/extract.py`, workers, ghi theo shard. `README.md`. Chi tiết trích xuất không rõ (chưa đọc `extract.py`).
- Làm sạch: không rõ thêm.
- Dedup: MinHash LSH 128 perm, shingle 5 từ, Jaccard thật >= 0.8, trên body chuẩn hóa theo metric. Kết quả: 623.852 doc ok -> 21.074 cụm; 47.086 doc (7,55%) thuộc cụm; 83% cụm đa domain; 401 cụm boilerplate tách riêng. `docs/PROGRESS.md` (dòng 144); code `src/r2ai/dupes/cluster.py`, `scan.py`, `postprocess.py`.
- Chunk: packing theo token BGE-M3 với mục tiêu 128/256/400; overlap một đoạn; đoạn > 1.5x target được tách theo câu. Field `answer`/`body` và `doc_ids_group` được giữ. `src/r2ai/index/chunk.py` (đầu file); `reports/s12_vi_chunker_compare.json`.
- Metadata: doc_ids, url, domain, title, question, answer, body, paragraphs, status, n_tokens_bge_m3. `src/r2ai/index/chunk.py` (DOC_COLS).
- Index: dense + sparse BGE-M3 trên chunk; zh có index riêng (cùng model, cùng chunker, không dịch). `docs/PROGRESS.md` (mục Z-append).
- Chunk nộp: verbatim slice; lựa chọn độ dài chunk có đo (xem mục 4 bên dưới).

**Độ phủ**
- vi: 624.596/653.970 ok (bản v1). `docs/PROGRESS.md` (dòng 76).
- Extract mới: 632.208 doc (623.852 ok, 8.356 thin). `docs/PROGRESS.md` (dòng 47).
- zh full crawl: 921.845/1.736.107 URL ok (53,1%) tại 11/10. `docs/PROGRESS.md` (dòng 36).
- zh mẫu 2%: 35.686 doc, 15/41 domain có dữ liệu. `docs/PROGRESS.md` (dòng 84).
- Corpus v2: 740.598 vi, 240.000 zh, 240.000 en. `reports/s9_corpus_v2.md`.

**Site bị chặn**
- robots chặn: vov.vn (9.784 URL), baolangson.vn (1.981 URL); ghi `blocked_4xx`, reason `robots_disallowed`. `docs/PROGRESS.md` (dòng 58).
- nhathuoclongchau.com.vn: Cloudflare 403 khi crawl trực tiếp (5/5 trang mẫu, robots không chặn 79.598 URL). Dùng Common Crawl columnar index: 85,64% URL có bản CC. `docs/PROGRESS.md` (dòng 36, 85).
- Slug-only fallback: URL vi chưa crawl được; BM25 trên slug không dấu, thêm 20 ID/query. `docs/PROGRESS.md` (dòng 162).
- Có fallback `r2ai/cc_coverage/` (`cc_coverage/crawls.py`, `index_scan.py`).

**Số đo LB**
- Xem mục 4 bên dưới (ĐO).

### 4. CS (chimsedibui_RAG-for-Medical)

**Quản lý dữ liệu**
- Crawl resume: `data/crawl_log.jsonl` (id, url, status; ok, http_<code>, error, empty, robots_blocked, blocked, garbled). Chỉ `ok` mới bị bỏ qua khi resume. `medrag/ingest/crawl.py` (dòng 1-6, 41-60, 105-107).
- docs.jsonl append-only, giữ bản extract đầu tiên. `medrag/ingest/clean.py` (dòng 13-18).
- Raw bytes chỉ giữ khi `--keep-raw` (`data/html/{id}.bin`). `medrag/ingest/crawl.py` (dòng 1-3).
- Không có manifest hash, snapshot, hay ledger lỗi tách riêng. Không rõ thêm.
- Submission: `medrag/submission.py`; không rõ nơi lưu.

**Tổ chức dữ liệu**
- Trích xuất: decode charset theo thứ tự BOM/utf-8, rồi charset khai báo, tránh lỗi GBK. PDF dùng pypdf. `medrag/ingest/extract.py` (dòng 1-30).
- Làm sạch: boilerplate theo host. Dòng lặp lại ở >= 8 doc và >= 5% doc của host bị coi là template và bỏ. `medrag/ingest/clean.py` (dòng 22-41).
- Dedup: simhash 64-bit trên token từ, LSH banding, văn bản >= 60 token; cụm theo docs đã làm sạch. Mặc định tắt (`dedupe_clusters: False`, "có thể mất recall"). `medrag/indexing/dedup.py`; `medrag/config.py` (dòng 50).
- Chunk: child tối đa 220 token, overlap 40, không cắt qua heading; parent tối đa 700 token; context "Title > H1 > H2" chỉ dùng để embed/rerank, không đưa vào chunk_text. `medrag/config.py` (dòng 20-22); `medrag/indexing/chunking.py` (dòng 1-8).
- Schema: Doc(id, url, lang, title, text, source, extractor); Chunk(chunk_id, doc_id, ordinal, text, context, parent_text, lang). `medrag/schemas.py`.
- Index: BM25 theo ngôn ngữ (bigram cho vi/zh); dense bge-m3 trong Qdrant; retrieve theo ngôn ngữ rồi gộp (stratify=True, per_stratum=30). `README.md`; `medrag/config.py` (dòng 42-44).
- Chọn doc: ngưỡng cosine, không có số đo. `medrag/config.py` (dòng 45-51).

**Độ phủ**
- Không có số. Ghi rõ: "Đã kiểm thử ... 3 URL wiki thật. Chưa chạy: index dense, rerank, query expansion." `README.md` (mục Trạng thái).

**Site bị chặn**
- robots cache theo host (`medrag/ingest/net.py`, dòng 40-61). Retry 429/5xx với Retry-After (`net.py` dòng 32-33).
- Không thấy xử lý CAPTCHA, Cloudflare, Common Crawl hay Wayback.

**Số đo LB**
- Không có. `README.md` (mục Trạng thái) nói chưa chạy index dense và rerank.

### 5. HU (huu7725_dataa)

**Quản lý dữ liệu**
- Manifest corpus: `corpus_data_vi_en_zh/corpus_manifest.json` ghi reason, complete, total_unique_ids (1.194.469), counts (saved 1.042.154, retryable 6.962, terminal 177, unvisited 145.176), max_id, source_fingerprint (bytes, mtime_ns), shards (file, bytes). Đây là điểm mạnh về inventory.
- `corpus_data_vi_en_zh/system_snapshot_20261013.json` ghi SHA256 snapshot đã index (1.041.242 tài liệu, 4.328.424 chunk). `competition/README.md` (dòng 9).
- Audit: `corpus_data_vi_en_zh/corpus_audit.json` kiểm số dòng, ID duy nhất, text_characters, short_text (dưới 100 ký tự), hosts. Không có issues.
- Lưu trạng thái crawl: `crawl_state.sqlite` (vị trí host, ghi mỗi 30 giây), `crawl_status.json` (làm mới 10 giây), `crawl_failures.jsonl`. `README.md`; `crawl_local.py` (dòng 138).
- PubMed: có `pubmed_api_provenance.jsonl` và `pubmed_api_cache` với checksum, ghi PMID/title/abstract. `REVISED_CORPUS_20261012.md`.
- Shard Parquet ảnh chụp cố định: 133 file, 2.696.523.216 byte. `CRAWL_REPAIR_20261012.md`.
- Submission: `submission.zip` một JSON ở root; `submission_validation.json` kiểm ID, chunk, SHA ZIP. `competition/README.md` (dòng 40-46).
- Mô hình: `competition/models.json` ghi weights_sha256; `evidence/` có revision và config của model. Đây là điểm tốt về provenance model.

**Tổ chức dữ liệu**
- Trích xuất: trafilatura, kèm kiểm tra tiêu đề kiểu "just a moment / captcha / access denied" để loại trang chặn. `crawl_local.py` (dòng 185-211).
- Làm sạch: không rõ thêm.
- Dedup: không thấy near-dup. Có loại dòng trùng ID (25.543 dòng) khi xuất snapshot. `CRAWL_REPAIR_20261012.md`.
- Chunk: 240 token, overlap 48, span ký tự được giữ, kiểm đoạn không bị viết lại. `competition/config.json` (dòng 20-22); `competition/README.md` (dòng 69).
- Split train/holdout theo hash toàn văn để không lộ nội dung. `competition/README.md` (dòng 93).
- Metadata: doc_id, url, clean_text. Không thấy ICD hay quality tier.
- Index: `corpus.sqlite` gồm ID/URL, tài liệu nén, vị trí chunk, chỉ mục từ khóa; vector int8 `.npy` theo shard 25.000 dòng. `competition/README.md` (dòng 45-46).

**Độ phủ**
- Crawl dừng ở 1.042.154/1.194.469 ID duy nhất; 145.176 ID chưa thăm. `corpus_manifest.json`.
- PubMed: 239.355/240.000 đã lưu. `CRAWL_STALL_REPAIR_20261013.md`.
- Tên miền lớn nhất: suckhoecongdongonline.vn 154.503 đã lưu; medlatec.vn 24.710; thanhnien.vn 72.111; vov.vn 9.784 chưa có (đã ghi chú). `corpus_audit.json`; `CRAWL_STALL_REPAIR_20261013.md`.

**Site bị chặn**
- Host bị hoãn sau một số lỗi liên tiếp (VOV từ 32 xuống 8 lỗi). Các URL tạm thời giữ trạng thái retryable. `CRAWL_STALL_REPAIR_20261013.md`; `README.md`.
- Ngăn 400/404/410/451 không retry trừ khi `--retry-failed`. `README.md`.
- NCBI chặn; PubMed lấy qua Europe PMC REST, theo PMID, batch 50. `REVISED_CORPUS_20261012.md`.
- Không thấy Common Crawl hay Wayback.

**Số đo LB**
- Lượt chấm đầu ngày 15/10 trả 0 điểm vì ID tài liệu gửi dưới dạng chuỗi số: `unknown_doc: 13.353`. Đã đổi sang integer. `competition/README.md` (dòng 52-54).
- File `results/submission_score_diagnosis_20261015.json` ghi doc F2 0.1208, chunk F2 0.1012 trên 593 query chấm, còn `unknown_doc` 9.169. Không rõ đây là lượt LB nào. Ghi chú: nguồn là `scoring_result (1).zip` trong Downloads.

### 6. NE (neiths_med-doc-retrieval)

**Quản lý dữ liệu**
- Crawl JSONL theo ba shard: 18.798 + 8.354 + 20.854 = 48.006 bài thô; sau làm sạch còn 40.273 bài hợp lệ dạng Parquet ZSTD mức 9 (58 MB). `README.md` (khoảng dòng 137-144, 82-84).
- Resume crawl bằng `--resume`; PubMed có cache đĩa `pubmed_cache.jsonl`. `TEAM_GUIDE.md`; `README.md`.
- Đồng bộ lên Hugging Face Storage Bucket bằng `hf sync`. `README.md` (dòng 85).
- Không thấy hash manifest hay ledger lỗi có cấu trúc. `TEAM_GUIDE.md` nói `data/` là gitignored.

**Tổ chức dữ liệu**
- Trích xuất: trafilatura, fallback BeautifulSoup; loại class quảng cáo, liên quan, tác giả. `src/crawler/url_scraper.py` (dòng 122-152); `src/crawler/extractors/`.
- Làm sạch: 6 bộ regex trong `configs/boilerplate_patterns.yaml`; sửa mojibake GB18030 cho zh. `README.md` (mục 1).
- Dedup: không thấy trong phần đã đọc. Chỉ có dedup URL khi lấy PubMed (`src/crawler/pubmed.py`, dòng 306-315).
- Chunk: ký tự, tối đa 800, overlap 128, tối thiểu 60, tách câu; có viết tắt y tế. Ghi chú README nói `max_chunks_per_doc: 2`, còn `configs/config.yaml` ghi 4. `configs/config.yaml` (dòng 19-22, 54); `src/ingestion/chunker.py`. `scripts/stitch_submission_chunks.py` ghép chunk liền kề đến 1.400-2.500 ký tự.
- ICD-10: có module `src/ingestion/icd10_tagger.py` (phân loại vào 22 chương ICD-10), nhưng không thấy `pipeline.py` gọi. Ontology 10.002 thực thể. `README.md` (mục 4).
- Index: Qdrant, "VI & ZH" tách hay không không rõ. Có BM25 sparse. `src/retrieval/`.

**Độ phủ**
- Không có số phủ corpus. Có 48.006 bài thô và 40.273 bài hợp lệ (xem trên).

**Site bị chặn**
- Retry 429 theo Retry-After; 403 ghi "anti-bot", bỏ qua. `src/crawler/url_scraper.py` (dòng 97-111).
- Không thấy CAPTCHA, Cloudflare, Common Crawl hay Wayback.

**Số đo LB**
- Badge và README ghi Macro F2 0.8775 trên "mock validation" (doc 0.9202, chunk 0.8349). Đây là tập mock, không phải LB. `README.md` (dòng 9, khoảng 280-290). Không dùng để so sánh với LB.

### 7. PD (phongdaohuy0905CSE_Cross-Lingual-Medical-Retrieval)

**Quản lý dữ liệu**
- Crawl ghi `out/part-*.jsonl.gz` (id, url, host, lang, title, text, xv) và `progress.tsv` (id, trạng thái, xv). Mỗi id có thể có nhiều dòng; lấy dòng xv cao nhất. `crawl.py` (dòng 1-19).
- Chạy lại an toàn, `--retry-failed`, `--redo-stale` theo phiên bản quy tắc trích xuất (xv). `crawl.py` (dòng 18-19).
- Raw HTML gọn `raw/raw-*.jsonl.gz` khi `--save-raw`, để `--reextract` không cần mạng. `crawl.py` (dòng 13-16).
- Thư mục lưu `_meta.json`, `_config.json`, `_DONE.json`, `<run>.json` để biết cấu hình đã dùng. `README.md` (khoảng dòng 35-40).
- `configs/baseline.yaml` và `README.md` ghi rõ tham số từng bước.
- Không thấy hash manifest hay ledger tách riêng.

**Tổ chức dữ liệu**
- Trích xuất: quy tắc riêng theo tên miền (`SITE_RULES`) cho khoảng 96% URL; còn lại trafilatura. `crawl.py` (docstring đầu file).
- Phân loại trạng thái: FINAL_HTTP (400, 401, 403, 404, 410, 451) không thử lại; RETRY_HTTP (429, 5xx, 520-524) có thử lại. `crawl.py` (dòng 42-43).
- Làm sạch: `mir/clean.py` áp dụng `clean_lines.json` đã duyệt theo nhóm host; đếm trùng bằng MD5 nhưng chỉ ghi `dup_extra_copies`, không xóa. `mir/clean.py` (dòng 150-182).
- Chunk: ba chiến lược recursive, structure (mặc định, 400 token, min 40, overlap 0), parent_child (cha 480, con 128). Bất biến: `text == corpus_clean.text[char_start:char_end]`. `mir/chunk.py` (dòng 1-13); `configs/baseline.yaml` (dòng 42-49).
- Index: FAISS IVF-SQ8 cho dense, BM25 (pyvi/jieba), LSR cho sparse; RRF 3 nhánh. `README.md` (sơ đồ đầu).
- Metadata: doc_id, host, lang, title, text. Không có ICD hay quality tier.

**Độ phủ**
- Không có số. Chỉ có "sample 2.000 bài, 50 câu hỏi". `configs/baseline.yaml` (dòng 11-12).

**Site bị chặn**
- Phát hiện CAPTCHA bằng regex (AJ-Captcha, slideVerify, turnstile, recaptcha, hcaptcha), dừng nhóm domain đó, không giải CAPTCHA. `crawl.py` (dòng 531).
- `cf-mitigated` từ Cloudflare thì ghi `blocked_cf`. Trang quá nhỏ bị ghi `blocked_tiny`. `crawl.py` (dòng 589-606).
- Dùng `curl_cffi` để giả lập TLS, cảnh báo nếu thiếu. `crawl.py` (dòng 654).
- Không thấy Common Crawl hay Wayback.

**Số đo LB**
- Không có. `README.md` không ghi điểm LB.

### 8. QU (quyen244_medial-retrieval)

**Quản lý dữ liệu**
- Lưu raw doc theo domain: `data/raw_documents/{domain}/{doc_id}.json`; tên file là official corpus id, đã kiểm 60/60 khớp. `README.md` (mục Data).
- Log tiến độ: `logs/crawl-vi.progress`. `docs/crawl-status.md` là báo cáo trạng thái theo ngôn ngữ, ngày 2026-10-17.
- `src/ingestion/storage.py` ghi đường dẫn theo domain để dễ xem lại, re-extract, bỏ từng site.
- Không thấy hash manifest, snapshot có SHA, hay ledger lỗi. Không rõ thêm.

**Tổ chức dữ liệu**
- Trích xuất: trafilatura thay bộ tự viết (`docs/crawl-status.md`, dòng 160). Có `src/ingestion/extraction.py` và `measure.py` đo chất lượng theo domain trước khi crawl.
- Làm sạch markdown: bỏ `**`, `##` nhưng giữ nguyên chữ và số để khớp LCS. `docs/chunking_pipeline_architecture.md` (dòng 104).
- Dedup: không thấy trong phần đã đọc. Không rõ.
- Chunk theo ngôn ngữ: vi và en 800 ký tự (tối đa 1000); zh 600 (tối đa 800). Heading promote, sentence splitter bảo vệ viết tắt, overlap 10-15% theo câu. `README.md` (mục Chunking); `docs/chunking_comparison.md` (dòng 18-21).
- Metadata: dùng `source.language_category` (curated theo domain), không dùng `source.language` (theo trang, hay sai). `README.md` (mục Language identification).
- Retrieval: lọc theo ngôn ngữ riêng: vi top 50, en top 30, zh top 20; RRF; rerank xuống 20 chunk. Router còn để trống. `README.md` (mục Pipeline).

**Độ phủ**
- Ngày 2026-10-17, trên 1.220.598 tài liệu: vi 735.265 (60,24%); zh 107.631 (8,82%); en 240.000 (19,66%); tổng 88,72%. `docs/crawl-status.md`.
- Bản README cũ (trước đó) ghi tổng 45,3%. Không rõ bản nào mới hơn ngoài ngày ghi trong file.

**Site bị chặn**
- nhathuoclongchau.com.vn: trước là HTTP 403 Cloudflare, rồi 75h qua Playwright; sau đó chuyển sang giả lập TLS (`curl_cffi`), chạy 4,4h. `docs/crawl-status.md` (dòng 30, 146).
- laodong.vn: JS cookie challenge, trước trích được 0 ký tự; sửa bằng crawler laodong riêng. `docs/crawl-status.md` (dòng 36).
- zysjonline.com: Cloudflare 403, cả Playwright lẫn curl_cffi đều không qua; giữ trạng thái blocked. `docs/crawl-status.md` (dòng 91).
- 120ask.com (98.160 URL): origin chết (521 cả trang chủ), Wayback không có. Bị chặn. `docs/crawl-status.md` (dòng 87).
- familydoctor.com.cn: cần IP Trung Quốc, bỏ. `docs/crawl-status.md` (dòng 90).
- Playwright bị bỏ hẳn: "không còn được dùng nữa". `docs/crawl-status.md` (dòng 185).

**Số đo LB**
- Không có điểm LB trong các file đã đọc. Không rõ.

---

## (b) Bảng so sánh: Họ làm gì | Ta làm gì | Ta nên học gì không

Ghi chú: "Ta làm gì" theo mô tả trong yêu cầu (crawl.sqlite + docs.jsonl -> docs_clean.jsonl -> processed_v3 parquet; clean_v4; chunk 1000/1500/4500 char; BM25 trên whole doc; submission chunk_text = whole doc; coverage VI 98,35%, ZH 89,85%, EN 99,87%). Không tra cứu lại setup của ta.

| Chủ đề | Họ làm gì | Ta làm gì | Ta nên học gì không |
|---|---|---|---|
| Trạng thái crawl | SQLite WAL có bảng urls/domains (TV `docs/crawl.md`); manifest SQLite có shard_items (PSX `crawler/manifest.py`); crawl signature và attempt bất biến (VMB `docs/DATA_PIPELINE.md`) | crawl.sqlite | Một phần: học ghi reason code và khóa run bằng signature (VMB, PSX). |
| Snapshot và hash | SHA256 từng file, freeze theo build, inventory có SHA (VMB `docs/FREEZE_NEW_DATA.md`; HU `corpus_manifest.json`, `system_snapshot`; TV `reports/s9_corpus_v2.md`) | Chưa có manifest, submission rải rác | Nên học: manifest JSON có SHA cho từng processed shard và từng submission. Chi phí thấp. |
| Raw pages | Gzip JSONL giữ body hash (VMB); raw HTML gọn để re-extract (PD `--save-raw`, CS `--keep-raw`); PSX bỏ raw | Không rõ có giữ raw | Nên học: giữ raw gọn để trích xuất lại mà không tải lại. |
| Trích xuất | trafilatura + BS fallback (VMB, NE, HU, QU); tự viết theo domain (PSX, PD 96% URL); pypdf cho PDF (VMB, CS) | cleaned docs, rule theo host | Một phần: giữ trafilatura làm nền; rule theo domain cho các host lớn đã đo (PD). |
| Làm sạch boilerplate | Chữ ký neo theo domain (VMB `site_cleanup.py`); dòng lặp theo host >= 8 doc và >= 5% (CS `clean.py`); regex YAML + UI duyệt (NE); clean_lines đã duyệt theo nhóm (PD) | clean_v4 luật | Nên học: thống kê dòng lặp theo host (CS) để tìm template mới; rẻ, dễ kiểm. |
| Dedup near-dup | MinHash Jaccard >= 0.8 trên body, 21.074 cụm (TV `src/r2ai/dupes/`); simhash LSH tùy chọn, mặc định tắt (CS); sha256 content_hash (PSX, VMB) | Chỉ exact (content_key) | Học: phát hiện cụm bằng MinHash và giữ ID để dùng sau. Không gộp (xem ĐO bên dưới). |
| Chunk: đơn vị và kích thước | Token BGE-M3: 128/256/400 (TV); 180/40 với parent 512 (VMB); 220/40 với parent 700 (CS); 240/48 (HU); 400/overlap 48 (PSX); ký tự 800/128 (NE); ký tự 800/1000 vi-en, 600/800 zh (QU) | Char 1000/1500/4500 cho BGE | Học: đo theo token BGE-M3 thay vì ký tự; giữ span ký tự để verbatim. |
| Chunk: heading và parent-child | Heading neo chính xác (VMB, PD `structure`); parent-child (VMB, PD, CS) | Heading tùy chọn | Nên học: chunk theo heading, có parent để chọn đơn vị nộp (CS `output_unit`). Chưa đo. |
| Metadata | Quality tier HIGH/MED/LOW, language confidence, section spans (VMB); language_category theo domain (QU); ICD-10 module không gắn vào pipeline (NE); context Title>H1>H2 (CS) | id, host, lang, title, text, n_chars, removed_chars, content_key, dup_of | Học: quality tier và section spans (VMB). ICD-10: không rõ hiệu quả, chưa đo. |
| Tách index theo ngôn ngữ | Lọc theo ngôn ngữ rồi RRF: vi 50/en 30/zh 20 (QU); stratify theo ngôn ngữ (CS); zh index riêng (TV) | Không rõ | Học: pool theo ngôn ngữ đảm bảo EN/ZH không bị cosine VI lấn. Chưa đo ở repo nào. |
| Chunk_text nộp | Verbatim chunk (VMB, TV, QU, HU); chunk ghép đến 1.400-2.500 ký tự (NE); passage expansion 768/1024 (PSX) | Whole doc | Ta đang ở cực "dài nhất". ĐO TV: chunk full thắng chunk ngắn (xem dưới). Giữ whole doc, nhưng cân nhắc thêm chunk để tăng chunk F2. |
| Chặn Cloudflare/captcha | curl_cffi giả lập TLS (PD, QU); dừng domain khi gặp captcha, không giải (PD); Common Crawl cho host Cloudflare (TV, 85,64% URL có bản) | Không rõ | Học: Common Crawl có provenance là phương án khi host chặn. Chỉ dùng khi ghi được WARC và URL khớp corpus. |
| Robots và host chết | robots_disallowed ghi riêng (TV, PSX, CS); host bị hoãn sau 8 lỗi (HU); Wayback không có (QU) | Không rõ | Học: ghi reason riêng cho robots, host chết, captcha; cho phép retry có điều kiện. |
| Slug-only | BM25 trên slug không dấu, 20 ID/query (TV `docs/PROGRESS.md` dòng 162) | Không có | Học: fallback có ĐO (+0,0029 Final). Xem ý tưởng 2. |
| Độ phủ báo cáo | TV: vi 624.596/653.970, zh 53,1%; QU: 88,72% tổng; HU: 1.042.154/1.194.469 | VI 98,35%, ZH 89,85%, EN 99,87% | Ta đã cao hơn các repo khác về VI và EN; ZH là chỗ còn trống lớn nhất. |
| Submission và tái lập | TV: LOG + SHA + replay byte diff = 0 (`docs/PROGRESS.md` dòng 44); VMB: freeze + config lock; HU: validation JSON + SHA ZIP; PSX: manifest theo run | Thư mục rải rác, không manifest | Nên học: LOG có điểm, SHA ZIP, validator, replay. Chi phí thấp, tránh nộp nhầm. |
| Kiểu ID | HU: ID phải là integer, nộp chuỗi số thì 0 điểm (`competition/README.md` dòng 52-54) | Không rõ | Nên học: validator kiểm kiểu integer trước khi nộp. |

---

## (c) 5 ý tưởng hữu ích nhất, xếp theo tác động kỳ vọng

1. **[ĐO] Lấp độ phủ bằng nguồn thay thế có provenance (Common Crawl) cho host bị chặn.** TV: cc-A 0,2223 so với VS-append 0,2168 (chênh +0,0055 Final). Ghi chú: hai bản khác nhau nhiều thành phần, nên không tách được riêng ảnh hưởng của CC. Nguồn: `docs/PROGRESS.md` dòng 11-12. Áp dụng: ZH còn 89,85% (phần trống lớn nhất). Chỉ áp dụng cho URL có WARC khớp corpus và có provenance đầy đủ.

2. **[ĐO] Fallback cho URL chưa có text: BM25 trên slug để chọn thêm ID.** TV: VS-append (thêm 20 ID/query từ slug) đạt 0,2168, so với RD150 0,2139 (+0,0029 Final). Ghi chú: đo qua điểm LB, không phải recall đã gắn nhãn. Nguồn: `docs/PROGRESS.md` dòng 12, 162. Áp dụng cho URL vi/en/zh chưa crawl được.

3. **[ĐO] Validator kiểu dữ liệu trước khi nộp: ID doc phải là integer.** HU: lượt chấm đầu bằng 0 điểm vì toàn bộ ID bị gửi dưới dạng chuỗi số, scorer diễn giải thành `pubmed:<id>`, báo `unknown_doc: 13.353`. Nguồn: `competition/README.md` dòng 52-54. Tác động kỳ vọng: rất lớn nếu ta mắc lỗi, bằng 0 nếu đã đúng. Chi phí thấp, nên làm.

4. **[ĐO] Phát hiện cụm near-dup bằng MinHash, nhưng mở rộng thay vì gộp.** TV: 47.086 doc (7,55%) thuộc 21.074 cụm. Mở rộng thêm 2.000 ID cùng cụm (H1-expand) đạt 0,2109, so với D50 0,2107 (+0,0002 Final, Doc R +0,0013). Gộp còn 1 doc/cụm (H1-dedup) chưa được nộp để đo LB. Nguồn: `docs/PROGRESS.md` dòng 144-146. Tác động nhỏ; làm sau các mục 1-3.

5. **[SUY] Pool theo ngôn ngữ trước khi RRF.** QU lấy vi top 50, en top 30, zh top 20 rồi RRF; CS dùng stratify per_stratum 30; TV có zh index riêng và đã đo được +0,0056 Final khi thêm doc zh (C1 0,2655 so với S6-f 0,2599, `docs/PROGRESS.md` dòng 8, 210). Lưu ý: phần +0,0056 đó gắn với nội dung zh được thêm, không phải với việc tách pool. Chưa repo nào đo riêng việc tách pool. Tác động kỳ vọng: chưa biết.

Ghi chú thêm (không đưa vào top 5): TV đo chunk dài thắng chunk ngắn (early baseline, K=5: c2 0,0158, full 0,0273; `docs/PROGRESS.md` dòng 83). Điều này ủng hộ việc ta đang nộp whole doc, nhưng đây là bằng chứng cũ, chưa tách được ảnh hưởng của số doc.

---

## Tóm tắt điểm đo được (ĐO)

| Repo | Thay đổi | Điểm đo | Nguồn |
|---|---|---|---|
| VMB | BGE + BM25 whole-doc baseline | FINAL 0,0081 (doc 0,0115, chunk 0,0048) | `README.md` dòng 106-107 |
| TV | Dedup cụm, expand, cùng chunk | 0,2109 vs D50 0,2107 | `docs/PROGRESS.md` dòng 144-146 |
| TV | CC cho longchau | cc-A 0,2223 vs VS-append 0,2168 | `docs/PROGRESS.md` dòng 11-12 |
| TV | Slug fallback 20 ID/query | VS-append 0,2168 vs RD150 0,2139 | `docs/PROGRESS.md` dòng 12 |
| TV | Thêm doc zh | C1 0,2655 vs S6-f 0,2599 | `docs/PROGRESS.md` dòng 8, 210 |
| TV | Chunk ngắn vs dài (early) | c2 0,0158 vs full 0,0273 (K=5) | `docs/PROGRESS.md` dòng 83 |
| HU | ID chuỗi số | 0 điểm, sửa thành integer | `competition/README.md` dòng 52-54 |
| NE | Mock validation | Macro F2 0,8775 (không phải LB) | `README.md` dòng 9, 280-290 |
| CS, PSX, PD, QU | Không có | không rõ | không có số đo trong file |

---

## Không rõ / cần kiểm thêm

- Ta chưa rõ có giữ raw HTML hay không; báo cáo này không kiểm tra setup của ta.
- Không repo nào báo điểm LB gắn với chọn lọc quality tier, ICD-10 hay tách index theo ngôn ngữ.
- Điểm 0,1208 / 0,1012 trong HU không rõ là lượt LB nào.
- TV `extract.py` và các file scripts khác chưa đọc đầy đủ; mô tả trích xuất của TV chỉ dựa trên README và docs.
