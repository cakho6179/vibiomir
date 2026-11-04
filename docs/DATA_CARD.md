# Data card — ViBioMIR docstore v2.1

Cập nhật: 25/10/2026. Phạm vi: `data/docstore/v2.1`, dùng làm **nguồn dữ liệu duy nhất**
cho hệ thống từ 25/10 đến hết cuộc thi. `v2` chỉ còn là parent để tái lập/so sánh,
không phải fallback runtime.

## 1. Danh tính và trạng thái

| Trường | Giá trị |
|---|---|
| Version | `v2.1` |
| Built | `2026-10-25 16:39:58` |
| Git commit lúc build | `f4c93c9c844bfbca22adca4089afa7abb8fbdcb4` (dirty) |
| Parent | `v2` |
| Manifest | `data/docstore/v2.1/MANIFEST.json` |
| Validation | `data/docstore/v2.1/validation.json` — **0 errors**, 1 warning |
| Vai trò | Docstore chính/duy nhất cho retrieval, rerank, chunk và submission |
| Raw data | Bất biến; mọi thay đổi đi qua builder incremental, không sửa raw |

## 2. Quy mô

| Chỉ số | v2.1 | So với v2 |
|---|---:|---:|
| Documents | 1.192.673 | +5.760 |
| Unique-content docs | 1.187.751 | +5.760 |
| Passages | 3.607.930 | +14.703 |
| Sections | 1.654.080 | +6.989 |
| Tổng ký tự body | 3.122.749.308 | +17.743.221 |
| Furniture chars | 90.991.975 | +1.803.072 |
| Exact duplicate docs | 4.922 | 0 |
| Thin docs (<500 ký tự) | 121.685 (10,20%) | +2.514 |
| Empty docs | 5 | 0 |
| Sections >4.800 ký tự | 0 | 0 |
| Docs ngoài official links | 0 | 0 |
| Official links còn thiếu | 27.925 | −5.760 |

Phân bố ngôn ngữ: VI **732.246**, EN **239.895**, ZH **220.532**.

Phân bố nguồn: `crawl` 1.182.666; `tmh` 5.760; `cc` 2.278; `wayback` 1.400;
`direct` 370; `pubmed` 177; `lc` 22.

## 3. Thành phần file

| File/thư mục | Nội dung |
|---|---|
| `corpus/part-00..09.jsonl.gz` | Body chuẩn hoá; mỗi dòng `{id,url,t,x}`; `part-09` là phần thêm của v2.1 |
| `docs_meta.parquet` | Metadata theo `id`: URL, host, lang, source, `content_key`, `dup_of`, thin/furniture/passage counts |
| `docs_meta_unique.parquet` | Một đại diện cho mỗi `content_key` |
| `passages.parquet` | `(doc_id, pid, start, end, heading)`; offset body, passage 1.500/400 |
| `sections.parquet` | `(doc_id, sid, pid_first, pid_last, start, end, heading)`; section tối đa 4.800 |
| `furniture.parquet` | Span boilerplate/menu/footer đã nhận diện |
| `templates.json` | Template model kế thừa v2; không fit trên external text |
| `validation.json` | Kết quả validator độc lập |
| `MANIFEST.json` | Input hashes, tham số build, counts, provenance |

`data/docstore/v2.1_kaggle/` là bản staging để upload Kaggle: có thêm
`query.parquet`, đã upload thành dataset `tthiutoko/vibiomir-docstore-v21`.
Không dùng staging làm nguồn local; nguồn local là `data/docstore/v2.1`.

## 4. Provenance

### 4.0 Nguồn từng phần

| `source` | Docs | Nguồn / cách lấy | Ghi chú |
|---|---:|---|---|
| `crawl` | 1.182.666 | Crawler nội bộ trên official link list; raw `data/raw/crawl/docs.jsonl` + `crawl.sqlite`, đóng băng 20/10 | Nguồn chính; code crawler trong repo |
| `tmh` | 5.760 | `tmh1008/ViBioMIR-crawled-corpus` @ `18cb5a5` | Chỉ add/refresh; license trống |
| `cc` | 2.278 | Common Crawl CDX/columnar recovery (`kaggle/cc_recover/*`) | Upstream không có commit cố định; lưu raw recovered |
| `wayback` | 1.400 | Internet Archive Wayback CDX (`r2ai2026/crawl/wayback.py`) | Upstream snapshot theo timestamp; raw recovered |
| `direct` | 370 | Recrawl trực tiếp các URL VI còn thiếu | Raw `data/raw/recovered/direct_*` |
| `pubmed` | 177 | NCBI E-utilities (`r2ai2026/crawl/pubmed.py`) | Chỉ EN; raw `pubmed_2026-10-22.jsonl` |
| `lc` | 22 | Long Châu recrawl (delta v1) | Nằm trong `part-08` của v1/delta; giữ nguyên khi lên v2.1 |

Tổng khớp `MANIFEST.json["counts"]["by_source"]`. Ngoài `tmh`, các nguồn còn lại
là crawl/recovery nội bộ từ official links, không phải dataset external có
revision riêng; raw đã đóng băng và không sửa.

### 4.1 Base v2

- 1.186.913 docs; 1.181.991 unique-content docs.
- Sinh từ corpus crawl BTC + các nguồn đã ghi trong `MANIFEST.json` của v2.
- Không thay byte của tài liệu không liên quan khi build v2.1.

### 4.2 External HF (chỉ phần thêm/refresh)

- Dataset: `tmh1008/ViBioMIR-crawled-corpus`
- Revision: `18cb5a544322d8d7b903ca2b6ceb3138628223b4` (`lastModified` 2026-10-22)
- Dataset card: **không có README/dataset card**, `license=""`.
- Chỉ dùng làm **corpus**, không dùng làm qrels, query labels hay candidate ngoài official links.
- Tài liệu thêm vào được gán `source="tmh"`; ID/URL vẫn là ID/URL chính thức trong
  `links_corpus_vi_en_zh.parquet`.

Phần đã chọn:

- Missing: **5.760** docs (VI 31, ZH 5.729), 5.046.661 ký tự.
- Refresh: chọn **4.502**, áp dụng **4.475**, bỏ 27; body VI
  7.714.440 → 20.460.495 ký tự (+12.746.055).
- Không refresh body ZH; chỉ thêm tài liệu ZH thiếu.

Chi tiết gate:
`data/external/hf_tmh/v2_1_selection.json`.

## 5. License và citation

- **Dữ liệu BTC** (query, link list, corpus crawl): dùng theo thể lệ cuộc thi và
  hướng dẫn của BTC. Không redistribute ngoài phạm vi cuộc thi.
- **Text crawl**: bản quyền thuộc các website gốc; docstore chỉ lưu bản phục vụ
  cuộc thi. Khi public code/report, không đóng gói toàn bộ corpus text nếu BTC
  không cho phép.
- **Wayback / Common Crawl / PubMed**: dùng theo điều khoản của Internet Archive,
  Common Crawl và NCBI E-utilities; chỉ lưu phần text phục vụ cuộc thi, không
  redistribute thành dataset public.
- **`tmh1008/ViBioMIR-crawled-corpus`**: không khai báo license. Đã ghi provenance
  và revision; cần xác nhận với BTC trước khi public nếu có đóng gói dữ liệu.
  Không dùng làm qrels/nhãn.
- **ICD-10/lexicon** (nếu dùng ở query side): nguồn Cục Quản lý Khám, chữa bệnh
  (icd.kcb.vn), WHO ICD-10, ICD-10-CN; repo trung gian không thấy LICENSE riêng.
  Không đóng gói lexicon nếu chưa xác nhận.
- **ViHealthQA / hungnm**: chỉ là ngữ liệu query, không phải qrels/candidate.
  ViHealthQA không có license rõ; hungnm là Apache-2.0.

Citation gợi ý:

```text
BTC ViBioMIR 2026 official query/link corpus.
tmh1008/ViBioMIR-crawled-corpus, revision 18cb5a544322d8d7b903ca2b6ceb3138628223b4.
```

## 6. Tái lập

Từ root repo, với external parquet đã tải:

```bash
# 1) Chọn tập add/refresh từ HF (cần trỏ đúng thư mục parquet)
VIBIOMIR_HF_DIR=/path/to/hf_tmh_conv/default/train \
  python scripts/data/select_hf_v2_1.py --force

# 2) Build incremental từ v2; ghi đè v2.1 có kiểm soát
python scripts/data/build_docstore_v2_1.py --force

# 3) Validate độc lập
python scripts/data/validate_docstore.py \
  --store data/docstore/v2.1 \
  --base data/docstore/v2 \
  --json-out data/docstore/v2.1/validation.json
```

Lưu ý:

- `select_hf_v2_1.py` cần `VIBIOMIR_HF_DIR` hoặc `--hf-dir`; không hard-code
  đường dẫn máy cá nhân.
- Builder cần `data/external/hf_tmh/v2_1_add.jsonl.gz` và
  `v2_1_refresh.jsonl.gz`; đây là output của bước 1.
- Không chạy builder khi chưa có audit/selection; không sửa raw data.
- Index/embedding không nằm trong data card này; chúng được build từ v2.1 bởi
  pipeline retrieval (xem `configs/system_v2_1.yaml` và `docs/THUYET_MINH_SAN_PHAM.md` §1.3).

## 7. Hạn chế đã biết

- Không có qrels/gold doc IDs; mọi “gold” hiện tại chỉ là gate chất lượng nội dung.
- 27.925 official links vẫn thiếu; 5.760 link đã cứu được, phần còn lại chủ yếu
  403/empty hoặc host không qua gate.
- 5 body rỗng kế thừa v2 (`1305, 1788, 1866, 2681, 2901`) được giữ để không mất ID;
  validator coi là warning.
- 121.685 thin docs, tăng 2.514 do cố ý thêm ZH 200–500 ký tự. Nếu ưu tiên
  precision, có thể nâng ngưỡng ZH ở version sau.
- 4.922 exact duplicate docs vẫn có mặt trong `docs_meta` với `dup_of`; dùng
  `docs_meta_unique` khi cần unique-content.
- HF không có license rõ; đây là rủi ro provenance cần xác nhận trước public.

## 8. Retrieval index v2.1 (dense + BM25)

Index du?c build m?t l?n t? docstore v2.1 v? luu ngo?i Git; runtime ch? encode/score query.

| Th?nh ph?n | Quy m? | File |
|---|---:|---|
| Dense VI (`bge-m3`, 1024-d, fp16) | 2.224.624 passage / 732.246 doc | `index/dense/vi/{emb.npy,ids.npy}` |
| Dense ZH | 557.732 passage / 220.532 doc | `index/dense/zh/{emb.npy,ids.npy}` |
| Dense EN | 337.744 passage / 239.895 doc | `index/dense/en/{emb.npy,ids.npy}` |
| BM25 VI doc | 732.246 unit/doc, vocab 533.407 | `index/bm25/vi_doc/index/` |
| BM25 VI section | 1.208.509 unit / 732.246 doc, vocab 535.407 | `index/bm25/vi_sec/index/` |
| BM25 ZH | 220.532 doc, vocab 469.684 | `index/bm25/zh/index/` |
| BM25 EN | 239.895 doc, vocab 218.184 | `index/bm25/en/index/` |

- Dense VI h?p 4 shard a,b,c,d; kh?ng tr?ng doc; ph? d?ng t?p `docs_meta.lang == vi`.
- T?ng dung lu?ng index local ~10,37 GB; dataset chuy?n giao: `tthiutoko/vibiomir-v21-index`.
- `configs/system_v2_1_precomputed.yaml` d?ng index n?y; `dense_search.py` stream
  `emb.npy` theo block d? kh?ng c?n d?a to?n b? ma tr?n VI (4,3 GiB) l?n GPU.
- Query-side v?n d?ng Qwen3.5-9B cho query frame; kh?ng d?ng model d? embed corpus l?c runtime.

