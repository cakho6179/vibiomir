# CODEMAP — giai đoạn → file chuẩn → lệnh

Cập nhật 22/10 sau đợt dọn cấu trúc (commit "Restructure scripts…"). Mã cũ không còn dùng nằm ở
`_temp/2026-10-22/cleanup/` và trong lịch sử git (commit `bffa280`).

## 1. Dữ liệu
| Việc | File | Lệnh / ghi chú |
|---|---|---|
| Đường dẫn dùng chung | `r2ai2026/paths.py` | `RAW`, `QUERIES`, `LINKS`, `DOCS_RAW`, `CRAWL_DB`, `RECOVERED`, `DOCS_CLEAN`, `DOCS_PROCESSED` |
| Corpus sạch v2 | `scripts/data/build_clean_corpus.py` | raw/crawl/docs.jsonl → data/docs_clean.jsonl |
| Corpus processed v3 | `scripts/data/build_processed_corpus.py` + `r2ai2026/processing.py` | gỡ template theo host, content_key, dup_of → data/processed_v3/ |
| Gói dataset Kaggle | `scripts/data/pack_corpus_dataset.py` | → kaggle/corpus_dataset (part-*.jsonl.gz + query.parquet) |
| Gói phần bổ sung | `scripts/data/pack_corpus_delta.py --tag dN` | doc chưa có trong dataset đã tải lên |

## 2. Crawl / khôi phục (đầu ra vào data/raw)
| Việc | File |
|---|---|
| Crawler async | `r2ai2026/fetch.py`, `r2ai2026/store.py`, `scripts/crawl/fetch.py` |
| Recrawl (direct / Wayback) | `scripts/crawl/recrawl.py`, `r2ai2026/wayback.py`; kernel: `kaggle/recrawl_driver/kernel_recrawl.py` |
| Common Crawl | `kaggle/cc_recover/cc_recover.py` (CDX), `kaggle/cc_recover/cc_columnar.py` (columnar index) |
| PubMed | `r2ai2026/pubmed.py` |
| Gộp kết quả crawl cũ | `scripts/crawl/merge_crawled.py`, `scripts/crawl/merge_kaggle_recrawl.py` (lịch sử; raw/crawl nay đã đóng băng) |
| Kiểm tra sống | `scripts/crawl/smoke_live.py`, `scripts/crawl/host_report.py` |

## 3. Truy hồi & xếp hạng (Kaggle)
| Việc | Nguồn kernel | Ghi chú |
|---|---|---|
| BM25 (E1v2/E3v2, b1–b5, c1/s1/s2) | `kaggle/bm25_exp/bm25_experiment.py` (+ `kaggle/common/docorg.py`) | hướng BM25 đã dừng tinh chỉnh |
| BGE-M3 / Qwen3-Emb dense | `kaggle/dense_exp/dense_embed.py` | |
| Reranker bge-reranker-v2-m3 | `kaggle/rerank_exp/rerank.py` | ứng viên `kaggle/rerank_exp/cands_*` |
| Dịch query (Qwen2.5-7B) | `kaggle/translate_exp/translate_queries.py` | |
| Dựng zip nộp | `kaggle/build_exp/build_submission.py` | RRF, độ sâu doc/chunk, cắt ngưỡng, xếp lại query, LZMA 256 MiB |

## 4. Vận hành Kaggle
| Việc | File |
|---|---|
| Đẩy / chờ / tải nhiều tài khoản | skill `kaggle-multi-account` (`kaggle_acc.py wrap/push/status/wait`), sổ `kaggle/kaggle_jobs.json` |
| Tài khoản (cũ) | `scripts/kaggle_ops/kaggle_accounts.py` |
| Chuẩn bị kernel theo loại | `scripts/kaggle_ops/prep_kaggle_{bm25,dense,recrawl,crawler,script}.py` |
| Chờ & tải (cũ) | `scripts/kaggle_ops/kaggle_wait_fetch.py`, `kaggle_recrawl_orchestrate.py` |

## 5. Nộp bài
| Việc | File |
|---|---|
| Gom zip + MANIFEST | `scripts/submit/collect_submissions.py` (điểm lấy từ EXPERIMENTS.md bảng A) |
| Dựng zip local từ ranking (nhỏ) | `scripts/submit/build_from_rankings.py` |
| Đóng gói / kiểm định dạng | `r2ai2026/submit.py`; chấm local: `r2ai2026/scoring.py` |

## 6. Phân tích
| Việc | File |
|---|---|
| Chương ICD theo hệ | `scripts/analysis/icd_systems.py` |
| Kiểm toán corpus / boilerplate | `scripts/analysis/audit_corpus.py`, `audit_boilerplate.py` |
| Probe khung query/doc | `experiments/2026-10-22_frame_probe/` |

## 7. Tên cũ → tên mới (đợt dọn 22/10)
`scripts/r2ai2026_<tên>.py` → `scripts/<nhóm>/<tên>.py` với nhóm:
- data: build_clean_corpus, build_processed_corpus, pack_corpus_dataset, pack_corpus_delta
- crawl: fetch, recrawl, merge_crawled, merge_kaggle_recrawl, purge_synthetic, host_report, smoke_live
- kaggle_ops: kaggle_accounts, kaggle_wait_fetch, kaggle_recrawl_orchestrate, prep_kaggle_{bm25,dense,recrawl,crawler,script}
- submit: collect_submissions, build_from_rankings
- analysis: icd_systems, audit_corpus, audit_boilerplate, benchmark_scoring, leaderboard, eval_retrieval, compare_chunk_policies

Đã chuyển vào `_temp/2026-10-22/cleanup/` (pipeline local thời kỳ đầu, đã thay bằng kernel Kaggle): build_index,
build_partitioned_index, build_web_index_e1, run, run_e1_submission, devproxy, embed, dense_search, dense_rerank, hybrid,
rerank, crosslingual_run, build_sota_v2, build_decision_v4, build_decision_v4_1_fulltext, clean_corpus, kaggle_sync,
prep_kaggle_data, move_to_temp, push_when_ready, submit_competition; `reproduce.py`; 41 probe `research/results/_data/*.py`.
Kết quả probe (json/txt/png) giữ ở `research/results/crawl_probes/`.

Lưu ý: kernel recrawl tìm `code/scripts/r2ai2026_recrawl.py` trong dataset — `prep_kaggle_recrawl.py` chép
`scripts/crawl/recrawl.py` vào đúng tên đó. Script tự tìm gốc dự án bằng thư mục chứa `r2ai2026/`.
