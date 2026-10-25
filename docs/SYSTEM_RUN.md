# Chạy hệ thống ViBioMIR (entry point duy nhất)

```
python -m r2ai2026 run --config configs/system_v1.yaml            # chạy đủ các stage, có cache
python -m r2ai2026 run --config configs/system_v1.yaml --dry-run  # in lệnh sẽ chạy
python -m r2ai2026 run --config configs/system_v1.yaml --stages rerank,build --force
python -m r2ai2026 run --config configs/system_v1.yaml --smoke --allow-cpu --var work=runs/system/smoke
```

- Mã: `r2ai2026/system/runner.py` (bộ điều phối); script stage: `r2ai2026/stages/`. Cấu hình: `configs/*.yaml`. Test: `tests/test_system.py`.
- Mỗi stage ghi vào `<work>/<stage>/`, kèm `.done.json` (hash cấu hình đã resolve). Stage được bỏ qua khi output còn và cấu hình không đổi ⇒ chạy tiếp được, hoặc chia máy: stage GPU chạy trên Kaggle, chép thư mục stage về.
- `<work>/manifest.json`: commit git, phiên bản Python, thời gian từng stage.
- `--var name=value`: ghi đè `data`, `queries`, `work` hoặc biến template. Đường dẫn có `*` lấy kết quả glob đầu tiên (ví dụ `/kaggle/input/**/query.parquet`).

## Đầu vào
| Khoá | Nội dung | Mặc định |
|---|---|---|
| `data` | kho doc: `corpus/part-*.jsonl[.gz]` (`{"id","t","x"}`), `sections.parquet` | `data/docstore/v1` |
| `queries` | `query.parquet` (id, query) do BTC cấp | `data/raw/btc/query.parquet` |

`prepare: true` dựng `<work>/data/` = bản sao `query.parquet` + liên kết tới kho (junction trên Windows, symlink trên Linux), vì các script tìm cả hai dưới cùng một gốc.

## System v1 (`configs/system_v1.yaml`) = công thức của #6751
| Stage | Script | GPU | Output |
|---|---|---|---|
| bm25 | `r2ai2026/stages/bm25_experiment.py e1` (mức doc, chỉ VI, top 200) | – | `ranking.parquet` |
| dense | `r2ai2026/stages/dense_embed.py` bge-m3, đoạn 1.500 ký tự | ✓ | `ranking.parquet` |
| cands | builtin: RRF k=60, trọng số 1:2, top 150 | – | `cands.parquet` |
| rerank | `r2ai2026/stages/rerank.py` bge-reranker-v2-m3, doc = max đoạn | ✓ | `rerank.parquet` (score, best_passage) |
| build | `r2ai2026/stages/build_submission.py`: giữ doc có logit ≥ −2 (ít nhất 20), tối đa 150; chunk = cửa sổ 4.500 ký tự quanh đoạn tốt nhất; LZMA | – | `system_v1_n150.zip` |

Khác #6751 ở chỗ kho text: v1 dùng docstore v1 thay cho docs_clean + delta ⇒ điểm có thể lệch nhẹ; đo bằng 1 lượt nộp khi chạy đủ.

## Chạy trên Kaggle
1. `python scripts/kaggle_ops/pack_code.py --user <user>` ⇒ `kaggle/datasets/code_<user>/` (zip mã + CODE_VERSION) ⇒ tải lên làm dataset `vibiomir-code`.
2. Kernel `kaggle/system_run/run_system.py` (giải nén mã rồi gọi entry point), ví dụ:
   `--config configs/system_v1.yaml --stages dense --var data=/kaggle/input/**/vibiomir-docstore-v1 --var queries=/kaggle/input/**/query.parquet`.
3. Tải thư mục stage về `<work>/<stage>/` trên máy, chạy tiếp các stage còn lại.

## Các stage sẽ thêm (sau khi v1 tái lập được)
query frame (Qwen3.5-9B) · BM25/dense theo section và theo câu hỏi con · truy hồi theo thực thể · decision model (Qwen3-Reranker, `docs/PLAN_DECISION_MODEL_2026-10-25.md`) · cắt theo F2 kỳ vọng · chunk theo section.
