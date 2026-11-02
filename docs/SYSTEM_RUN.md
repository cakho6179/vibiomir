# Chạy hệ thống ViBioMIR (entry point duy nhất)

```
python -m r2ai2026 run --config configs/system_v1.yaml            # chạy đủ các stage, có cache
python -m r2ai2026 run --config configs/system_v1.yaml --dry-run  # in lệnh sẽ chạy
python -m r2ai2026 run --config configs/system_v1.yaml --stages rerank,build --force
python -m r2ai2026 run --config configs/system_v1.yaml --smoke --allow-cpu --var work=runs/system/smoke

# v2.1 (docstore duy nhất; query time chỉ encode/score)
python -m r2ai2026 run --config configs/system_v2_1.yaml
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

## System v2.1 (`configs/system_v2_1.yaml`)

Cùng stage/recipe với system v1 nhưng trỏ `data: data/docstore/v2.1`:

```
python -m r2ai2026 run --config configs/system_v2_1.yaml
python -m r2ai2026 run --config configs/system_v2_1.yaml --dry-run
python -m r2ai2026 run --config configs/system_v2_1.yaml --stages bm25,dense --force
```

v2.1 thêm 5.760 tài liệu thiếu và thay 4.475 body VI; validation 0 errors.
**v2.1 là docstore duy nhất** từ 28/10; `v2` chỉ còn là nguồn cha để tái dựng, không còn là fallback runtime.
Toàn bộ index dense/BM25 và chunk assets đã dựng trên v2.1 cho cả VI/ZH/EN; query time chỉ encode/score query.

## System v3 (`configs/system_v3.yaml`) — pipeline v3, `docs/ARCHITECTURE_V3_2026-10-27.md`

```
python -m r2ai2026 run --config configs/system_v3.yaml            # cả chuỗi
python -m r2ai2026 run --config configs/system_v3.yaml --dry-run  # in lệnh sẽ chạy
python -m r2ai2026 run --config configs/system_v3.yaml --stages decide,materialize
```

Chuỗi 18 stage: `frame` (Qwen3.5-9B) → `variants` → BM25 `vi_doc`/`vi_sec`/`zh`/`en` (`--load-index`) → dense `vi`/`zh`/`en`
(`dense_search.py`, `--emb`) → `pool_vi` (builtin cands, RRF 2:1:1) → `pool_sets` → `pool_cands` → `rr3` + `rr4`
(`--keep-all`) + `m3m` → `pas` → `decide` → `materialize`.

- Chỉ số trong `vars.index` là tài sản đã lưu: không nhúng lại gì. Mọi thứ từ query frame tới zip được tính lại vào
  `vars.work`. Chia máy bằng `--stages` (stage nào có output rồi thì được bỏ qua, chép thư mục stage từ Kaggle về là chạy tiếp).
- `decide` đọc đặc trưng **của lượt chạy này** (`--pool-file/--rr/--m3m/--e1`). Chạy riêng stage `decide` không kèm bốn
  tham số đó thì nó dựng lại danh sách sản xuất — tái lập đúng #7230 (`r8_m3rr197`, 197 doc + 148 chunk, FINAL .2983).
- Hai stage mỏng mới bọc script sản xuất (không chuyển logic, vì `decide_lists` bị 6 script khác import và
  `materialize_lists` bị nhúng source vào kernel Kaggle): `r2ai2026/stages/decide_lists.py`,
  `r2ai2026/stages/materialize_lists.py`.
- Hướng 29/10: **không thêm candidate ngoài pool**. Pool là đầu vào cố định; việc còn lại là lọc pool bằng
  p(gold)/feature. `lex_head3` chỉ được dùng như một feature cho doc **đã nằm trong pool**, không dùng để union
  doc ngoài pool; không dùng giao nhiều kênh (consensus) vì làm mất recall. Mỗi tầng lọc chỉ được phép bỏ ≤1%
  recall gốc, và chỉ tin sau khi đo được bằng p(gold)/proxy. Stage materialize tự dựng input root: chép `lists.parquet` + `query.parquet`
  vào `<out>/input/` và **liên kết** corpus thành `store` (junction, glob `**` đi xuyên được — đã kiểm trên NTFS).
- `pas` đã nối vào config (`scripts/analysis/pas_agg.py`): `pool_pas.parquet` cho các thứ tự `p*/m3p*/m3rp`.
  Chưa nối: `doc2query`, `judge`/`decide` (LLM chấm cặp).
- Mặc định job `m3rr197=m3rr:197|m3rr:197:4500` = "chunk = doc", cửa sổ 4500. **197 chỉ là điểm vận hành để so sánh,
  không phải mục tiêu**: độ sâu doc/chunk cuối phải chọn động theo score/title, không hard-code 197. Chunk size hiện
  giữ cố định 4500. **Trần zip 100 MB (payload 1.024 MB):** 197 chunk × 4500 ký tự = 1.039 MB nên bị server từ chối;
  với cửa sổ 4500, ~150–160 chunk mới an toàn; materialize phải tự cắt theo ngân sách zip thay vì chỉ báo `over_limit`.

## Chạy trên Kaggle
1. `python scripts/kaggle_ops/pack_code.py --user <user>` ⇒ `kaggle/datasets/code_<user>/` (zip mã + CODE_VERSION) ⇒ tải lên làm dataset `vibiomir-code`.
2. Kernel `kaggle/system_run/run_system.py` (giải nén mã rồi gọi entry point), ví dụ:
   `--config configs/system_v1.yaml --stages dense --var data=/kaggle/input/**/vibiomir-docstore-v1 --var queries=/kaggle/input/**/query.parquet`.
3. Tải thư mục stage về `<work>/<stage>/` trên máy, chạy tiếp các stage còn lại.

## Các stage sẽ thêm (sau khi v1 tái lập được)
query frame (Qwen3.5-9B) · BM25/dense theo section và theo câu hỏi con · truy hồi theo thực thể · decision model (Qwen3-Reranker, `docs/PLAN_DECISION_MODEL_2026-10-25.md`) · cắt theo F2 kỳ vọng · chunk theo section.


## Ghi chú 28/10: BM25/e1_seed

- BM25 `e1` full 1.200 query đã chạy local: `runs/bm25_vi_full/ranking.parquet`.
  Top-40 vẫn nhiễu; **không dùng BM25 e1 top-40 làm doc channel**.
- Entry id 4060 có owner **BM25 Baseline** (filename `test_submission_random_cut.zip`)
  là baseline chính thức của BTC: FINAL 0,3764, Docs F2 0,5587, Chunks F2 0,1941.
  BM25 `e1` của ta không tái lập được baseline này; cần thử tokenizer/index khác
  (e3 word segmentation, clean text) hoặc dùng làm mốc tham chiếu, không copy doc.
- `runs/system/v3/decide_e1/lists.parquet` job `e1_seed` trùng chính xác list r24
  (1.200/1.200 query). Nó là seed cho thí nghiệm e1, không phải ranking BM25 mới.
- Hướng tiếp theo là cascade pruning (xem `docs/WORKING_NOTES_DRAFT.md` mục 11),
  dùng nhiều tín hiệu sẵn có trước khi gọi cross-encoder trên vùng uncertain.
