# ViBioMIR — đội Băng Cá Khô (R2AI 2026, Stage 3)

Hệ thống truy hồi tài liệu y khoa đa ngữ (VI/ZH/EN) và trích đoạn cho 1.200 câu hỏi tiếng Việt.

**Bài nộp được báo cáo:** private #7892 `pN3_base_pqchunks.zip`. Đây là cùng file với public #7589 (FINAL 0,3484).

- Thuyết minh phương pháp, dữ liệu, mô hình và mã: [`docs/THUYET_MINH_SAN_PHAM.md`](docs/THUYET_MINH_SAN_PHAM.md).
- File này: cài đặt, tải dữ liệu và checkpoint, và các lệnh chạy lại từ đầu.

## 0. Tóm tắt các mức tái lập

| Mức | Làm gì | Phần cứng | Thời gian | Kết quả |
|---|---|---|---|---|
| **A** | Từ đặc trưng đã chia sẻ → p(gold) → danh sách docs/chunks của 5 bài nộp | CPU, 8 GB RAM | ~5 phút | khớp từng dòng với bài đã nộp (đã kiểm) |
| **B** | Danh sách → văn bản chunk → zip nộp | CPU (kernel Kaggle CPU hoặc máy) | ~47 phút trên kernel Kaggle CPU | `pN3_base.zip` |
| **C** | Từ câu hỏi + chỉ mục → khung truy vấn → truy hồi → pool → reranker/BGE-M3 → đặc trưng của mức A | GPU 16 GB (Kaggle T4) | khoảng 40 kernel Kaggle chạy song song trên nhiều tài khoản | các file đầu vào của mức A |
| **D** | Từ dữ liệu thô → docstore v2.1 → chỉ mục | CPU + GPU | nhiều ngày (crawl) | `data/docstore/v2.1` |

Mỗi mức dùng đầu ra đã chia sẻ của mức sau, nên có thể bắt đầu từ bất kỳ mức nào.

## 1. Môi trường

- Python **3.11+**; đội dùng 3.13.3 trên Windows 11. Mã cũng chạy trên Linux.
- CPU: `pip install -r requirements.txt`. Phiên bản đã ghim là phiên bản đội đã chạy và kiểm.
- GPU (chỉ mức C): `pip install -r requirements-gpu.txt`, hoặc dùng image Python mặc định của Kaggle (Python 3.13, GPU T4/P100, bật internet). Các kernel đã chạy đều dùng image đó, chỉ cài thêm `jieba` và `pyvi`.

```bash
git clone <repo> vibiomir   # hoặc giải nén vibiomir_code_<commit>.zip
cd vibiomir
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests                            # tuỳ chọn
```

Mọi lệnh dưới đây chạy **từ thư mục gốc repo**.

## 2. Tải dữ liệu

Thư mục chia sẻ: **`<LINK_DRIVE>`**. Nội dung từng gói ở `docs/THUYET_MINH_SAN_PHAM.md` §2.4.

| File trên Drive | Giải nén / đặt vào |
|---|---|
| `artifacts_A_pgold.zip`, `artifacts_B_materialize.zip`, `artifacts_C_doc2query.zip`, `artifacts_D_retrieval.zip` | giải nén tại **gốc repo** (đường dẫn bên trong đã là đường dẫn repo) |
| `docstore_v2.1/` | `data/docstore/v2.1/` |
| `index_v2.1/` | `data/docstore/v2.1/index/` |
| `MANIFEST.tsv` | gốc repo |

```bash
unzip -o artifacts_A_pgold.zip        # tương tự cho B, C, D
python scripts/release/verify_manifest.py MANIFEST.tsv               # sha256 các file đã giải nén
python scripts/release/verify_manifest.py MANIFEST.tsv --big         # thêm docstore + index (chậm)
```

`data/raw/btc/query.parquet` là file câu hỏi của BTC (có trong gói A). Mức D cần thêm `links_corpus_vi_en_zh.parquet` của BTC.

## 3. Tải checkpoint

Không cần cho mức A và B. Mức C tự tải khi chạy. Muốn tải trước, đúng revision đã dùng:

```bash
huggingface-cli download BAAI/bge-m3              --revision 5617a9f61b028005a4858fdac845db406aefb181
huggingface-cli download BAAI/bge-reranker-v2-m3  --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
huggingface-cli download Qwen/Qwen3.5-9B          --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
huggingface-cli download Qwen/Qwen3.5-4B          --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a   # chỉ doc2query
```

Không dùng API hay mô hình đóng nào. Không fine-tune.

## 4. Mức A — dựng lại danh sách nộp (CPU)

Cần: gói A (thêm gói C nếu dựng bài doc2query).

```bash
# A1. p(gold) và độ sâu N_q theo câu hỏi. Đã có sẵn bản dùng khi nộp ở runs/reports/pgold_feats/base2.
#     Lệnh này fit lại từ đầu (~3,5 phút); 5 danh sách thêm sau ngày fit phải loại ra để ra đúng bản đã dùng.
python scripts/analysis/pgold_all.py --lists runs/reports/pgold_lists \
    --out runs/reports/pgold_feats/base2_refit --exclude 7584,7589,7608,7609,7632

# A2. Danh sách docs + chunks: job pN3_base (#7892) và N1_n125 (#7916)
python scripts/analysis/pn3_lists.py --model base2 --out out/lists_pn3.parquet

# A3. So với bài đã nộp
python scripts/release/check_lists.py out/lists_pn3.parquet kaggle/datasets/mat_lists22/lists.parquet --job pN3_base
python scripts/release/check_lists.py out/lists_pn3.parquet kaggle/datasets/mat_lists30/lists.parquet --job N1_n125
#   có zip đã nộp thì so trực tiếp:  ... out/lists_pn3.parquet pN3_base_pqchunks.zip --job pN3_base
```

Kết quả mong đợi (đã chạy 02/11/2026 trong thư mục trống):
- A3 in `same order 1200/1200` cho cả docs và chunks.
- Nếu A2 dùng `--model base2_refit`: tập docs khớp 1.199/1.200 câu hỏi (q785 thêm 1 doc, do sai số tối ưu); chunks khớp 1.200/1.200.

Bốn bài private còn lại (cần gói C):

```bash
python scripts/analysis/dq_lists.py --base out/lists_pn3.parquet --out out/lists_dq29.parquet \
    --job S1_dqd20:dense_head=20 --job S2_dqb20:bm25_head=20 --gate 0                      # #7893, #7894
python scripts/analysis/dq_lists.py --base out/lists_pn3.parquet --out out/lists_dq31.parquet \
    --job F2_dqd20b20g:dense=20,bm25=20 --chunks 40 --gate 200                             # #7917
python scripts/release/check_lists.py out/lists_dq29.parquet kaggle/datasets/mat_lists29/lists.parquet --job S1_dqd20
python scripts/release/check_lists.py out/lists_dq31.parquet kaggle/datasets/mat_lists31/lists.parquet --job F2_dqd20b20g
```

## 5. Mức B — văn bản chunk và file zip

Văn bản chunk là cửa sổ ≤ 4.500 ký tự trích nguyên văn từ docstore.
- Mã: `materialize_lists.py` + `chunk_stair.py`, nhúng nguyên văn trong kernel `kaggle/kernels_generic/vibiomir-mat-st3-acc2/run.py`.
- Code `r2ai2026` mà kernel import: bản cố định `kaggle/chunk_v21_assets/vibiomir-code.zip` (CODE_VERSION `06f71bf-dirty`).

**Trên Kaggle (cách đội đã chạy):**
1. Tạo các dataset `vibiomir-docstore-v21` (`data/docstore/v2.1/corpus` + `query.parquet`), `vibiomir-chunk-v21-assets` (`kaggle/chunk_v21_assets/`), `vibiomir-pas-scores` (`kaggle/datasets/pas_scores/`) và `vibiomir-mat-lists22` (thư mục chứa `lists.parquet`).
2. Sửa `id` và `dataset_sources` trong `kernel-metadata.json` theo tài khoản của bạn.
3. Chạy: `kaggle kernels push -p kaggle/kernels_generic/vibiomir-mat-st3-acc2`.
4. Output: `pN3_base.zip`, `N1_n125.zip`, `stats.json`.

**Trên máy (cùng mã, xuất từ kernel; đường này chưa được chạy thử, đường đã chạy là kernel ở trên):**

```bash
python scripts/release/extract_bundle.py kaggle/kernels_generic/vibiomir-mat-st3-acc2 --out tools/mat
mkdir -p work/mat_in
cp out/lists_pn3.parquet work/mat_in/lists.parquet
cp data/raw/btc/query.parquet kaggle/datasets/pas_scores/pas_scores.parquet work/mat_in/
cp -r data/docstore/v2.1/corpus work/mat_in/corpus          # hoặc symlink / junction
unzip -q kaggle/chunk_v21_assets/vibiomir-code.zip -d work/mat_in/vibiomir-code
python tools/mat/materialize_lists.py --input work/mat_in --out work/zips
python scripts/release/check_lists.py out/lists_pn3.parquet work/zips/pN3_base.zip --job pN3_base
```

Lưu ý: `--input` phải là thư mục riêng như trên. Script lấy `lists.parquet` và `part-*.jsonl*` đầu tiên tìm thấy (đệ quy) dưới thư mục đó.

## 6. Mức C — chạy lại các stage GPU

Mỗi stage là một kernel trong `kaggle/kernels_generic/`. `run.py` của kernel chứa nguyên văn mã và tham số đã chạy (xem bằng `scripts/release/extract_bundle.py`). Mã trong kernel là chuẩn. Ví dụ `rerank.py` trong kernel `rr3-p*` có tuỳ chọn `--best-passages` mà `r2ai2026/stages/rerank.py` không có.

Cách chạy chung:
1. Tạo dataset đầu vào (docstore, danh sách ứng viên…).
2. Sửa `id` và `dataset_sources`/`kernel_sources` trong `kernel-metadata.json` theo tài khoản của bạn.
3. Chạy: `kaggle kernels push -p <thư mục>`.
4. Tải output: `kaggle kernels output <user>/<kernel> -p <đích>`.

| # | Stage | Kernel (thư mục) | Đầu vào | Đầu ra → đặt tại |
|---|---|---|---|---|
| C0 | chỉ mục + ranking câu hỏi gốc (có thể bỏ qua: chỉ mục có trong `index_v2.1`) | `vibiomir-v21-m3-vi-{a,b,c,d}`, `-m3-zh`, `-m3-en` (BGE-M3, `dense_embed.py --save-emb`), `vibiomir-v21-bm25-{vi-doc,vi-sec,zh,en}` (`bm25_experiment.py --save-index`) | docstore v2.1 | `data/docstore/v2.1/index/`, `data/rankings_v21/{m3,bm25}_*.parquet` |
| C1 | khung truy vấn (Qwen3.5-9B) | `vibiomir-query-frame-v3-acc1` | `query.parquet` | `kaggle/outputs/query-frame-v3/query_frame.parquet` |
| C2 | biến thể truy vấn (CPU) | `python r2ai2026/stages/query_variants.py --queries data/raw/btc/query.parquet --frame kaggle/outputs/query-frame-v3/query_frame.parquet --out kaggle/datasets/qvariants_v3_acc1/query_variants.parquet` | C1 | `qv_zh.parquet` = các dòng `variant ∈ {zh_q, zh_kw}`; `qv_vi_bm25.parquet` = `{orig, kw, sub0..4, acro}` |
| C3 | truy hồi ZH/EN theo biến thể | `vibiomir-qv3-m3-{zh,en}`, `vibiomir-qv3-bm25-{zh,en}` | C0 (chỉ mục), C2 | `kaggle/outputs/qv3-*/ranking.parquet` |
| C4 | pool VI (CPU) | RRF k=60, 2:1:1 trên `data/rankings_v21/{m3_vi,bm25_vi_doc,bm25_vi_sec}.parquet`, cắt 5.000 (`r2ai2026/system/runner.py: builtin_cands`, `weights=[2,1,1]`, `vi_only`, `depth=5000`, `k=5000`) | C0 | `data/rankings_v21/pool5000_vi.parquet` |
| C5 | pool 3 ngôn ngữ (CPU) | `python scripts/kaggle_ops/lang_pool_sets.py --vi 1500 --zh 500 --en 500 --additions kaggle/datasets/probe_inputs/additions.parquet` rồi `python scripts/kaggle_ops/rerank_pool_cands.py --sets vi1500,zh500,en500 --out kaggle/datasets/rerank_pool` | C3, C4 | `kaggle/datasets/rerank_pool/cands.parquet` (3.000.000 cặp) |
| C6 | rr3: bge-reranker-v2-m3, 8 phần | `vibiomir-rr3-p{0..7}-acc*` | docstore, C5 | `kaggle/outputs/rr3-p*/rerank.parquet` |
| C7 | m3m: BGE-M3 ba đầu, 4 phần | `vibiomir-m3m-p{0..3}-acc*` | docstore, C5 | `kaggle/outputs/m3m-p*/m3multi.parquet` |
| C8 | rr4: reranker trên mọi đoạn của 1.200 ứng viên đầu (`rerank.py --keep-all`), 8 phần → hồ sơ đoạn | `vibiomir-rr4-p{0..7}-acc*` | docstore, C5 | `kaggle/outputs/rr4-p*/passages.parquet`; nối 8 phần lại = `kaggle/datasets/pas_scores/pas_scores.parquet` (4.942.553 dòng, khớp hoàn toàn) |
| C9 | doc2query (chỉ #7893/#7894/#7917) | `vibiomir-dq-g-acc{1..4}` (phần đầu), `vibiomir-dqt-p{0..7}` (phần đuôi); đầu vào từ `scripts/kaggle_ops/doc2query_inputs.py`; gộp bằng `scripts/analysis/dq_prepare.py`; điểm: `vibiomir-dq-emb*`, `vibiomir-dq-score-all-acc1`, `scripts/analysis/doc2query_score.py`, `dq_bm25.py` | docstore | `kaggle/outputs/dq-all*`, `runs/reports/dq_*` |

Sau C6–C7, mức A chạy được ngay: `pgold_all.py` đọc `rerank_pool/cands.parquet`, `rr3-p*` và `m3m-p*`. Danh sách đã chấm và điểm public dùng để hiệu chỉnh nằm ở `runs/reports/pgold_lists/`. Muốn dựng lại từ các zip đã nộp: `scripts/analysis/sub_doc_lists.py`.

Ghi chú về độ khớp khi chạy lại:
- C1 và C9 giải mã tham lam, nhưng khác GPU hoặc phiên bản thư viện có thể lệch vài token. Đầu ra gốc đã chia sẻ (gói D, C).
- C4 dựng lại bằng `builtin_cands` khác bản gốc 257/1,8 triệu cặp ở ngưỡng 1.500, do các ứng viên hoà điểm RRF. Bản gốc được chia sẻ.
- C2 (biến thể truy vấn) và C5 (pool) chạy lại từ đầu ra đã chia sẻ khớp hoàn toàn (đã kiểm).

## 7. Mức D — docstore và chỉ mục từ dữ liệu thô

Xem `docs/DATA_CARD.md` §6 và `docs/DATA_PIPELINE.md`. Tóm tắt:
1. Crawl danh sách URL của BTC (`r2ai2026/crawl/`) và cứu URL chết (Common Crawl, Wayback, PubMed).
2. Chạy `scripts/data/build_docstore.py` (v1/v2).
3. Chạy `scripts/data/select_hf_v2_1.py` và `build_docstore_v2_1.py` (v2.1).
4. Chạy `scripts/data/validate_docstore.py`.
5. Dựng chỉ mục bằng các kernel C0, gộp shard dense bằng `scripts/data/merge_dense_shards.py`.

Crawl web không tái lập byte-for-byte (trang thay đổi theo thời gian). Vì vậy docstore v2.1 được chia sẻ nguyên bản, có `MANIFEST.json` ghi hash đầu vào.

## 8. Đóng gói lại bản phát hành

```bash
python scripts/release/pack_release.py --hash-big   # release/: code zip, artifacts_*.zip, MANIFEST.tsv
```

## 9. Cấu trúc repo

```
r2ai2026/            thư viện: core (định dạng bài nộp, chấm nội bộ), corpus, crawl, stages (mã từng stage), system (runner)
scripts/data/        dựng docstore từ dữ liệu thô
scripts/kaggle_ops/  đầu vào cho kernel: pool ứng viên, đầu vào doc2query, đóng gói mã
scripts/analysis/    p(gold), danh sách nộp, doc2query
scripts/release/     đóng gói, kiểm sha256, so danh sách, xuất mã từ kernel
scripts/submit/      nộp bài và lấy điểm Codabench
kaggle/kernels_generic/  mọi kernel Kaggle đã chạy (mã nhúng trong run.py)
kaggle/pinned/       bản cố định của mã materialize
configs/             cấu hình runner và bảng mở rộng truy vấn
docs/                thuyết minh, thẻ dữ liệu, thiết kế; README nội bộ cũ: docs/README_NOI_BO.md
EXPERIMENTS.md       sổ điểm mọi bài nộp; WORKLOG.md: nhật ký
```
