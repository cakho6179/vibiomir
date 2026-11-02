# Hệ thống cuối — 5 bài private (phase 45)

Cập nhật 02/11/2026. Tài liệu nguồn cho phần nộp hệ thống (hạn 04/11 23:59 ICT) và working notes paper.
Mỗi danh sách nộp đều dựng lại được bằng script trong repo; các phép kiểm "khớp từng dòng" ghi ở mục 4.

## 1. Năm bài private

| Mã | File | Ý tưởng | docs/q | chunks/q |
|---|---|---|---|---|
| #7892 | `pN3_base_pqchunks.zip` | Nền: chọn độ sâu theo p(gold), = #7589 (public FINAL .3484) | 157,4 | 162,7 |
| #7893 | `S1_dqd20.zip` | Nền + 20 doc doc2query dense (phần đầu) | 177,4 | 162,7 |
| #7894 | `S2_dqb20.zip` | Nền + 20 doc doc2query BM25 (phần đầu) | 177,4 | 162,7 |
| #7916 | `N1_n125.zip` | Docs sâu 1,25× theo p(gold); chunks = nền | 196,7 | 162,7 |
| #7917 | `F2_dqd20b20g.zip` | Nền + 20 dense + 20 BM25 doc2query trên toàn bộ doc, có cổng đuôi | 197,4 | 162,7 |

Luật phase 45 là "Add": đội chọn bài hiện lên leaderboard. Hết 01/11, server vẫn chưa chấm bài nào (mọi bài nộp từ 31/10 16:58Z đứng ở "Submitting").

## 2. Chuỗi xử lý

```
docstore v2.1 (1.192.673 doc VI/ZH/EN)
  └─ ứng viên: BM25 + BGE-M3 dense theo ngôn ngữ, query frame Qwen3.5-9B (từ khoá/dịch ZH, EN)
       → 2.500 ứng viên/query (vi1500 / zh500 / en500)                      [configs/system_v3.yaml, stages pool_*]
  └─ đặc trưng trên pool 1.200 doc/query: bge-reranker-v2-m3 (rr3), BGE-M3 dense+sparse+colbert (m3all)
  └─ p(gold) "base2": logit = a_lang + e·log1p(rank) + in_pool·(h + b·rr3 + c·m3z + d·m3rel)
       fit trên P/R/F2 tổng hợp (public) của 27 danh sách doc của chính đội   [scripts/analysis/pgold_all.py]
       → runs/reports/pgold_feats/base2/{cands_p,n_per_query,fit}.parquet/json
  └─ độ sâu theo query: N_q = argmax E[F2] (gold mỗi query ngẫu nhiên, sd σ)    [pgold_fit.f2_exp]
  └─ danh sách:
       pN3_base, N1_n125                                                      [scripts/analysis/pn3_lists.py]
       S1, S2, H1, H2, F1, F2 (+ doc2query)                                    [scripts/analysis/dq_lists.py]
  └─ zip: kernel Kaggle materialize_lists (cửa sổ 4.500 ký tự "seed" quanh đoạn rerank tốt nhất, LZMA)
       kernels_generic/vibiomir-mat-st{3,10,11,12}-acc2
```

### 2.1 Docs
- Ứng viên xếp theo p(gold) giảm dần; lấy N_q đầu (pN3), hoặc round(1,25·N_q) (N1).
- N_q tối đa F2 kỳ vọng của từng query. Tham số fit: γ .536, σ .698. Logit ngoài ứng viên VI ≈ −27,7, tức là mô hình không đặt gold VI ngoài 2.500 ứng viên.

### 2.2 Chunks
- Một chunk/doc cho n_c doc có p cao nhất. n_c = argmax E[F2] chunk, với p_c = 0,77·p và số gold chunk = 75/71·g_exp (tỉ lệ |gold chunk|/|gold doc| công khai).
- Văn bản chunk là cửa sổ ≤ 4.500 ký tự trích nguyên văn từ doc. Cửa sổ bám theo profile điểm đoạn của reranker ("seed" = `stair_chunk`); khi thiếu profile thì lùi về cửa sổ từ vựng.
- Ngân sách zip ≈ 100 MB (server trả 413 khi vượt) giới hạn khoảng 163 chunk/query.

### 2.3 Doc2query (S1, S2, F2)
- Sinh câu hỏi: Qwen3.5-4B, 3 câu/doc, tiếng Việt.
  - Phần đầu: 63.298 doc, 189.140 câu (`kaggle/outputs/dq-all`).
  - Phần đuôi: 125.929 doc, 375.724 câu (8 kernel `vibiomir-dqt-p0..7`).
- Dense: BGE-M3. Điểm (query, doc) = cosine lớn nhất giữa query và các câu của doc.
  - Phần đầu: `doc2query_score.py`.
  - Đầu + đuôi: kernel `hytahy/vibiomir-dq-score-all` → `runs/reports/dq_cross_all`.
- BM25 trên câu hỏi: token là âm tiết, giữ dấu, k1 1,2, b .75. Điểm doc = max theo câu (`dq_bm25.py` → `runs/reports/dq_bm25{,_all}`).
- Ghép vào nền: thêm k doc đầu chưa có trong nền, đặt sau đuôi danh sách. Mỗi doc thêm thay 1 chunk p thấp nhất của nền, nên tổng số chunk giữ nguyên.
- Cổng đuôi (F1, F2): doc phần đuôi chỉ được nhận khi kênh còn lại cũng xếp nó vào top 200, vì S1/S2 chỉ đo được mật độ gold của phần đầu.

## 3. Tuân thủ luật

### 3.1 Mô hình
Tất cả đều open-weight, ≤ 15B và phát hành trước 01/08/2026. Ngày lấy từ HF API `createdAt` (kiểm 02/11).

| Mô hình | Vai trò | Kích thước | Phát hành | License |
|---|---|---|---|---|
| BAAI/bge-m3 | dense/sparse/colbert ứng viên, đặc trưng m3all, embed doc2query | ~0,57B | 2024-01-27 | MIT |
| BAAI/bge-reranker-v2-m3 | rr3, profile đoạn cho chunk | 0,57B | 2024-03-15 | Apache-2.0 |
| Qwen/Qwen3.5-9B | query frame (từ khoá, dịch ZH/EN) | 9,65B | 2026-02-27 | Apache-2.0 |
| Qwen/Qwen3.5-4B | sinh câu hỏi doc2query | 4,66B | 2026-02-27 | Apache-2.0 |

Không có LLM đóng nào trong hệ thống nộp.

### 3.2 Dữ liệu (`docs/DATA_CARD.md`)
- Chỉ dùng tài liệu thuộc danh sách link chính thức của BTC (`links_corpus_vi_en_zh.parquet`, 1.220.598 URL). Không có doc ngoài danh sách đó.
- Nguồn nội dung:

  | Nguồn | Số doc | Ghi chú |
  |---|---|---|
  | crawl nội bộ | 1.182.666 | |
  | HF `tmh1008/ViBioMIR-crawled-corpus` @ `18cb5a5` | 5.760 | chỉ làm corpus |
  | Common Crawl | 2.278 | |
  | Internet Archive Wayback | 1.400 | |
  | recrawl trực tiếp | 370 | |
  | NCBI E-utilities (PubMed) | 177 | |
  | Long Châu recrawl | 22 | |

### 3.3 Cam kết và công bố
- **MedKB của BTC.** Đầu cuộc thi (19–20/10) đội có một lần probe MedKB. Lần đó chỉ ghi lại thống kê tổng hợp: độ dài chunk trung vị ~3,3k ký tự, tối đa ~4,8k; chunk có trường `similar_queries`; ~88% URL kết quả nằm trong corpus (WORKLOG 20/10, `ROADMAP_SYSTEM.md` §2).
  - Không nội dung, kết quả hay `similar_queries` nào của MedKB đi vào bất kỳ danh sách nộp nào. Không gọi API MedKB ở thời điểm chạy.
  - Câu hỏi doc2query do Qwen3.5-4B tự sinh từ văn bản doc.
  - Cửa sổ chunk 4.500 ký tự được chọn bằng chính các bài nộp public: #6933 (6.000) và #6934 (7.500) cho chunks F2 như nhau, tức là bão hoà từ 4.500.
- **Thống kê gold.** Hai con số dùng ở dạng tổng hợp: |gold| trung bình 71 doc / 75 chunk mỗi query, và public = 600/1.200 query. Chúng được đọc từ một file output của scorer chính thức (`details.jsonl`) do một đội khác công khai trong repo của họ (`research/refs/`, `docs/RESEARCH_REFS.md`).
  - Chỉ dùng ba số trung bình đó, cụ thể là hệ số 75/71 trong `pn3_lists.py`.
  - Không dùng nhãn từng cặp (query, doc) hay từng query từ file chấm của đội mình hay đội khác.
- **Leaderboard.** Thông tin lấy từ leaderboard chỉ là P/R/F2 tổng hợp trên public split của các bài đội đã nộp. Nó được dùng để hiệu chỉnh p(gold) (14 tham số, 27 danh sách) và độ sâu N_q. Không có tham số nào ở mức từng query hay từng doc được học từ nhãn.
- Không có doc nào lấy từ danh sách nộp của đội khác.

## 4. Tái lập

```
# p(gold) base2 (đầu vào: pool, rr3, m3m, các danh sách đã chấm runs/reports/pgold_lists)
# base2 fit trên 27 danh sách; 5 danh sách thêm sau phải loại ra để dựng lại đúng bản đã dùng
python scripts/analysis/pgold_all.py --lists runs/reports/pgold_lists --out runs/reports/pgold_feats/base2 \
    --exclude 7584,7589,7608,7609,7632

# pN3_base + N1_n125
python scripts/analysis/pn3_lists.py --out kaggle/datasets/mat_lists22/lists.parquet

# doc2query
python scripts/analysis/dq_bm25.py --questions kaggle/outputs/dq-all --out runs/reports/dq_bm25
python scripts/analysis/dq_bm25.py --questions kaggle/outputs/dq-all2 --out runs/reports/dq_bm25_all
python scripts/analysis/doc2query_score.py --emb kaggle/outputs/dq-embed --topk 1000 --out runs/reports/dq_cross
#   dense đầu + đuôi: kernel kaggle/kernels_generic/vibiomir-dq-score-all-acc1 -> runs/reports/dq_cross_all
python scripts/analysis/dq_lists.py --out kaggle/datasets/mat_lists29/lists.parquet \
    --job S1_dqd20:dense_head=20 --job S2_dqb20:bm25_head=20 --gate 0
python scripts/analysis/dq_lists.py --out kaggle/datasets/mat_lists31/lists.parquet \
    --job F1_dqd50g:dense=50 --job F2_dqd20b20g:dense=20,bm25=20 --chunks 40 --gate 200

# zip: tải lists lên dataset Kaggle, chạy kernel materialize (vd. vibiomir-mat-st12-acc2)
```

Các phép kiểm đã chạy (02/11), so với các lists đã dùng để đóng zip nộp:

| Script | So với | Kết quả |
|---|---|---|
| `pn3_lists.py` | `mat_lists22` job pN3_base | 384.020 dòng, khớp 100% |
| `pn3_lists.py` | `mat_lists30` job N1_n125 | 431.225 dòng, khớp 100% |
| `dq_lists.py` | `mat_lists29` S1_dqd20, S2_dqb20 | 408.020 dòng mỗi job, khớp 100% |
| `dq_lists.py` | `mat_lists30` H2_dqd20b20 | 432.073 dòng, khớp 100% |
| `dq_lists.py` | F2 | chính là `mat_lists31` đã nộp |
| `pgold_all.py --exclude …` | `pgold_feats/base2` | 3.000.000 p, lệch tối đa 1,1e-5; N_q lệch 1 doc ở 1/1.200 query (sai số tối ưu); ~3,7 phút CPU |

| 5 zip đã nộp trong `submissions/` | job tương ứng trong lists | 1.200/1.200 query: thứ tự `relevant_docs` và `doc_id` chunk khớp hoàn toàn, không chunk nào bị rơi |

Code dựng văn bản chunk: các kernel materialize nhúng `materialize_lists` và `stair_chunk` trong `run.py` của bundle (đã commit). Chúng import `r2ai2026` từ dataset `tthiutoko/vibiomir-chunk-v21-assets` (`vibiomir-code/`, CODE_VERSION `06f71bf-dirty`). Hai file được dùng có lưu bản cố định ở `kaggle/pinned/chunk-v21-assets/`, vì working tree đã đổi sau đó.

Riêng các stage ứng viên/đặc trưng của system v3 chưa được kiểm khớp từng dòng. Chúng chạy trên Kaggle, output lưu ở `kaggle/outputs/`.
