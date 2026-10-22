# Song song hóa hệ thống trên Kaggle — so sánh phương án (2026-10-20)

Nhãn: **[ĐO]** đo trên máy/log thật · **[TÀI LIỆU]** nguồn thứ cấp · **[CHƯA KIỂM]** cần thử bằng 1 lần push.

## 1. Tài nguyên hiện có
| | Local | Kaggle CPU kernel | Kaggle GPU kernel |
|---|---|---|---|
| CPU / RAM | Ryzen 7 7435HS 8C/16T, 23.7 GB (≈9 GB trống khi đang chạy crawl + IDE) [ĐO] | 4 lõi, 30 GB [TÀI LIỆU] | 2×T4 16 GB, ~29 GB RAM [TÀI LIỆU] |
| Thời gian | không giới hạn | 12 h/phiên [TÀI LIỆU] | 12 h/phiên, ~30 h GPU/tuần [TÀI LIỆU] |
| Ổ đĩa | 51 GB trống | 20 GB output `/kaggle/working` [TÀI LIỆU] | như CPU |
| IP | 1 IP nhà (đã bị Long Châu cấm ~22 h, 39.net bật captcha) | mỗi kernel 1 IP datacenter riêng | như CPU |
| Số phiên song song | — | **[CHƯA KIỂM]** (p1–p5 từng chạy cùng ngày) | **[CHƯA KIỂM]** |

## 2. Bằng chứng crawl từ IP Kaggle (5 kernel cũ, log thật) [ĐO]
| Kernel | Lấy được | Tốc độ/kernel |
|---|---|---|
| p1 vinmec + hellobacsi | 37,722 / 43,209 (87%) | 17 URL/s |
| p3 báo/portal VN | 141,835 / 180,585 (78%) | 13 URL/s |
| p4 120ask (khi site còn sống) | 98,145 / 98,159 | 11 URL/s |
| p5 39.net, familydoctor… | 32,851 / 71,323 (46%), **đứng hẳn sau đó** = bị chặn giữa chừng | 17 URL/s |
| p2 Long Châu | **0** / 73,422 — Cloudflare chặn IP datacenter | — |

So với local: crawl trực tiếp ~6–7 URL/s, Wayback ~3 URL/s/IP [ĐO].

## 3. Các phương án
| | A. Tất cả local | B. Tất cả trên Kaggle | **C. Lai (đề xuất)** |
|---|---|---|---|
| Treo máy | **Cao**: index BM25 toàn văn 1.05M doc + quét file 4 GB ăn hết RAM trống | Không | Không (local chỉ điều phối) |
| GPU (BGE-M3, reranker) | Không có | Có | Có |
| Crawl | 1 IP, dễ bị cấm | N IP song song; Long Châu trực tiếp = 0 | Kaggle cho host chịu IP datacenter + Wayback chia shard; local giữ Wayback Long Châu đang chạy |
| Chạy nhiều thí nghiệm cùng lúc (E1/E2/E3) | Tuần tự | Song song mỗi biến thể 1 kernel | Song song |
| Chi phí chuyển dữ liệu | 0 | Upload corpus 1 lần (~1.1 GB nén), sau đó dữ liệu nằm trên Kaggle | Như B; chỉ kéo về local file nộp (~50 MB) |
| Tốc độ debug | Nhanh | Chậm (push → chờ → đọc log, ~2–5 phút/vòng) | Debug trên mẫu nhỏ ở local, chạy thật trên Kaggle |

**Kết luận: phương án C.** Local là "bảng điều khiển" + dry-run trên mẫu nhỏ (golden rule #7); mọi việc nặng (index, embedding, chạy thí nghiệm, crawl đa IP) chạy trên Kaggle.

## 4. Kiến trúc C
```
local (điều phối, code, dry-run)                      Kaggle
 ├─ kaggle datasets version  ──►  dataset vibiomir-corpus   (docs_clean shards *.jsonl.gz, versioned)
 ├─ kaggle datasets version  ──►  dataset vibiomir-code     (package r2ai2026 + scripts)
 ├─ kaggle kernels push      ──►  crawl-shard-{1..N}  (CPU, internet, mỗi shard 1 IP) ──► output parquet
 │                                     └─(kernel_sources)─► merge-clean (CPU) ──► corpus mới
 ├─ kaggle kernels push      ──►  bm25-E1 / E2 / E3 (CPU, song song) ──► submission_*.zip + stats
 ├─ kaggle kernels push      ──►  embed-bge-m3 (GPU) ──► vectors ──► hybrid/rerank (GPU)
 └─ kaggle kernels output    ◄──  chỉ kéo về file nộp (~50 MB) → nộp tay Codabench
```
- Ghép kernel bằng `kernel_sources` (output kernel trước là input kernel sau), nên corpus không phải đi vòng về local.
- Corpus chia shard (vd. 8 × ~150 MB nén) để versioning chỉ upload shard thay đổi (doc crawl thêm = shard mới).
- Mọi kernel đọc code từ dataset `vibiomir-code`, nên local và Kaggle chạy **cùng một code**, không copy-paste vào notebook.

## 5. Phân công cụ thể
| Việc | Chạy ở đâu | Lý do |
|---|---|---|
| Wayback Long Châu (69k, đang chạy ~6 h) | local, giữ nguyên | đang chạy tốt 95%; có thể chia thêm 3–4 shard Wayback lên Kaggle để rút còn ~1.5 h |
| Long Châu trực tiếp | không làm | 0% từ Kaggle, IP nhà bị cấm |
| 39.net, ask.39.net | Kaggle shard nhỏ, tốc độ thấp (≤1 req/s/kernel) | IP mới; p5 bị chặn khi chạy 17/s |
| Dựng corpus sạch + index BM25 toàn văn | Kaggle CPU (30 GB) | tránh treo máy local |
| E1/E2/E3 (biến thể BM25) | Kaggle CPU, **3 kernel song song** | 1 vòng thay vì 3 |
| BGE-M3 embedding, reranker | Kaggle GPU | không có GPU local |
| Đóng gói + nộp | local | nộp tay trên Codabench |

## 6. Việc cần kiểm trước khi chuyển hẳn
1. **Số kernel chạy song song tối đa**: push 3 kernel CPU cùng lúc, xem cái thứ mấy bị xếp hàng.
2. **Tốc độ upload** dataset từ local: thử 1 shard ~150 MB.
3. Chạy E1 trên Kaggle với cùng code đã dry-run ở local (100 query, corpus mẫu), so sánh kết quả giống hệt.

## 7. Thứ tự triển khai (đề xuất)
1. Tạo dataset `vibiomir-code` + `vibiomir-corpus` (shard từ `docs_clean.jsonl`). Việc đơn giản → giao Haiku.
2. Viết `kaggle/kernels/bm25_experiment/` (tham số: tokenizer, đơn vị doc/chunk, N) và dry-run local trên mẫu.
3. Push E1 (+E2, E3 song song) → kéo zip về → người dùng nộp.
4. Song song: shard Wayback Long Châu lên Kaggle nếu job local còn > 3 h.
