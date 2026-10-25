# 2026-10-24_slice_probe — Probe chỉ-doc: thứ tự reranker và mật độ gold của các tầng union

- **Trạng thái:** đã đo (#6767–6769, 24/10)
- **Câu hỏi:** (1) Cùng số doc, thứ tự bge-reranker-v2-m3 (r1) có chọn được nhiều gold hơn thứ tự w12 không? (2) Mật độ gold
  của các tầng union nằm ngoài danh sách chính là bao nhiêu?
- **Giả thuyết:** r1 xếp tốt hơn (trùng chương ICD 90,0% so với 88,8%); mật độ một tầng < 6% trừ khi nhiều họ đồng thuận.
- **Quyết định phụ thuộc:** r1ord Docs P > base ⇒ mở rộng reranker lên union top-400 (kernel `vibiomir-rerank-r2p0..3`) và
  dùng ở L2. Tầng có P > F2/5 ≈ 6% ⇒ thêm vào danh sách doc chính; ngược lại cắt.

## Cách làm
- Base: `submissions/w12T2_VI_win3000_n150.zip` (88,2 doc/query, chunk = cửa sổ 3000).
- Mọi probe giữ **nguyên từng byte danh sách chunk** của base, chỉ thay `relevant_docs` ⇒ Chunks F2 không đổi, Docs P/R
  trả lời đúng một câu hỏi.
- `make_probes.py` → `submissions/probe_{r1ord,sliceA,sliceB}_w12T2_VI.zip`, thống kê `probe_stats.json`:
  - `r1ord`: mỗi query lấy đúng N_q như base, top-N_q theo điểm r1 trong pool của r1 (E1v2 top-100 ∪ M3 top-100, VI).
  - `sliceA`: doc union được ≥ 2 họ tìm thấy, ngoài base, hạng union ≤ 500.
  - `sliceB`: doc union chỉ họ m3 tìm thấy, ngoài base, hạng union 100–500.
- Cùng việc trên Kaggle (job builder): `"keep": {"spec": "union_scored_v1.parquet", "query": ...}`, `"exclude": {...}`
  (`kaggle/build_exp/build_submission.py: doc_lists`).
- Đọc kết quả: với một tầng, P (macro) = mật độ gold; query có tầng rỗng tính P = 0 ⇒ hiệu chỉnh × 1200 / số query không
  rỗng. R × 71 = số gold/query mà tầng chứa. Chỉ dùng số tổng hợp trên LB, không dùng nhãn từng cặp.

## Kết quả [ĐO]
| Probe | doc/q | Docs P | Docs R | Ghi chú |
|---|---|---|---|---|
| base #6750 | 88,2 | 0,2495 | 0,3823 | Docs F2 0,3132 |
| r1ord #6769 | 88,2 | 0,2398 | 0,3690 | Docs F2 0,3022 (−0,011) |
| sliceA #6768 | 225,3 | 0,0696 | 0,2175 | mật độ 7,0%; 15,4 gold/q |
| sliceB #6767 | 57,6 | 0,0593 (≈ 0,0597 sau hiệu chỉnh 8 query rỗng) | 0,0339 | ≈ 6,0%; 2,4 gold/q |

## Kết luận
- Thứ tự r1 **kém** w12 ở cùng N ⇒ không mở rộng reranker (bỏ r2), r1 không dùng làm khoá xếp chính.
- sliceA vượt ngưỡng F2/5 ≈ 6,3% ⇒ thêm vào danh sách doc (ước Docs F2 ~0,33); nên cắt theo hạng union trong tầng
  (phần đầu dày hơn) ⇒ bước sau: base ∪ sliceA top-k với k ∈ {75, 150, 225}.
- sliceB ≈ ngưỡng, ít gold ⇒ bỏ.
