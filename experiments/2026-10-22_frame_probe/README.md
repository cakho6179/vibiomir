# 2026-10-22_frame_probe — Tín hiệu "đúng bệnh / đúng ý" có tách hệ tốt khỏi hệ kém không?

- **Trạng thái:** xong
- **Câu hỏi:** Loại tài liệu, mã ICD 3 ký tự và ý hỏi ↔ heading có đi cùng chiều precision LB (BM25 .178 < w12 .354) và giảm theo hạng không?
- **Giả thuyết:** gold = bệnh × ý hỏi (slide BTC) ⇒ hệ tốt hơn có tỉ lệ khớp cao hơn.
- **Quyết định phụ thuộc:** tín hiệu nào đơn điệu thì đưa vào thiết kế kho doc (`docs/DOCSTORE_DESIGN.md`); không thì bỏ.

## Cách làm
- Dữ liệu: `data/raw/btc/*`, `data/processed_v3/`, ranking `kaggle/outputs/vibiomir-bm25-e1v2`, `vibiomir-build-depth-a` (w12), `vibiomir-build-depth-c` (rd).
- Mã: `host_mix.py` (host + URL rác), `probe_frame.py` (mã ICD qua ontology `research/refs/neiths_med-doc-retrieval`, 8 ý hỏi bằng từ khoá; doc = tiêu đề + heading + 600 ký tự đầu).
- Lệnh (từ `vibiomir/`): `$env:PYTHONIOENCODING="utf-8"; python experiments/2026-10-22_frame_probe/probe_frame.py`

## Kết quả [ĐO]
| Tập | URL rác | Khớp mã ICD | Khớp ý hỏi |
|---|---|---|---|
| BM25 e1v2 @1–28 (P .178) | 1,2% | .553 | .682 |
| w12 @1–28 (P .354) | 1,4% | .615 | .717 |
| rd @1–28 | 1,4% | .640 | .727 |
| w12 @29–90 | — | .540 | .663 |
| w12 @91–150 | 1,3% (@1–150) | .477 | .634 |
Độ phủ: mã ICD chỉ 272/1.200 query; ý hỏi 1.005/1.200. Host top-28 gần như giống nhau giữa các hệ.

## Kết luận
Loại tài liệu không phải đòn bẩy; **đúng bệnh** và **đúng ý** đều đơn điệu theo P và theo hạng ⇒ là trục của kho doc.
Cần LLM mở để phủ thực thể query (23% → ≥80%) và so ý ở mức mục. Số đã ghi `EXPERIMENTS.md` mục C.
