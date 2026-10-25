# Kế hoạch tầng lấy doc (L1): 3000 VI + 1500 EN + 1500 ZH / query — 25/10

## 1. Docstore đang có gì

| | v1 (`data/docstore/v1`, 24/10) | v2 (`data/docstore/v2`, 25/10) | v3 |
|---|---|---|---|
| Doc | 1,185,389 (vi 730,691 · en 239,895 · zh 214,803) | 1,186,913 (vi 732,215 · en/zh như v1); bản bỏ trùng nội dung `docs_meta_unique` 1,181,991 | **chưa có trên đĩa** |
| Nguồn | crawl 1,182,666 + wayback 1,340 + cc 1,009 + direct 197 + pubmed 177 | v1 + delta `part-08` **chỉ VI**: 1,524 doc (cc 1,269 · direct 173 · wayback 60 · lc 22) | |
| Passage (≤1.500 ký tự) | 3,582,922 | 3,593,227 | |
| Section (2.000–4.800 ký tự, theo heading) | 1,643,339 | 1,647,091 | |
| Khác | `furniture.parquet` (menu/footer đã tách, 2,9% ký tự), `templates.json` (48 host), `slug_docs` 9,877 doc chỉ có slug URL, `MANIFEST.json` (hash đầu vào) | như v1 + `delta_duplicates.parquet`, `delta_report.json` | |

`data/processed_v3/` **không phải** docstore v3: đó là corpus processed cũ (20/10, từ `docs_clean.jsonl`).

### Độ phủ so với link BTC (1,220,598 id)
| Ngôn ngữ | Link | Thiếu trong v2 | Doc thin (<500 ký tự) trong v2 |
|---|---|---|---|
| VI | 740,598 | 8,383 | 14,531 |
| EN | 240,000 | 105 | 9,059 |
| ZH | 240,000 | **25,197** | **95,581 / 214,803 (44%)** |

⇒ Với ZH, chất lượng kho chứ không phải code truy hồi mới là giới hạn: gần một nửa không gian id ZH thiếu hoặc thin.

### Nguồn chưa gộp (ứng viên cho v3)
- `data/raw/recovered/wayback_zh_2026-10-24.jsonl`: 1,103 id ZH. v2 đã bỏ qua vì delta chỉ nhận VI (`skipped_lang` 442).
- `kaggle/outputs/vibiomir-hf-recover/recovered_hf.jsonl.gz`: 1,800 dòng VI, nhưng có 81 dòng nhị phân (JPEG) và 1,187 dòng của thaythuocvietnam.vn (1,073 dòng cùng một tiền tố, tức trang template). Phần dùng được chỉ khoảng vài trăm dòng (hellobacsi 280, phunusuckhoe 66, medlatec 49…). Cần lọc template trước khi gộp. Audit của agent kia (`docs/EXTERNAL_HF_TMH_AUDIT_2026-10-25.md`) ra 49 doc dùng được; cần đối chiếu hai con số này.

## 2. Sáu module đã cất (`_temp/2026-10-25/cleanup/r2ai2026/`)

| Module | Nội dung | Kết luận |
|---|---|---|
| `fusion.py` | RRF, hybrid min-max | Trùng `builtin_cands`. Thứ còn thiếu là quota theo ngôn ngữ ⇒ thêm vào `builtin_cands`, không hồi module |
| `multistack.py` | Ghép nhiều luồng (web VI, ZH, slug, dense, pubmed) | Import `bm25` (đã cất), trọng số đoán tay. Giữ làm ghi chú thiết kế; runner đã lo phần điều phối |
| `clir_transfer.py` | Từ điển VI→ZH 86 mục, viết tay | **Tận dụng**: gộp vào một bảng mở rộng truy vấn tất định |
| `query_expansion.py` | 46 viết tắt xét nghiệm/bệnh (VI/EN) + prompt HyDE/decompose/step-back | **Tận dụng bảng viết tắt**. Prompt đã có `query_frame` thay |
| `translate.py` | MarianMT vi→en | Bỏ: chỉ có EN, kém hơn bản dịch LLM đang có |
| `decision.py` | Cắt theo score; JEV với `k_gold=2`, `max_docs` 4–5 | Viết sai giả định (|G| ≈ 71) ⇒ dùng nguyên là có hại. Luật dừng theo F2 kỳ vọng biên dùng lại được ở stage cắt, khi có xác suất đã hiệu chỉnh và |G| thật. Không thuộc tầng lấy doc |

Hai bảng tận dụng được đều viết tay nên sạch giấy phép. Lexicon ICD (`data/external/lexicons/`) không đưa vào hệ thống nộp vì repo gốc không có LICENSE.

## 3. Cần sửa gì để ra 3000 / 1500 / 1500

1. `bm25_experiment.py`, `dense_embed.py`: thay `--vi-only` bằng `--lang vi|en|zh`, cho `--topk` theo quota.
2. Truy vấn EN/ZH: `kaggle/outputs/vibiomir-translate/query_{en,zh}.parquet` (Qwen2.5-7B) chưa phải stage. Để chạy được trên server BTC, đưa việc dịch vào stage, ví dụ `query_frame` sinh thêm keyword EN/ZH bằng Qwen3.5-9B. Bge-m3 dense thì dùng thẳng câu hỏi VI cho cả ba ngôn ngữ.
3. `bm25-e1en` / `bm25-e1zh` cũ chỉ có top 200 ⇒ phải chạy lại sâu 1500.
4. `builtin_cands`: thêm `quota: {vi: 3000, en: 1500, zh: 1500}` (RRF riêng từng ngôn ngữ rồi nối).
5. Config mới `configs/system_v2_pool.yaml`: các luồng BM25 (VI, EN, ZH) + dense (theo ngôn ngữ) → pool 6000.

## 4. Hai điểm cần quyết

1. **EN/ZH có thể gần như không có gold.** Probe 20/10 (14 EN + 14 ZH): Docs P .017, R .0105 ≈ 0,5–0,75 gold/query, trong khi VI top 28 có P ≈ .35. Trước khi dựng luồng EN/ZH sâu, nên đo trần recall bằng **1 lượt nộp probe** kiểu #6739: zip chỉ gồm EN 1500 + ZH 1500 doc (chunk tối thiểu). Có thể làm thêm 1 probe VI@3000. Cần kiểm giới hạn doc/query và dung lượng zip trước; mỗi probe tốn 1 lượt public và cần bạn duyệt.
2. **Ngân sách judge.** 6000 × 1200 = 7,2 triệu cặp, gấp khoảng 40 lần G0 (180k cặp, đang chạy trên 3 GPU). Phải có tầng lọc rẻ ở giữa (bge-reranker-v2-m3 hoặc student 0.6B) đưa pool về khoảng 300/query rồi mới chạy judge 4B/8B. Nghĩa là "lấy 6000" là tầng recall, không phải judge hết.
