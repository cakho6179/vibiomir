# Kế hoạch: Kho doc + Khung query (L0) — bản để hình dung, CHƯA xây (22/10)

Nhãn: **[ĐO]** số đo thật · **[SUY]** suy luận · **[MH]** ví dụ minh hoạ (viết tay, chưa chạy).
Quyết định của user: tự lên kế hoạch/thiết kế/nghiên cứu; **chưa dùng nhãn** (kể cả file chấm) — chỉ kiểm chứng không nhãn + A/B LB.

---

## 1. Vì sao cần, và cần cái gì

Slide BTC: gold = **(bệnh / nhóm ICD) × (ý hỏi)**. "Cùng bệnh, khác ý" ⇒ sai. Hệ hiện tại chỉ biết "văn bản giống câu hỏi"
(BM25/M3/reranker), không biết *bệnh nào* và *ý nào*. Kho doc + khung query cho mỗi cặp (query, doc) ba câu trả lời có cấu trúc:

```
   query ──► KHUNG QUERY                         doc ──► HỒ SƠ DOC
             bệnh/triệu chứng: [mã ICD…]                  bệnh chính: [mã ICD…]
             ý hỏi: [nguyên nhân, điều trị…]              các mục: [nguyên nhân §2, điều trị §4…]
             đối tượng: [trẻ em / thai phụ…]              đối tượng, loại tài liệu, cụm trùng
                         │                                           │
                         └──────────────►  SO KHỚP  ◄────────────────┘
                                 ① đúng bệnh?  ② có mục trả lời đúng ý?  ③ đúng đối tượng?
                                             │
                                  đặc trưng cho L2/L3 (cộng vào điểm reranker, chọn ngưỡng)
```

### Số đo đã có [ĐO]
| Đo | Kết quả | Hệ quả cho thiết kế |
|---|---|---|
| URL rác (xổ số, bóng đá, thời tiết…) | 3,2% corpus; chỉ 1,2–1,4% trong top-28/150 mọi hệ | lọc *loại tài liệu* gần như không giúp |
| Host trong top-28: BM25 (P .178) vs w12 (P .354) | gần như y hệt | sai nằm **trong cùng nguồn, cùng chủ đề** |
| Trùng **chương** ICD: BM25 79,7% → M3 88,8% → reranker 90,0% | giải thích ~1/3 khoảng cách P | chương quá thô ⇒ cần **bệnh cụ thể** |
| Trùng **mã ICD 3 ký tự** (luật, mới phủ 272/1.200 query) | BM25 .553 < w12 .615 < rd .640; theo hạng .615 → .540 → .477 | tín hiệu đúng chiều ✔ — nhưng phải phủ được nhiều query hơn (LLM) |
| Trùng **ý hỏi ↔ heading** (8 ý, luật từ khoá) | BM25 .682 < w12 .717 < rd .727; theo hạng .717 → .663 → .634 | đúng chiều ✔, biên nhỏ vì luật thô ⇒ cần mức *mục*, không mức *doc* |
| Doc có ≥3 mục theo khung chuẩn | 3,9% (Medlatec 30%, Long Châu tiêm chủng 36%, báo chí ≈0%) | mục phải suy từ heading tự do và câu |
| Query | 1.200, trung vị 81 ký tự, văn hỏi-đáp bệnh nhân | ý hỏi nhận ra được qua mẫu câu |

---

## 2. Hình dung bằng một ví dụ [MH]

**Query 229:** "Đau bụng trên rốn kèm đi ngoài nhiều lần, phân ướt cảnh báo bệnh lý gì?"
```json
{ "qid": 229,
  "entities":  [{"name": "đau bụng thượng vị", "icd": "R10", "role": "triệu chứng"},
                {"name": "tiêu chảy",          "icd": "A09", "role": "triệu chứng"}],
  "chapters":  ["XVIII", "I"],
  "intents":   ["cause", "diagnosis"],          // "cảnh báo bệnh lý gì" = hỏi nguyên nhân/bệnh nào
  "audience":  [],
  "has_test_result": false,
  "core": "đau bụng trên rốn kèm tiêu chảy là bệnh gì" }
```
**Hai doc ứng viên** (cả hai đều được M3 xếp cao):
| | Doc A "Đau bụng trên rốn là dấu hiệu bệnh gì?" (bệnh viện) | Doc B "Cách chữa tiêu chảy cho trẻ tại nhà" (nhà thuốc) |
|---|---|---|
| doc_type | disease | disease |
| entities | R10 (tiêu đề), K29, K25 | A09 (tiêu đề), E86 |
| sections | §1 định nghĩa · **§2 nguyên nhân** · §3 triệu chứng đi kèm · §4 khi nào đi khám | §1 dấu hiệu · **§2 điều trị tại nhà** · §3 dinh dưỡng |
| audience | — | **trẻ em** |
| ① đúng bệnh | ✔ R10 | ~ A09 (một nửa) |
| ② đúng ý | ✔ có mục *nguyên nhân* | ✘ chỉ có *điều trị* |
| ③ đúng đối tượng | ✔ (query không giới hạn) | ✘ trẻ em, query người lớn |
| ⇒ | **giữ, đẩy lên** | **hạ, cắt ở ngưỡng** |

Đó chính là trường hợp "cùng bệnh, khác ý" mà retriever hiện tại không phân biệt được.

---

## 3. Lược đồ dữ liệu

```
data/docstore/
  docs.parquet          1 dòng / doc         (VI 740k trước; ZH/EN sau)
  sections.parquet      1 dòng / mục         (doc_id, idx, heading, sec_type, start, end)
  dup_clusters.parquet  doc_id → cluster_id  (trùng hẳn + gần trùng)
  query_frame.parquet   1 dòng / query       (1.200)
  pair_features/*.parquet  (qid, doc_id, f1…fn) cho ~200 ứng viên/query
```
| Bảng.cột | Giá trị | Cách lấy |
|---|---|---|
| docs.doc_type | `qa` · `disease` · `drug` · `nutrition` · `procedure_service` · `vaccine` · `news_health` · `news_other` · `junk` | luật trên host + đoạn URL + tiêu đề |
| docs.entities | [(mã ICD 3 ký tự, điểm)]; rỗng = `unknown` (không phạt) | khớp dài nhất ontology ICD-10 tam ngữ (10k) + từ đời thường; trọng số tiêu đề 3 · heading 2 · 600 ký tự đầu 1 |
| docs.audience | thai phụ · sau sinh · sơ sinh · trẻ em · người già · nam · nữ | từ khoá trên tiêu đề + heading |
| sections.sec_type | định nghĩa · nguyên nhân · triệu chứng · chẩn đoán/xét nghiệm · điều trị · thuốc/liều · phòng ngừa/tiêm chủng · dinh dưỡng/chăm sóc · biến chứng/tiên lượng · đối tượng đặc biệt · **hỏi-đáp (giữ nguyên cặp)** · khác | heading (dòng ngắn, đánh số, in đậm) → từ khoá; không có heading → câu đầu đoạn |
| dup_clusters | content_key (đã có) ∪ MinHash 5-shingle, Jaccard ≥ 0,8 | để mở rộng id khi nộp (đội khác đo: +0,0013 doc R) |
| query_frame | entities, chapters, intents (đa nhãn, cùng bảng sec_type), audience, has_test_result, core | luật + LLM mở Qwen2.5-7B-Instruct (≤15B) xuất JSON |

**Nguyên tắc:** kho doc chỉ là *metadata bên cạnh* — **text gốc giữ nguyên** cho `chunk_text` (chunk = toàn văn, không giới hạn ký tự).

---

## 4. Lộ trình xây (khi user cho phép bắt đầu)

```
 Pha 1 (nhẹ, ~½ ngày)          Pha 2 (Kaggle CPU, ~1 ngày)          Pha 3 (Kaggle GPU ngắn)        Pha 4 (đo & dùng)
 ┌──────────────────┐          ┌───────────────────────────┐        ┌────────────────────┐        ┌───────────────────────┐
 │D1 doc_type (luật)│          │D2 heading + sec_type + QA │        │Q1 khung query      │        │F1 đặc trưng cặp       │
 │   url + tiêu đề  │ ───────► │D3 entities / audience     │ ─────► │   Qwen2.5-7B JSON  │ ─────► │   ~200 ứng viên/query │
 │Q0 khung query    │          │D4 MinHash cụm trùng       │        │   1.200 query      │        │kiểm chứng không nhãn  │
 │   bằng luật      │          │   (740k doc VI, 3 GB)     │        └────────────────────┘        │A/B LB qua builder     │
 └──────────────────┘          └───────────────────────────┘                                      └───────────────────────┘
```
| Pha | Đầu ra | Cổng qua pha (không nhãn) |
|---|---|---|
| 1 | docs.doc_type; query_frame v0 | Haiku đọc 30 mẫu/loại: sai ≤ 15% |
| 2 | sections, entities, audience, dup_clusters | ≥ 60% doc VI có ≥ 1 mục gán loại; entity phủ ≥ 50% doc |
| 3 | query_frame v1 (LLM) | entity phủ ≥ 80% query (hiện 23%); Haiku đọc 50 query: đúng ý ≥ 85% |
| 4 | pair_features + bài nộp A/B | đặc trưng đơn điệu theo P đã biết (BM25 < M3 < w12 < rd) **và** giảm theo hạng; rồi mới tốn lượt nộp |

**Cách dùng trong bài nộp (Pha 4):** điểm mới = điểm reranker + w₁·đúng_bệnh + w₂·đúng_ý + w₃·đúng_đối_tượng;
ngưỡng giữ doc chọn theo đường P(N) từ depth probe. Chỉ 2–3 tham số ⇒ chỉnh bằng vài lượt LB, không cần nhãn.

---

## 5. Rủi ro và cách xử lý
| Rủi ro | Xử lý |
|---|---|
| Ontology ICD dùng tên hàn lâm, query dùng từ đời thường (mới phủ 23% query) | LLM trích thực thể + chuẩn hoá về mã; bảng từ đời thường mở rộng |
| Gán ý hỏi bằng luật quá thô (biên .68 → .73) | so ý ở mức **mục** (sec_type), không mức doc; LLM cho query |
| Doc báo chí không có heading | sec_type theo câu đầu đoạn; không có ⇒ `khác` (không phạt) |
| Phạt nhầm doc đúng ("unknown") | mọi đặc trưng là **mềm** (cộng điểm), không lọc cứng — đúng lời mentor |
| Tốn GPU | chỉ 1.200 query qua LLM; doc dùng luật trên CPU |
