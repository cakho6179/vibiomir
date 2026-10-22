# Lộ trình hệ thống ViBioMIR → Top 1 (lập 2026-10-22, thay thế PLAN_TARGET_0.6.md; cập nhật trưa 22/10 sau nghiên cứu đội khác)

Nhãn: **[ĐO]** số đo thật · **[LUẬT]** trang luật/nộp bài chính thức · **[SUY]** suy luận, cần kiểm.

## 0. Mục tiêu và ràng buộc cứng
- **Mục tiêu (user 22/10): vượt cả baseline BTC 0.3764 lẫn Top 1 — đích FINAL ≥ 0.45** (vd. docs F2 ≥ 0.47 + chunks F2 ≥ 0.43; baseline mạnh ở docs 0.5587 nhưng yếu ở chunks 0.1941, ta có lợi thế chunk toàn văn ⇒ chunk F2 ≈ docs F2).
- Vượt Top 1 `toptire` **0.4071** (docs P/R 0.232/0.571 ⇒ họ nộp ≈ 113 doc/query [SUY]). BM25 của BTC: FINAL 0.3764, docs F2 0.5587 (P 0.88 / R 0.515), chunks F2 0.1941.
- **Model [LUẬT]:** chỉ model công khai trọng số, phát hành trước 01/08/2026, ≤ 15B. **Cấm LLM đóng** (GPT-4o, Gemini…)
  ⇒ Jev (TypeSafe, model đóng qua API) **không được nằm trong hệ thống nộp**; dùng đúng *mẫu thiết kế* của Jev
  (câu hỏi nguyên tử Noul/Score + ngưỡng tin cậy + ghép bằng code) nhưng chạy bằng model mở (mục 3, L3).
- **Định dạng [LUẬT]:** `relevant_docs` và `relevant_chunks` là **hai danh sách tách rời**. Số doc mỗi query tùy ý
  (nhiều hay ít hơn 28); doc id gần như không tốn dung lượng. Giới hạn upload ~100 MB [ĐO: 413 ở 159 MB] chỉ ràng buộc
  **tổng chữ trong chunk_text**. chunk_text phải trích nguyên văn; quy ước của ta: chunk = toàn văn doc, không cắt.
- 1.200 query dùng cho cả hai vòng; **public chấm 600 query** [ĐO: scoring_result], private chấm 600 còn lại, chỉ 5 lượt.
- **|G| trung bình 71 doc / 75 chunk mỗi query** [ĐO: details.jsonl của scorer, `docs/RESEARCH_REFS.md` mục 3]. Public đóng 31/10, nộp hệ thống 04/11,
  kết quả 11/11, phải có working-notes paper.
- Mọi việc nặng chạy Kaggle (skill `kaggle-multi-account`); local chỉ điều phối, dry-run, ghép file nhỏ.

## 1. Toán của metric — quyết định số doc/chunk mỗi query
Với F2 (β = 2): `F2 = 5·TP / (4·|G| + N)` (TP = số đúng trong N mục nộp, |G| = số gold).
Thêm một mục có xác suất đúng p làm E[F2] tăng **khi và chỉ khi p > F2_hiện_tại / 5** (Ye et al. 2012, "Optimizing
F-measures"; Lipton et al. 2014: ngưỡng tối ưu F_β = F*/(1+β²)).
- Với F2 ≈ 0.3–0.45 ⇒ ngưỡng **p ≈ 0.06–0.09**: nên nộp *mọi* doc có xác suất gold ≥ ~8%, tức danh sách **sâu**
  (50–150 doc), không phải 28 cố định. Đúng với Top 1 (N≈113, P 0.23).
- Ví dụ từ số đo: hệ tốt nhất ở N=28 có P 0.354 ⇒ TP ≈ 9.9 (|G| thật trung bình 71) ⇒ docs F2 ≈ 0.22. Nếu P(100) ≈ 0.20 thì
  TP ≈ 20 ⇒ docs F2 ≈ 0.34; P(150) ≈ 0.17 ⇒ ≈ 0.37 [SUY: P(N) phải đo].
- Cấp chunk: chunk toàn văn của doc gold khớp mọi chunk gold của doc đó (LCS ≥ 40%) ⇒ chunk P ≈ doc P [ĐO: 0.351 vs
  0.354]. Độ sâu chunk bị giới hạn bởi byte ⇒ chunk list = K doc có p cao nhất vừa ngân sách. [ĐO của đội khác] LZMA dict
  256 MiB được server chấm; K ≈ 117–133 chunk toàn văn/query vừa ≤ 102,6 MB; chunk F2 tăng đều theo K (19 → 150: 0.128 → 0.26).
- ⇒ Hệ thống cần **xác suất đã hiệu chỉnh** p(gold | q, d), không chỉ thứ hạng. Đây là vai trò của "decision model".

## 2. Hiểu biết về dữ liệu và cách BTC tổ chức (nghiên cứu 21–22/10)
| Nguồn | Phát hiện |
|---|---|
| Slide mentor (12/10) | Pipeline BTC: URL → thu thập → metadata + văn bản → làm sạch/chuẩn hoá → chunking → index. Lấy **vùng bài** (không lấy menu, liên quan); **chunk theo mục**, giữ tiêu đề mục với đoạn dưới nó; giữ cặp **hỏi–đáp** chung chunk; khử trùng nhưng giữ ánh xạ mọi id; metadata (tiêu đề, chuyên mục, ngôn ngữ, ngày, loại tài liệu); nhóm **ICD-10**; baseline = BM25/TF-IDF, BGE-M3, RRF, bge-reranker-v2-m3; ví dụ tách từ "triệu_chứng sỏi_thận". |
| HF dataset card | Corpus 1,220,598 link = 740,598 vi + 240,000 en (PubMed) + 240,000 zh; trích dẫn "MedKB" (medkb.tinix.ai = KB của BTC). |
| Probe MedKB đã lưu (chỉ thống kê) | Chunk trung vị ~3.3k ký tự, max ~4.8k; 97.5% `news`, có `book` (bach_khoa_y_hoc_2010), 1 pubmed; nguồn: Long Châu, SKĐS, Medlatec, HelloBacsi, Thanh Niên, Vinmec…; mỗi chunk có `similar_queries` (kiểu doc2query) và `section_types`; 88% URL kết quả nằm trong corpus. |
| Phân bố host top-28 của ta | Gần giống MedKB (Long Châu 23%, SKĐS 13%, Medlatec 11%, Vinmec 8%) ⇒ không phải lệch host [ĐO]. |
| Query | Câu hỏi bệnh nhân, trung vị 18 từ, P95 84, dài tới 283 từ; nhiều ý; **không** chép nguyên văn từ corpus (3/40 mẫu) [ĐO]. |
| Nhiễu còn sót trong text v3 | Medlatec: danh sách "bài liên quan" có ngày ở đầu bài; Long Châu: tiểu sử dược sĩ đầu bài + chữ ký cuối; YouMed: link tham khảo; Vinmec hỏi–đáp: câu chào/cảm ơn mẫu; suckhoecongdong: dòng `\| \|` tin liên quan, widget chia sẻ ⇒ `kaggle/common/docorg.py: clean_v4` (đã kiểm trên 30 doc mẫu). |
| Đội khác — nhật ký LB (22/10) | TranVu2005 (0.2655): đòn bẩy chính là **độ sâu** doc ~115–150 và chunk toàn văn ~100–130 (LZMA 256 MiB), chunk chỉ từ doc đã chọn, id trùng nội dung +, zh theo ngưỡng reranker +0.0056, rerank sâu 150–300 giảm. quyen244: augment/tách ý query bằng LLM giảm. Chi tiết `docs/RESEARCH_REFS.md`. |
| Phân nhóm hệ cơ quan (ICD-10) | Trùng chương query–doc đi cùng chiều precision (BM25 79.7% → M3 88.8% → reranker 90.0%) nhưng chỉ giải thích ~1/3 khoảng cách ⇒ đặc trưng mềm. Khung mục chuẩn chỉ ở 3.9% doc (Medlatec/Vinmec/Long Châu tiêm chủng) ⇒ đặc trưng khía cạnh, không tái tổ chức corpus [ĐO]. |
| Đội khác (GitHub, 21/10) | VietMedBridge: chunk con 180 token / cha 512, parser theo heading, Qwen3-Reranker-8B, HyDE/PICO; quyen244: chunk 800–1000 ký tự, BGE-M3 + bge-reranker, quota theo ngôn ngữ. Không ai công bố điểm cao. |
| Model VN mở | AITeamVN/Vietnamese_Embedding (BGE-M3 fine-tune, +16 điểm Acc@1 trên Zalo legal), AITeamVN/Vietnamese_Reranker (từ bge-reranker-v2-m3) — cần kiểm ngày phát hành < 01/08/2026. |

Câu hỏi mở lớn nhất: **vì sao BM25 của BTC đạt P 0.88 còn BM25 của ta 0.18?** Giả thuyết hàng đầu [SUY]: gold được
lấy từ pool các hệ của BTC (BM25 + BGE-M3 trên chunk MedKB) rồi lọc người/LLM ⇒ doc ngoài pool không bao giờ là gold
⇒ phải **tái tạo cách tổ chức dữ liệu + tokenizer của BTC**. Các biến thể đang chờ LB phân xử (mục 5, ngày 1).

## 2b. Tổ chức data theo slide BTC (checklist: `research/reports/btc_slides_data_org.md`)
Trục chính: **bệnh/nhóm ICD × ý hỏi (loại mục)** — slide tr.23/40: chunk "Cùng bệnh, khác ý" không đúng; câu hỏi triệu chứng phải ra mục Triệu chứng.
| Slide | Ta đã có | Cần làm |
|---|---|---|
| tr.12–14, 19 vùng bài, bỏ giao diện | processed_v3 + clean_v4 (phía index) | giữ |
| tr.17–18 metadata (tóm tắt, chuyên mục, loại tài liệu, nhóm chủ đề, ngày) | id, host, lang, title | thêm loại tài liệu (tr.11: hỏi đáp / kiến thức / sách-cẩm nang / thuốc-dược liệu / tin tức), nhóm ICD, chuyên mục nếu có |
| tr.20 giữ dấu, phủ định, số–đơn vị; lưu gốc | ✓ | — |
| tr.21 tách từ cho lexical (triệu_chứng sỏi_thận) | e3v2, b2 (pyvi) chờ LB | chọn theo LB |
| tr.14, 22–23 chunk theo mục, giữ heading | s1/s2 (BM25) chờ LB | gắn **loại mục** cho từng chunk; dense trên chunk theo mục nếu LB ủng hộ |
| tr.24 giữ cặp hỏi–đáp chung đoạn | chưa riêng | luật cho trang hỏi–đáp |
| tr.25 khử trùng giữ ánh xạ mọi id | content_key/dup_of (trùng y hệt) | mở rộng id trùng khi nộp; MinHash gần trùng |
| tr.28–31 ICD đa nhãn, "chưa xác định", không lọc chặt | script phân tích | đưa vào kho doc + query, làm đặc trưng |
| tr.40–41 chọn đoạn theo từng ý | — | phân loại ý hỏi của query (đa ý) ↔ loại mục; câu hỏi nguyên tử "đúng bệnh?", "đúng ý?" ở L2 |

## 3. Kiến trúc hệ thống đích
```
L0 DỮ LIỆU (Kaggle CPU)                     giữ text gốc cho chunk_text; mọi bước dưới chỉ phục vụ index
   corpus v3 → clean_v4 → đơn vị index: doc | section chunk (≤4500, theo mục, tiêu đề + heading) | passage
   metadata: host, loại (hỏi-đáp/bài kiến thức/thuốc/tin), tiêu đề, heading, chương ICD-10 (`scripts/r2ai2026_icd_systems.py`),
   khía cạnh mục (nguyên nhân/triệu chứng/điều trị…), cụm trùng nội dung (`content_key`)
L1 TRUY HỒI RỘNG (recall@200–300)
   BM25 bản sao BTC (biến thể thắng ở LB) · BGE-M3 dense (cách chunk thắng ở LB) · [M3 sparse/colbert]
   query: gốc · bản làm sạch (bỏ lời chào/kể bệnh) · LLM viết lại ngắn + tách ý (Qwen2.5-7B, đã có kernel dịch)
   → hợp ứng viên (RRF) 200–300 doc/query
L2 ĐẶC TRƯNG CHO TỪNG (q, d)
   hạng/điểm từng retriever · bge-reranker-v2-m3 max-passage toàn văn [có r1] · Qwen3-Reranker-8B P(yes)
   chương ICD trùng/khác · khía cạnh query ↔ loại mục của doc · host · độ dài · số retriever đồng thuận
   câu hỏi nguyên tử kiểu Jev bằng LLM mở (logprob yes/no):
     a) `doc` bàn về đúng bệnh/vấn đề trong `query`?   b) `doc` trả lời đúng ý hỏi (nguyên nhân/điều trị/…)?
     c) đúng đối tượng (thai phụ, trẻ em, tuổi)?        d) `doc` là bài y khoa (không phải tin hành chính)?
L3 QUYẾT ĐỊNH (thay cho Jev, model mở)
   p(gold) = hiệu chỉnh(đặc trưng)  — logistic đơn điệu, ít tham số
   docs: lấy mọi d có p ≥ ngưỡng ≈ F2*/5 (tối ưu E[F2] theo từng query, |G| ước từ Σp)
   chunks: K doc p cao nhất (chỉ trong relevant_docs) sao cho zip LZMA-256MiB ≤ ~101 MB (K ≈ 100–130) — chunk = toàn văn
   mở rộng: thêm id trùng nội dung của doc đã chọn vào relevant_docs
   cổng tin cậy: query có p phẳng (không chắc) → dùng danh sách mặc định theo độ sâu hiệu chỉnh
L4 ĐÓNG GÓI (Kaggle CPU builder) → kiểm định dạng + dung lượng → user nộp tay
```
**Hiệu chỉnh khi không có nhãn:** (i) nhãn bạc từ LLM mở trên mẫu (q, d); (ii) số P/R tổng hợp từ LB ở nhiều độ sâu
("depth probe" chỉ kéo dài `relevant_docs`, giữ nguyên chunk ⇒ không tốn dung lượng); khớp hàm hạng→precision cho từng
thành phần, rồi chỉ tinh chỉnh 2–3 tham số bằng LB.

## 4. Kế hoạch tính toán Kaggle
| Việc | Loại | Ước lượng [dựa trên số đo cũ] |
|---|---|---|
| BM25 biến thể (clean/sections/tokenizer) | CPU, 1 kernel/biến thể | 15–25 phút (b1 1395 s, b5 876 s) |
| BGE-M3 trên section chunks VI | GPU 2×T4, 3 kernel | ~2–2.5 h/kernel (m3L 8.1 ks) |
| bge-reranker top-200 hợp nhất | GPU, 3–4 kernel chia query | r1 (178 doc/q) đã chạy được |
| Qwen3-Reranker-8B / LLM judge top-60 | GPU, 5 tài khoản × 2 | ~72k cặp; ước 3–6 h tổng — đo trên 20 query trước |
| Builder zip | CPU | ~20 phút (build-v2: 22 phút) |
Quota GPU ~30 h/tuần/tài khoản [TÀI LIỆU]: ưu tiên GPU cho L2 (reranker/judge), dense chỉ chạy lại khi LB chứng minh cách chunk mới tốt hơn.

## 5. Lịch (10 lượt/ngày; mỗi lượt trả lời đúng 1 câu hỏi)
| Ngày | Nộp (câu hỏi) | Song song trên Kaggle |
|---|---|---|
| **22/10** | Thành phần @28 (`bm25-b1/b2/b4/b5`, `b3_VI`, `e3v2_VI`, `m3c_VI`, `m3L_VI`, `rerank_r1_VI`, `c1/s1/s2`) **+ depth probe** w12: d100/150/200 c28, d150 c60/90/120 (đội khác đã chứng minh LZMA 256 MiB được chấm) | `vibiomir-build-depth-a/b` đang chạy |
| 23/10 | Depth probe còn lại + bản sâu tốt nhất với thành phần thắng; mở rộng id trùng | LLM judge thử 20 query; Qwen3-Reranker; đặc trưng ICD/khía cạnh |
| 24–25/10 | Fusion các thành phần thắng; decision v1 (ngưỡng theo p hiệu chỉnh, docs biến thiên/query); chunk depth với LZMA | query rewrite/tách ý; M3 trên section chunks nếu LB ủng hộ |
| 26–28/10 | decision v2 (đặc trưng L2 đầy đủ), tinh chỉnh ngưỡng/ngân sách chunk | chạy toàn bộ 1.200 query |
| 29–30/10 | Chốt 2–3 ứng viên, kiểm độ ổn định | đóng gói pipeline tái lập |
| 01–04/11 | Private: 5 lượt cho các cấu hình đã chốt; nộp hệ thống + working notes | — |

## 6. Rủi ro
- LZMA (method 14, dict 256 MiB): đội khác đã nộp nhiều lần được chấm [ĐO của họ]; `b3_VI_n28` xác nhận lại cho file của ta.
- Trần upload thực tế ~102,6 MB (đội khác); giữ ≤ 101 MB cho an toàn.
- `m3L_VI_n28.zip` 96.6 MB có thể dính 413.
- Hiệu chỉnh từ số tổng hợp LB dễ quá khớp public ⇒ giữ ít tham số, ưu tiên đặc trưng đơn điệu.
- Jev/LLM đóng: không dùng trong hệ thống nộp (vi phạm luật model).

## 7. Ghi sổ
Số đo → `EXPERIMENTS.md`; việc làm/quyết định → `WORKLOG.md`; mọi kernel ghi `stats.json`.
