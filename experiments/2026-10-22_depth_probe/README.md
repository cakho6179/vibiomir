# 2026-10-22_depth_probe — Nộp sâu hơn 28 doc được bao nhiêu? Cắt theo ngưỡng reranker có lợi không?

- **Trạng thái:** đang chạy (đã có 2/9 điểm)
- **Câu hỏi:** P và R thay đổi thế nào khi tăng số doc (28 → 88–200) và số chunk toàn văn (28 → 60–90)? Đưa reranker vào thứ tự hay chỉ dùng nó để cắt?
- **Giả thuyết:** |G| ≈ 71 doc/query ⇒ 28 doc quá nông; F2 thưởng recall ⇒ đi sâu tăng điểm mạnh (đội 0,2655 đi 115–150 doc).
- **Quyết định phụ thuộc:** độ sâu + ngưỡng τ cho bài chính; reranker vào thứ tự hay chỉ cắt.

## Cách làm
- Builder: `kaggle/build_exp/build_submission.py`, kernel `tthiutoko/vibiomir-build-depth-{a,b,c,d}` (acc2), dataset `vibiomir-rankings`.
- w12 = RRF(BM25 E1v2, BGE-M3 gộp 5 shard) trọng số 1:2; rd = RRF(E1v2, M3, reranker r1) 1:2:3.
- Cắt T2/T1: giữ doc có logit bge-reranker-v2-m3 ≥ −2 / ≥ −1 (tối thiểu 20 doc đầu), trần 150 doc, 90 chunk; "r" = xếp lại thứ tự query cho LZMA.

## Kết quả [ĐO] (LB public, user nộp tay)
| Submission | Bài | doc/q | chunk/q | MB | FINAL | Docs P / R | Chunks P / R |
|---|---|---|---|---|---|---|---|
| (tham chiếu 20/10) | w12 n28 | 28 | 28 | 33.7 | 0.1997 | 0.3544 / 0.2029 | 0.3511 / 0.2101 |
| #6536 22/10 16:53 | w12T2_VI_d150c90r | 88.2 | 75.4 | 71.3 | **0.3108** | 0.2495 / 0.3823 | 0.2635 / 0.3698 |
| #6537 22/10 16:54 | rdT2_VI_d150c90r | 103.4 | 77.9 | 71.5 | 0.3054 | 0.2225 / 0.3900 | 0.2586 / 0.3722 |
| chờ | rd_VI_d150c90r, w12_VI_d150c90, w12_VI_d150c28, w12_VI_d200c28, w12_VI_d150c60, w12_VI_d100c28, rdT1_VI_d150c90r | | | | | | |

## Kết luận (tạm)
1. Đi sâu là đòn bẩy lớn nhất: 0.1997 → 0.3108 (+0.111) chỉ nhờ 28 → 88 doc / 75 chunk.
2. Từ hạng 29 đến 88 ước ~12 doc gold/query thêm ⇒ precision dải đó ≈ 0.2 — vẫn > F2/5 ≈ 0.06 ⇒ còn nên đi sâu hơn.
3. Đưa reranker vào thứ tự (rdT2) **không** hơn giữ thứ tự w12 (−0.0054) dù thêm 15 doc; cần rd_VI_d150c90r và w12_VI_d150c90 để tách hiệu ứng.

## Phân tích #6536 vs #6537 [ĐO offline + SUY] — `diff_6536_6537.py`
| | #6536 w12T2 | #6537 rdT2 |
|---|---|---|
| doc/query | 88.2 | 103.4 |
| chung | 86.8 | 86.8 |
| chỉ có ở bài này | 1.4 (logit reranker trung vị −3.06) | **16.5** (logit trung vị −0.28; hạng w12 trung vị 215, 99% nằm ngoài top-150 w12) |
| chunk/query (doc chung) | 75.4 (68.7) | 77.9 (68.7) |
Hai bài gần như là **cùng một danh sách + 16,5 doc** reranker thích nhưng retriever xếp hạng 161–263.
Ước lượng (giả định docs F2 và chunk F2 giảm đều, hiệu chỉnh macro bằng tỉ lệ đo ở #6536): #6536 trúng ≈ 22,0 doc gold/query;
#6537 ≈ 22,7 ⇒ 16,5 doc thêm chỉ mang ≈ 0,7 gold ⇒ **precision ≈ 4%**, dưới ngưỡng có lợi F2/5 ≈ 6% ⇒ điểm giảm.
Đường precision theo hạng (ước): hạng 1–28 ≈ 0.35 · hạng 29–88 (đã cắt) ≈ 0.20 · hạng 161–263 (logit ≥ −2) ≈ 0.04.
⇒ (a) **hạng của retriever** là tín hiệu mạnh; reranker một mình không cứu được doc retriever xếp thấp;
(b) gold dồn ở top retriever ⇒ đi sâu hơn trên *cùng* retriever có lợi giảm nhanh — gold còn thiếu (R chỉ 0.38) phần lớn
**không nằm trong top ~250 của w12** ⇒ cần nguồn ứng viên khác (BM25 sâu/khác, truy vấn khác), không chỉ đào sâu.
**Kiểm chứng bằng LB (#6537 đủ dòng):** Docs R +0.0077 ≈ +0,55 gold/q cho 16,5 doc thêm ⇒ precision ≈ 3,3% — xác nhận.
Số đã ghi `EXPERIMENTS.md` bảng A.
