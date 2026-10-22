# Thí nghiệm — quy ước

Mỗi thí nghiệm trả lời **một câu hỏi**. Mỗi thí nghiệm = một thư mục `experiments/<YYYY-MM-DD>_<slug>/`; tên thư mục là ID.
Danh sách mọi thí nghiệm: `REGISTRY.md`. Số đo gốc vẫn ghi `EXPERIMENTS.md` (cột/ghi chú dẫn về ID).

## Trong mỗi thư mục
| File | Bắt buộc | Nội dung |
|---|---|---|
| `README.md` | ✔ | theo `_TEMPLATE.md`: câu hỏi · giả thuyết · cách làm · kết quả · kết luận/quyết định |
| mã phân tích local | nếu có | script nhỏ, chạy từ thư mục gốc `vibiomir/`, đọc dữ liệu qua `r2ai2026/paths.py` |
| `jobs.json` / tham số kernel | nếu chạy Kaggle | cấu hình đã đẩy (vd. jobs của `build_submission.py`), ref kernel `user/slug` |
| kết quả nhỏ (< 1 MB) | nếu có | bảng `.md`/`.tsv`/`.json`; kết quả lớn để ở `kaggle/outputs/` hoặc `runs/`, ghi đường dẫn |

## Quy tắc
1. Mã nguồn chung (kernel, thư viện) **không** chép vào thư mục thí nghiệm — ghi đường dẫn + commit git.
2. Nhãn số liệu: **[ĐO]** đo thật · **[SUY]** suy luận · **[MH]** minh hoạ.
3. Thí nghiệm tốn lượt nộp LB phải ghi trước: câu hỏi nó trả lời và quyết định sẽ đổi thế nào theo kết quả.
4. Trạng thái trong REGISTRY: `kế hoạch` → `đang chạy` → `xong` / `bỏ`.
5. Không dùng nhãn gold từng cặp (kể cả file chấm) — chỉ số tổng hợp từ LB.
