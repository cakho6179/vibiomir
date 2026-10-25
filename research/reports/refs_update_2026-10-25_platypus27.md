# Review cập nhật repo Platypus27-coder_VietMedBridge (2026-10-25)

Phạm vi: 10 commit mới từ `5ae79c98f65195837ab1b799ef4591ea27a5689f` đến HEAD. Diff: 45 file, +8274/-56. Chỉ đọc code, không chạy code, không cài package, không tải dữ liệu.

## 1. Các commit mới làm gì

Đây là hệ thống hàng đợi embedding trên Kaggle và đóng gói dữ liệu, không phải pipeline thu thập dữ liệu mới.

- `4783d4f` đến `22f3309`: thêm script kiểm tiến độ read-only (`scripts/check_maximize_progress_cell.py`), chốt text raw đã phục hồi (`scripts/finalize_ready_text_cell.py`), và cơ chế journal cho SQL checkpoint (`docs/MAXIMIZE_COLLECTED_DATA.md`).
- `d1d4e49`, `372e8da`: xuất full/recovery delta có binding, có resume.
- `9966fac`, `ede229b`: notebook embedding full-data cho 2 tài khoản Kaggle (`notebooks/kaggle/04_full_data_account_{0,1}.ipynb`, `scripts/full_queue_embeddings.py`).
- `639b412`, `f759b59`: tải output Kaggle qua paged signed URL, có resume từ file cache (`scripts/kaggle_output_paged.py`, `scripts/kaggle_output_transfer.py`, `scripts/kaggle_output_guard.py`).
- `9d4724a`, `2b9b70c`: mở rộng lên 3 tài khoản Kaggle, mỗi tài khoản 2 GPU T4, dùng cùng thư viện text đã đóng băng (`notebooks/kaggle/04_full_data_3acc_account_{0,1,2}.ipynb`, `scripts/write_full_queue_notebooks.py`).

Kiến trúc chính: thư viện input text bất biến (`input_library.json`, `full_package.json`) tách khỏi plan GPU (`plan v2`). Mỗi lượt chỉ đổi plan, không upload lại text. Mỗi task chỉ giao cho một GPU worker theo hash, nên text trùng được dùng lại vector.

## 2. Nguồn dữ liệu mới

Không thấy URL corpus hay dataset HF/Kaggle mới chứa text hoặc nhãn bên ngoài. Các nguồn đã xuất hiện trong prev commit:
- Google Drive: `/content/drive/MyDrive/VietMedBridge` và shortcut `1uKKHYEcMPodnJIscKPi6jsoNBJvyhFGK`. Có sẵn từ trước.
- Kho archive TAR raw của BTC (AI Guru, 10 nguồn intake, 13 nguồn đã chốt theo doc). Có sẵn từ trước.
- Dataset Kaggle riêng tư do repo tự tạo, không phải dữ liệu công khai: `vmb-full-inputs-*`, `vmb-full-plan-*`, `vmb-new-*`, `vmb-api-json-*`, `vmb-output-pages-*`.
- Các URL còn lại: `kaggle.com/docs/notebooks` (tham khảo giới hạn 20 GB output), `googleapis.com/drive/v3/files` (Drive API), và GitHub repo của chính họ. Một comment trong `kaggle_output_transfer.py` nhắc tới "public Hub resolver" (Hugging Face), nhưng tôi không xác minh được đây có tải file HF thật hay không.

## 3. Model

- `BAAI/bge-m3`: dense, chunk policy BGE 180/40, parent 512, tokenizer pin cố định (`docs/MAXIMIZE_COLLECTED_DATA.md`).
- `Qwen/Qwen3-Embedding-8B`: encoder thứ hai (`scripts/qwen_models.py`, `QwenEncoder`).
- Có cache vector `BGE COMPLETE` được kế thừa, và namespace vector theo model.

## 4. Điểm số

Không có điểm retrieval hay public LB nào trong diff và docs mới. Doc tự nói rõ: "không hứa tăng coverage hoặc điểm trước khi chạy thực tế và đánh giá retrieval". Các con số chỉ là số lượng: 798.948 ID có nội dung giữ lại; 2.304 ID mới có text; 97.758 ID BTC có raw metadata; 802.028 ID BTC trong phạm vi full; 96.982 ID mục tiêu cứu raw.

## 5. Đáng học cho dự án của chúng ta

Chúng ta đã chạy multi-account Kaggle kernel và BM25 section, bge-m3 dense. Các điểm sau có thể tái sử dụng:
- Tách thư viện input bất biến khỏi plan GPU. Upload text một lần, đối chiếu `job.json`, và đổi plan mà không upload lại (`scripts/full_queue_embeddings.py`).
- Giao task theo hash để tránh encode trùng giữa các GPU worker, và ưu tiên cùng worker cho các task có tập key giống nhau.
- Ngân sách output theo tài khoản: 18 GB trên giới hạn 20 GB của `/kaggle/working`. Plan có trần `ELASTIC_OUTPUT_BYTES`.
- Tải output bằng paged signed URL, có resume từ cache, thay vì một API call cho mỗi file (`scripts/kaggle_output_paged.py`).
- Checkpoint SQL: một snapshot đã kiểm SHA cộng tối đa 64 journal đã seal. Tiện cho nhánh raw dài.
- Giới hạn gzip/zstd giải nén 64 MiB, file lớn đưa vào review.
- Chunk policy BGE 180/40 với parent 512 là một điểm để so với chunk BM25 section của chúng ta.

Kết luận: không có nguồn dữ liệu mới hoặc điểm số đáng tin để dùng trực tiếp. Phần đáng học là hạ tầng queue và resume trên Kaggle đa tài khoản.
