# Thư nộp thuyết minh sản phẩm (bản nháp, hạn 23:59 04/11/2026 ICT)

**Gửi:** hr@aiguru.com.vn
**Tiêu đề:** [R2AI 2026 – Stage 3] Thuyết minh sản phẩm – đội Băng Cá Khô (ViBioMIR)

---

Kính gửi Ban Tổ chức,

Đội **Băng Cá Khô** (tài khoản Codabench `ca_kho`, cuộc thi ViBioMIR) xin gửi bộ tài liệu thuyết minh sản phẩm theo hướng dẫn của BTC.

**Bài nộp được báo cáo:** private **#7892** (`pN3_base_pqchunks.zip`), đã gắn lên leaderboard private. Đây là cùng file với bài public #7589 (FINAL 0,3484).

**Đường link** (đã mở quyền xem cho người có link):
- Mã nguồn: file `vibiomir_code_<commit>.zip` trong cùng thư mục Drive dưới đây (`git archive` của commit nộp)
- Dữ liệu, chỉ mục, đầu ra trung gian và `MANIFEST.tsv` (sha256 từng file): https://drive.google.com/drive/folders/1_VUdqZ4eTv9w_hHRPDWsLLNNmqxyYDaj?usp=sharing

Bản PDF để đọc nhanh (có sơ đồ luồng xử lý): `THUYET_MINH_SAN_PHAM.pdf` và `README.pdf` ở thư mục gốc trên Drive.

**Nội dung theo mục hướng dẫn:**

| Mục | Ở đâu |
|---|---|
| 1.1 Tổng quan hệ thống (kiến trúc, sơ đồ luồng, chỉ mục, truy hồi, xếp hạng lại, chọn đầu ra) | `docs/THUYET_MINH_SAN_PHAM.md` §1 |
| 1.2 Dữ liệu (nguồn, thu thập, làm sạch, phân đoạn, cách truy cập) | `docs/THUYET_MINH_SAN_PHAM.md` §2, `docs/DATA_CARD.md` |
| 1.3 Mô hình (vai trò, revision checkpoint, link tải chính thức, cấu hình, prompt) | `docs/THUYET_MINH_SAN_PHAM.md` §3 |
| 1.4 Mã nguồn (toàn bộ mã, thư viện, cấu hình) | repo; `requirements.txt`, `requirements-gpu.txt`; `docs/THUYET_MINH_SAN_PHAM.md` §4 |
| 1.5 Hướng dẫn (cài đặt, tải dữ liệu/checkpoint, chỉ mục, lệnh chạy) | `README.md` |

**Tóm tắt hệ thống:**
- Pipeline: truy hồi đa ngữ BM25 + BGE-M3 → pool 2.500 ứng viên/câu hỏi → bge-reranker-v2-m3 và BGE-M3 → mô hình xác suất liên quan → số tài liệu chọn theo từng câu hỏi để cực đại F2 kỳ vọng → chunk là cửa sổ nguyên văn ≤ 4.500 ký tự.
- Mô hình: chỉ dùng mô hình mở ≤ 15B, gồm BGE-M3, bge-reranker-v2-m3, Qwen3.5-9B và Qwen3.5-4B. Không dùng API đóng, không dùng MedKB.
- Tái lập:
  - Từ dữ liệu chia sẻ, danh sách tài liệu và chunk của cả 5 bài private dựng lại khớp 100% trên CPU trong vài phút (README, mức A).
  - Các bước GPU chạy lại được bằng các kernel Kaggle có trong repo.

Các công bố về tuân thủ (MedKB, số liệu thống kê tổng hợp, dùng leaderboard) nằm ở `docs/THUYET_MINH_SAN_PHAM.md` §5.

Trân trọng,
Đội Băng Cá Khô
