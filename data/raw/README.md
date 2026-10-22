# data/raw — dữ liệu gốc, CHỈ ĐỌC

Mọi thứ khác trong `data/` (docs_clean.jsonl, processed_v3/, docstore/ …) là **dẫn xuất** và phải dựng lại được từ đây
bằng script trong git. Quy tắc:
1. Không sửa, không ghi đè, không xoá file trong `raw/` (file đặt thuộc tính read-only).
2. Dữ liệu mới (crawl thêm, khôi phục từ Common Crawl/Wayback/PubMed) **thêm thành file mới** trong `recovered/`,
   tên `<nguồn>_<YYYY-MM-DD>.jsonl`, rồi ghi một dòng vào `MANIFEST.md`.
3. Script chỉ đọc đường dẫn qua `r2ai2026/paths.py` (`RAW`, `QUERIES`, `LINKS`, `DOCS_RAW`, `CRAWL_DB`, `RECOVERED`).

| Thư mục | Nội dung | Schema |
|---|---|---|
| `btc/` | file BTC phát hành (HF dataset, tải 18/10) | links: (id, url) 1.220.598 dòng; query: (id, query) 1.200 dòng |
| `crawl/` | đầu ra crawler của ta, **đóng băng 20/10 23:53** | docs.jsonl: {id, url, title, text, …} theo crawler; crawl.sqlite: trạng thái từng id |
| `recovered/` | doc khôi phục sau 20/10, mỗi nguồn một file | cc_kernels/pubmed: {id, url, t, x, …}; direct_vi: schema recrawler {id, url, final_url, host, kind, status, ok, title, text, lang, …} |
