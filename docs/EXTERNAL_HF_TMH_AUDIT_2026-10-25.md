# Audit `tmh1008/ViBioMIR-crawled-corpus` — 25/10/2026

## 1. Nguồn, quy mô, pháp lý

- URL: <https://huggingface.co/datasets/tmh1008/ViBioMIR-crawled-corpus>
- Revision đã audit: `18cb5a544322d8d7b903ca2b6ceb3138628223b4`
  (`lastModified` 2026-10-22).
- 1.174.600 dòng; 998 file parquet ở nhánh `main` (1,58 GB); nhánh
  `refs/convert/parquet` có 1.176 file (1,91 GB), dùng cho audit này.
- Schema: `id:int64`, `url`, `domain`, `status_code:int64`,
  `success:bool`, `title`, `content`, `content_length:int64`.
- **Không có README/dataset card và không khai báo license**
  (`dataset_info.license=""`). Dataset public, không gate.
- BTC cho phép external data nhưng phải cite đầy đủ provenance. Vì không có
  license rõ ràng, chỉ nên dùng nếu thật sự cần; không dùng làm qrels hay
  nhãn, và phải ghi rõ nguồn/revision nếu đưa vào hệ thống.

## 2. Độ phủ so với docstore v2

| Chỉ số | Giá trị |
|---|---:|
| Docstore v2 docs | 1.186.913 |
| Link list BTC | 1.220.598 |
| Link còn thiếu trong v2 | 33.685 |
| HF unique IDs | 1.174.594 |
| HF ID trùng v2 | 1.140.956 |
| ID thiếu của v2 có mặt trong HF | 33.638 / 33.685 |
| ID thiếu vắng hoàn toàn | 47 (toàn bộ là PubMed EN) |
| HF ID ngoài link list | **0** |
| HF duplicate ID | 6 |
| HF `success=True` / `False` | 878.943 / 295.657 |
| Missing rows `success=True` / `False` | 15.364 / 18.274 |
| Missing rows usable sau gate | 15.175 |
| Junk bị loại | 101 (`binary`) |
| Unique content trong candidates | 14.576 |
| Trùng content với v2 | 37 |
| Trùng content trong nội bộ candidates | 562 |

HF phủ gần như toàn bộ ID còn thiếu, nhưng **không có nghĩa là lấy được
nội dung**: 18.274/33.638 missing rows là failure hoặc quá ngắn; chỉ 15.175
dòng qua gate.

## 3. Theo ngôn ngữ — điểm quyết định

| Ngôn ngữ | Missing links | Missing rows có trong HF | Usable rows | Unique content |
|---|---:|---:|---:|---:|
| VI | 8.383 | 8.383 | **334** | **128** |
| ZH | 25.197 | 25.197 | 14.841 | 14.448 |
| EN | 105 | 58 | 0 | 0 |

Sau khi loại template junk của `thaythuocvietnam.vn` (79 unique, body chỉ là
menu/header), **chỉ còn 49 tài liệu VI unique** dùng được, trong đó 2 thin.
Đây là mức tăng không đáng kể so với 732.215 tài liệu VI của v2.

Phần lớn recovery là **tiếng Trung**: 14.841 rows / 14.448 unique, chủ yếu
`www.120ask.com`. EN/ZH retrieval đã bị bỏ, nên phần này không tạo giá trị
trực tiếp cho hệ thống VI hiện tại.

## 4. Các host VI thiếu lớn — HF không cứu được

| Host | Missing trong v2 | HF usable | Ghi chú |
|---|---:|---:|---|
| `pmc-ecm-healthblog.beta.pharmacity.io` | 5.333 | **0** | toàn bộ 403, content rỗng |
| `nhathuoclongchau.com.vn` | 2.514 | **0** | 79.598 rows nhưng 79.582 fail; 16 success đã có trong v2 |
| `thaythuocvietnam.vn` | 285 | 180 rows / 79 unique | body là menu/header, không phải bài; **0 good** |
| `thanhnien.vn` | 48 | 48 rows / 16 unique | dùng được |
| `phunusuckhoe.giadinhonline.vn` | 45 | 45 / 9 | dùng được |
| `laodong.vn` | 56 | 0 | không qua gate |
| `www.vinmec.com` | 19 | 1 | dùng được |
| `medlatec.vn` | 10 | 4 | dùng được |
| `zysjonline.com` | 6.591 (ZH) | 0 | toàn bộ 403 |
| `article.iiyi.com` | 729 (ZH) | 0 | 723 status 0, 6 status 521 |
| `www.120ask.com` | 13.349 (ZH) | 10.760 | ZH, không dùng cho VI |

49 tài liệu VI tốt nhất nằm ở: `thanhnien.vn` 16,
`phunusuckhoe.giadinhonline.vn` 9, `benhvienvietduc.org` 6,
`medlatec.vn` 4, `baohaiphong.vn` 3, `baolangson.vn` 2,
`khoahocphothong.vn` 2, `suckhoecongdongonline.vn` 2,
`vietnamnet.vn` 2, `baovinhlong.com.vn` 1,
`phuyen.baodaklak.vn` 1, `www.vinmec.com` 1.

## 5. Body dài hơn cho tài liệu đã có — không nên tự động thay

Có 113.726 tài liệu đã có trong v2 mà content HF dài hơn >25% và >=500 ký tự:

- VI: 37.666
- ZH: 75.796
- EN: 264

Top VI: `suckhoecongdongonline.vn` 16.544, `thanhnien.vn` 4.791,
`baogialai.com.vn` 2.363, `phuyen.baodaklak.vn` 2.083,
`www.vietnamplus.vn` 2.022, `suckhoedoisong.vn` 1.415,
`khoahocphothong.vn` 1.103, `tuoitre.vn` 943, `thaythuocvietnam.vn` 890,
`tienphong.vn` 858, `baonghean.vn` 755, `www.vinmec.com` 385,
`medlatec.vn` 124, `hellobacsi.com` 102.

Chất lượng không đồng nhất:

- Tốt hơn: `thanhnien.vn`, `suckhoeviet.org.vn`, `suckhoedoisong.vn` — HF có
  toàn bài, v2 chỉ có đoạn mở đầu.
- Không tốt hơn: `www.vinmec.com` (trang trắc nghiệm + boilerplate),
  `hellobacsi.com` (trang category), `medlatec.vn` (lẫn teaser bài khác),
  `thaythuocvietnam.vn` (menu/header).
- Vì vậy không thể thay body hàng loạt. Muốn làm phải có cleaning theo host,
  rebuild passages/sections và rebuild index; chi phí lớn, rủi ro cao, không
  phù hợp trước mốc 31/10.

## 6. Kết luận và quyết định

1. **Không merge HF vào docstore v2.** Giá trị VI thực tế chỉ 49 tài liệu
   unique; 14.448 tài liệu ZH không dùng vì EN/ZH retrieval đã bỏ.
2. Các lỗ hổng VI lớn nhất (`pharmacity`, `longchau`, `zys`) vẫn bị chặn
   403 và content rỗng trong HF; không có cách cứu từ nguồn này.
3. HF có 0 ID ngoài link list và 6 duplicate ID — dữ liệu sạch về ID, nhưng
   không có license rõ ràng; nếu dùng phải cite URL + revision, và chỉ dùng
   như corpus, tuyệt đối không dùng làm qrels.
4. Giữ `data/docstore/v2` làm bản cuối. Nếu sau này cần tăng độ phủ, ưu tiên
   crawl/recrawl có kiểm soát hoặc nguồn khác có license; không dùng HF làm
   nguồn chính.
5. Body refresh từ HF chỉ là hướng nghiên cứu sau deadline, không đưa vào
   pipeline 31/10.

## 7. Tái lập

```bash
python scripts/data/audit_external_corpus.py \
  --corpus C:/Users/NOS/AppData/Local/Temp/opencode/hf_tmh_conv/default/train \
  --store data/docstore/v2 \
  --out data/external/hf_tmh
```

Output:

- `data/external/hf_tmh/audit.json`
- `data/external/hf_tmh/host_coverage.csv`
- `data/external/hf_tmh/candidates.parquet`
- `data/external/hf_tmh/candidates.jsonl.gz`
