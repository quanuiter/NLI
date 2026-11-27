# Dữ liệu nộp kèm đồ án NLI tiếng Việt

Gói `data_NLI.zip` (tạo bằng `python scripts/package_data.py`) chứa đúng dữ liệu nhóm dùng trong báo cáo.
Dữ liệu tạo lại được từ nguồn gốc bằng `python scripts/prepare_data.py` (tập con XNLI 100.000 cặp chọn phân tầng theo nhãn, seed 42).

## processed/ – dữ liệu đã chuẩn hóa (JSON Lines, mỗi dòng: id, premise, hypothesis, label)

| File | Số cặp | Nguồn | Dùng ở |
|---|---|---|---|
| `xnli_train_100k.jsonl` | 100.000 | XNLI tiếng Việt, train (dịch máy) – tập con phân tầng | S1, S3 (bước 1) |
| `xnli_dev.jsonl` | 2.490 | XNLI tiếng Việt, dev | chọn siêu tham số S1 |
| `xnli_test.jsonl` | 5.010 | XNLI tiếng Việt, test | kiểm tra S1 |
| `vianli_train.jsonl` | 8.012 | ViANLI train | S2, S3 (bước 2) |
| `vianli_dev.jsonl` | 1.000 | ViANLI dev | chọn siêu tham số S2, S3 |
| `vianli_test.jsonl` | 1.000 | ViANLI test | kiểm tra |
| `vinli_test.jsonl` | 2.264 | ViNLI test, đã bỏ nhãn `other` | chỉ kiểm tra (ngoài miền) |
| `*_seg.jsonl` (nếu có) | như trên | bản đã tách từ bằng VnCoreNLP | chỉ PhoBERT (M4) |

Chuẩn hóa: Unicode NFC, giải mã HTML entity, bỏ khoảng trắng thừa và dấu cách trước dấu câu (`src/data.py`). Nhãn: `entailment`, `neutral`, `contradiction`.

## manual/ – phân tích thủ công 50 mẫu (mục 2 báo cáo)

| File | Nội dung |
|---|---|
| `50_mau.csv` | 25 mẫu XNLI (Quân phân tích) + 25 mẫu ViANLI (Quang phân tích): nhãn gốc, thẻ hiện tượng, giải thích, ghi chú |
| `gan_nhan_doc_lap_A.csv`, `gan_nhan_doc_lap_B.csv` | Nhãn mỗi người tự gán cho 25 mẫu của người kia khi chưa xem nhãn gốc (tính kappa) |
| `huong_dan_gan_the.md` | Bộ thẻ hiện tượng và cách gắn |
| `ghi_chu_tach_tu.md` | Ghi chú về tách từ tiếng Việt |

## Nguồn và giấy phép

| Bộ | Nguồn | Giấy phép |
|---|---|---|
| XNLI | Conneau et al. (2018), huggingface.co/datasets/facebook/xnli (config `vi`) | CC BY-NC 4.0 |
| ViANLI | Huynh, Nguyen & Nguyen (2025), huggingface.co/datasets/uitnlp/ViANLI | CC BY-NC-SA 4.0 |
| ViNLI | Huynh, Nguyen & Nguyen (2022), bản đăng lại huggingface.co/datasets/presencesw/vinli_4_label; bản chính thức tại nlp.uit.edu.vn | chỉ dùng cho nghiên cứu |

Chỉ dùng cho mục đích học tập trong môn CS221.
