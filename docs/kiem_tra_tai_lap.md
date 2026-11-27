# Kiểm tra tái lập code và số liệu (W6-03)

Kiểm tra ngày 27/11 trên máy CPU sạch (clone mới từ GitHub, cài `requirements.txt`, tải dữ liệu lại từ nguồn).
Mục tiêu: mọi con số trong báo cáo truy được về `results/`, và phần chạy được trên CPU cho lại đúng kết quả đã lưu.

## 1. Dữ liệu tạo lại từ nguồn

- `python scripts/prepare_data.py` tải lại XNLI, ViANLI, ViNLI từ Hugging Face: đủ 7 file, số cặp đúng như báo cáo
  (100.000 / 2.490 / 5.010 / 8.012 / 1.000 / 1.000 / 2.264).
- Id và nhãn của tập test tạo lại **trùng 100%** với file dự đoán đã lưu (XNLI test so với `M4_S1_xnli_test`,
  ViANLI test so với `M4_S2_vianli_test`, ViNLI test so với `M4_S2_vinli_test` và `M1_S2_vinli_test`).
- `python tests/test_core.py`: 3/3 bài kiểm tra đạt (chuẩn hóa nhãn, tập con 100k phân tầng và tất định, độ đo và lưu kết quả).

## 2. Chạy lại mô hình trên CPU

| Mô hình | Cách chạy lại | Kết quả |
|---|---|---|
| M0, M1, M1-h, ablation M1 | Chạy nguyên văn các ô code của `notebooks/02_m1.ipynb` (trừ ô kết nối Colab) trong một bản sao repo | [ĐIỀN] |
| Ensemble E1–E4, bảng tổng hợp, hình, phân tích lỗi | `ensemble.py`, `make_table.py`, `plot_results.py`, `error_analysis.py` | File kết quả giống hệt bản đã lưu, chỉ khác trường `created_at` |

## 3. Mô hình cần GPU (M2–M7): kiểm tra bằng file kết quả

Các mô hình này huấn luyện trên Colab T4; chạy lại toàn bộ cần khoảng 8 giờ GPU nên không chạy lại hết trong đợt kiểm tra.
Thay vào đó đã kiểm tra:
- Mỗi số trong báo cáo có file `results/*.json` kèm cấu hình, seed, thời gian và file dự đoán từng cặp câu (`results/preds/`).
- Accuracy và macro-F1 trong **81/81** file JSON khớp với file dự đoán (tính lại từ cột `label`, `pred`), số cặp đúng.
- Lệnh chạy lại từng mô hình ghi ở README, mục "Chạy lại toàn bộ kết quả"; notebook `03`–`08` đã được chạy từ đầu trên Colab ít nhất một lần
  (M6, M7 do Quang chạy theo notebook trên tài khoản Colab của mình, khác máy viết code, nên notebook chạy được ngoài máy của người viết).
- PhoBERT, XLM-R S2 khớp số công bố trong khoảng 1–2 điểm (bài ViANLI: 44,3 và 44,8; nhóm: 44,4 ± 1,6 và 43,6).

## 4. Đối chiếu số liệu trong báo cáo

- `python scripts/verify_report.py docs/bao_cao_muc4.md`: **106/106** số trong bảng kết quả chính (accuracy, macro-F1, độ lệch chuẩn) khớp `results/`.
- Khi chép mục 4 vào file báo cáo hoặc sửa bảng, chạy lại lệnh này.

## 5. Còn phải làm khi có bản báo cáo cuối

- [ ] Đối chiếu các số ở mục 2 (EDA, kappa), mục 3 (ví dụ minh họa), mục 5 (nhóm câu, ví dụ đúng/sai) với `results/analysis/` và `data/manual/`.
- [ ] Kiểm tra link GitHub trong báo cáo trỏ đúng repo, nhánh `main`.
