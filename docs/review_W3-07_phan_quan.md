# W3-07 – Review code chéo: phần Quân review code của Quang

Phạm vi: `notebooks/01_eda.ipynb`, `notebooks/02_m1.ipynb` (M0, M1, M1-h, ablation). Review ngày 27/11, kèm chạy lại M0/M1/M1-h/ablation
trên CPU (chi tiết ở `docs/kiem_tra_tai_lap.md`).

## Danh sách kiểm tra (theo `docs/huong_dan/W3-07.md`)

| Mục | Kết quả |
|---|---|
| Dữ liệu chỉ lấy qua `load_scenario` / `load_split` | Đạt. EDA dùng `load_split`, M1 dùng `load_scenario("S1"/"S2")` |
| Không dùng test để chọn siêu tham số | Đạt. Lưới C (LR) và alpha (NB), lưới C của M1-h và ablation đều tính macro-F1 trên `d['dev']`; test chỉ đánh giá cấu hình đã chọn |
| Không dùng cột `id` làm đặc trưng | Đạt. Đặc trưng chỉ lấy từ `premise`, `hypothesis` (TF-IDF + 5 đặc trưng so sánh) |
| TF-IDF chỉ học từ vựng trên train | Đạt. `fit_transform` trên train, `transform` trên dev/test |
| Đặc trưng không âm (dùng được cho Naive Bayes) | Đạt. Chênh lệch độ dài lấy trị tuyệt đối, chia 50 |
| Có `set_seed`, chạy lại ra đúng kết quả cũ | Đạt. M0, M1, M1-h S2 trùng hoàn toàn khi chạy lại; M1-h S1 lệch ≤ 0,2 điểm do khác phiên bản scikit-learn |
| Lưu bằng `evaluate_and_save`, `config` ghi đủ siêu tham số | Đạt. Có `clf`, `C`/`alpha`, danh sách đặc trưng, macro-F1 dev |
| Mọi số trong báo cáo có file trong `results/` | Đạt (`scripts/verify_report.py`) |

## Chỗ đã sửa (2)
1. `02_m1.ipynb`: dùng `os.makedirs` mà không `import os` (trên Colab không lộ vì ô kết nối Drive đã import). Đã thêm `import os` vào ô import.
2. `01_eda.ipynb`: thư mục lưu hình ghi cứng `/content/NLI/results/figures` (chỉ đúng trên Colab). Đã đổi thành `RESULTS_DIR / 'figures'`.

## Ghi chú, không cần sửa
- Đặc trưng "trùng từ" của M1 là tỉ lệ từ của hypothesis có trong premise, khác Jaccard ở EDA và phân tích lỗi. Khi viết mục 3 nên ghi rõ công thức của từng chỗ.
- `max_iter=2000` của Logistic Regression đủ cho C ≤ 0,1; với C ≥ 1 trên 100.000 cặp, lời giải lbfgs hơi khác giữa các phiên bản scikit-learn. Không ảnh hưởng cấu hình được chọn.
