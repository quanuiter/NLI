# Bản nháp báo cáo mục 4 – Cài đặt và đánh giá (W5-01)

Bản nháp soạn trước ngày chốt số liệu (W4-09). Chỗ đánh dấu **[CHỜ]** điền khi Quang xong M0, M1, M1-h (W2-05 → W2-07)
và seed thêm cho M2, M3 (W4-05). Khi chốt xong, chép vào mục 4 của BaoCao_NLI theo [quy ước báo cáo](huong_dan/quy_uoc_bao_cao.md).
Cột "Nguồn" chỉ để đối chiếu, khi chép vào báo cáo thì chuyển thành nhận xét Google Tài liệu.

---

## 4. Cài đặt và đánh giá

### 4.1. Môi trường thực nghiệm

Toàn bộ code viết bằng Python, dùng chung một khung cho mọi mô hình: đọc dữ liệu và tiền xử lý (`src/data.py`),
tính độ đo và lưu kết quả (`src/evaluate.py`). Mỗi lần chạy lưu một file kết quả JSON gồm độ đo, cấu hình, thời gian chạy
và một file dự đoán cho từng mẫu, nên mọi con số trong báo cáo đều truy lại được.

| Thành phần | Dùng cho | Chi tiết |
|---|---|---|
| Google Colab, GPU NVIDIA T4 (16 GB) | M3, M4, M5 | Huấn luyện fp16 cho M4, M5 |
| CPU 4 nhân | M0, M1, M2 | Mô hình nhỏ, không cần GPU |
| Thư viện | | PyTorch, Hugging Face Transformers, scikit-learn, Gensim (fastText), py_vncorenlp (tách từ). **[CHỜ: phiên bản trên Colab]** |
| Mã nguồn | | github.com/quanuiter/NLI; dữ liệu đã xử lý, kết quả và checkpoint trên Google Drive của nhóm |

Để lấy phiên bản thư viện, chạy một ô trên Colab:
`import torch, transformers, sklearn, gensim; print(torch.__version__, transformers.__version__, sklearn.__version__, gensim.__version__)`.

**Tái lập:** mọi lần chạy cố định seed (`set_seed` cho random, NumPy, PyTorch). Tập con 100.000 cặp XNLI được chọn
một lần (phân tầng theo nhãn, seed 42) và lưu lại để mọi mô hình dùng chung.

### 4.2. Dữ liệu và kịch bản

Ba kịch bản (chi tiết ở mục 2 và 3.3):

| Kịch bản | Huấn luyện | Chọn mô hình (dev) | Kiểm tra |
|---|---|---|---|
| S1 | XNLI tiếng Việt, 100.000 cặp | XNLI dev (2.490) | XNLI test (5.010), ViANLI test (1.000), ViNLI test (2.264) |
| S2 | ViANLI train (8.012) | ViANLI dev (1.000) | ViANLI test, ViNLI test |
| S3 | XNLI 100.000 cặp, rồi học tiếp ViANLI train | ViANLI dev | ViANLI test, ViNLI test |

Tập test không được dùng ở bất kỳ bước chọn siêu tham số hay chọn checkpoint nào.

### 4.3. Độ đo đánh giá

Với mỗi nhãn c, gọi TP_c là số mẫu nhãn c được đoán đúng, FP_c là số mẫu nhãn khác bị đoán thành c, FN_c là số mẫu nhãn c bị đoán thành nhãn khác:

    Precision_c = TP_c / (TP_c + FP_c)      Recall_c = TP_c / (TP_c + FN_c)
    F1_c = 2 · Precision_c · Recall_c / (Precision_c + Recall_c)
    Macro-F1 = (F1_entailment + F1_neutral + F1_contradiction) / 3
    Accuracy = số mẫu đoán đúng / tổng số mẫu

- **Độ đo chính là macro-F1**, vì nó coi 3 nhãn quan trọng như nhau. Một mô hình chỉ đoán một nhãn vẫn có accuracy 33%
  nhưng macro-F1 chỉ khoảng 17%. Báo cáo cả accuracy để so với số liệu công bố (các bài báo thường chỉ báo accuracy).
- **Confusion matrix** (hàng là nhãn thật, cột là nhãn dự đoán) dùng để phân tích lỗi ở mục 5.
- **Nhiều seed:** PhoBERT ở S2, S3 chạy 3 seed (42, 123, 2024), báo cáo trung bình ± độ lệch chuẩn. **[CHỜ: M2, M3 thêm seed ở W4-05.]**
  Các mô hình còn lại chạy seed 42.

### 4.4. Siêu tham số và cách chọn

Mọi siêu tham số được chọn theo macro-F1 trên tập dev. Kết quả thử từng cấu hình lưu trong `results/analysis/`.

| Mô hình | Siêu tham số chọn | Cách chọn | Nguồn |
|---|---|---|---|
| M0 | Nhãn đa số của tập train | – | **[CHỜ]** |
| M1 | **[CHỜ: bộ phân loại, C hoặc alpha, danh sách đặc trưng]** | Lưới trên dev | **[CHỜ]** |
| M2 | 2 lớp ẩn × 512, dropout 0,5, Adam lr 3·10⁻⁴, batch 128, dừng sớm (patience 3) | Lưới 16 cấu hình trên dev S2, kiểm tra lại 3 seed | `m2_tuning_S2.json` |
| M3 | BiLSTM 128 mỗi chiều, FFNN 512, dropout 0,3, embedding đóng băng, Adam lr 10⁻³, batch 64 | Lưới 8 cấu hình trên dev S2 (kiểm tra lại 3 seed), thử đóng băng embedding trên dev S1 | `m3_tuning_S2.json`, `m3_seedcheck_S2.json`, `m3_tuning_S1.json` |
| M4, M5 | S1: lr 2·10⁻⁵, batch 32, 3 epoch. S2, S3: lr 10⁻⁵, batch 16, 7 epoch, dropout lớp phân loại 0,4. AdamW, weight decay 0,01, warmup 10%, tối đa 256 token | So 2 cấu hình trên dev S2 (0,444 và 0,448); S1 dùng cấu hình ngắn vì 100.000 cặp | `M4_tuning_S2_seed42.json`, `token_lengths.json` |

### 4.5. Kết quả chính

**Bảng 4.x – Accuracy / macro-F1 (%) trên các tập test.** PhoBERT S2, S3: trung bình ± độ lệch chuẩn 3 seed.

| Mô hình | S1 → XNLI | S1 → ViANLI | S1 → ViNLI | S2 → ViANLI | S2 → ViNLI | S3 → ViANLI | S3 → ViNLI |
|---|---|---|---|---|---|---|---|
| M0 nhãn đa số | **[CHỜ]** | **[CHỜ]** | **[CHỜ]** | **[CHỜ]** | **[CHỜ]** | – | – |
| M1 TF-IDF + LR/NB | **[CHỜ]** | **[CHỜ]** | **[CHỜ]** | **[CHỜ]** | **[CHỜ]** | – | – |
| M1-h (chỉ hypothesis) | **[CHỜ]** | **[CHỜ]** | **[CHỜ]** | **[CHỜ]** | **[CHỜ]** | – | – |
| M2 embedding + FFNN | 53,0 / 53,0 | 27,5 / 27,2 | 48,4 / 47,0 | 39,7 / 37,5 | 32,6 / 30,2 | – | – |
| M3 BiLSTM | 63,7 / 63,5 | 25,7 / 24,5 | 52,6 / 51,9 | 38,0 / 37,1 | 37,3 / 36,5 | – | – |
| M4 PhoBERT-base | **77,1 / 77,2** | 27,8 / 27,3 | **70,9 / 70,4** | **44,4 ± 1,6 / 44,1 ± 1,7** | **54,4 ± 1,6 / 54,4 ± 1,5** | **44,6 ± 0,5 / 44,3 ± 0,5** | **69,9 ± 1,4 / 69,6 ± 1,5** |
| M5 XLM-R-base | 75,2 / 75,2 | 27,0 / 26,7 | 67,8 / 67,4 | 43,6 / 41,0 | 32,5 / 28,3 | 43,5 / 42,7 | 63,3 / 63,3 |

Nguồn: `results/<mô hình>_<kịch bản>_<tập test>_seed<seed>.json`; bảng tổng hợp bằng `summarize()` trong `src/evaluate.py`
(**[CHỜ]** thay bằng `results/summary_bang.csv` khi Quang xong W3-06). Mức ngẫu nhiên: 33,3%.

**Hình 4.x – So sánh accuracy các mô hình** (`results/figures/so_sanh_mo_hinh.png`, vẽ bằng `scripts/plot_results.py`;
M0, M1 tự được thêm vào hình khi có kết quả).

Nhận xét ngắn cho mục 4 (phân tích chi tiết ở mục 5):
- Trên XNLI, accuracy tăng dần theo độ phức tạp của mô hình: M2 53,0% → M3 63,7% → PhoBERT 77,1%.
- Học trên XNLI thì **mọi mô hình đều thấp hơn mức ngẫu nhiên trên ViANLI** (25,7–27,8%), nhưng vẫn đạt 48–71% trên ViNLI.
- Trên ViANLI, khoảng cách giữa các mô hình nhỏ hơn nhiều: mô hình tốt nhất (PhoBERT) chỉ đạt khoảng 44%.

### 4.6. So sánh với số liệu công bố

| Thiết lập | Nhóm | Công bố | Nguồn công bố | Ghi chú |
|---|---|---|---|---|
| PhoBERT-base, S2 → ViANLI | 44,4 ± 1,6 | 44,3 | ViANLI, Bảng 10 | Cùng thiết lập, so sánh trực tiếp được |
| XLM-R-base, S2 → ViANLI | 43,6 | 44,8 | ViANLI, Bảng 10 | Cùng thiết lập; nhóm chạy 1 seed |
| PhoBERT-base, S1 → XNLI | 77,1 | 78,5 | PhoBERT, Bảng 3 | Bài báo dùng đủ 392.702 cặp, nhóm dùng 100.000 cặp |
| XLM-R-base, S1 → XNLI | 75,2 | 76,5 | XLM-R, Bảng 1 | Bài báo fine-tune trên tiếng Anh, nhóm fine-tune trên tiếng Việt |

Kết quả của nhóm khớp với số công bố trong khoảng 1–2 điểm, tức nằm trong mức dao động giữa các seed. Đây là bằng chứng cách cài đặt đúng.

### 4.7. Chi phí tính toán

| Mô hình | Số tham số | Thời gian huấn luyện S2 | Thời gian huấn luyện S1 | Dự đoán 1.000 cặp (ViANLI test) | Thiết bị |
|---|---|---|---|---|---|
| M0 | 0 | **[CHỜ]** | **[CHỜ]** | **[CHỜ]** | CPU |
| M1 | **[CHỜ]** | **[CHỜ]** | **[CHỜ]** | **[CHỜ]** | CPU |
| M2 | 0,88 triệu | 2 giây | 54 giây | 0,05 giây | CPU |
| M3 | 8,97 triệu (0,97 triệu được huấn luyện) | 26 giây | 7,6 phút | 0,08 giây | GPU T4 |
| M4 PhoBERT | 135 triệu | 6,8 phút | 24,5 phút | 1,8 giây | GPU T4 |
| M5 XLM-R | 278 triệu | 11,3 phút | 33,5 phút | 1,9 giây | GPU T4 |

Nguồn: trường `train_seconds`, `predict_seconds_vianli_test`, `n_params` trong file kết quả seed 42.
Thời gian M2 tính cả bước tính vector câu. Thời gian M3 trở đi đo trên GPU nên không so trực tiếp được với M2 (CPU).

- PhoBERT tốt hơn M3 khoảng 6 điểm trên ViANLI và 13 điểm trên XNLI, nhưng lớn gấp 15 lần và huấn luyện lâu gấp khoảng 15 lần.
- XLM-R lớn gấp đôi PhoBERT, huấn luyện chậm hơn 37% (S1) đến 67% (S2), mà kết quả không tốt hơn.
- Bảng này dùng lại cho bảng ưu/nhược điểm ở mục 5.
