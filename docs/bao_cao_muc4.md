# Bản nháp báo cáo mục 4 – Cài đặt và đánh giá (W5-01)

Bản đã điền đủ số liệu M0–M5 (13/11). Chỉ còn **[CHỜ]** phiên bản thư viện trên Colab. Sau ngày chốt số liệu (W4-09)
chép vào mục 4 của BaoCao_NLI theo [quy ước báo cáo](huong_dan/quy_uoc_bao_cao.md).
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
| CPU 4 nhân | M0, M1, M2 | Mô hình nhỏ, không cần GPU (M1 dùng scikit-learn 1.9) |
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

**Lưu ý về cách chia của ViANLI:** 956/1.000 cặp test (963/1.000 cặp dev) có premise đã xuất hiện trong tập train,
nhưng chỉ 4,5% cặp test có nhãn trùng với một nhãn đã gặp của cùng premise đó trong train. Nói cách khác, cùng một premise
được người viết đối kháng dùng để tạo các hypothesis khác nhau, và các hypothesis đó được chia vào train, dev, test với nhãn khác nhau.
Một luật "đoán nhãn chưa từng gặp với premise này" đúng 96,3% trên 564 cặp test mà premise chỉ còn đúng 1 nhãn chưa gặp trong train. Nhóm **không** dùng
thông tin này ở bất kỳ mô hình nào, nhưng nó ảnh hưởng đến cách đọc kết quả (mục 5): mô hình học thuộc premise sẽ bị phạt
trên dev/test, điển hình là Naive Bayes ở S2 chỉ đạt macro-F1 0,08–0,13 trên dev. Chi tiết: `docs/phan_tich_loi_transformer.md`, mục (f).

### 4.3. Độ đo đánh giá

Với mỗi nhãn c, gọi TP_c là số mẫu nhãn c được đoán đúng, FP_c là số mẫu nhãn khác bị đoán thành c, FN_c là số mẫu nhãn c bị đoán thành nhãn khác:

    Precision_c = TP_c / (TP_c + FP_c)      Recall_c = TP_c / (TP_c + FN_c)
    F1_c = 2 · Precision_c · Recall_c / (Precision_c + Recall_c)
    Macro-F1 = (F1_entailment + F1_neutral + F1_contradiction) / 3
    Accuracy = số mẫu đoán đúng / tổng số mẫu

- **Độ đo chính là macro-F1**, vì nó coi 3 nhãn quan trọng như nhau. Một mô hình chỉ đoán một nhãn vẫn có accuracy 33%
  nhưng macro-F1 chỉ khoảng 17%. Báo cáo cả accuracy để so với số liệu công bố (các bài báo thường chỉ báo accuracy).
- **Confusion matrix** (hàng là nhãn thật, cột là nhãn dự đoán) dùng để phân tích lỗi ở mục 5.
- **Nhiều seed:** M2, M3, PhoBERT ở S2 và PhoBERT ở S3 chạy 3 seed (42, 123, 2024), báo cáo trung bình ± độ lệch chuẩn.
  Các thiết lập còn lại chạy seed 42 (S1 tốn thời gian; M0, M1 không phụ thuộc seed; XLM-R chạy 1 seed do giới hạn GPU).

### 4.4. Siêu tham số và cách chọn

Mọi siêu tham số được chọn theo macro-F1 trên tập dev. Kết quả thử từng cấu hình lưu trong `results/analysis/`.

| Mô hình | Siêu tham số chọn | Cách chọn | Nguồn |
|---|---|---|---|
| M0 | Nhãn đa số của tập train: S1 contradiction, S2 neutral (các nhãn gần như cân bằng nên chọn nhãn nào cũng cho accuracy ≈ 33%) | – | `M0_*.json` (trường `config`) |
| M1 | Logistic Regression; S1: C = 0,1, S2: C = 0,01. Đặc trưng: TF-IDF 1–2-gram riêng cho premise và hypothesis (min_df 2, sublinear tf) + 5 đặc trưng so sánh (tỉ lệ từ trùng, số từ phủ định ở P và ở H, chênh lệch độ dài / 50, H có số mới) | Lưới 7 cấu hình trên dev (LR: C ∈ {0,01; 0,1; 1; 10}, Naive Bayes: alpha ∈ {0,1; 0,5; 1}); LR thắng ở cả hai kịch bản (dev 0,593 ở S1, 0,387 ở S2). Bỏ từng nhóm đặc trưng: bỏ tỉ lệ từ trùng làm dev giảm 5,4 điểm (S1) và 7,9 điểm (S2) | `m1_chon_sieu_tham_so.csv`, `m1_ablation.csv` |
| M1-h | Logistic Regression chỉ trên TF-IDF của hypothesis, C = 1 | Lưới C ∈ {0,1; 1; 10} trên dev | `M1h_*.json` |
| M2 | 2 lớp ẩn × 512, dropout 0,5, Adam lr 3·10⁻⁴, batch 128, dừng sớm (patience 3) | Lưới 16 cấu hình trên dev S2, kiểm tra lại 3 seed | `m2_tuning_S2.json` |
| M3 | BiLSTM 128 mỗi chiều, FFNN 512, dropout 0,3, embedding đóng băng, Adam lr 10⁻³, batch 64 | Lưới 8 cấu hình trên dev S2 (kiểm tra lại 3 seed), thử đóng băng embedding trên dev S1 | `m3_tuning_S2.json`, `m3_seedcheck_S2.json`, `m3_tuning_S1.json` |
| M4, M5 | S1: lr 2·10⁻⁵, batch 32, 3 epoch. S2, S3: lr 10⁻⁵, batch 16, 7 epoch, dropout lớp phân loại 0,4. AdamW, weight decay 0,01, warmup 10%, tối đa 256 token | So 2 cấu hình trên dev S2 (0,444 và 0,448); S1 dùng cấu hình ngắn vì 100.000 cặp | `M4_tuning_S2_seed42.json`, `token_lengths.json` |

### 4.5. Kết quả chính

**Bảng 4.x – Accuracy / macro-F1 (%) trên các tập test.** M2, M3, PhoBERT ở S2 và PhoBERT ở S3: trung bình ± độ lệch chuẩn 3 seed.

| Mô hình | S1 → XNLI | S1 → ViANLI | S1 → ViNLI | S2 → ViANLI | S2 → ViNLI | S3 → ViANLI | S3 → ViNLI |
|---|---|---|---|---|---|---|---|
| M0 nhãn đa số | 33,3 / 16,7 | 33,3 / 16,7 | 32,6 / 16,4 | 33,3 / 16,7 | 34,3 / 17,0 | – | – |
| M1 TF-IDF + LR | 59,1 / 59,1 | 21,4 / 21,6 | 51,7 / 50,5 | 45,1 / 41,1 | 27,9 / 27,2 | – | – |
| M1-h (chỉ hypothesis) | 53,8 / 53,7 | 33,2 / 31,6 | 44,4 / 43,5 | 31,7 / 30,9 | 42,9 / 42,2 | – | – |
| M2 embedding + FFNN | 53,0 / 53,0 | 27,5 / 27,2 | 48,4 / 47,0 | 37,0 ± 2,5 / 35,8 ± 1,7 | 33,7 ± 1,3 / 32,0 ± 2,0 | – | – |
| M3 BiLSTM | 63,7 / 63,5 | 25,7 / 24,5 | 52,6 / 51,9 | 36,8 ± 2,8 / 36,1 ± 2,1 | 38,5 ± 1,3 / 37,4 ± 1,9 | – | – |
| M4 PhoBERT-base | **77,1 / 77,2** | 27,8 / 27,3 | **70,9 / 70,4** | **44,4 ± 1,6 / 44,1 ± 1,7** | **54,4 ± 1,6 / 54,4 ± 1,5** | **44,6 ± 0,5 / 44,3 ± 0,5** | **69,9 ± 1,4 / 69,6 ± 1,5** |
| M5 XLM-R-base | 75,2 / 75,2 | 27,0 / 26,7 | 67,8 / 67,4 | 43,6 / 41,0 | 32,5 / 28,3 | 43,5 / 42,7 | 63,3 / 63,3 |

Nguồn: `results/<mô hình>_<kịch bản>_<tập test>_seed<seed>.json`; bảng tổng hợp `results/summary_bang_acc.csv` (accuracy)
và `results/summary_bang.csv` (macro-F1), tạo bằng `scripts/make_table.py`. Mức ngẫu nhiên: 33,3%.

**Hình 4.x – So sánh accuracy các mô hình trên ViANLI test và ViNLI test** (`results/figures/so_sanh_mo_hinh.png`,
vẽ bằng `scripts/plot_results.py`).

Nhận xét ngắn cho mục 4 (phân tích chi tiết ở mục 5):
- Trên XNLI, các mô hình có học đều vượt xa M0 (33,3%), và PhoBERT tốt nhất (77,1%). M1 (59,1%) hơn M2 (53,0%):
  lấy trung bình vector từ làm mất thông tin từ nào trùng giữa hai câu, còn M1 có sẵn đặc trưng này.
- **M1-h chỉ nhìn hypothesis mà đạt 53,8% trên XNLI**, hơn M0 20 điểm: XNLI có lộ nhãn qua hypothesis.
  Trên ViANLI, M1-h chỉ đạt 31,7% (≈ mức ngẫu nhiên), tức cách xây dựng đối kháng đã loại được kiểu lộ nhãn này.
- Học trên XNLI thì **mọi mô hình đều thấp hơn mức ngẫu nhiên trên ViANLI** (M1 21,4%, các mô hình nơron 25,7–27,8%),
  nhưng vẫn đạt 48–71% trên ViNLI.
- Học trên ViANLI (S2), M1 đạt 45,1%, ngang PhoBERT (44,4%), nhưng chỉ đạt 27,9% trên ViNLI, thấp hơn M0.
  PhoBERT S2 vẫn đạt 54,4% trên ViNLI. M1 học được mẹo riêng của ViANLI chứ không học được suy luận (phân tích ở mục 5).
- Trên ViANLI, khoảng cách giữa các mô hình nhỏ: mô hình tốt nhất chỉ đạt khoảng 44–45%. Học trước trên XNLI (S3)
  không tăng điểm ViANLI nhưng giữ được điểm ViNLI (PhoBERT 69,9% so với 54,4% ở S2).

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
| M0 | 0 (chỉ lưu 1 nhãn) | < 0,01 giây | < 0,01 giây | < 0,01 giây | CPU |
| M1 | 0,21 triệu (S2), 0,82 triệu (S1) | 1,7 giây | 31 giây | 0,1 giây | CPU |
| M2 | 0,88 triệu | 2 giây | 54 giây | 0,05 giây | CPU |
| M3 | 8,97 triệu (0,97 triệu được huấn luyện) | 26 giây | 7,6 phút | 0,08 giây | GPU T4 |
| M4 PhoBERT | 135 triệu | 6,8 phút | 24,5 phút | 1,8 giây | GPU T4 |
| M5 XLM-R | 278 triệu | 11,3 phút | 33,5 phút | 1,9 giây | GPU T4 |

Nguồn: trường `train_seconds`, `predict_seconds_vianli_test`, `n_params` trong file kết quả seed 42. File kết quả M0, M1
không ghi thời gian, nên nhóm đo lại bằng cách chạy lại đúng code của notebook `02_m1.ipynb` trên CPU (dự đoán trùng 100%
với file kết quả đã lưu). Thời gian M1 tính cả bước tạo đặc trưng TF-IDF; số tham số M1 = số trọng số của Logistic Regression
(3 nhãn × số đặc trưng, cộng 3 hệ số chặn), phụ thuộc kích thước từ vựng nên khác nhau giữa S1 và S2.
Thời gian M2 tính cả bước tính vector câu. Thời gian M3 trở đi đo trên GPU nên không so trực tiếp được với M2 (CPU).

- M1 nhỏ và nhanh nhất trong các mô hình có học, huấn luyện trên ViANLI chưa tới 2 giây, mà ngang PhoBERT trên ViANLI test.
  Nhưng M1 thua PhoBERT 18 điểm trên XNLI và 27 điểm trên ViNLI (S2), nên lợi thế này chỉ có trên ViANLI.
- PhoBERT tốt hơn M3 khoảng 8 điểm trên ViANLI và 13 điểm trên XNLI, nhưng lớn gấp 15 lần và huấn luyện lâu gấp khoảng 15 lần.
- XLM-R lớn gấp đôi PhoBERT, huấn luyện chậm hơn 37% (S1) đến 67% (S2), mà kết quả không tốt hơn.
- Bảng này dùng lại cho bảng ưu/nhược điểm ở mục 5.
