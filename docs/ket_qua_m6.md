# Kết quả M6 (mDeBERTa đã học NLI) và ensemble (W4-10, W4-11)

Quang chạy `notebooks/07_m6.ipynb` trên Colab T4 ngày 14/11. Quân đồng bộ từ Drive và chạy lại `scripts/ensemble.py`: kết quả khớp với bản trên Colab.

## 1. M6

Mô hình: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (279 triệu tham số), đã học NLI trên khoảng 2,7 triệu cặp câu
ở 27 ngôn ngữ (có tiếng Việt, gồm MNLI, ANLI, WANLI… dịch máy và XNLI dev). Tập test XNLI, ViANLI, ViNLI không nằm trong dữ liệu đó.

**Chọn cấu hình S2 trên dev (seed 42)** – `results/analysis/M6_tuning_S2_seed42.json`

| Cấu hình | Dev macro-F1 tốt nhất | Epoch tốt nhất | Thời gian |
|---|---|---|---|
| A (lr 2e-5, batch 32, 3 epoch) | 0,507 | 2 | 11 phút |
| **B (lr 1e-5, batch 16, 7 epoch, dropout lớp phân loại 0,4)** | **0,511** | 4 | 36 phút |

Hai cấu hình gần như ngang nhau; B được chọn theo quy tắc (dev cao hơn). Từ epoch 5, loss train tiếp tục giảm (0,33 → 0,14) nhưng dev giảm: học quá khớp, checkpoint epoch 4 được giữ.
Dev PhoBERT S2 là 0,448, M6 hơn khoảng 6 điểm ngay trên dev.

**Kết quả test (accuracy / macro-F1, %)**

| Thiết lập | XNLI | ViANLI | ViNLI |
|---|---|---|---|
| M6 S0 – không fine-tune | 79,3 / 79,4 | 38,7 / 38,0 | **74,1 / 74,1** |
| M6 S2 – fine-tune trên ViANLI (seed 42) | – | **48,7 / 48,8** | 71,1 / 71,1 |
| PhoBERT S2 (3 seed) để so | – | 44,4 ± 1,6 / 44,1 ± 1,7 | 54,4 ± 1,6 / 54,4 ± 1,5 |
| PhoBERT S3 (3 seed) để so | – | 44,6 ± 0,5 / 44,3 ± 0,5 | 69,9 ± 1,4 / 69,6 ± 1,5 |
| PhoBERT S1 để so | 77,1 / 77,2 | 27,8 / 27,3 | 70,9 / 70,4 |

Nhận xét:
- **M6 S2 là mô hình tốt nhất của nhóm trên ViANLI: 48,7%, hơn PhoBERT 4,3 điểm** (khoảng 2,7 lần độ lệch chuẩn giữa các seed của PhoBERT),
  F1 cả 3 nhãn đều 47–51 (không lệch nhãn như XLM-R). Đồng thời giữ được ViNLI 71,1%, ngang PhoBERT S3.
- **Chưa fine-tune (S0), M6 đã hơn mức ngẫu nhiên trên ViANLI (38,7%)**, trong khi mọi mô hình học XNLI của nhóm (S1) đều dưới mức ngẫu nhiên.
  Khác biệt chính: dữ liệu NLI của M6 có cả ANLI (đối kháng, dịch máy), tức đã gặp kiểu câu "bẫy" tương tự. Đây là bằng chứng
  cho nhận định ở mục 5: mô hình học XNLI bị mẹo trùng từ, còn học thêm dữ liệu đối kháng thì bớt.
- Fine-tune trên ViANLI tăng ViANLI 10 điểm nhưng giảm ViNLI 3 điểm (74,1 → 71,1): đánh đổi nhỏ hơn nhiều so với PhoBERT
  (S1 → S2: ViNLI giảm 16,5 điểm).
- **Hạn chế:** M6 S2 mới chạy 1 seed (seed 123, 2024 chưa chạy). Lợi thế 4,3 điểm lớn hơn dao động seed của PhoBERT nhưng
  chưa đo được dao động của chính M6. Dữ liệu huấn luyện trước của M6 lớn hơn rất nhiều so với các mô hình khác, nên đây là so sánh
  "mô hình có sẵn kiến thức NLI" với "mô hình học từ đầu trên dữ liệu của nhóm", không phải so kiến trúc.

## 2. Ensemble – `results/analysis/ensemble.json`

Bỏ phiếu đa số, thành viên cố định trước khi xem test, phá hòa theo macro-F1 dev.

| Mã | Thành viên | ViANLI | ViNLI | Thành viên tốt nhất (ViANLI / ViNLI) |
|---|---|---|---|---|
| E1 | PhoBERT S2, 3 seed | 43,9 | 54,8 | 44,4 / 54,4 |
| E2 | PhoBERT S3, 3 seed | 44,8 | 70,2 | 44,6 / 69,9 |
| E3 | PhoBERT S2 + XLM-R S2 + M6 S2 | 48,0 | 60,6 | 48,7 / 71,1 |
| E4 | PhoBERT S3 + XLM-R S3 + M6 S2 | 47,2 | **72,5** | 48,7 / 71,1 |

- Ensemble **không** giúp trên ViANLI: E3, E4 thấp hơn M6 một mình. Khi một thành viên mạnh hơn hẳn hai thành viên còn lại
  (M6 48,7 so với 43–45), hai mô hình yếu hơn thường cùng sai và bỏ phiếu thắng M6.
- E4 tăng nhẹ trên ViNLI (72,5 so với 71,1 của M6), vì PhoBERT S3 và XLM-R S3 cũng tốt trên ViNLI (63–70%).
- Bỏ phiếu giữa các seed của cùng một mô hình (E1, E2) gần như không đổi: các seed sai ở cùng những câu.
- Kết luận cho báo cáo: cách cải thiện hiệu quả là **mô hình đã học NLI trên nhiều dữ liệu (có dữ liệu đối kháng)**, không phải kết hợp các mô hình đã có.
