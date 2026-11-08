# M5 – XLM-R base: kết quả (W4-04)

`xlm-roberta-base`, cùng code và cùng cấu hình với PhoBERT (M4) để so sánh công bằng giữa mô hình **đa ngữ** và **đơn ngữ**:
S1 cấu hình A, S2 và S3 cấu hình B, S3 bắt đầu từ checkpoint S1 của chính XLM-R. XLM-R nhận văn bản thô, không tách từ.
Mô tả mô hình: `docs/mo_ta_m2_m5.md` mục 6. Chạy trên Colab T4 ngày 08/11, seed 42.

## 1. Kết quả test (seed 42)

| Kịch bản | Tập test | M5 XLM-R acc | M5 macro-F1 | M4 PhoBERT acc (seed 42) |
|---|---|---|---|---|
| S1 | XNLI test | 75,2 | 75,2 | 77,1 |
| S1 | ViANLI test | 27,0 | 26,7 | 27,8 |
| S1 | ViNLI test | 67,8 | 67,4 | 70,9 |
| S2 | ViANLI test | 43,6 | 41,0 | 46,0 (3 seed: 44,4 ± 1,6) |
| S2 | ViNLI test | 32,5 | 28,3 | 52,7 |
| S3 | ViANLI test | 43,5 | 42,7 | 44,4 (3 seed: 44,6 ± 0,5) |
| S3 | ViNLI test | 63,3 | 63,3 | 71,4 |

Thời gian huấn luyện: S1 34 phút, S2 và S3 khoảng 11 phút (PhoBERT: 25 và 7 phút). XLM-R có 278 triệu tham số, PhoBERT 135 triệu,
và chuỗi token của XLM-R dài hơn khoảng 30% (`results/analysis/token_lengths.json`).

Số công bố để đối chiếu: XLM-R-base trên ViANLI test (S2) 44,8% acc / 39,9 macro-F1 (bài ViANLI, Bảng 10);
trên XNLI-vi 76,5% khi fine-tune bằng tiếng Anh (bài XLM-R, Bảng 1).

## 2. Nhận xét (dùng cho mục 5 báo cáo)

1. **PhoBERT tốt hơn hoặc ngang XLM-R ở mọi dòng** dù chỉ bằng một nửa số tham số và học trước trên lượng dữ liệu nhỏ hơn nhiều
   (20 GB so với 2,5 TB). Khớp với kết luận của bài PhoBERT: mô hình đơn ngữ có lợi thế với tiếng Việt, một phần nhờ tách từ.
2. **XLM-R ở S2 gần như không tổng quát sang ViNLI (32,5%, ngang mức ngẫu nhiên).** Confusion matrix trên ViNLI:
   XLM-R gần như không đoán *contradiction* (110/2.264 câu; F1 contradiction 11,1). Trên ViANLI cũng vậy (F1 contradiction 21,0).
   Với 8.000 cặp đối kháng, XLM-R không học được dấu hiệu mâu thuẫn dùng được cho văn bản khác. PhoBERT cùng kịch bản đạt 52,7% trên ViNLI.
3. **Kết quả S2 của nhóm (43,6%) khớp số công bố (44,8%)**, chênh 1,2 điểm, trong mức dao động giữa các seed thấy ở PhoBERT (±1,6).
4. **Giống PhoBERT: S1 → ViANLI thấp hơn mức ngẫu nhiên (27,0%); S3 không hơn S2 trên ViANLI nhưng giữ được ViNLI (63,3%).**
   Hiện tượng lặp lại ở cả hai mô hình Transformer, nên đây là đặc điểm của dữ liệu chứ không phải của một mô hình.
5. Câu ví dụ Tim Cook: XLM-R S2 đoán *neutral* (sai), PhoBERT đoán đúng *contradiction*.

Chỉ có seed 42 cho XLM-R (kế hoạch chỉ chạy 3 seed cho PhoBERT, W4-03). Khi so PhoBERT với XLM-R nên ghi rõ điều này.

## 3. Chạy lại

```bash
python scripts/run_transformer.py --model M5 --scenario S1 --config A
python scripts/run_transformer.py --model M5 --scenario S2 S3 --config B
```
Trên Colab: `notebooks/06_tuan4.ipynb`, ô W4-04. Checkpoint S1 (khoảng 1,1 GB) ở Drive chung `NLI/checkpoints/M5_S1_seed42/`.
