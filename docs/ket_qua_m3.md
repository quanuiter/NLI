# M3 – BiLSTM + max-pooling: cách làm và kết quả (W3-01)

## 1. Mô hình

```
premise    → âm tiết → chỉ số từ → embedding fastText 300 (đóng băng) → BiLSTM 1 lớp (mỗi chiều 128) → max-pooling → u (256)
hypothesis → (cùng embedding, cùng BiLSTM)                                                            → v (256)
[u, v, |u − v|, u * v] (1024) → FFNN 1 lớp ẩn 512, ReLU, dropout 0,3 → softmax 3 nhãn
```

- **Embedding:** dùng lại ma trận fastText của M2 (`data/processed/emb_fasttext.npz`). Âm tiết không có vector dùng vector 0.
- **Độ dài tối đa:** 64 âm tiết, đủ cho khoảng 97–99% premise (XNLI train khoảng 96%) và gần như toàn bộ hypothesis.
  Mỗi batch chỉ đệm tới câu dài nhất trong batch.
- **Max-pooling có mặt nạ:** vị trí đệm không được chọn làm giá trị lớn nhất, nên vector của một câu không phụ thuộc vào câu khác trong batch
  (đã kiểm tra trong `notebooks/04_m3.ipynb`).
- **Khác M2:** BiLSTM đọc câu theo thứ tự hai chiều. Ví dụ "Việt Nam thắng Thái Lan" và "Thái Lan thắng Việt Nam":
  M2 (trung bình) cho hai vector giống hệt nhau, M3 cho hai vector khác nhau.
- **Huấn luyện:** cross-entropy, Adam lr 1e-3, batch 64, cắt gradient 5, tối đa 15 epoch, dừng sớm khi macro-F1 dev không tăng 3 epoch.
- **Số tham số:** 8,97 triệu, trong đó 0,97 triệu được huấn luyện (ma trận embedding 8 triệu tham số bị đóng băng). M2: 0,88 triệu.

## 2. Chọn siêu tham số trên dev

- **S2** (`scripts/tune_m3.py`, 8 cấu hình, `results/analysis/m3_tuning_S2.json`): đóng băng embedding tốt hơn ở cả 4 cặp so sánh.
  Hai cấu hình tốt nhất được chạy lại với 3 seed (`results/analysis/m3_seedcheck_S2.json`):

  | Cấu hình (đóng băng embedding) | Dev macro-F1, seed 42 / 123 / 2024 | Trung bình |
  |---|---|---|
  | **hidden 128, dropout 0,3** (chọn) | 0,388 / 0,379 / 0,378 | **0,382 ± 0,006** |
  | hidden 128, dropout 0,5 | 0,398 / 0,371 / 0,366 | 0,378 ± 0,017 |

  Cấu hình dropout 0,5 đứng đầu ở seed 42 chỉ do may. Tính trung bình 3 seed thì dropout 0,3 cao hơn và ổn định hơn.
- **S1** (`results/analysis/m3_tuning_S1.json`, chạy trên Colab GPU): chỉ thử đóng băng hay không, vì S1 chạy lâu.
  Đóng băng 0,637, không đóng băng 0,626 → giữ đóng băng. Không đóng băng thì mô hình quá khớp rất nhanh (tốt nhất ở epoch 2).
- Cả hai kịch bản dùng chung một cấu hình (`m3.DEFAULT_CONFIG`).

## 3. Kết quả test (seed 42, chạy trên Colab GPU T4)

| Kịch bản | Tập test | M2 acc | M2 macro-F1 | **M3 acc** | **M3 macro-F1** | M3 F1 E / N / C |
|---|---|---|---|---|---|---|
| S1 (học XNLI) | XNLI test | 53,0 | 53,0 | **63,7** | **63,5** | 64,7 / 58,8 / 67,0 |
| S1 | ViANLI test | 27,5 | 27,2 | 25,7 | 24,5 | 31,4 / 20,1 / 21,9 |
| S1 | ViNLI test | 48,4 | 47,0 | **52,6** | **51,9** | 54,5 / 56,6 / 44,6 |
| S2 (học ViANLI) | ViANLI test | 39,7 | 37,5 | 38,0 | 37,1 | 46,1 / 32,2 / 33,1 |
| S2 | ViNLI test | 32,6 | 30,2 | **37,3** | **36,5** | 39,8 / 40,8 / 29,0 |

Mức ngẫu nhiên: 33,3%. Đối chiếu: PhoBERT-base trên ViANLI test (S2) đạt 44,3% (bài ViANLI, Bảng 10).
Thời gian huấn luyện trên GPU T4: S1 khoảng 7,5 phút, S2 khoảng 26 giây. M2 chạy trên CPU nên thời gian không so trực tiếp được.

Chỉ có 1 seed. Chạy thêm seed 123 và 2024 ở W4-05 để biết chênh lệch nào vượt mức nhiễu.

## 4. Nhận xét (dùng cho mục 5 báo cáo)

1. **Trên XNLI, M3 hơn M2 khoảng 10 điểm (63,7% so với 53,0%), chủ yếu nhờ nhãn contradiction (F1 54,2 → 67,0).**
   Nhóm câu trong XNLI test mà hypothesis có từ phủ định và nhãn là contradiction (761 câu): M2 đúng 71,1%, M3 đúng 89,9%.
   Max-pooling giữ được tín hiệu mạnh của từ "không", trong khi lấy trung bình như M2 làm tín hiệu này bị loãng.
   Nhóm câu trùng nhiều từ (Jaccard ≥ 0,5) nhưng nhãn là contradiction (29 câu): M2 13,8%, M3 48,3%. Đây là trường hợp
   đọc theo thứ tự có ích (thẻ `ROLE`), nhưng số câu ít nên chỉ xem là gợi ý.
2. **S1 → ViANLI vẫn thấp hơn mức ngẫu nhiên, và M3 còn thấp hơn M2 (25,7% so với 27,5%).** Mô hình mạnh hơn trên XNLI thì học
   quy luật "trùng từ nhiều → entailment" của XNLI chắc hơn, nên sai nhiều hơn trên ViANLI, nơi quy luật này bị đảo ngược:
   227/333 câu neutral bị đoán là entailment (M2: 201/333). Đây là cùng hiện tượng mà bài ANLI mô tả.
3. **Trên ViANLI (S2), M3 ngang M2 về tổng thể (macro-F1 37,1 so với 37,5) nhưng phân bố lỗi khác:**
   F1 contradiction tăng từ 21,9 lên 33,1, còn F1 neutral giảm từ 41,8 xuống 32,2.
   Câu có từ phủ định với nhãn contradiction (57 câu): M2 7,0%, M3 24,6%. Như vậy M3 có học được tín hiệu phủ định,
   nhưng 8.012 câu đối kháng không đủ để cải thiện tổng thể.
4. **M3 tổng quát sang ViNLI tốt hơn M2 ở cả hai kịch bản:** S1 52,6% so với 48,4%, S2 37,3% so với 32,6%.
5. **M3 trên ViANLI (38,0%) vẫn kém PhoBERT công bố (44,3%) khoảng 6 điểm.** Một phần là vì M2, M3 mã hóa từng câu riêng rồi mới so sánh,
   còn PhoBERT đọc hai câu cùng lúc (self-attention giữa mọi từ của hai câu). So sánh trực tiếp ở W3-03.

## 5. Chạy lại

```bash
python scripts/prepare_embeddings.py                          # 1 lần (dùng chung với M2)
python scripts/tune_m3.py --scenario S2                       # chọn siêu tham số trên dev
python scripts/tune_m3.py --scenario S1 --grid freeze_embedding
python scripts/run_m3.py --scenario S2 S1 --seeds 42          # W3-01
python scripts/run_m3.py --scenario S2 --seeds 123 2024       # thêm seed (W4-05)
```
Trên Colab: `notebooks/04_m3.ipynb` (nên chọn GPU T4). Checkpoint lưu ở Drive chung `NLI/checkpoints/M3_*.pt`
(khoảng 4 MB mỗi file vì không lưu ma trận embedding đóng băng). Checkpoint không đưa lên GitHub.
