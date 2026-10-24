# M4 – PhoBERT: cách làm và kết quả (W3-02, W3-03)

## 1. Mô hình

```
premise, hypothesis (đã tách từ bằng VnCoreNLP)
→ tokenizer PhoBERT: <s> premise </s></s> hypothesis </s>, tối đa 256 token
→ PhoBERT-base (12 lớp Transformer, 135 triệu tham số) đọc cả cặp câu cùng lúc
→ vector của token <s> → lớp phân loại (dense + tanh + dropout) → softmax 3 nhãn
```

- **Huấn luyện:** Hugging Face Trainer, AdamW, weight decay 0,01, warmup 10% số bước, fp16, Colab GPU T4.
  Đánh giá dev sau mỗi epoch, giữ checkpoint có macro-F1 dev cao nhất.
- **`max_len` = 256:** 99% cặp câu dài dưới 105 token, chỉ 17/100.000 cặp XNLI train dài hơn 256
  (`results/analysis/token_lengths.json`). Mỗi batch chỉ đệm tới cặp dài nhất trong batch.
- **Khác M2, M3:** M2, M3 mã hóa từng câu thành vector riêng rồi mới so sánh. PhoBERT đọc hai câu cùng lúc,
  self-attention cho phép mỗi token của hypothesis "nhìn" trực tiếp mọi token của premise.

## 2. Chọn cấu hình trên dev (S2)

`results/analysis/M4_tuning_S2_seed42.json`. Mỗi cấu hình lưu riêng ở `results/tuning/`, chọn theo macro-F1 dev.

| Cấu hình | lr | batch | epoch | dropout lớp phân loại | Dev macro-F1 theo epoch | Tốt nhất | Thời gian |
|---|---|---|---|---|---|---|---|
| A | 2e-5 | 32 | 3 | 0,1 | 0,406 / 0,436 / 0,444 | 0,444 (epoch 3) | 2,3 phút |
| **B** (bài ViANLI) | 1e-5 | 16 | 7 | 0,4 | 0,307 / 0,416 / **0,448** / 0,434 / 0,437 / 0,446 / 0,437 | **0,448** (epoch 3) | 6,8 phút |

- Chọn B, nhưng hai cấu hình chênh nhau 0,004, trong phạm vi nhiễu của dev 1.000 câu.
- B đạt tốt nhất ở epoch 3, sau đó loss train vẫn giảm (0,92 → 0,57) mà dev không tăng: mô hình bắt đầu quá khớp.

**Lỗi đã sửa ở lần chạy đầu (24/10):** dropout 0,4 bị áp cho cả 12 lớp của PhoBERT thay vì chỉ lớp phân loại,
nên mô hình học rất chậm (dev 0,34 sau 7 epoch, test ViANLI 33,9%). Kiểm chứng trên CPU (1.500 mẫu XNLI, 2 epoch):
dropout bên trong 0,4 → dev 0,317; giữ 0,1 → dev 0,419.

## 3. Kết quả test (S2, seed 42)

| Tập test | Accuracy | Macro-F1 | F1 E / N / C |
|---|---|---|---|
| ViANLI test | **46,0** | **45,9** | 52,1 / 46,4 / 39,2 |
| ViNLI test | 52,7 | 52,8 | 44,8 / 55,2 / 58,3 |

So sánh trên ViANLI test (S2):

| Mô hình | Accuracy | Macro-F1 |
|---|---|---|
| M2 (embedding trung bình + FFNN) | 39,7 | 37,5 |
| M3 (BiLSTM) | 38,0 | 37,1 |
| **M4 PhoBERT-base (nhóm)** | **46,0** | **45,9** |
| PhoBERT-base (bài ViANLI, Bảng 10) | 44,3 | 41,9 |
| NLIMoE, mô hình tốt nhất của bài ViANLI | 47,3 | – |

## 4. Nhận xét (dùng cho mục 5 báo cáo)

1. **Kết quả của nhóm khớp với số công bố:** 46,0% so với 44,3% của bài ViANLI với gần đúng cấu hình. Chênh 1,7 điểm,
   trong khoảng dao động thường gặp giữa các seed. Đây là bằng chứng code và dữ liệu (kể cả bước tách từ) đúng.
   Chạy thêm 2 seed ở W4-03 để biết chênh lệch này có ổn định không.
2. **PhoBERT hơn M2, M3 khoảng 6–8 điểm**, lớn nhất ở nhãn contradiction:
   - câu có từ phủ định với nhãn contradiction (57 câu): M2 7,0%, M3 24,6%, **M4 50,9%**;
   - câu trùng nhiều từ (Jaccard ≥ 0,5) nhưng nhãn neutral (56 câu): M2 55,4%, M3 39,3%, **M4 67,9%**.
   Đọc hai câu cùng lúc giúp mô hình so trực tiếp từng phần của hypothesis với premise.
3. **Dù vậy, ViANLI vẫn rất khó:** 46% chỉ hơn mức ngẫu nhiên 13 điểm. Contradiction vẫn là nhãn yếu nhất
   (F1 39,2; 138/333 câu contradiction bị đoán là neutral), khớp với nhận xét của bài ViANLI (mục 7.8).
4. **Học trên ViANLI rồi kiểm tra trên ViNLI được 52,7%**, cao hơn hẳn M2 (32,6%) và M3 (37,3%).
   Phần lớn khả năng này đến từ việc PhoBERT đã được huấn luyện trước trên văn bản tiếng Việt lớn.

## 5. Chạy lại

```bash
python scripts/run_transformer.py --token-lengths                         # đo độ dài token (chọn max_len)
python scripts/run_transformer.py --model M4 --scenario S2 --smoke        # chạy thử (W3-02)
python scripts/run_transformer.py --model M4 --scenario S2 --config A B   # W3-03: so A, B trên dev, giữ cấu hình tốt hơn
```
Trên Colab: `notebooks/05_phobert.ipynb` (bắt buộc GPU). Checkpoint ở Drive chung `NLI/checkpoints/M4_S2_seed42/`
(khoảng 0,5 GB), không đưa lên GitHub.
