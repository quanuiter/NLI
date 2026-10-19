# M2 – Embedding trung bình + FFNN: cách làm và kết quả (W2-08, W2-09)

## 1. Word embeddings (W2-08)

- **Embedding:** fastText `cc.vi.300` (Grave et al., 2018), 300 chiều, mức âm tiết, đọc bằng **Gensim**.
  PhoW2V (VinAI) không tải được vì máy chủ `public.vinai.io` không truy cập được lúc làm.
- **Lọc vocab:** chỉ giữ token có trong dữ liệu đồ án: 26.671 / 41.220 token → `data/processed/emb_fasttext.npz`.
- **Tra vector theo thứ tự:**
  1. nguyên dạng;
  2. chữ thường;
  3. **dấu kiểu cũ**: fastText (giống VnCoreNLP/PhoBERT) viết `hoá, toà, khoẻ, thuỷ`, còn dữ liệu viết `hóa, tòa, khỏe, thủy`. Không đổi thì các từ này không có vector.
- **OOV** (`results/analysis/oov_fasttext.json`):

| Tập | OOV theo lượt | OOV theo từ | Ví dụ từ không có vector |
|---|---|---|---|
| XNLI train 100k | 0,99% | 38,0% | lsc, inglethorp, poirot (tên riêng dịch máy) |
| XNLI test | 0,42% | 4,5% | Wolverstone, Binalshibh |
| ViANLI train | 0,55% | 3,7% | Covid, nCoV, VinFast, Tourneo |
| ViANLI test | 0,56% | 2,3% | Covid, Sinopharm, BioNTech |
| ViNLI test | 0,63% | 1,7% | Covid, Bolsonaro |

  Đổi dấu kiểu cũ làm OOV giảm khoảng 1/3 (ViANLI train: 0,87% → 0,55%). Phần OOV còn lại chủ yếu là tên riêng.
  XNLI train có OOV theo từ cao (38%) vì có nhiều tên riêng và lỗi dịch hiếm gặp, nhưng chúng chỉ chiếm 1% lượt xuất hiện.
- **Kiểm tra chất lượng vector** (từ gần nhất theo cosine): `tăng` → giảm, vọt · `thắng` → thua, giành · `bệnh` → chữa, mắc.
  Lưu ý: từ trái nghĩa nằm gần nhau ("tăng"–"giảm", "thắng"–"thua") vì chúng xuất hiện trong ngữ cảnh giống nhau.
  Đây là điểm yếu của embedding với thẻ `ANT`.

## 2. Mô hình M2 (W2-09)

```
premise    → âm tiết → vector fastText → trung bình → u (300)
hypothesis → âm tiết → vector fastText → trung bình → v (300)
[u, v, |u − v|, u * v, cos(u, v)] (1201) → chuẩn hóa → FFNN 2 lớp ẩn 512, ReLU, dropout 0,5 → softmax 3 nhãn
```

- **Huấn luyện:** cross-entropy, Adam, batch 128, tối đa 30 epoch, dừng sớm khi macro-F1 dev không tăng 3 epoch.
- **Chọn siêu tham số trên dev** (`scripts/tune_m2.py`, lưới 16 cấu hình, `results/analysis/m2_tuning_S2.json`),
  sau đó kiểm tra lại 4 cấu hình tốt nhất với 3 seed:

| Cấu hình | S2 dev macro-F1 | S1 dev macro-F1 |
|---|---|---|
| **2 lớp × 512, dropout 0,5, lr 3e-4** (chọn) | 0,367 ± 0,014 | **0,549 ± 0,002** |
| 1 lớp × 512, dropout 0,5, lr 3e-4 | 0,355 ± 0,013 | 0,547 ± 0,004 |
| 1 lớp × 512, dropout 0,3, lr 1e-3 | 0,352 ± 0,011 | 0,537 ± 0,010 |
| 2 lớp × 512, dropout 0,5, lr 1e-4 | 0,369 ± 0,005 | – |

  Ở S2 các cấu hình chênh nhau trong phạm vi nhiễu (dev 1.000 câu). Chọn cấu hình tốt nhất ở S1 và dùng chung cho cả hai kịch bản.
  ViANLI dừng ở epoch 1–3 (mô hình học thuộc train rất nhanh), XNLI dừng ở epoch khoảng 11.

## 3. Kết quả test (seed 42)

| Kịch bản | Tập test | Accuracy | Macro-F1 | F1 E / N / C |
|---|---|---|---|---|
| S1 (học XNLI) | XNLI test | 53,0 | 53,0 | 52,7 / 52,3 / 54,2 |
| S1 | **ViANLI test** | **27,5** | 27,2 | 29,8 / 24,7 / 27,3 |
| S1 | ViNLI test | 48,4 | 47,0 | 51,9 / 52,4 / 36,6 |
| S2 (học ViANLI) | ViANLI test | 39,7 | 37,5 | 48,9 / 41,8 / 21,9 |
| S2 | ViNLI test | 32,6 | 30,2 | 33,8 / 39,2 / 17,7 |

Mức đoán bừa: 33,3%. Đối chiếu: PhoBERT-base trên ViANLI test (S2) đạt 44,3% (bài ViANLI, Bảng 10).

## 4. Nhận xét (dùng cho mục 5 báo cáo)

1. **S1 → ViANLI thấp hơn đoán bừa (27,5%) vì mô hình học được mẹo "trùng từ nhiều = entailment", và ViANLI bị đảo ngược quy luật này.**
   Độ trùng từ Jaccard (%) giữa premise và hypothesis, theo nhãn:

   | Nhãn | XNLI train | XNLI test | ViANLI train | ViANLI test | ViNLI test |
   |---|---|---|---|---|---|
   | entailment | **32,5** | **22,9** | 21,6 | 21,4 | **30,9** |
   | neutral | 20,1 | 15,8 | **31,3** | **33,5** | 21,8 |
   | contradiction | 21,9 | 16,4 | 31,7 | 30,6 | 24,5 |

   - Ở XNLI và ViNLI, entailment có độ trùng cao nhất. Ở ViANLI thì thấp nhất, vì người viết cố tình không chép từ của premise (bài ViANLI, Bảng 6).
   - Confusion matrix S1 → ViANLI: 201/333 câu neutral bị đoán là entailment, 145/334 câu entailment bị đoán là neutral.
   - Đây chính là hiện tượng ANLI mô tả: mô hình học từ dữ liệu thường bị dữ liệu đối kháng đánh bại (RoBERTa học SNLI + MNLI chỉ đạt 31% trên ANLI).
2. **Contradiction là nhãn yếu nhất ở hầu hết các lần chạy** (F1 18–37% ngoài XNLI). Lấy trung bình vector làm "không" chỉ còn là 1 trong khoảng 20 vector bị cộng trung bình, nên tín hiệu phủ định gần như mất. Đây là căn cứ để kỳ vọng M3 (BiLSTM, giữ thứ tự từ) và PhoBERT làm tốt hơn ở nhãn này.
3. **S2 không chuyển sang ViNLI (32,6%, bằng đoán bừa).** 8.000 câu đối kháng không đủ để mô hình đơn giản học được quy luật chung.
   Ngược lại, S1 (100.000 câu dịch) chuyển sang ViNLI được 48,4%.
4. **M2 ở S2 (39,7%) kém PhoBERT công bố (44,3%) khoảng 5 điểm.** Đây là khoảng cách nhỏ hơn mong đợi giữa mô hình chỉ lấy trung bình embedding và mô hình Transformer, cho thấy ViANLI khó với cả hai.

## 5. Chạy lại

```bash
python scripts/prepare_embeddings.py          # 1 lần, ~10 phút (tải 1,2 GB)
python scripts/tune_m2.py --scenario S2       # chọn siêu tham số trên dev
python scripts/run_m2.py --scenario S2 S1 --seeds 42          # W2-09
python scripts/run_m2.py --scenario S2 --seeds 123 2024       # thêm seed (W4-05)
```
Trên Colab: `notebooks/03_m2.ipynb`. Chạy trên CPU khoảng 1–2 phút (không tính bước tải embedding).
