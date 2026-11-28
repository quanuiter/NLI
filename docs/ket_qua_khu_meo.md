# Khử mẹo trùng từ (W7-01) – kết quả

## 1. Mô hình mẹo một mình (M1s, đã có, chạy trên CPU)

Logistic Regression chỉ dùng 5 đặc trưng bề mặt của M1 (tỉ lệ từ của hypothesis có trong premise, số từ phủ định ở P và ở H,
chênh lệch độ dài / 50, H có số mới), không có TF-IDF. Đây cũng là "mô hình mẹo" dùng trong PoE. Lệnh: `run_bias_only()` trong `src/debias.py`.

| Học trên | XNLI test | ViANLI test | ViNLI test |
|---|---|---|---|
| S1 (XNLI 100k) | 53,5 / 53,2 | **21,4** / 21,7 | 48,0 / 46,8 |
| S2 (ViANLI) | – | **46,4** / 44,5 | 27,7 / 27,5 |

(accuracy / macro-F1, %; mức ngẫu nhiên 33,3%)

- Chỉ 5 con số bề mặt đã đạt 53,5% trên XNLI, tức gần 70% hiệu năng của PhoBERT S1 (77,1%) đến được bằng mẹo.
- Học trên XNLI thì mẹo **dưới mức ngẫu nhiên trên ViANLI (21,4%)**, đúng bằng M1 S1 (21,4%) và thấp hơn PhoBERT S1 (27,8%).
- Học trên ViANLI thì mẹo đạt **46,4%, cao hơn PhoBERT S2 (44,4%)**, nhưng sang ViNLI chỉ 27,7%: quy luật trùng từ của hai bộ dữ liệu ngược nhau.
- Hệ số học được trên XNLI (`config.coef` trong `results/M1s_S1_*.json`): trùng từ +2,77 cho entailment, −2,08 cho neutral;
  phủ định trong H +0,73 cho contradiction. Đúng mẹo "trùng nhiều từ → suy ra được, có phủ định → mâu thuẫn".

## 2. PhoBERT khử mẹo (M4P) – chờ Quang chạy `notebooks/09_khu_meo.ipynb`

So M4P với PhoBERT thường (M4) ở S1, mỗi bên 3 seed. Kỳ vọng nếu nhận định đúng: ViANLI tăng (hướng về hoặc vượt 33%),
XNLI giảm nhẹ. [CHỜ]
