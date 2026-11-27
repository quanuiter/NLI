# Kết quả M7: LLM Qwen2.5-7B-Instruct phân loại bằng prompt (W4-12)

Quang chạy `notebooks/08_llm.ipynb` trên Colab T4 (18–19/11); Quân đồng bộ từ Drive ngày 22/11. Không huấn luyện: mô hình 4-bit đọc prompt
trắc nghiệm A (chắc chắn đúng) / B (có thể đúng hoặc sai) / C (chắc chắn sai), chọn chữ có xác suất cao nhất (`src/llm.py`).
Few-shot (FS): 6 ví dụ từ ViANLI train, 2 mỗi nhãn, 3 bộ ví dụ khác nhau (seed 42, 123, 2024); `demo_ids` ghi trong từng file kết quả.

## 1. Kết quả (accuracy / macro-F1, %)

| Thiết lập | XNLI test | ViANLI test | ViNLI test |
|---|---|---|---|
| M7 zero-shot (S0) | 63,0 / 61,4 | 42,7 / 39,8 | 67,4 / 67,3 |
| M7 few-shot 6 ví dụ (FS, 3 bộ ví dụ) | – | 45,9 ± 0,2 / 44,0 ± 0,6 | 72,7 ± 1,0 / 72,7 ± 1,1 |
| PhoBERT S2 (3 seed) | – | 44,4 ± 1,6 | 54,4 ± 1,6 |
| PhoBERT S3 (3 seed) | – | 44,6 ± 0,5 | 69,9 ± 1,4 |
| M6 S2 (3 seed) | – | 47,7 ± 0,9 | 70,9 ± 0,5 |

Kiểm định t (Welch, 3 lần chạy mỗi bên): ViANLI M7 FS so với PhoBERT S2 p = 0,25 (không khác biệt), so với PhoBERT S3 p = 0,025,
so với M6 p = 0,059 (M6 nhỉnh hơn nhưng chưa có ý nghĩa thống kê). ViNLI: M7 FS so với M6 p = 0,07, so với PhoBERT S2 p < 0,001.

## 2. Nhận xét

- **Không huấn luyện mà ngang mô hình fine-tune.** Chỉ với 6 ví dụ, M7 đạt 45,9% trên ViANLI (ngang PhoBERT 44,4%, dưới M6 47,7%) và 72,7% trên ViNLI,
  cao nhất nhóm (M6 70,9%). Điểm cực ổn định giữa các bộ ví dụ (± 0,2 trên ViANLI).
- **Zero-shot thua rõ trên XNLI** (63,0% so với PhoBERT 77,1%): LLM tổng quát không tự biết cách gán nhãn của bộ dữ liệu.
- **Lệch nhãn về "neutral" (đáp án B).** Zero-shot đoán neutral cho 731/1.000 câu ViANLI và 3.270/5.010 câu XNLI. Few-shot giảm còn 558–597/1.000,
  nhưng vẫn rất ít đoán contradiction (80–102/1.000, F1 contradiction 31–35). Mô hình chọn "có thể đúng hoặc sai" khi không chắc.
- **M7 không dùng mẹo trùng từ của ViANLI.** Ở nhóm câu trùng ít từ (ViANLI, chủ yếu entailment), mô hình học ViANLI đạt 47–58% nhờ mẹo, M7 chỉ 27,0% (S0) và 38,6% (FS).
  Ngược lại, trên ViNLI nhóm trùng nhiều từ, M7 FS đạt 75,4% trong khi PhoBERT S2 chỉ 40,1% (mẹo ngược). Điểm của M7 đến từ hiểu câu hơn là quy luật bề mặt,
  nhưng lại yếu ở câu cần nhận ra mâu thuẫn.
- **Chi phí:** không huấn luyện, nhưng dự đoán chậm nhất: khoảng 0,26 giây mỗi cặp (ViANLI 1.000 cặp: 262 giây), chậm hơn PhoBERT khoảng 150 lần; 7,6 tỷ tham số (4-bit, khoảng 5,5 GB).
- **Hạn chế:** dữ liệu huấn luyện trước của Qwen có thể đã chứa văn bản giống XNLI/ViNLI (không kiểm tra được); prompt chỉ có một phiên bản, không thử nhiều cách hỏi
  (để không chọn prompt theo test).

## 3. File
`results/M7_*.json`, `results/preds/M7_*.csv`, `results/figures/M7_*.png`; nhóm câu: cột `M7 S0`, `M7 FS` trong `results/analysis/loi_nhom_cau.csv`.
