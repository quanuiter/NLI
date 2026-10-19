# NLI – Suy luận ngôn ngữ tự nhiên tiếng Việt

Đồ án môn CS221 – Xử lý ngôn ngữ tự nhiên (UIT). Phân loại quan hệ ngữ nghĩa giữa cặp câu
(premise, hypothesis) thành **entailment / neutral / contradiction** trên hai bộ dữ liệu huấn luyện
và một bộ test phụ:

| Bộ | Nguồn | Train / Dev / Test | Ghi chú |
|---|---|---|---|
| XNLI-vi | [facebook/xnli](https://huggingface.co/datasets/facebook/xnli) (config `vi`) | 392.702 (dùng tập con 100k) / 2.490 / 5.010 | train dịch máy, dev/test do người dịch |
| ViANLI | [uitnlp/ViANLI](https://huggingface.co/datasets/uitnlp/ViANLI) | 8.012 / 1.000 / 1.000 | dữ liệu đối kháng, giấy phép CC BY-NC-SA 4.0 |
| ViNLI (chỉ test) | [presencesw/vinli_4_label](https://huggingface.co/datasets/presencesw/vinli_4_label) (bản đăng lại; bản chính thức: [UIT NLP](https://nlp.uit.edu.vn/)) | – / – / 2.264 | tiếng Việt gốc, không đối kháng; đã bỏ nhãn `other` (727 mẫu) |

## Cấu trúc

```
src/
  data.py       load + chuẩn hóa dữ liệu, tập con XNLI 100k, ViNLI test, load_scenario("S1"|"S2"|"S3")
  evaluate.py   độ đo, lưu kết quả JSON + confusion matrix + dự đoán từng mẫu; summarize()
  utils.py      set_seed, timer, save_json/load_json
  segment.py    tách từ cho PhoBERT (VnCoreNLP, dự phòng underthesea)
  embeddings.py fastText cc.vi.300 qua Gensim, lọc theo vocab, báo cáo OOV
  m2.py         M2: embedding trung bình + FFNN
scripts/
  prepare_data.py   tải + chuẩn hóa -> data/processed/*.jsonl
  segment_data.py   tách từ -> data/processed/*_seg.jsonl
  sample_manual.py  chọn 50 mẫu phân tích thủ công -> data/manual/
  check_manual.py   kiểm tra file phân tích thủ công, tính kappa
  prepare_embeddings.py, tune_m2.py, run_m2.py   embedding và M2
docs/           tóm tắt bài báo, kết quả và nhận xét từng mô hình
notebooks/      mỗi mô hình một notebook (chạy trên Colab)
data/manual/    50 mẫu phân tích thủ công, hướng dẫn gắn thẻ
results/        kết quả từng lần chạy (json), figures/ (confusion matrix), preds/ (dự đoán)
tests/          kiểm tra nhanh khung code
```

## Cài đặt và chuẩn bị dữ liệu

```bash
pip install -r requirements.txt
python scripts/prepare_data.py      # ~1 phút, tạo data/processed/
python scripts/segment_data.py      # tách từ cho PhoBERT, cần Java (Colab có sẵn)
python tests/test_core.py           # kiểm tra khung code
```

Trên Colab: mở `notebooks/00_colab_setup.ipynb`.

## Quy ước dùng chung

**Tiền xử lý** (`src/data.py`, áp dụng cho mọi mô hình): Unicode NFC, giải mã HTML entity, bỏ khoảng trắng thừa,
bỏ dấu cách trước dấu câu. Bước cuối cần thiết vì ~91% câu XNLI train (dịch máy) có dạng `từ .`, trong khi dev/test
gần như không có.

**Nhãn**: luôn là chuỗi `entailment | neutral | contradiction` (`LABELS`, `LABEL2ID`, `ID2LABEL`).

**Kịch bản**

| Mã | Train | Dev | Test |
|---|---|---|---|
| S1 | XNLI-vi 100k | XNLI-vi dev | XNLI-vi test + ViANLI test + ViNLI test |
| S2 | ViANLI train | ViANLI dev | ViANLI test + ViNLI test |
| S3 | XNLI-vi 100k → ViANLI train | ViANLI dev | ViANLI test + ViNLI test |

ViNLI test chỉ dùng để đánh giá khả năng tổng quát sang dữ liệu tiếng Việt gốc không đối kháng
(`load_scenario(..., include_vinli=False)` để bỏ). Đã kiểm tra: không có cặp câu nào trùng giữa ViNLI test và ViANLI.

```python
from src.data import load_scenario
from src.evaluate import evaluate_and_save
from src.utils import set_seed

set_seed(42)
data = load_scenario("S2")                 # load_scenario("S2", segmented=True) cho PhoBERT
# ... huấn luyện trên data["train"], chọn siêu tham số trên data["dev"] (KHÔNG dùng test) ...
for test_name, test_df in data["test"].items():
    y_pred = model.predict(test_df)
    evaluate_and_save(test_df["label"], y_pred, model="M1", scenario="S2", test_set=test_name,
                      seed=42, config={"C": 1.0}, ids=test_df["id"],
                      premise=test_df["premise"], hypothesis=test_df["hypothesis"])
```

**Tên mô hình**: `M0` nhãn đa số · `M1` TF-IDF + NB/LR · `M1h` chỉ hypothesis · `M2` embedding + FFNN ·
`M3` BiLSTM · `M4` PhoBERT · `M5` XLM-R. **Seed**: 42, 123, 2024 (`src.utils.SEEDS`).

Tổng hợp bảng kết quả: `from src.evaluate import summarize; summarize()`.

## Giấy phép

Mã nguồn: MIT (xem `LICENSE`). Dữ liệu trích trong `data/manual/` giữ giấy phép gốc của từng bộ
(ViANLI: CC BY-NC-SA 4.0; ViNLI: chỉ dùng cho nghiên cứu), chỉ dùng cho mục đích học tập.

## Tài liệu tham khảo dữ liệu

- Conneau et al. (2018). XNLI: Evaluating Cross-lingual Sentence Representations. EMNLP.
- Huynh, Nguyen & Nguyen (2022). ViNLI: A Vietnamese Corpus for Studies on Open-Domain Natural Language Inference. COLING.
- Huynh, Nguyen & Nguyen (2025). A New Benchmark Dataset and Mixture-of-Experts Language Models for Adversarial
  Natural Language Inference in Vietnamese. Expert Systems with Applications.
