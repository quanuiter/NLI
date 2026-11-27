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
  m3.py         M3: BiLSTM + max-pooling
  transformer.py M4 (PhoBERT), M5 (XLM-R), M6 (mDeBERTa đã học NLI): fine-tune cặp câu bằng Hugging Face Trainer
  llm.py        M7: LLM Qwen2.5-7B-Instruct (4-bit) phân loại bằng prompt, zero-shot và few-shot
scripts/
  prepare_data.py   tải + chuẩn hóa -> data/processed/*.jsonl
  segment_data.py   tách từ -> data/processed/*_seg.jsonl
  sample_manual.py  chọn 50 mẫu phân tích thủ công -> data/manual/
  check_manual.py   kiểm tra file phân tích thủ công, tính kappa
  check_tasks.py    kiểm tra sản phẩm từng task (kết quả, notebook, ablation)
  prepare_embeddings.py, tune_m2.py, run_m2.py   embedding và M2
  tune_m3.py, run_m3.py   M3
  run_transformer.py      M4, M5, M6 (--smoke: chạy thử; --token-lengths: đo độ dài token; --scenario S0: M6 không fine-tune)
  run_llm.py, check_llm.py   M7 (LLM bằng prompt)
  ensemble.py       E1–E4: bỏ phiếu đa số giữa các mô hình
  make_table.py     bảng tổng hợp results/summary*.csv
  plot_results.py   hình so sánh results/figures/so_sanh_mo_hinh.png
  error_analysis.py phân tích lỗi theo nhóm câu, danh sách mẫu -> results/analysis/loi_*
  verify_report.py  đối chiếu số trong báo cáo với results/ (kiểm tra tái lập)
  package_data.py   đóng gói dữ liệu nộp kèm -> dist/data_NLI.zip
docs/           tóm tắt bài báo, kết quả và nhận xét từng mô hình
docs/huong_dan/ hướng dẫn từng bước cho từng task trong kế hoạch
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
`M3` BiLSTM · `M4` PhoBERT · `M5` XLM-R · `M6` mDeBERTa-v3 đã học NLI · `M7` Qwen2.5-7B-Instruct bằng prompt ·
`E1`–`E4` ensemble. **Seed**: 42, 123, 2024 (`src.utils.SEEDS`). Kịch bản thêm: `S0` = không huấn luyện trên dữ liệu
của nhóm (M6, M7 zero-shot), `FS` = M7 few-shot (6 ví dụ từ ViANLI train).

Tổng hợp bảng kết quả: `from src.evaluate import summarize; summarize()`.

## Chạy lại toàn bộ kết quả

Mỗi lần chạy lưu `results/<mô hình>_<kịch bản>_<tập test>_seed<seed>.json` (độ đo, cấu hình, thời gian) và
`results/preds/….csv` (dự đoán từng cặp câu). Lần chạy nào đã có kết quả thì script bỏ qua (`--rerun` để chạy lại).

| Bước | Notebook (Colab) | Lệnh tương đương | Thiết bị, thời gian |
|---|---|---|---|
| Dữ liệu | `00_colab_setup.ipynb` | `python scripts/prepare_data.py && python scripts/segment_data.py` | CPU, ~20 phút (tách từ) |
| EDA | `01_eda.ipynb` | – | CPU |
| M0, M1, M1-h, ablation | `02_m1.ipynb` | – | CPU, ~10 phút |
| M2 | `03_m2.ipynb` | `python scripts/prepare_embeddings.py && python scripts/run_m2.py` | CPU, vài phút |
| M3 | `04_m3.ipynb` | `python scripts/run_m3.py` | GPU T4, ~10 phút |
| M4 PhoBERT | `05_phobert.ipynb`, `06_tuan4.ipynb` | `python scripts/run_transformer.py --model M4 --scenario S1 S2 S3 …` | GPU T4, ~2 giờ |
| M5 XLM-R | `06_tuan4.ipynb` | `python scripts/run_transformer.py --model M5 …` | GPU T4, ~1 giờ |
| M6 mDeBERTa | `07_m6.ipynb` | `python scripts/run_transformer.py --model M6 --scenario S0` rồi `--scenario S2 --config A B` | GPU T4, ~1,5 giờ |
| M7 LLM | `08_llm.ipynb` | `python scripts/run_llm.py --mode zero few --seeds 42 123 2024` | GPU T4, ~1,5 giờ |
| Tổng hợp | – | `python scripts/ensemble.py && python scripts/make_table.py && python scripts/plot_results.py && python scripts/error_analysis.py` | CPU, vài giây |
| Kiểm tra | – | `python scripts/verify_report.py` | CPU |

Kết quả đã kiểm tra tái lập: xem `docs/kiem_tra_tai_lap.md`.

## Kết quả chính (accuracy %, ± độ lệch chuẩn qua 3 seed)

| Mô hình | S1 → XNLI | S1 → ViANLI | S1 → ViNLI | S2 → ViANLI | S2 → ViNLI | S3 → ViANLI | S3 → ViNLI |
|---|---|---|---|---|---|---|---|
| M0 nhãn đa số | 33,3 | 33,3 | 32,6 | 33,3 | 34,3 | – | – |
| M1 TF-IDF + LR | 59,1 | 21,4 | 51,7 | 45,1 | 27,9 | – | – |
| M1-h chỉ hypothesis | 53,8 | 33,2 | 44,4 | 31,7 | 42,9 | – | – |
| M2 embedding + FFNN | 53,0 | 27,5 | 48,4 | 37,0 ± 2,5 | 33,7 ± 1,3 | – | – |
| M3 BiLSTM | 63,7 | 25,7 | 52,6 | 36,8 ± 2,8 | 38,5 ± 1,3 | – | – |
| M4 PhoBERT-base | 77,1 | 27,8 | 70,9 | 44,4 ± 1,6 | 54,4 ± 1,6 | 44,6 ± 0,5 | 69,9 ± 1,4 |
| M5 XLM-R-base | 75,2 | 27,0 | 67,8 | 43,6 | 32,5 | 43,5 | 63,3 |
| M6 mDeBERTa (S0 \| S2) | 79,3 | 38,7 | 74,1 | 47,7 ± 0,9 | 70,9 ± 0,5 | – | – |
| M7 Qwen2.5-7B (S0 \| FS) | 63,0 | 42,7 | 67,4 | 45,9 ± 0,2 | 72,7 ± 1,0 | – | – |

M6, M7: ba cột S1 là kết quả không huấn luyện trên dữ liệu của nhóm (S0); hai cột S2 là M6 fine-tune trên ViANLI
và M7 với 6 ví dụ trong prompt. Bảng đầy đủ (cả macro-F1): `results/summary_bang_acc.csv`, `results/summary_bang.csv`.
Phân tích: `docs/phan_tich_loi_transformer.md`, `docs/ket_qua_m6.md`, `docs/ket_qua_m7.md`.

## Giấy phép

Mã nguồn: MIT (xem `LICENSE`). Dữ liệu trích trong `data/manual/` giữ giấy phép gốc của từng bộ
(ViANLI: CC BY-NC-SA 4.0; ViNLI: chỉ dùng cho nghiên cứu), chỉ dùng cho mục đích học tập.

## Tài liệu tham khảo dữ liệu

- Conneau et al. (2018). XNLI: Evaluating Cross-lingual Sentence Representations. EMNLP.
- Huynh, Nguyen & Nguyen (2022). ViNLI: A Vietnamese Corpus for Studies on Open-Domain Natural Language Inference. COLING.
- Huynh, Nguyen & Nguyen (2025). A New Benchmark Dataset and Mixture-of-Experts Language Models for Adversarial
  Natural Language Inference in Vietnamese. Expert Systems with Applications.
