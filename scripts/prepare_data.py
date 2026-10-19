"""Tải XNLI-vi + ViANLI + ViNLI test, chuẩn hóa, lấy tập con XNLI 100k và lưu vào data/processed/.

Chạy: python scripts/prepare_data.py
Kết quả:
    data/processed/xnli_train_100k.jsonl, xnli_dev.jsonl, xnli_test.jsonl
    data/processed/vianli_train.jsonl, vianli_dev.jsonl, vianli_test.jsonl
    data/processed/vinli_test.jsonl   (ViNLI bỏ nhãn "other", chỉ dùng để đánh giá)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATASET_SPLITS, load_split, processed_path  # noqa: E402


def main():
    for dataset, splits in DATASET_SPLITS.items():
        for split in splits:
            df = load_split(dataset, split)
            subset = dataset == "xnli" and split == "train"
            dist = df["label"].value_counts().reindex(["entailment", "neutral", "contradiction"]).tolist()
            print(f"{processed_path(dataset, split, subset=subset).name:<28} {len(df):>7} mẫu  "
                  f"entailment/neutral/contradiction = {dist}")


if __name__ == "__main__":
    main()
