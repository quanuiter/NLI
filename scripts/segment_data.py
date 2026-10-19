"""Tách từ toàn bộ dữ liệu đã xử lý (dùng cho PhoBERT) và in mẫu để kiểm tra tay.

Chạy: python scripts/segment_data.py [--backend vncorenlp|underthesea] [--check 20]
Cần chạy scripts/prepare_data.py trước. Kết quả: data/processed/*_seg.jsonl
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATASET_SPLITS, load_split, processed_path, save_jsonl  # noqa: E402
from src.segment import Segmenter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="vncorenlp", choices=["vncorenlp", "underthesea"])
    ap.add_argument("--check", type=int, default=20, help="số câu in ra để kiểm tra tay")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    seg = Segmenter(args.backend)
    for dataset, splits in DATASET_SPLITS.items():
        for split in splits:
            subset = dataset == "xnli" and split == "train"
            out = processed_path(dataset, split, segmented=True, subset=subset)
            if out.exists() and not args.overwrite:
                print(f"Bỏ qua {out.name} (đã có, dùng --overwrite để làm lại)")
                continue
            df = load_split(dataset, split)
            start = time.time()
            # XNLI lặp lại mỗi premise 3 lần -> chỉ tách từ các câu khác nhau
            texts = sorted(set(df["premise"]) | set(df["hypothesis"]))
            cache = {t: seg(t) for t in texts}
            df["premise"] = df["premise"].map(cache)
            df["hypothesis"] = df["hypothesis"].map(cache)
            save_jsonl(df, out)
            print(f"{out.name:<32} {len(texts):>7} câu khác nhau  {time.time() - start:.0f}s")

    if args.check:
        print(f"\n=== {args.check} câu ngẫu nhiên từ ViANLI train để kiểm tra tay ===")
        raw = load_split("vianli", "train")
        segd = load_split("vianli", "train", segmented=True)
        for i in raw.sample(n=args.check, random_state=0).index:
            print(f"- {raw.at[i, 'hypothesis']}\n  {segd.at[i, 'hypothesis']}")


if __name__ == "__main__":
    main()
