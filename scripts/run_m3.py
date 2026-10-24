"""Chạy M3 (BiLSTM + max-pooling).

Ví dụ:
    python scripts/run_m3.py --scenario S2                     # 1 seed
    python scripts/run_m3.py --scenario S2 S1 --seeds 42 123 2024
Cần chạy scripts/prepare_embeddings.py trước. Kết quả: results/M3_*.json, figures/, preds/, checkpoints/M3_*.pt
Cấu hình lấy từ m3.config_for(kịch bản) (đã chọn trên dev bằng scripts/tune_m3.py).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import m3  # noqa: E402
from src.evaluate import summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", nargs="+", default=["S2"], choices=["S1", "S2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = ap.parse_args()

    for scenario in args.scenario:
        for seed in args.seeds:
            m3.run(scenario, seed)

    table = summarize()
    print("\n", table[table["model"] == "M3"].to_string(index=False))


if __name__ == "__main__":
    main()
