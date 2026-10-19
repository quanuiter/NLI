"""Chạy M2 (embedding trung bình + FFNN).

Ví dụ:
    python scripts/run_m2.py --scenario S2                     # 1 seed
    python scripts/run_m2.py --scenario S2 S1 --seeds 42 123 2024
    python scripts/run_m2.py --scenario S2 --hidden 256 --layers 2   # thử cấu hình khác (chọn theo dev)
Cần chạy scripts/prepare_embeddings.py trước. Kết quả: results/M2_*.json, figures/, preds/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import m2  # noqa: E402
from src.evaluate import summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", nargs="+", default=["S2"], choices=["S1", "S2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    for key, value in m2.DEFAULT_CONFIG.items():
        if key != "embedding":
            ap.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in m2.DEFAULT_CONFIG if k != "embedding"}

    for scenario in args.scenario:
        for seed in args.seeds:
            m2.run(scenario, seed, cfg)

    table = summarize()
    print("\n", table[table["model"] == "M2"].to_string(index=False))


if __name__ == "__main__":
    main()
