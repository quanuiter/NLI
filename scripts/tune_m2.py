"""Chọn siêu tham số M2 trên tập DEV (không đụng tới test).

Chạy: python scripts/tune_m2.py --scenario S2
Thử lưới nhỏ (số lớp ẩn, kích thước, dropout, learning rate), in bảng macro-F1 dev,
lưu results/analysis/m2_tuning_<kịch bản>.json. Cấu hình tốt nhất dùng cho scripts/run_m2.py.
"""

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import m2  # noqa: E402
from src.data import LABEL2ID, load_scenario  # noqa: E402
from src.embeddings import load_embedding  # noqa: E402
from src.evaluate import RESULTS_DIR  # noqa: E402
from src.utils import save_json  # noqa: E402

GRID = {"layers": [1, 2], "hidden": [256, 512], "dropout": [0.3, 0.5], "lr": [1e-3, 3e-4]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S2", choices=["S1", "S2"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = load_scenario(args.scenario, include_vinli=False)
    word2idx, matrix = load_embedding()
    X = {k: m2.pair_features(data[k]["premise"].tolist(), data[k]["hypothesis"].tolist(), word2idx, matrix)
         for k in ("train", "dev")}
    y = {k: data[k]["label"].map(LABEL2ID).to_numpy() for k in ("train", "dev")}

    rows = []
    for values in itertools.product(*GRID.values()):
        cfg = {**m2.DEFAULT_CONFIG, **dict(zip(GRID, values))}
        _, _, info = m2.train_ffnn(X["train"], y["train"], X["dev"], y["dev"], cfg, args.seed, verbose=False)
        rows.append({**dict(zip(GRID, values)), "best_epoch": info["best_epoch"], "dev_macro_f1": info["dev_macro_f1"]})
        print(rows[-1])

    table = pd.DataFrame(rows).sort_values("dev_macro_f1", ascending=False)
    print("\n", table.to_string(index=False))
    save_json({"scenario": args.scenario, "seed": args.seed, "grid": GRID, "results": table.to_dict("records")},
              RESULTS_DIR / "analysis" / f"m2_tuning_{args.scenario}.json")


if __name__ == "__main__":
    main()
