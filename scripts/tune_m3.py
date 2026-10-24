"""Chọn siêu tham số M3 trên tập DEV (không đụng tới test).

Chạy: python scripts/tune_m3.py --scenario S2
      python scripts/tune_m3.py --scenario S1 --grid freeze_embedding   # S1 chậm: chỉ thử một vài siêu tham số
In bảng macro-F1 dev, lưu results/analysis/m3_tuning_<kịch bản>.json. Cấu hình tốt nhất ghi vào m3.DEFAULT_CONFIG
hoặc m3.SCENARIO_CONFIG.
"""

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import m3  # noqa: E402
from src.data import LABEL2ID, load_scenario  # noqa: E402
from src.embeddings import load_embedding  # noqa: E402
from src.evaluate import RESULTS_DIR  # noqa: E402
from src.utils import save_json  # noqa: E402

GRID = {"hidden": [128, 256], "dropout": [0.3, 0.5], "freeze_embedding": [True, False]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S2", choices=["S1", "S2"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", nargs="*", default=list(GRID), help="chỉ thử các siêu tham số này, còn lại giữ mặc định")
    args = ap.parse_args()
    grid = {k: GRID[k] for k in args.grid}

    data = load_scenario(args.scenario, include_vinli=False)
    word2idx, matrix = load_embedding()
    base = m3.config_for(args.scenario)
    enc = {k: (m3.encode(data[k]["premise"], word2idx, base["max_len"]),
               m3.encode(data[k]["hypothesis"], word2idx, base["max_len"]),
               data[k]["label"].map(LABEL2ID).to_numpy()) for k in ("train", "dev")}

    rows = []
    for values in itertools.product(*grid.values()):
        cfg = {**base, **dict(zip(grid, values))}
        _, info = m3.train_m3(enc["train"], enc["dev"], matrix, cfg, args.seed, verbose=False)
        rows.append({**dict(zip(grid, values)), "best_epoch": info["best_epoch"],
                     "dev_macro_f1": info["dev_macro_f1"], "history": info["history"]})
        print({k: v for k, v in rows[-1].items() if k != "history"}, flush=True)

    table = pd.DataFrame(rows).sort_values("dev_macro_f1", ascending=False)
    print("\n", table.drop(columns="history").to_string(index=False))
    save_json({"scenario": args.scenario, "seed": args.seed, "base_config": base, "grid": grid,
               "results": table.to_dict("records")},
              RESULTS_DIR / "analysis" / f"m3_tuning_{args.scenario}.json")


if __name__ == "__main__":
    main()
