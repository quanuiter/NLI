"""Kiểm tra kết quả các task bổ sung tuần 7. Chạy bằng lệnh riêng (!python ...) để thấy file vừa ghi vào Drive.

    python scripts/check_w7.py --task 01     # W7-01: PhoBERT khử mẹo (M4P) S1 3 seed + PhoBERT S1 seed 123, 2024
    python scripts/check_w7.py --task 02     # W7-02: thêm seed M2, M3 (S1), XLM-R (S1, S2, S3)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import RESULTS_DIR  # noqa: E402

S1 = ["xnli_test", "vianli_test", "vinli_test"]
S2 = ["vianli_test", "vinli_test"]
NEED = {
    "01": [("M4P", "S1", t, s) for t in S1 for s in (42, 123, 2024)] + [("M4", "S1", t, s) for t in S1 for s in (123, 2024)],
    "02": [(m, "S1", t, s) for m in ("M2", "M3", "M5") for t in S1 for s in (123, 2024)]
          + [("M5", sc, t, s) for sc in ("S2", "S3") for t in S2 for s in (123, 2024)],
}

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=list(NEED), required=True)
args = ap.parse_args()
need = NEED[args.task]
missing, rows = [], []
for m, sc, t, s in need:
    f = RESULTS_DIR / f"{m}_{sc}_{t}_seed{s}.json"
    if not f.exists():
        missing.append(f.name)
        continue
    r = json.load(open(f))
    rows.append(f"{f.stem}: accuracy {r['accuracy'] * 100:.1f}, macro-F1 {r['macro_f1'] * 100:.1f}")
print("\n".join(rows))
print(f"THIẾU {len(missing)} file:", missing) if missing else print(f"Đủ {len(need)} file kết quả. Xong W7-{args.task}.")
