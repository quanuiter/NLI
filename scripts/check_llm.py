"""Kiểm tra kết quả M7 (W4-12). Chạy bằng lệnh riêng (!python ...) để luôn thấy file vừa ghi vào Drive.

    python scripts/check_llm.py --smoke     # sau bước chạy thử
    python scripts/check_llm.py             # sau bước 3: đủ 9 file chưa
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import RESULTS_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

if args.smoke:
    files = sorted((RESULTS_DIR / "smoke").glob("M7_*.json"))
    if not files:
        print("CHƯA CÓ KẾT QUẢ CHẠY THỬ: ô chạy thử bị lỗi. Kéo lên xem dòng lỗi cuối, gửi cho Quân.")
    for f in files:
        r = json.load(open(f)); c = r["config"]
        one_label = max(c["pred_distribution"].values()) == r["n"]
        print(f"{r['scenario']} {r['test_set']}: n={r['n']}, accuracy={r['accuracy']:.3f}, dự đoán={c['pred_distribution']}, "
              f"4-bit={c['load_in_4bit']}, thiết bị={c['device']}" + ("  ← TOÀN MỘT NHÃN, nhắn Quân" if one_label else ""))
    if len(files) >= 5:
        print("OK, chạy tiếp bước 2")
else:
    need = ([f"M7_S0_{t}_seed42.json" for t in ["xnli_test", "vianli_test", "vinli_test"]] +
            [f"M7_FS_{t}_seed{s}.json" for t in ["vianli_test", "vinli_test"] for s in [42, 123, 2024]])
    missing = [f for f in need if not (RESULTS_DIR / f).exists()]
    print("THIẾU:", missing) if missing else print("Đủ 9 file kết quả. Xong W4-12.")
