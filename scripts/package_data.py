"""Đóng gói dữ liệu nộp kèm đồ án (task W6-04).

Chạy: python scripts/package_data.py
Ra:   dist/data_NLI.zip gồm README.md (data/README.md), processed/*.jsonl, manual/*.
Thiếu data/processed thì chạy python scripts/prepare_data.py trước.
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPECTED = ["xnli_train_100k", "xnli_dev", "xnli_test", "vianli_train", "vianli_dev", "vianli_test", "vinli_test"]


def main():
    missing = [n for n in EXPECTED if not (DATA / "processed" / f"{n}.jsonl").exists()]
    if missing:
        sys.exit(f"Thiếu data/processed: {missing}. Chạy python scripts/prepare_data.py trước.")
    out = ROOT / "dist" / "data_NLI.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(DATA / "README.md", "data_NLI/README.md")
        for f in sorted((DATA / "processed").glob("*.jsonl")):
            z.write(f, f"data_NLI/processed/{f.name}")
        for f in sorted((DATA / "manual").iterdir()):
            if f.is_file():
                z.write(f, f"data_NLI/manual/{f.name}")
        names = z.namelist()
    print(f"Đã tạo {out} ({out.stat().st_size / 1e6:.1f} MB, {len(names)} file)")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()
