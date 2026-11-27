"""Đối chiếu số liệu trong bảng kết quả của báo cáo với results/ (task W6-03).

Chạy: python scripts/verify_report.py [docs/bao_cao_muc4.md]
Đọc bảng "Accuracy / macro-F1" (dòng bắt đầu bằng "| M0", "| M1", ..., "| E4") và so từng ô "acc / F1"
(kể cả "± độ lệch chuẩn") với summarize() làm tròn 1 chữ số. In mọi ô lệch; mã thoát 1 nếu có ô lệch.
Quy ước của bảng: M6, M7 ghi kết quả S0 (không huấn luyện) ở cột S1 kèm "*"; M7 few-shot ghi ở cột S2 kèm "†".
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import summarize  # noqa: E402

COLS = [("S1", "xnli_test"), ("S1", "vianli_test"), ("S1", "vinli_test"), ("S2", "vianli_test"), ("S2", "vinli_test"),
        ("S3", "vianli_test"), ("S3", "vinli_test")]
ROWS = [("M0", "| M0"), ("M1", "| M1 "), ("M1h", "| M1-h"), ("M2", "| M2"), ("M3", "| M3"), ("M4", "| M4"),
        ("M5", "| M5"), ("M6", "| M6"), ("M7", "| M7"), ("E3", "| E3"), ("E4", "| E4")]
NUM = r"(\d+,\d)(?:\s*±\s*(\d+,\d))?"


def fmt(x):
    return f"{x * 100:.1f}".replace(".", ",")


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/bao_cao_muc4.md")
    lines = path.read_text(encoding="utf-8").splitlines()
    t = summarize().set_index(["model", "scenario", "test_set"])
    checked, bad = 0, []
    for model, prefix in ROWS:
        line = next((l for l in lines if l.startswith(prefix) and l.count("|") >= 9), None)
        if line is None:
            bad.append(f"{model}: không thấy dòng trong bảng")
            continue
        cells = [c.strip() for c in line.strip("|").split("|")[1:8]]
        for (sc, ts), cell in zip(COLS, cells):
            if cell in ("–", "-", ""):
                continue
            if model in ("M6", "M7") and sc == "S1":
                sc = "S0"
            if model == "M7" and sc == "S2":
                sc = "FS"
            m = re.findall(NUM, cell.replace("*", "").replace("†", ""))
            key = (model, sc, ts)
            if len(m) != 2 or key not in t.index:
                bad.append(f"{model} {sc} {ts}: ô '{cell}' không đọc được hoặc không có kết quả")
                continue
            r = t.loc[key]
            want = [(fmt(r["accuracy_mean"]), fmt(r["accuracy_std"]) if r["accuracy_count"] > 1 else ""),
                    (fmt(r["macro_f1_mean"]), fmt(r["macro_f1_std"]) if r["macro_f1_count"] > 1 else "")]
            for (got_mean, got_std), (w_mean, w_std), name in zip(m, want, ["accuracy", "macro-F1"]):
                checked += 1
                if got_mean != w_mean or got_std != w_std:
                    bad.append(f"{model} {sc} {ts} {name}: báo cáo {got_mean} ± {got_std or '–'}, results {w_mean} ± {w_std or '–'}")
    print(f"Đã đối chiếu {checked} số trong {path}.")
    for b in bad:
        print("LỆCH:", b)
    print("Khớp hết." if not bad else f"{len(bad)} chỗ lệch.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
