"""Tổng hợp mọi file results/*.json thành bảng kết quả cho báo cáo.

Chạy:  python scripts/make_table.py
Ra:    results/summary.csv           (mô hình × kịch bản × tập kiểm tra: trung bình, độ lệch chuẩn qua các seed)
       results/summary_bang.csv      (bảng macro-F1 (%) định dạng sẵn để đưa vào báo cáo)
       results/summary_bang_acc.csv  (bảng accuracy (%) cùng định dạng)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # để import được src/ khi chạy từ thư mục gốc repo

from src.evaluate import RESULTS_DIR, summarize

THU_TU_MO_HINH = ["M0", "M1", "M1h", "M2", "M3", "M4", "M5"]
THU_TU_COT = [("S1", "xnli_test"), ("S1", "vianli_test"), ("S1", "vinli_test"),
              ("S2", "vianli_test"), ("S2", "vinli_test"),
              ("S3", "vianli_test"), ("S3", "vinli_test")]


def dinh_dang(mean, std, n):
    """0.4431, 0.0052, 3 -> '44,3 ± 0,5'; chỉ có 1 seed -> '44,3'."""
    s = f"{mean * 100:.1f}"
    if n > 1:
        s += f" ± {std * 100:.1f}"
    return s.replace(".", ",")


def xoay_bang(bang, cot_mean, cot_std, cot_count):
    """Xoay bảng dài thành bảng rộng: dòng = mô hình, cột = (kịch bản · tập kiểm tra), ô = 'trung bình ± độ lệch chuẩn'.
    Dùng chung cho bảng macro-F1 và bảng accuracy, chỉ khác tên 3 cột đầu vào."""
    bang = bang.copy()
    bang["o"] = [dinh_dang(m, s, n) for m, s, n in zip(bang[cot_mean], bang[cot_std], bang[cot_count])]
    rong = bang.pivot_table(index="model", columns=["scenario", "test_set"], values="o", aggfunc="first")
    cot = [c for c in THU_TU_COT if c in rong.columns]
    dong = [m for m in THU_TU_MO_HINH if m in rong.index]
    rong = rong.reindex(index=dong, columns=cot).fillna("–")          # ô chưa có kết quả hiện '–'
    rong.columns = [f"{s} · {t.replace('_test', '')}" for s, t in rong.columns]
    return rong


def main():
    bang = summarize()
    if bang.empty:
        print("Chưa có file kết quả nào trong", RESULTS_DIR)
        return
    bang.to_csv(RESULTS_DIR / "summary.csv", index=False)

    for do_do, ten_file in [("macro_f1", "summary_bang.csv"), ("accuracy", "summary_bang_acc.csv")]:
        rong = xoay_bang(bang, f"{do_do}_mean", f"{do_do}_std", f"{do_do}_count")
        rong.to_csv(RESULTS_DIR / ten_file, encoding="utf-8-sig")       # utf-8-sig: Excel đọc đúng tiếng Việt
        print(f"\n{do_do} (%) trên tập kiểm tra, trung bình ± độ lệch chuẩn qua các seed:")
        print(rong.to_string())


if __name__ == "__main__":
    main()
