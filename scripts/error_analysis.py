"""Phân tích lỗi các mô hình trên ViANLI test và ViNLI test (task W5-02; dùng lại được cho W4-08).

Chạy: python scripts/error_analysis.py
Đọc results/preds/*.csv của mọi mô hình đã có (M1 tự được thêm khi có kết quả), ghi ra:
    results/analysis/loi_nhom_cau.csv       accuracy theo nhóm câu (gắn thẻ tự động bằng luật) × mô hình
    results/analysis/loi_theo_nhan.csv      F1 theo nhãn × mô hình
    results/analysis/loi_mau_<loại>.csv     danh sách mẫu để diễn giải ở mục 5 báo cáo
    results/figures/loi_confusion_transformer.png
Mô hình có nhiều seed: accuracy trung bình qua các seed; "đúng" trong danh sách mẫu nghĩa là đúng ở mọi seed.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data import LABELS, load_split
from src.embeddings import tokenize
from src.evaluate import RESULTS_DIR
from src.utils import load_json

ANALYSIS = RESULTS_DIR / "analysis"
TEST_SETS = ["vianli_test", "vinli_test"]
# (tên hiển thị, mô hình, kịch bản)
RUNS = [("M1 S2", "M1", "S2"), ("M2 S2", "M2", "S2"), ("M3 S2", "M3", "S2"),
        ("PhoBERT S1", "M4", "S1"), ("PhoBERT S2", "M4", "S2"), ("PhoBERT S3", "M4", "S3"),
        ("XLM-R S1", "M5", "S1"), ("XLM-R S2", "M5", "S2"), ("XLM-R S3", "M5", "S3"),
        ("M6 S0", "M6", "S0"), ("M6 S2", "M6", "S2")]   # M6 tự được thêm khi có kết quả

NEG = re.compile(r"\b(?:không|chưa|chẳng|chả|đâu có)\b", re.IGNORECASE)
QUANT = re.compile(r"\b(?:tất cả|mọi|hầu hết|đa số|một số|chỉ|duy nhất|nhiều|ít|toàn bộ)\b", re.IGNORECASE)
DIGIT = re.compile(r"\d")


def jaccard(p: str, h: str) -> float:
    a, b = {t.lower() for t in tokenize(p)}, {t.lower() for t in tokenize(h)}
    return len(a & b) / max(1, len(a | b))


def tag(df: pd.DataFrame) -> pd.DataFrame:
    """Gắn thẻ hiện tượng bằng luật (không thay được phân tích thủ công, chỉ để đếm trên cả tập test)."""
    out = df.copy()
    out["jaccard"] = [jaccard(p, h) for p, h in zip(df["premise"], df["hypothesis"])]
    out["phu_dinh_H"] = df["hypothesis"].str.contains(NEG)
    out["phu_dinh_chi_P"] = df["premise"].str.contains(NEG) & ~out["phu_dinh_H"]
    out["so_trong_H"] = df["hypothesis"].str.contains(DIGIT)
    out["tu_chi_luong_H"] = df["hypothesis"].str.contains(QUANT)
    out["trung_tu_cao"] = out["jaccard"] >= 0.4
    out["trung_tu_thap"] = out["jaccard"] < 0.15
    return out


GROUPS = {"tất cả": None, "phủ định trong H": "phu_dinh_H", "phủ định chỉ ở P": "phu_dinh_chi_P",
          "có số trong H": "so_trong_H", "từ chỉ lượng trong H": "tu_chi_luong_H",
          "trùng từ cao (J ≥ 0,4)": "trung_tu_cao", "trùng từ thấp (J < 0,15)": "trung_tu_thap"}


def load_correct(model: str, scenario: str, test_set: str, ids: pd.Index):
    """Ma trận đúng/sai (hàng = mẫu, cột = seed) và dự đoán của seed nhỏ nhất (42); None nếu chưa có kết quả."""
    files = sorted((RESULTS_DIR / "preds").glob(f"{model}_{scenario}_{test_set}_seed*.csv"),
                   key=lambda f: int(f.stem.rsplit("seed", 1)[1]))   # seed 42 trước 123, 2024
    if not files:
        return None, None
    cols, first_pred = {}, None
    for f in files:
        p = pd.read_csv(f).set_index("id").loc[ids]
        cols[f.stem.rsplit("seed", 1)[1]] = p["correct"].astype(bool).values
        if first_pred is None:
            first_pred = p["pred"].values
    return pd.DataFrame(cols, index=ids), first_pred


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    group_rows, label_rows, all_tables = [], [], {}
    for test_set in TEST_SETS:
        df = tag(load_split(test_set.replace("_test", ""), "test")).set_index("id")
        correct, preds = {}, {}
        for name, model, scenario in RUNS:
            c, p = load_correct(model, scenario, test_set, df.index)
            if c is not None:
                correct[name], preds[name] = c, p
        all_tables[test_set] = (df, correct, preds)
        for gname, col in GROUPS.items():
            mask = np.ones(len(df), bool) if col is None else df[col].values
            row = {"tap_test": test_set, "nhom": gname, "so_mau": int(mask.sum()),
                   "nhan_E_N_C": "/".join(str(int((df["label"].values[mask] == l).sum())) for l in LABELS)}
            for name, c in correct.items():
                row[name] = round(float(c.values[mask].mean()) * 100, 1)
            group_rows.append(row)
        for name, c in correct.items():
            model, scenario = next((m, s) for n, m, s in RUNS if n == name)
            seeds = sorted((RESULTS_DIR).glob(f"{model}_{scenario}_{test_set}_seed*.json"))
            f1 = np.mean([[load_json(s)["per_class"][l]["f1"] for l in LABELS] for s in seeds], axis=0)
            label_rows.append({"tap_test": test_set, "mo_hinh": name, "so_seed": len(seeds),
                               **{f"F1_{l}": round(v * 100, 1) for l, v in zip(LABELS, f1)}})

    groups = pd.DataFrame(group_rows)
    groups.to_csv(ANALYSIS / "loi_nhom_cau.csv", index=False, encoding="utf-8-sig")
    labels = pd.DataFrame(label_rows)
    labels.to_csv(ANALYSIS / "loi_theo_nhan.csv", index=False, encoding="utf-8-sig")
    print(groups.to_string(index=False), "\n")
    print(labels.to_string(index=False), "\n")

    # Danh sách mẫu trên ViANLI test ("đúng"/"sai" = ở mọi seed)
    df, correct, preds = all_tables["vianli_test"]
    always = {n: c.all(axis=1) for n, c in correct.items()}
    never = {n: ~c.any(axis=1) for n, c in correct.items()}
    classic = [n for n in ("M1 S2", "M2 S2", "M3 S2") if n in correct]
    show = ["premise", "hypothesis", "label", "jaccard", "phu_dinh_H", "so_trong_H", "tu_chi_luong_H"]

    def save(kind: str, mask: pd.Series, n: int = 15):
        out = df.loc[mask, show].copy()
        for name in correct:
            out[f"du_doan_{name}"] = pd.Series(preds[name], index=df.index)[mask]
        out = out.assign(do_dai=out["premise"].str.len() + out["hypothesis"].str.len()).sort_values("do_dai")
        out.drop(columns="do_dai").head(n).to_csv(ANALYSIS / f"loi_mau_{kind}.csv", encoding="utf-8-sig")
        print(f"{kind}: {int(mask.sum())} mẫu")

    classic_wrong = np.logical_and.reduce([never[n] for n in classic])
    classic_right = np.logical_or.reduce([always[n] for n in classic])
    save("co_dien_sai_phobert_dung", classic_wrong & always["PhoBERT S2"])
    save("phobert_sai_co_dien_dung", never["PhoBERT S2"] & classic_right)
    save("s3_dung_s2_sai", always["PhoBERT S3"] & never["PhoBERT S2"])
    save("s2_dung_s3_sai", always["PhoBERT S2"] & never["PhoBERT S3"])
    save("phobert_dung_xlmr_sai", always["PhoBERT S2"] & never["XLM-R S2"])
    save("xlmr_dung_phobert_sai", always["XLM-R S2"] & never["PhoBERT S2"])
    save("moi_mo_hinh_deu_sai", np.logical_and.reduce([never[n] for n in correct]))

    # Confusion matrix của PhoBERT và XLM-R (seed 42) trên 2 tập test
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in ("PhoBERT S1", "PhoBERT S2", "PhoBERT S3", "XLM-R S1", "XLM-R S2", "XLM-R S3")]
    fig, axes = plt.subplots(2, len(names), figsize=(3.1 * len(names), 6.4))
    short = ["E", "N", "C"]
    for i, test_set in enumerate(TEST_SETS):
        df, _, preds = all_tables[test_set]
        for j, name in enumerate(names):
            ax = axes[i, j]
            if name not in preds:
                ax.axis("off")
                continue
            m = pd.crosstab(pd.Categorical(df["label"], LABELS), pd.Categorical(preds[name], LABELS),
                            dropna=False).values
            ax.imshow(m, cmap="Blues")
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, m[a, b], ha="center", va="center", color="white" if m[a, b] > m.max() / 2 else "black")
            ax.set_xticks(range(3), short)
            ax.set_yticks(range(3), short)
            ax.set_title(f"{name} · {test_set.replace('_test', '')}", fontsize=10)
            if j == 0:
                ax.set_ylabel("Nhãn thật")
            if i == 1:
                ax.set_xlabel("Dự đoán")
    fig.suptitle("Confusion matrix (seed 42). E = entailment, N = neutral, C = contradiction", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "loi_confusion_transformer.png", dpi=150)
    print("Đã lưu hình results/figures/loi_confusion_transformer.png")


if __name__ == "__main__":
    main()
