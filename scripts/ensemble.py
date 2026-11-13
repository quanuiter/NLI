"""Kết hợp nhiều mô hình bằng bỏ phiếu đa số (task W4-11).

Chạy: python scripts/ensemble.py        (không cần GPU, chỉ đọc results/preds/*.csv)
Ra:   results/E<k>_<kịch bản>_<tập test>_seed0.json + results/preds/... (cùng định dạng các mô hình khác, seed 0 = không áp dụng)
      results/analysis/ensemble.json    (thành viên, cách phá hòa, so với từng thành viên)

Thành phần mỗi ensemble được cố định TRƯỚC khi nhìn kết quả test (danh sách ENSEMBLES dưới đây), không chọn theo test.
Bỏ phiếu 2 tầng:
  1. Trong một mô hình chạy nhiều seed: nhãn được nhiều seed đoán nhất; hòa thì theo seed có macro-F1 dev cao nhất.
  2. Giữa các mô hình: nhãn được nhiều mô hình đoán nhất; hòa thì theo mô hình có macro-F1 dev (trung bình seed) cao nhất.
Chỉ dùng macro-F1 dev (đã lưu trong file kết quả) để phá hòa, nên không dùng thông tin của tập test.
Ensemble nào còn thiếu thành viên (ví dụ chưa chạy M6) thì bỏ qua, chạy lại script khi đủ.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.evaluate import RESULTS_DIR, evaluate_and_save
from src.utils import load_json, save_json

TEST_SETS = ["vianli_test", "vinli_test"]
# (mã, mô tả, kịch bản ghi vào bảng, thành viên (mô hình, kịch bản))
ENSEMBLES = [
    ("E1", "PhoBERT S2, bỏ phiếu 3 seed", "S2", [("M4", "S2")]),
    ("E2", "PhoBERT S3, bỏ phiếu 3 seed", "S3", [("M4", "S3")]),
    ("E3", "PhoBERT S2 + XLM-R S2 + M6 S2 (chỉ học ViANLI ở bước cuối)", "S2", [("M4", "S2"), ("M5", "S2"), ("M6", "S2")]),
    ("E4", "PhoBERT S3 + XLM-R S3 + M6 S2 (đều học NLI trước rồi học ViANLI)", "S3",
     [("M4", "S3"), ("M5", "S3"), ("M6", "S2")]),
]


def seed_of(path: Path) -> int:
    return int(path.stem.rsplit("seed", 1)[1])


def vote(preds: list[np.ndarray], priority: list[float]) -> np.ndarray:
    """Bỏ phiếu đa số theo từng mẫu; hòa thì lấy dự đoán của người có priority cao nhất trong số các nhãn hòa."""
    order = np.argsort(priority)[::-1]
    out = []
    for votes in zip(*preds):
        counts = Counter(votes)
        top = max(counts.values())
        tied = {label for label, c in counts.items() if c == top}
        out.append(next(votes[i] for i in order if votes[i] in tied))
    return np.array(out)


def member(model: str, scenario: str, test_set: str):
    """Dự đoán đã gộp seed của một mô hình, macro-F1 dev trung bình, accuracy test trung bình; None nếu chưa có."""
    files = sorted((RESULTS_DIR / "preds").glob(f"{model}_{scenario}_{test_set}_seed*.csv"), key=seed_of)
    if not files:
        return None
    frames = [pd.read_csv(f).set_index("id") for f in files]
    devs = [load_json(RESULTS_DIR / f"{f.stem}.json")["config"].get("dev_macro_f1", 0.0) for f in files]
    ids = frames[0].index
    pred = vote([fr.loc[ids, "pred"].values for fr in frames], devs)
    return {"ids": ids, "frame": frames[0], "pred": pred, "dev": float(np.mean(devs)), "seeds": [seed_of(f) for f in files],
            "test_acc_mean": float(np.mean([fr["correct"].mean() for fr in frames]))}


def main():
    report = []
    for code, desc, scenario, members in ENSEMBLES:
        for test_set in TEST_SETS:
            parts = {f"{m} {s}": member(m, s, test_set) for m, s in members}
            missing = [name for name, p in parts.items() if p is None]
            if missing:
                print(f"Bỏ qua {code} trên {test_set}: chưa có {', '.join(missing)}")
                continue
            first = next(iter(parts.values()))
            ids = first["ids"]
            preds = [pd.Series(p["pred"], index=p["ids"]).loc[ids].values for p in parts.values()]
            final = preds[0] if len(preds) == 1 else vote(preds, [p["dev"] for p in parts.values()])
            df = first["frame"].loc[ids]
            res = evaluate_and_save(df["label"], final, model=code, scenario=scenario, test_set=test_set, seed=0,
                                    config={"ensemble": desc, "members": {n: p["seeds"] for n, p in parts.items()},
                                            "rule": "bỏ phiếu đa số 2 tầng, hòa theo macro-F1 dev"},
                                    ids=ids, premise=df["premise"], hypothesis=df["hypothesis"], verbose=False)
            best_single = max(p["test_acc_mean"] for p in parts.values())
            report.append({"ensemble": code, "mo_ta": desc, "test_set": test_set,
                           "accuracy": round(res["accuracy"], 4), "macro_f1": round(res["macro_f1"], 4),
                           "thanh_vien": {n: {"seeds": p["seeds"], "dev_macro_f1": round(p["dev"], 4),
                                              "test_accuracy_tb_seed": round(p["test_acc_mean"], 4)}
                                          for n, p in parts.items()},
                           "hon_thanh_vien_tot_nhat": round(res["accuracy"] - best_single, 4)})
            print(f"{code} {test_set}: accuracy {res['accuracy'] * 100:.1f}, macro-F1 {res['macro_f1'] * 100:.1f} "
                  f"(thành viên tốt nhất {best_single * 100:.1f})  — {desc}")
    save_json({"note": __doc__.split("Thành phần")[1].strip(), "results": report},
              RESULTS_DIR / "analysis" / "ensemble.json")


if __name__ == "__main__":
    main()
