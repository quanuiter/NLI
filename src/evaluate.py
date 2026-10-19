"""Đánh giá và lưu kết quả theo một định dạng chung cho mọi mô hình.

Cách dùng:
    from src.evaluate import evaluate_and_save
    evaluate_and_save(y_true, y_pred, model="M1", scenario="S1", test_set="vianli_test",
                      seed=42, config={"C": 1.0}, ids=test_df["id"])

Mỗi lần gọi tạo ra:
    results/<model>_<scenario>_<test_set>_seed<seed>.json   độ đo + cấu hình
    results/figures/<cùng tên>.png                         confusion matrix
    results/preds/<cùng tên>.csv                           dự đoán từng mẫu (cho phân tích lỗi)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .data import ID2LABEL, LABELS, ROOT
from .utils import load_json, save_json

RESULTS_DIR = ROOT / "results"


def _to_labels(values) -> list[str]:
    out = []
    for v in values:
        if isinstance(v, str):
            out.append(v)
        else:
            out.append(ID2LABEL[int(v)])
    return out


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy, macro-F1, precision/recall/F1 từng nhãn và confusion matrix (hàng = nhãn thật)."""
    y_true, y_pred = _to_labels(y_true), _to_labels(y_pred)
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "per_class": {
            label: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(support[i])}
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": {"labels": LABELS, "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()},
    }


def run_name(model: str, scenario: str, test_set: str, seed: int) -> str:
    return f"{model}_{scenario}_{test_set}_seed{seed}"


def plot_confusion_matrix(matrix, title: str, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=20)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title(title, fontsize=10)
    vmax = max(max(row) for row in matrix) or 1
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", color="white" if v > vmax / 2 else "black")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_and_save(y_true, y_pred, model: str, scenario: str, test_set: str, seed: int = 42,
                      config: dict | None = None, ids=None, premise=None, hypothesis=None,
                      out_dir=RESULTS_DIR, verbose: bool = True) -> dict:
    """Tính độ đo, lưu JSON + confusion matrix + file dự đoán. Trả về dict kết quả.

    model: "M0".."M5" (hoặc "M1h"); scenario: "S1" | "S2" | "S3";
    test_set: "xnli_test" | "vianli_test"; config: siêu tham số, thời gian chạy...
    """
    out_dir = Path(out_dir)
    name = run_name(model, scenario, test_set, seed)
    metrics = compute_metrics(y_true, y_pred)
    result = {
        "model": model, "scenario": scenario, "test_set": test_set, "seed": seed,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "config": config or {},
        **metrics,
    }
    save_json(result, out_dir / f"{name}.json")
    plot_confusion_matrix(metrics["confusion_matrix"]["matrix"], f"{model} · {scenario} · {test_set}",
                          out_dir / "figures" / f"{name}.png")

    preds = pd.DataFrame({"label": _to_labels(y_true), "pred": _to_labels(y_pred)})
    if ids is not None:
        preds.insert(0, "id", list(ids))
    if premise is not None:
        preds["premise"] = list(premise)
    if hypothesis is not None:
        preds["hypothesis"] = list(hypothesis)
    preds["correct"] = preds["label"] == preds["pred"]
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / "preds" / f"{name}.csv", index=False, encoding="utf-8")

    if verbose:
        print(f"[{name}] accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}")
    return result


def load_results(out_dir=RESULTS_DIR) -> pd.DataFrame:
    """Gom mọi file results/*.json thành một bảng (mỗi dòng một lần chạy)."""
    rows = []
    for path in sorted(Path(out_dir).glob("*.json")):
        r = load_json(path)
        row = {k: r[k] for k in ("model", "scenario", "test_set", "seed", "n", "accuracy", "macro_f1")}
        for label in LABELS:
            row[f"f1_{label}"] = r["per_class"][label]["f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(out_dir=RESULTS_DIR) -> pd.DataFrame:
    """Trung bình ± độ lệch chuẩn qua các seed, theo (model, scenario, test_set)."""
    df = load_results(out_dir)
    if df.empty:
        return df
    g = df.groupby(["model", "scenario", "test_set"])
    out = g[["accuracy", "macro_f1"]].agg(["mean", "std", "count"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()
