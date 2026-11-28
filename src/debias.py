"""Khử mẹo bề mặt khi fine-tune bằng product of experts (PoE) – task W7-01.

Ý tưởng (Clark et al., 2019; He et al., 2019): huấn luyện trước một "mô hình mẹo" chỉ nhìn các đặc trưng bề mặt
(tỉ lệ từ trùng, từ phủ định, chênh lệch độ dài, số mới – đúng 5 đặc trưng so sánh của M1). Khi fine-tune mô hình chính,
xác suất của hai mô hình được nhân với nhau (cộng log-xác suất) rồi mới tính cross-entropy:

    loss = CE( log_softmax(logits_chính) + log p_mẹo , nhãn )

Mẫu nào mô hình mẹo đã đoán đúng với xác suất cao thì đóng góp ít vào gradient của mô hình chính, nên mô hình chính
phải học từ những mẫu mà mẹo không giải được. Khi dự đoán chỉ dùng mô hình chính (bỏ mô hình mẹo).
Mô hình mẹo chỉ học trên tập train của kịch bản; không dùng dev/test.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .data import LABELS
from .embeddings import tokenize

NEG = re.compile(r"\b(không|chưa|chẳng|chả|đâu có)\b", re.IGNORECASE)
FEATURES = ["overlap", "neg_p", "neg_h", "do_dai", "so_moi"]


def compare_features(df: pd.DataFrame) -> np.ndarray:
    """5 đặc trưng so sánh của M1 (giống hệt notebooks/02_m1.ipynb), tính trên văn bản thô (chưa tách từ)."""
    rows = []
    for p, h in zip(df["premise"], df["hypothesis"]):
        tp, th = set(t.lower() for t in tokenize(p)), set(t.lower() for t in tokenize(h))
        overlap = len(tp & th) / max(1, len(th))
        neg_p, neg_h = len(NEG.findall(p)), len(NEG.findall(h))
        do_dai = abs(len(tokenize(p)) - len(tokenize(h))) / 50
        so_moi = 1.0 if (set(re.findall(r"\d+", h)) - set(re.findall(r"\d+", p))) else 0.0
        rows.append([overlap, neg_p, neg_h, do_dai, so_moi])
    return np.array(rows, dtype=np.float64)


class BiasModel:
    """Logistic Regression trên 5 đặc trưng bề mặt; predict_logp trả log-xác suất theo thứ tự LABELS."""

    def __init__(self, C: float = 1.0):
        from sklearn.linear_model import LogisticRegression

        self.clf = LogisticRegression(C=C, max_iter=2000)

    def fit(self, train_raw: pd.DataFrame) -> "BiasModel":
        self.clf.fit(compare_features(train_raw), train_raw["label"])
        return self

    def predict_logp(self, df_raw: pd.DataFrame) -> np.ndarray:
        proba = self.clf.predict_proba(compare_features(df_raw))
        order = [list(self.clf.classes_).index(l) for l in LABELS]
        return np.log(np.clip(proba[:, order], 1e-12, 1.0)).astype(np.float32)

    def predict(self, df_raw: pd.DataFrame) -> list[str]:
        return [LABELS[i] for i in self.predict_logp(df_raw).argmax(axis=1)]


def run_bias_only(scenario: str = "S1", out_dir=None, verbose: bool = True) -> list[dict]:
    """Đánh giá riêng mô hình mẹo (lưu là M1s: M1 chỉ đặc trưng so sánh) để biết mẹo một mình đạt bao nhiêu."""
    from .data import load_scenario
    from .evaluate import RESULTS_DIR, evaluate_and_save

    data = load_scenario(scenario)
    bias = BiasModel().fit(data["train"])
    coef = {l: dict(zip(FEATURES, np.round(bias.clf.coef_[list(bias.clf.classes_).index(l)], 3).tolist())) for l in LABELS}
    results = []
    for name, df in data["test"].items():
        results.append(evaluate_and_save(df["label"], bias.predict(df), model="M1s", scenario=scenario, test_set=name,
                                         seed=42, config={"clf": "LR", "C": 1.0, "features": FEATURES, "coef": coef,
                                                          "dung_lam": "mô hình mẹo cho PoE (src/debias.py)"},
                                         ids=df["id"], premise=df["premise"], hypothesis=df["hypothesis"],
                                         out_dir=out_dir or RESULTS_DIR, verbose=verbose))
    return results
