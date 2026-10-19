"""Kiểm tra nhanh khung code chung. Chạy: python -m pytest -q  (hoặc python tests/test_core.py)"""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import normalize_label, normalize_text, stratified_subset  # noqa: E402
from src.embeddings import old_style_tone, tokenize  # noqa: E402
from src.evaluate import compute_metrics, evaluate_and_save, summarize  # noqa: E402


def test_normalize_text():
    assert normalize_text("Sản phẩm và địa lý .") == "Sản phẩm và địa lý."
    assert normalize_text("  a   b ,c ( d ) ") == "a b,c (d)"
    assert normalize_text("&quot;x&quot;") == '"x"'
    decomposed = "Hoà"  # "Hoà" viết bằng dấu tổ hợp (NFD)
    assert normalize_text(decomposed) == "Hoà"


def test_tokenize_and_old_style_tone():
    assert tokenize("Hà Nội, ngày 3/4.") == ["Hà", "Nội", "ngày", "3", "4"]
    assert [old_style_tone(w) for w in ["hóa", "Hòa", "khỏe", "thủy"]] == ["hoá", "Hoà", "khoẻ", "thuỷ"]
    assert [old_style_tone(w) for w in ["hoàng", "quý", "khoa"]] == ["hoàng", "quý", "khoa"]


def test_normalize_label():
    assert [normalize_label(x) for x in (0, 1, 2)] == ["entailment", "neutral", "contradiction"]
    assert normalize_label(" Neutral ") == "neutral"


def test_stratified_subset_keeps_ratio_and_is_deterministic():
    df = pd.DataFrame({"label": ["entailment"] * 600 + ["neutral"] * 300 + ["contradiction"] * 100,
                       "x": range(1000)})
    a = stratified_subset(df, 100, seed=42)
    b = stratified_subset(df, 100, seed=42)
    assert len(a) == 100
    assert a["label"].value_counts().to_dict() == {"entailment": 60, "neutral": 30, "contradiction": 10}
    assert a["x"].tolist() == b["x"].tolist()


def test_metrics_and_saving():
    y_true = ["entailment", "neutral", "contradiction", "neutral"]
    y_pred = [0, 1, 1, 1]  # nhãn số cũng được chấp nhận
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.75
    assert m["confusion_matrix"]["matrix"][2] == [0, 1, 0]
    with tempfile.TemporaryDirectory() as d:
        evaluate_and_save(y_true, y_pred, model="M0", scenario="S2", test_set="vianli_test", seed=42,
                          config={"note": "test"}, ids=["a", "b", "c", "d"], out_dir=d, verbose=False)
        name = "M0_S2_vianli_test_seed42"
        assert json.loads((Path(d) / f"{name}.json").read_text(encoding="utf-8"))["config"] == {"note": "test"}
        assert (Path(d) / "figures" / f"{name}.png").exists()
        assert pd.read_csv(Path(d) / "preds" / f"{name}.csv")["correct"].sum() == 3
        assert summarize(d)["accuracy_mean"].iloc[0] == 0.75


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
