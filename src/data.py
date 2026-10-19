"""Load, chuẩn hóa và lưu dữ liệu NLI tiếng Việt (XNLI-vi, ViANLI, ViNLI test).

Mọi mô hình trong đồ án đọc dữ liệu qua module này để dùng chung một cách
tiền xử lý và cùng một tập con XNLI 100k.

Định dạng chung: DataFrame với các cột
    id, premise, hypothesis, label, source, split
trong đó label là chuỗi: "entailment" | "neutral" | "contradiction".
"""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path

import pandas as pd

LABELS = ["entailment", "neutral", "contradiction"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

DATASETS = ("xnli", "vianli", "vinli")
SPLITS = ("train", "dev", "test")
# ViNLI chỉ dùng làm tập test phụ (không huấn luyện)
DATASET_SPLITS = {"xnli": SPLITS, "vianli": SPLITS, "vinli": ("test",)}
COLUMNS = ["id", "premise", "hypothesis", "label", "source", "split"]

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

XNLI_SUBSET_SIZE = 100_000
XNLI_SUBSET_SEED = 42

_VIANLI_REPO = "uitnlp/ViANLI"
_VIANLI_FILES = {"train": "vianli_train.jsonl", "dev": "vianli_dev.jsonl", "test": "vianli_test.jsonl"}
_XNLI_SPLITS = {"train": "train", "dev": "validation", "test": "test"}
# Bản đăng lại trên Hugging Face, số mẫu khớp bài báo ViNLI (COLING 2022).
# Bản chính thức: nhóm UIT NLP (https://nlp.uit.edu.vn/).
_VINLI_REPO = "presencesw/vinli_4_label"


# ---------------------------------------------------------------- chuẩn hóa

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:%)\]}])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[{])\s+")
_MULTI_SPACE = re.compile(r"\s+")


def normalize_text(text: str, fix_punct_spacing: bool = True) -> str:
    """Chuẩn hóa một câu.

    - Unicode NFC (ViANLI có ~1,7% câu chưa ở dạng NFC).
    - Giải mã HTML entity (&amp; ...) và đổi ngoặc kép kiểu `` '' thành ".
    - Bỏ khoảng trắng thừa.
    - fix_punct_spacing: bỏ dấu cách trước dấu câu ("từ ." -> "từ."). XNLI train
      (dịch máy) có ~91% câu bị tách dấu câu như vậy, còn dev/test gần như không;
      nếu không sửa, mô hình học S1 sẽ gặp lệch định dạng giữa train và test.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = html.unescape(text)
    text = text.replace("``", '"').replace("''", '"').replace(" ", " ")
    text = _MULTI_SPACE.sub(" ", text).strip()
    if fix_punct_spacing:
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    return text


def normalize_label(label) -> str:
    """Đổi nhãn số (XNLI: 0/1/2) hoặc chuỗi (ViANLI) về chuỗi thống nhất."""
    if isinstance(label, str):
        label = label.strip().lower()
        if label in LABEL2ID:
            return label
        raise ValueError(f"Nhãn không hợp lệ: {label!r}")
    label = int(label)
    if label not in ID2LABEL:
        raise ValueError(f"Nhãn không hợp lệ: {label!r}")
    return ID2LABEL[label]


def _finalize(df: pd.DataFrame, source: str, split: str) -> pd.DataFrame:
    df = df.copy()
    df["premise"] = df["premise"].map(normalize_text)
    df["hypothesis"] = df["hypothesis"].map(normalize_text)
    df["label"] = df["label"].map(normalize_label)
    df["source"] = source
    df["split"] = split
    return df[COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------- tải dữ liệu gốc

def load_xnli_raw(split: str) -> pd.DataFrame:
    """Tải XNLI tiếng Việt từ Hugging Face (facebook/xnli, config vi) và chuẩn hóa."""
    from datasets import load_dataset

    _check_split(split)
    ds = load_dataset("facebook/xnli", "vi", split=_XNLI_SPLITS[split])
    df = ds.to_pandas()
    df.insert(0, "id", [f"xnli_{split}_{i}" for i in range(len(df))])
    return _finalize(df, "xnli", split)


def load_vianli_raw(split: str) -> pd.DataFrame:
    """Tải ViANLI từ Hugging Face (uitnlp/ViANLI, file jsonl) và chuẩn hóa."""
    from huggingface_hub import hf_hub_download

    _check_split(split)
    path = hf_hub_download(_VIANLI_REPO, _VIANLI_FILES[split], repo_type="dataset")
    df = pd.read_json(path, lines=True).rename(columns={"uid": "id"})
    return _finalize(df, "vianli", split)


def load_vinli_raw(split: str = "test") -> pd.DataFrame:
    """Tải ViNLI (Huynh et al., COLING 2022) và bỏ nhãn "other" để về dạng 3 nhãn.

    ViNLI có 4 nhãn; "other" là cặp câu không liên quan nhau, không có trong XNLI/ViANLI.
    id giữ theo thứ tự trong file gốc (vinli_test_<i>) để tra lại được.
    """
    from datasets import load_dataset

    if split not in DATASET_SPLITS["vinli"]:
        raise ValueError("ViNLI chỉ dùng split test")
    df = load_dataset(_VINLI_REPO, split=split).to_pandas()
    df = df.rename(columns={"sentence1": "premise", "sentence2": "hypothesis", "gold_label": "label"})
    df.insert(0, "id", [f"vinli_{split}_{i}" for i in range(len(df))])
    df = df[df["label"].str.strip().str.lower() != "other"]
    return _finalize(df, "vinli", split)


def stratified_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Lấy n mẫu, giữ nguyên tỉ lệ nhãn, cố định seed."""
    if n >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    frac = n / len(df)
    parts = [g.sample(n=round(len(g) * frac), random_state=seed) for _, g in df.groupby("label", sort=True)]
    out = pd.concat(parts)
    # bù/bớt vài mẫu do làm tròn để đúng n
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    elif len(out) < n:
        rest = df.drop(out.index).sample(n=n - len(out), random_state=seed)
        out = pd.concat([out, rest])
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------- đọc/ghi bản đã xử lý

def processed_path(dataset: str, split: str, segmented: bool = False, subset: bool = False) -> Path:
    name = f"{dataset}_{split}"
    if subset:
        name += f"_{XNLI_SUBSET_SIZE // 1000}k"
    if segmented:
        name += "_seg"
    return PROCESSED_DIR / f"{name}.jsonl"


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True, dtype={"id": str})


def load_split(dataset: str, split: str, segmented: bool = False, subset: bool | None = None) -> pd.DataFrame:
    """Đọc một split đã xử lý.

    dataset: "xnli" | "vianli"; split: "train" | "dev" | "test".
    segmented=True: bản đã tách từ (dùng cho PhoBERT), cần chạy scripts/segment_data.py trước.
    subset: mặc định True cho xnli/train (dùng tập 100k chung), False cho các split khác.
    Nếu chưa có file đã xử lý (và không cần bản tách từ) thì tải và xử lý luôn.
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset phải là một trong {DATASETS}")
    if split not in DATASET_SPLITS[dataset]:
        raise ValueError(f"{dataset} chỉ có các split {DATASET_SPLITS[dataset]}")
    if subset is None:
        subset = dataset == "xnli" and split == "train"
    path = processed_path(dataset, split, segmented=segmented, subset=subset)
    if path.exists():
        return read_jsonl(path)
    if segmented:
        raise FileNotFoundError(f"Chưa có {path}. Chạy: python scripts/segment_data.py")
    loader = {"xnli": load_xnli_raw, "vianli": load_vianli_raw, "vinli": load_vinli_raw}[dataset]
    df = loader(split)
    if subset:
        df = stratified_subset(df, XNLI_SUBSET_SIZE, XNLI_SUBSET_SEED)
    save_jsonl(df, path)
    return df


# ---------------------------------------------------------------- kịch bản thí nghiệm

def load_scenario(scenario: str, segmented: bool = False, include_vinli: bool = True) -> dict:
    """Trả về dữ liệu cho một kịch bản.

    S1: train XNLI-vi 100k, dev XNLI-vi, test XNLI-vi + ViANLI.
    S2: train ViANLI, dev ViANLI, test ViANLI.
    S3: train_stage1 XNLI-vi 100k -> train ViANLI (fine-tune tiếp), dev ViANLI, test ViANLI.
    include_vinli=True: mọi kịch bản có thêm tập test phụ "vinli_test" (tiếng Việt gốc, không đối kháng).

    Kết quả: {"train": df, "dev": df, "test": {tên_tập_test: df}, ("train_stage1": df với S3)}.
    """
    scenario = scenario.upper()
    ld = lambda d, s: load_split(d, s, segmented=segmented)  # noqa: E731
    if scenario == "S1":
        out = {"train": ld("xnli", "train"), "dev": ld("xnli", "dev"),
               "test": {"xnli_test": ld("xnli", "test"), "vianli_test": ld("vianli", "test")}}
    elif scenario == "S2":
        out = {"train": ld("vianli", "train"), "dev": ld("vianli", "dev"),
               "test": {"vianli_test": ld("vianli", "test")}}
    elif scenario == "S3":
        out = {"train_stage1": ld("xnli", "train"), "train": ld("vianli", "train"),
               "dev": ld("vianli", "dev"), "test": {"vianli_test": ld("vianli", "test")}}
    else:
        raise ValueError("scenario phải là S1, S2 hoặc S3")
    if include_vinli:
        out["test"]["vinli_test"] = ld("vinli", "test")
    return out


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split phải là một trong {SPLITS}")
