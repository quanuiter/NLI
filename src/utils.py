"""Tiện ích dùng chung: cố định seed, đo thời gian, đọc/ghi JSON."""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SEEDS = (42, 123, 2024)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Cố định seed cho random, numpy và torch (nếu có) để kết quả tái lập được."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@contextmanager
def timer(result: dict, key: str = "seconds"):
    """with timer(info, "train_seconds"): ...  -> ghi số giây chạy vào info[key]."""
    start = time.perf_counter()
    yield
    result[key] = round(time.perf_counter() - start, 2)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_to_builtin)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_builtin(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Không ghi được kiểu {type(obj)} ra JSON")
