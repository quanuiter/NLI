"""M2 – Embedding trung bình + mạng nơron truyền thẳng (FFNN) (task W2-09).

1. Mỗi câu -> vector câu = trung bình vector fastText của các âm tiết có vector (u cho premise, v cho hypothesis).
2. Đặc trưng cặp câu (theo InferSent, Conneau et al. 2017): [u, v, |u - v|, u * v, cos(u, v)] -> 4*300 + 1 = 1201 chiều.
   |u - v| và u * v cho mô hình thấy hai câu giống / khác nhau ở chiều nào, điều mà ghép [u, v] không làm rõ.
3. FFNN: Linear -> ReLU -> Dropout (1 hoặc 2 lớp ẩn) -> Linear(3), huấn luyện bằng cross-entropy + Adam.
   Dừng sớm (early stopping) theo macro-F1 trên dev, giữ trọng số tốt nhất.

Hạn chế đã biết (để phân tích ở báo cáo): lấy trung bình làm mất thứ tự từ, nên "A thắng B" và "B thắng A"
cho cùng vector câu (thẻ ROLE); từ phủ định "không" chỉ là 1 trong nhiều vector bị cộng trung bình (thẻ NEG).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .data import LABEL2ID, LABELS, load_scenario
from .embeddings import load_embedding, tokenize
from .evaluate import evaluate_and_save
from .utils import get_device, set_seed, timer

# Chọn trên dev bằng scripts/tune_m2.py rồi kiểm tra lại 3 seed (S1 dev 0,549 ± 0,002; S2 dev 0,367 ± 0,014)
DEFAULT_CONFIG = {
    "embedding": "fasttext",
    "hidden": 512,
    "layers": 2,
    "dropout": 0.5,
    "lr": 3e-4,
    "weight_decay": 0.0,
    "batch_size": 128,
    "max_epochs": 30,
    "patience": 3,
}


def sentence_vectors(texts, word2idx: dict[str, int], matrix: np.ndarray) -> np.ndarray:
    """Trung bình vector các token có trong word2idx; câu không có token nào -> vector 0."""
    out = np.zeros((len(texts), matrix.shape[1]), dtype=np.float32)
    for i, text in enumerate(texts):
        ids = [word2idx[t] for t in tokenize(text) if t in word2idx]
        if ids:
            out[i] = matrix[ids].mean(axis=0)
    return out


def pair_features(premise, hypothesis, word2idx, matrix) -> np.ndarray:
    u = sentence_vectors(premise, word2idx, matrix)
    v = sentence_vectors(hypothesis, word2idx, matrix)
    cos = (u * v).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-8)
    return np.hstack([u, v, np.abs(u - v), u * v, cos[:, None]]).astype(np.float32)


class FFNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float, n_classes: int = len(LABELS)):
        super().__init__()
        blocks, dim = [], in_dim
        for _ in range(layers):
            blocks += [nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            dim = hidden
        blocks.append(nn.Linear(dim, n_classes))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


def _predict(model: nn.Module, X: torch.Tensor, device: str, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            preds.append(model(X[i:i + batch_size].to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def train_ffnn(X_train, y_train, X_dev, y_dev, cfg: dict, seed: int, verbose: bool = True):
    """Huấn luyện FFNN, dừng sớm theo macro-F1 dev. Trả về (model, scaler, thông tin huấn luyện)."""
    set_seed(seed)
    device = get_device()
    # Chuẩn hóa đặc trưng theo train (u*v rất nhỏ so với u, v nên cần đưa về cùng thang đo)
    mean, std = X_train.mean(0), X_train.std(0) + 1e-6
    Xtr = torch.tensor((X_train - mean) / std)
    Xdv = torch.tensor((X_dev - mean) / std)
    ytr = torch.tensor(y_train, dtype=torch.long)

    model = FFNN(X_train.shape[1], cfg["hidden"], cfg["layers"], cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)

    best = {"dev_macro_f1": -1.0, "epoch": 0, "state": None}
    history = []
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(Xtr), generator=gen)
        total = 0.0
        for i in range(0, len(order), cfg["batch_size"]):
            idx = order[i:i + cfg["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx].to(device)), ytr[idx].to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        dev_f1 = f1_score(y_dev, _predict(model, Xdv, device), average="macro")
        history.append({"epoch": epoch, "train_loss": round(total / len(Xtr), 4), "dev_macro_f1": round(dev_f1, 4)})
        if verbose:
            print(f"  epoch {epoch:>2}  loss={total / len(Xtr):.4f}  dev_macro_f1={dev_f1:.4f}")
        if dev_f1 > best["dev_macro_f1"]:
            best = {"dev_macro_f1": dev_f1, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= cfg["patience"]:
            break
    model.load_state_dict(best["state"])
    info = {"best_epoch": best["epoch"], "dev_macro_f1": round(best["dev_macro_f1"], 4), "history": history,
            "n_params": sum(p.numel() for p in model.parameters())}
    return model, (mean, std), info


def run(scenario: str, seed: int = 42, cfg: dict | None = None, verbose: bool = True) -> list[dict]:
    """Chạy M2 cho một kịch bản (S1 hoặc S2), lưu kết quả mọi tập test qua evaluate_and_save."""
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    if scenario.upper() not in ("S1", "S2"):
        raise ValueError("M2 chỉ chạy S1 và S2 (S3 dành cho PhoBERT / XLM-R)")
    data = load_scenario(scenario)
    word2idx, matrix = load_embedding(cfg["embedding"])
    feats = lambda df: pair_features(df["premise"].tolist(), df["hypothesis"].tolist(), word2idx, matrix)  # noqa: E731
    labels = lambda df: df["label"].map(LABEL2ID).to_numpy()  # noqa: E731

    info = {}
    with timer(info, "feature_seconds"):
        X_train, X_dev = feats(data["train"]), feats(data["dev"])
    if verbose:
        print(f"M2 · {scenario} · seed {seed}: train {X_train.shape}, dev {X_dev.shape}")
    with timer(info, "train_seconds"):
        model, (mean, std), train_info = train_ffnn(X_train, labels(data["train"]), X_dev, labels(data["dev"]),
                                                    cfg, seed, verbose)
    device = get_device()
    results = []
    for test_name, df in data["test"].items():
        with timer(info, f"predict_seconds_{test_name}"):
            pred = _predict(model, torch.tensor((feats(df) - mean) / std), device)
        config = {**cfg, **info, **{k: v for k, v in train_info.items() if k != "history"},
                  "history": train_info["history"], "device": device}
        results.append(evaluate_and_save(df["label"], pred, model="M2", scenario=scenario.upper(), test_set=test_name,
                                         seed=seed, config=config, ids=df["id"], premise=df["premise"],
                                         hypothesis=df["hypothesis"], verbose=verbose))
    return results
