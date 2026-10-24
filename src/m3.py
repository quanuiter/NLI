"""M3 – BiLSTM + max-pooling (task W3-01).

1. Mỗi câu -> dãy chỉ số âm tiết (cùng bảng từ vựng fastText với M2), cắt ở MAX_LEN âm tiết.
2. Embedding 300 chiều khởi tạo bằng fastText -> BiLSTM 1 lớp (dùng chung trọng số cho premise và hypothesis)
   -> max-pooling theo thời gian, bỏ qua phần đệm -> vector câu 2*hidden chiều (u cho premise, v cho hypothesis).
3. Đặc trưng cặp câu theo InferSent (Conneau et al. 2017): [u, v, |u - v|, u * v] -> FFNN -> 3 nhãn.
   Huấn luyện bằng cross-entropy + Adam, dừng sớm theo macro-F1 trên dev, giữ trọng số tốt nhất.

Khác M2: BiLSTM đọc câu theo thứ tự (hai chiều), nên "A thắng B" và "B thắng A" cho hai vector khác nhau (thẻ ROLE);
max-pooling giữ được tín hiệu mạnh của một từ như "không" thay vì bị trung bình làm loãng (thẻ NEG).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .data import LABEL2ID, LABELS, ROOT, load_scenario
from .embeddings import load_embedding, tokenize
from .evaluate import evaluate_and_save
from .utils import get_device, set_seed, timer

CHECKPOINT_DIR = ROOT / "checkpoints"

# Chọn trên dev bằng scripts/tune_m3.py (xem docs/ket_qua_m3.md)
DEFAULT_CONFIG = {
    "embedding": "fasttext",
    "max_len": 64,
    "hidden": 256,
    "fc_hidden": 512,
    "dropout": 0.3,
    "freeze_embedding": True,
    "lr": 1e-3,
    "batch_size": 64,
    "max_epochs": 15,
    "patience": 3,
    "grad_clip": 5.0,
}
# Ghi đè riêng cho từng kịch bản nếu dev của kịch bản đó chọn cấu hình khác
SCENARIO_CONFIG: dict[str, dict] = {"S1": {}, "S2": {}}


def config_for(scenario: str, cfg: dict | None = None) -> dict:
    return {**DEFAULT_CONFIG, **SCENARIO_CONFIG.get(scenario.upper(), {}), **(cfg or {})}


def encode(texts, word2idx: dict[str, int], max_len: int) -> list[list[int]]:
    """Câu -> danh sách chỉ số; âm tiết không có vector -> 0 (vector 0, giống M2). Câu rỗng -> [0]."""
    out = []
    for text in texts:
        ids = [word2idx.get(t, 0) for t in tokenize(text)][:max_len]
        out.append(ids or [0])
    return out


def pad_batch(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Đệm về độ dài lớn nhất trong batch. Trả về (chỉ số [B, T], độ dài [B])."""
    lengths = torch.tensor([len(s) for s in seqs])
    batch = torch.zeros(len(seqs), int(lengths.max()), dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, :len(s)] = torch.tensor(s)
    return batch, lengths


class BiLSTMEncoder(nn.Module):
    """Embedding -> BiLSTM -> max-pooling có mặt nạ (vị trí đệm không được chọn làm giá trị lớn nhất)."""

    def __init__(self, matrix: np.ndarray, hidden: int, freeze: bool, dropout: float):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(matrix), freeze=freeze, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(matrix.shape[1], hidden, batch_first=True, bidirectional=True)
        self.out_dim = 2 * hidden

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embed(ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.shape[1])
        mask = torch.arange(ids.shape[1], device=ids.device)[None, :] < lengths.to(ids.device)[:, None]
        return out.masked_fill(~mask[:, :, None], float("-inf")).max(dim=1).values


class M3(nn.Module):
    def __init__(self, matrix: np.ndarray, cfg: dict, n_classes: int = len(LABELS)):
        super().__init__()
        self.encoder = BiLSTMEncoder(matrix, cfg["hidden"], cfg["freeze_embedding"], cfg["dropout"])
        d = self.encoder.out_dim
        self.classifier = nn.Sequential(nn.Linear(4 * d, cfg["fc_hidden"]), nn.ReLU(), nn.Dropout(cfg["dropout"]),
                                        nn.Linear(cfg["fc_hidden"], n_classes))

    def forward(self, p_ids, p_len, h_ids, h_len):
        u, v = self.encoder(p_ids, p_len), self.encoder(h_ids, h_len)
        return self.classifier(torch.cat([u, v, (u - v).abs(), u * v], dim=1))


def _batches(P, H, batch_size: int, order=None):
    order = range(len(P)) if order is None else order
    order = list(order)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        yield idx, pad_batch([P[j] for j in idx]), pad_batch([H[j] for j in idx])


def _predict(model: M3, P, H, device: str, batch_size: int = 512) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for _, (p, pl), (h, hl) in _batches(P, H, batch_size):
            preds.append(model(p.to(device), pl, h.to(device), hl).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def train_m3(train, dev, matrix: np.ndarray, cfg: dict, seed: int, verbose: bool = True):
    """train/dev = (P, H, y) đã mã hóa. Dừng sớm theo macro-F1 dev. Trả về (model, thông tin huấn luyện)."""
    set_seed(seed)
    device = get_device()
    model = M3(matrix, cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    (P, H, y), (Pd, Hd, yd) = train, dev
    y = torch.tensor(y, dtype=torch.long)

    best = {"dev_macro_f1": -1.0, "epoch": 0, "state": None}
    history = []
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        total = 0.0
        for idx, (p, pl), (h, hl) in _batches(P, H, cfg["batch_size"], torch.randperm(len(P), generator=gen).tolist()):
            opt.zero_grad()
            loss = loss_fn(model(p.to(device), pl, h.to(device), hl), y[idx].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            opt.step()
            total += loss.item() * len(idx)
        dev_f1 = f1_score(yd, _predict(model, Pd, Hd, device), average="macro")
        history.append({"epoch": epoch, "train_loss": round(total / len(P), 4), "dev_macro_f1": round(dev_f1, 4)})
        if verbose:
            print(f"  epoch {epoch:>2}  loss={total / len(P):.4f}  dev_macro_f1={dev_f1:.4f}", flush=True)
        if dev_f1 > best["dev_macro_f1"]:
            best = {"dev_macro_f1": dev_f1, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= cfg["patience"]:
            break
    model.load_state_dict(best["state"])
    info = {"best_epoch": best["epoch"], "dev_macro_f1": round(best["dev_macro_f1"], 4), "history": history,
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_trainable_params": sum(p.numel() for p in params)}
    return model, info


def save_checkpoint(model: M3, cfg: dict, path: Path) -> Path:
    """Lưu trọng số; nếu embedding bị đóng băng thì bỏ ma trận embedding (tạo lại được từ emb_fasttext.npz)."""
    state = model.state_dict()
    if cfg["freeze_embedding"]:
        state = {k: v for k, v in state.items() if not k.startswith("encoder.embed.")}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "state_dict": state}, path)
    return path


def load_checkpoint(path: Path) -> M3:
    ckpt = torch.load(path, map_location="cpu")
    _, matrix = load_embedding(ckpt["config"]["embedding"])
    model = M3(matrix, ckpt["config"])
    model.load_state_dict(ckpt["state_dict"], strict=not ckpt["config"]["freeze_embedding"])
    return model.eval()


def run(scenario: str, seed: int = 42, cfg: dict | None = None, verbose: bool = True,
        save_ckpt: bool = True) -> list[dict]:
    """Chạy M3 cho một kịch bản (S1 hoặc S2), lưu kết quả mọi tập test qua evaluate_and_save."""
    cfg = config_for(scenario, cfg)
    if scenario.upper() not in ("S1", "S2"):
        raise ValueError("M3 chỉ chạy S1 và S2 (S3 dành cho PhoBERT / XLM-R)")
    data = load_scenario(scenario)
    word2idx, matrix = load_embedding(cfg["embedding"])
    enc = lambda df: (encode(df["premise"], word2idx, cfg["max_len"]),  # noqa: E731
                      encode(df["hypothesis"], word2idx, cfg["max_len"]))
    labels = lambda df: df["label"].map(LABEL2ID).to_numpy()  # noqa: E731

    info = {}
    train = (*enc(data["train"]), labels(data["train"]))
    dev = (*enc(data["dev"]), labels(data["dev"]))
    if verbose:
        print(f"M3 · {scenario} · seed {seed}: train {len(train[0])}, dev {len(dev[0])}", flush=True)
    with timer(info, "train_seconds"):
        model, train_info = train_m3(train, dev, matrix, cfg, seed, verbose)
    device = get_device()
    if save_ckpt:
        info["checkpoint"] = str(save_checkpoint(model, cfg, CHECKPOINT_DIR / f"M3_{scenario.upper()}_seed{seed}.pt")
                                 .relative_to(ROOT))
    results = []
    for test_name, df in data["test"].items():
        with timer(info, f"predict_seconds_{test_name}"):
            pred = _predict(model, *enc(df), device)
        config = {**cfg, **info, **{k: v for k, v in train_info.items() if k != "history"},
                  "history": train_info["history"], "device": device}
        results.append(evaluate_and_save(df["label"], pred, model="M3", scenario=scenario.upper(), test_set=test_name,
                                         seed=seed, config=config, ids=df["id"], premise=df["premise"],
                                         hypothesis=df["hypothesis"], verbose=verbose))
    return results
