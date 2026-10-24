"""M4 (PhoBERT) và M5 (XLM-R): fine-tune mô hình Transformer cho cặp câu (task W3-02, dùng lại ở W3-03, W4-01–04).

1. Tokenizer ghép cặp câu thành <s> premise </s></s> hypothesis </s>. Mô hình đọc hai câu cùng lúc
   (self-attention giữa mọi token của hai câu), khác M2, M3 vốn mã hóa từng câu riêng rồi mới so sánh.
2. Thêm một lớp phân loại 3 nhãn lên trên, huấn luyện tiếp toàn bộ mô hình bằng Hugging Face Trainer.
   Đánh giá trên dev sau mỗi epoch, giữ checkpoint có macro-F1 dev cao nhất.
3. PhoBERT học trên văn bản đã tách từ (VnCoreNLP) nên đầu vào phải là bản đã tách từ (segmented=True).
   XLM-R dùng SentencePiece, nhận thẳng văn bản thô (segmented=False).
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from .data import ID2LABEL, LABEL2ID, LABELS, ROOT, load_scenario
from .evaluate import RESULTS_DIR, evaluate_and_save
from .utils import save_json, set_seed, timer

CHECKPOINT_DIR = ROOT / "checkpoints"

MODELS = {
    "M4": {"name": "vinai/phobert-base", "segmented": True},
    "M5": {"name": "xlm-roberta-base", "segmented": False},
}

# max_len 256: phủ gần hết cặp câu (PhoBERT: chỉ 17/100.000 cặp XNLI train dài hơn, xem results/analysis/token_lengths.json).
# Mỗi batch chỉ đệm tới câu dài nhất trong batch, nên max_len lớn không làm chậm.
CONFIGS = {
    # A: cấu hình thường dùng khi fine-tune BERT
    "A": {"lr": 2e-5, "batch_size": 32, "epochs": 3, "max_len": 256, "weight_decay": 0.01, "warmup_ratio": 0.1,
          "dropout": 0.1},
    # B: cấu hình của bài ViANLI (mục 6.2): lr 1e-5, batch 16, 7 epoch, max_len 256, dropout 0,4
    "B": {"lr": 1e-5, "batch_size": 16, "epochs": 7, "max_len": 256, "weight_decay": 0.01, "warmup_ratio": 0.1,
          "dropout": 0.4},
}


class PairDataset(torch.utils.data.Dataset):
    """Token hóa sẵn cặp câu (cắt ở max_len, chưa đệm; đệm theo batch ở DataCollatorWithPadding)."""

    def __init__(self, df, tokenizer, max_len: int):
        self.enc = tokenizer(list(df["premise"]), list(df["hypothesis"]), truncation=True, max_length=max_len)
        self.labels = df["label"].map(LABEL2ID).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def token_lengths(tokenizer, df) -> dict:
    """Phân bố độ dài (số token) của cặp câu, để chọn max_len."""
    lengths = np.array([len(x) for x in tokenizer(list(df["premise"]), list(df["hypothesis"]))["input_ids"]])
    return {"n": int(len(lengths)), "p50": float(np.percentile(lengths, 50)), "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)), "max": int(lengths.max()),
            "over_128": round(float((lengths > 128).mean()), 5), "over_256": round(float((lengths > 256).mean()), 5)}


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro"), "accuracy": float((preds == labels).mean())}


def _training_args(out_dir: Path, cfg: dict, seed: int, fp16: bool):
    from transformers import TrainingArguments

    kwargs = dict(
        output_dir=str(out_dir), learning_rate=cfg["lr"], per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=max(64, cfg["batch_size"]), num_train_epochs=cfg["epochs"],
        weight_decay=cfg["weight_decay"], save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True, save_total_limit=1,
        logging_strategy="epoch", seed=seed, data_seed=seed, fp16=fp16, report_to="none",
    )
    # Tên tham số đổi giữa các phiên bản transformers: evaluation_strategy -> eval_strategy (4.41),
    # warmup_ratio -> warmup_steps nhận số thực trong [0, 1) (5.x)
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "epoch"
    kwargs["warmup_ratio" if "warmup_ratio" in params else "warmup_steps"] = cfg["warmup_ratio"]
    return TrainingArguments(**kwargs)


def run(model_key: str, scenario: str, seed: int = 42, config: str = "A", cfg: dict | None = None,
        train_limit: int | None = None, dev_limit: int | None = None, test_limit: int | None = None,
        init_from: str | Path | None = None,
        out_dir: Path = RESULTS_DIR, keep_checkpoint: bool = True, verbose: bool = True) -> list[dict]:
    """Fine-tune M4/M5 cho một kịch bản, đánh giá mọi tập test qua evaluate_and_save.

    config: "A" hoặc "B" (CONFIGS), cfg ghi đè từng giá trị.
    train_limit / dev_limit / test_limit: chỉ lấy một phần dữ liệu để chạy thử (out_dir nên là results/smoke).
    init_from: thư mục checkpoint để học tiếp (kịch bản S3: từ checkpoint S1, tiếp tục trên ViANLI).
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

    if model_key not in MODELS:
        raise ValueError(f"model_key phải là một trong {list(MODELS)}")
    spec = MODELS[model_key]
    cfg = {**CONFIGS[config], **(cfg or {}), "config_name": config, "pretrained": spec["name"]}
    scenario = scenario.upper()
    set_seed(seed)

    data = load_scenario(scenario, segmented=spec["segmented"])
    train_df, dev_df = data["train"], data["dev"]
    if train_limit:
        train_df = train_df.sample(min(train_limit, len(train_df)), random_state=seed)
    if dev_limit:
        dev_df = dev_df.sample(min(dev_limit, len(dev_df)), random_state=seed)

    source = str(init_from) if init_from else spec["name"]
    tokenizer = AutoTokenizer.from_pretrained(spec["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        hidden_dropout_prob=cfg["dropout"], attention_probs_dropout_prob=0.1)
    fp16 = torch.cuda.is_available()

    tag = f"{model_key}_{scenario}_seed{seed}" + ("_smoke" if train_limit else "")
    work_dir = CHECKPOINT_DIR / f"_{tag}_tmp"
    trainer = Trainer(
        model=model, args=_training_args(work_dir, cfg, seed, fp16),
        train_dataset=PairDataset(train_df, tokenizer, cfg["max_len"]),
        eval_dataset=PairDataset(dev_df, tokenizer, cfg["max_len"]),
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=_compute_metrics,
        **({"processing_class": tokenizer} if "processing_class" in inspect.signature(Trainer.__init__).parameters
           else {"tokenizer": tokenizer}),
    )
    info = {"n_train": len(train_df), "n_dev": len(dev_df), "init_from": str(init_from) if init_from else None,
            "fp16": fp16, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    if verbose:
        print(f"{model_key} ({spec['name']}) · {scenario} · seed {seed} · cấu hình {config}: "
              f"train {len(train_df)}, dev {len(dev_df)}, {info['device']}", flush=True)
    with timer(info, "train_seconds"):
        trainer.train()

    history = [{"epoch": round(h["epoch"]), "dev_macro_f1": round(h["eval_macro_f1"], 4),
                "dev_accuracy": round(h["eval_accuracy"], 4)} for h in trainer.state.log_history if "eval_macro_f1" in h]
    losses = {round(h["epoch"]): round(h["loss"], 4) for h in trainer.state.log_history if "loss" in h}
    for h in history:
        h["train_loss"] = losses.get(h["epoch"])
    best = max(history, key=lambda h: h["dev_macro_f1"])
    info.update({"best_epoch": best["epoch"], "dev_macro_f1": best["dev_macro_f1"], "history": history,
                 "n_params": sum(p.numel() for p in trainer.model.parameters())})

    if keep_checkpoint:
        final_dir = CHECKPOINT_DIR / tag
        trainer.save_model(str(final_dir))
        info["checkpoint"] = str(final_dir.relative_to(ROOT) if final_dir.is_relative_to(ROOT) else final_dir)
    shutil.rmtree(work_dir, ignore_errors=True)

    results = []
    for test_name, df in data["test"].items():
        if test_limit:
            df = df.sample(min(test_limit, len(df)), random_state=seed)
        with timer(info, f"predict_seconds_{test_name}"):
            logits = trainer.predict(PairDataset(df, tokenizer, cfg["max_len"])).predictions
        pred = [ID2LABEL[int(i)] for i in np.argmax(logits, axis=-1)]
        results.append(evaluate_and_save(df["label"], pred, model=model_key, scenario=scenario, test_set=test_name,
                                         seed=seed, config={**cfg, **info}, ids=df["id"], premise=df["premise"],
                                         hypothesis=df["hypothesis"], out_dir=out_dir, verbose=verbose))
    return results


def save_token_lengths(out=RESULTS_DIR / "analysis" / "token_lengths.json") -> dict:
    """Đo độ dài token của cặp câu cho PhoBERT và XLM-R trên mọi tập (W3-02, chọn max_len)."""
    from transformers import AutoTokenizer

    from .data import DATASET_SPLITS, load_split

    report = {}
    for key, spec in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(spec["name"])
        for dataset, splits in DATASET_SPLITS.items():
            for split in splits:
                df = load_split(dataset, split, segmented=spec["segmented"])
                report[f"{key}/{dataset}_{split}"] = token_lengths(tokenizer, df)
    save_json(report, Path(out))
    return report
