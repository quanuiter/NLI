"""M4 (PhoBERT), M5 (XLM-R), M6 (mDeBERTa đã học NLI): fine-tune Transformer cho cặp câu (W3-02, W3-03, W4-01–04, W4-10).

1. Tokenizer ghép cặp câu thành <s> premise </s></s> hypothesis </s>. Mô hình đọc hai câu cùng lúc
   (self-attention giữa mọi token của hai câu), khác M2, M3 vốn mã hóa từng câu riêng rồi mới so sánh.
2. Thêm một lớp phân loại 3 nhãn lên trên, huấn luyện tiếp toàn bộ mô hình bằng Hugging Face Trainer.
   Đánh giá trên dev sau mỗi epoch, giữ checkpoint có macro-F1 dev cao nhất.
3. PhoBERT học trên văn bản đã tách từ (VnCoreNLP) nên đầu vào phải là bản đã tách từ (segmented=True).
   XLM-R dùng SentencePiece, nhận thẳng văn bản thô (segmented=False).
4. M6 bắt đầu từ mDeBERTa-v3-base đã được huấn luyện NLI trên khoảng 2,7 triệu cặp câu ở 27 ngôn ngữ (có tiếng Việt;
   gồm MNLI, ANLI, WANLI... dịch máy và XNLI dev), nên lớp phân loại 3 nhãn đã có sẵn. Ta giữ lớp đó (sắp lại thứ tự nhãn
   cho khớp LABELS) rồi fine-tune thêm trên ViANLI (S2), hoặc đánh giá thẳng không fine-tune (S0).
   Tập test XNLI không nằm trong dữ liệu huấn luyện của checkpoint này (theo model card), nên so được với M4, M5.
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
from .utils import load_json, save_json, set_seed, timer

CHECKPOINT_DIR = ROOT / "checkpoints"

MODELS = {
    "M4": {"name": "vinai/phobert-base", "segmented": True},
    "M5": {"name": "xlm-roberta-base", "segmented": False},
    "M6": {"name": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "segmented": False, "nli_head": True},
}

# max_len 256: phủ gần hết cặp câu (PhoBERT: chỉ 17/100.000 cặp XNLI train dài hơn, xem results/analysis/token_lengths.json).
# Mỗi batch chỉ đệm tới câu dài nhất trong batch, nên max_len lớn không làm chậm.
# hidden_dropout: dropout bên trong 12 lớp Transformer, giữ đúng giá trị lúc huấn luyện trước (0,1).
# classifier_dropout: dropout của lớp phân loại đặt lên trên. "Dropout 0,4" của bài ViANLI được hiểu là dropout này;
# đặt 0,4 cho cả 12 lớp làm mô hình học rất chậm (lần chạy 24/10: dev macro-F1 chỉ 0,34 sau 7 epoch).
CONFIGS = {
    # A: cấu hình thường dùng khi fine-tune BERT
    "A": {"lr": 2e-5, "batch_size": 32, "epochs": 3, "max_len": 256, "weight_decay": 0.01, "warmup_ratio": 0.1,
          "hidden_dropout": 0.1, "classifier_dropout": 0.1},
    # B: cấu hình của bài ViANLI (mục 6.2): lr 1e-5, batch 16, 7 epoch, max_len 256, dropout 0,4
    "B": {"lr": 1e-5, "batch_size": 16, "epochs": 7, "max_len": 256, "weight_decay": 0.01, "warmup_ratio": 0.1,
          "hidden_dropout": 0.1, "classifier_dropout": 0.4},
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


def _classifier_layer(model):
    """Lớp Linear cuối cùng ra 3 nhãn (RoBERTa: classifier.out_proj, DeBERTa: classifier)."""
    layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear) and m.out_features == len(LABELS)]
    return layers[-1]


def load_model(model_key: str, source: str | Path | None = None, cfg: dict | None = None):
    """Tạo mô hình phân loại 3 nhãn theo thứ tự LABELS.

    source: checkpoint để học tiếp (mặc định là mô hình huấn luyện trước của model_key).
    Mô hình đã có lớp phân loại NLI (M6): giữ trọng số lớp đó, chỉ sắp lại hàng cho đúng thứ tự nhãn của nhóm.
    """
    from transformers import AutoConfig, AutoModelForSequenceClassification

    spec = MODELS[model_key]
    cfg = cfg or {}
    source = str(source or spec["name"])
    config = AutoConfig.from_pretrained(source)
    own = {k.lower(): int(v) for k, v in config.label2id.items()}
    if "hidden_dropout" in cfg:
        # dropout lớp phân loại: classifier_dropout ở RoBERTa (M4, M5), cls_dropout ở DeBERTa (M6)
        config.hidden_dropout_prob = cfg["hidden_dropout"]
        config.cls_dropout = config.classifier_dropout = cfg["classifier_dropout"]
    if spec.get("nli_head") and set(own) == set(LABELS):
        model = AutoModelForSequenceClassification.from_pretrained(source, config=config)
        order = [own[label] for label in LABELS]
        if order != list(range(len(LABELS))):
            layer = _classifier_layer(model)
            with torch.no_grad():
                layer.weight.copy_(layer.weight[order].clone())
                layer.bias.copy_(layer.bias[order].clone())
        model.config.id2label, model.config.label2id = dict(ID2LABEL), dict(LABEL2ID)
        return model
    config.num_labels, config.id2label, config.label2id = len(LABELS), dict(ID2LABEL), dict(LABEL2ID)
    return AutoModelForSequenceClassification.from_pretrained(source, config=config)


def zero_shot(model_key: str = "M6", seed: int = 42, test_limit: int | None = None, out_dir: Path = RESULTS_DIR,
              verbose: bool = True) -> list[dict]:
    """Kịch bản S0: đánh giá thẳng mô hình đã học NLI sẵn trên mọi tập test, không fine-tune thêm."""
    from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

    spec = MODELS[model_key]
    if not spec.get("nli_head"):
        raise ValueError(f"{model_key} chưa học NLI, không đánh giá thẳng được")
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(spec["name"])
    model = load_model(model_key)
    fp16 = torch.cuda.is_available()
    args = TrainingArguments(output_dir=str(CHECKPOINT_DIR / "_zero_shot_tmp"), per_device_eval_batch_size=64,
                             fp16=fp16, report_to="none")
    trainer = Trainer(model=model, args=args, data_collator=DataCollatorWithPadding(tokenizer))
    info = {"pretrained": spec["name"], "fine_tuned": False, "fp16": fp16,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "n_params": sum(p.numel() for p in model.parameters())}
    results = []
    for test_name, df in load_scenario("S1", segmented=spec["segmented"])["test"].items():
        if test_limit:
            df = df.sample(min(test_limit, len(df)), random_state=seed)
        with timer(info, f"predict_seconds_{test_name}"):
            logits = trainer.predict(PairDataset(df, tokenizer, 256)).predictions
        pred = [ID2LABEL[int(i)] for i in np.argmax(logits, axis=-1)]
        results.append(evaluate_and_save(df["label"], pred, model=model_key, scenario="S0", test_set=test_name,
                                         seed=seed, config=dict(info), ids=df["id"], premise=df["premise"],
                                         hypothesis=df["hypothesis"], out_dir=out_dir, verbose=verbose))
    shutil.rmtree(CHECKPOINT_DIR / "_zero_shot_tmp", ignore_errors=True)
    return results


def run(model_key: str, scenario: str, seed: int = 42, config: str = "A", cfg: dict | None = None,
        train_limit: int | None = None, dev_limit: int | None = None, test_limit: int | None = None,
        init_from: str | Path | None = None,
        out_dir: Path = RESULTS_DIR, keep_checkpoint: bool = True, ckpt_suffix: str = "",
        fp16: bool | None = None, verbose: bool = True) -> list[dict]:
    """Fine-tune M4/M5 cho một kịch bản, đánh giá mọi tập test qua evaluate_and_save.

    config: "A" hoặc "B" (CONFIGS), cfg ghi đè từng giá trị.
    train_limit / dev_limit / test_limit: chỉ lấy một phần dữ liệu để chạy thử (out_dir nên là results/smoke).
    init_from: thư mục checkpoint để học tiếp (kịch bản S3: từ checkpoint S1, tiếp tục trên ViANLI).
    fp16: mặc định bật khi có GPU; tắt (--no-fp16) nếu loss bị nan.
    """
    from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer

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

    tokenizer = AutoTokenizer.from_pretrained(spec["name"])
    model = load_model(model_key, init_from, cfg)
    fp16 = torch.cuda.is_available() if fp16 is None else (fp16 and torch.cuda.is_available())

    tag = f"{model_key}_{scenario}_seed{seed}" + ("_smoke" if train_limit else "") + (f"_{ckpt_suffix}" if ckpt_suffix else "")
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
    if any(h["train_loss"] is None or not np.isfinite(h["train_loss"]) for h in history):
        print("CẢNH BÁO: loss bị nan/không ghi được. Nếu đang dùng fp16, chạy lại với --no-fp16.", flush=True)
    best = max(history, key=lambda h: h["dev_macro_f1"])
    info.update({"best_epoch": best["epoch"], "dev_macro_f1": best["dev_macro_f1"], "history": history,
                 "n_params": sum(p.numel() for p in trainer.model.parameters())})

    if keep_checkpoint:
        final_dir = CHECKPOINT_DIR / tag
        trainer.save_model(str(final_dir))
        info["checkpoint"] = _rel(final_dir)
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


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def tune(model_key: str, scenario: str, configs: list[str], seed: int = 42, init_from: str | Path | None = None,
         verbose: bool = True, **run_kwargs) -> dict:
    """So nhiều cấu hình trên dev rồi giữ cấu hình tốt nhất (W3-03).

    Mỗi cấu hình lưu kết quả riêng ở results/tuning/<mô hình>_<kịch bản>_seed<seed>_<cấu hình>/.
    Cấu hình có macro-F1 dev cao nhất được chép ra results/ (kết quả chính) và giữ checkpoint;
    checkpoint của cấu hình còn lại bị xóa. Việc chọn chỉ dựa trên dev, không nhìn tập test.
    Bảng so sánh: results/analysis/<mô hình>_tuning_<kịch bản>_seed<seed>.json
    """
    scenario = scenario.upper()
    rows = []
    for config in configs:
        sub = RESULTS_DIR / "tuning" / f"{model_key}_{scenario}_seed{seed}_{config}"
        res = run(model_key, scenario, seed, config=config, init_from=init_from, out_dir=sub, ckpt_suffix=config,
                  verbose=verbose, **run_kwargs)
        c = res[0]["config"]
        rows.append({"config": config, "dev_macro_f1": c["dev_macro_f1"], "best_epoch": c["best_epoch"],
                     "train_seconds": c["train_seconds"], "history": c["history"], "dir": sub,
                     "checkpoint": c.get("checkpoint"), "settings": {k: c[k] for k in CONFIGS[config]}})
    best = max(rows, key=lambda r: r["dev_macro_f1"])
    final_dir = CHECKPOINT_DIR / f"{model_key}_{scenario}_seed{seed}"
    for path in best["dir"].rglob("*"):
        if path.is_file():
            target = RESULTS_DIR / path.relative_to(best["dir"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if target.suffix == ".json" and target.parent == RESULTS_DIR and best["checkpoint"]:
                result = load_json(target)
                result["config"]["checkpoint"] = _rel(final_dir)   # checkpoint được đổi tên ở dưới
                save_json(result, target)
    for r in rows:
        ckpt = (ROOT / r["checkpoint"]) if r["checkpoint"] else None
        if ckpt is None or not ckpt.exists():
            continue
        if r is best:
            shutil.rmtree(final_dir, ignore_errors=True)
            ckpt.rename(final_dir)
        else:
            shutil.rmtree(ckpt, ignore_errors=True)
    report = {"model": model_key, "scenario": scenario, "seed": seed, "chosen": best["config"],
              "rule": "chọn cấu hình có macro-F1 dev cao nhất",
              "results": [{k: (_rel(v) if isinstance(v, Path) else v) for k, v in r.items()} for r in rows]}
    save_json(report, RESULTS_DIR / "analysis" / f"{model_key}_tuning_{scenario}_seed{seed}.json")
    if verbose:
        for r in rows:
            print(f"cấu hình {r['config']}: dev macro-F1 {r['dev_macro_f1']:.4f} (epoch {r['best_epoch']})")
        print(f"→ chọn {best['config']}, kết quả chép ra {RESULTS_DIR}, checkpoint {final_dir}")
    return report


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
