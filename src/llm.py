"""M7: mô hình ngôn ngữ lớn (LLM) phân loại NLI bằng prompt, không huấn luyện (task W4-12).

Cách làm
1. Mỗi cặp câu được đặt vào một prompt hỏi đáp trắc nghiệm (tiếng Việt):
       Tiền đề: ...  Giả thuyết: ...  Chọn A (suy ra được), B (không xác định), C (mâu thuẫn).
   Mô hình dạng chat (Qwen2.5-7B-Instruct) trả lời bằng một chữ cái.
2. Không sinh văn bản tự do: chỉ chạy một lượt tính và so xác suất của token tiếp theo là "A", "B" hay "C"
   (chọn chữ có xác suất lớn nhất). Cách này nhanh, luôn ra đúng 1 trong 3 nhãn, và cho xác suất để phân tích.
3. Hai chế độ:
   - zero-shot (kịch bản S0): chỉ có hướng dẫn, không có ví dụ;
   - few-shot (kịch bản FS): thêm k ví dụ có nhãn lấy ngẫu nhiên (cân bằng nhãn) từ ViANLI train, đặt thành các lượt hỏi–đáp
     trước câu cần hỏi. Mỗi seed chọn một bộ ví dụ khác nhau, nên độ lệch chuẩn qua seed đo độ nhạy với việc chọn ví dụ.
4. Phần đầu prompt (hướng dẫn + ví dụ) giống nhau cho mọi câu hỏi, nên được tính một lần và dùng lại (KV cache);
   mỗi câu chỉ cần tính phần đuôi. Few-shot vì vậy chỉ chậm hơn zero-shot một chút.
5. Mô hình 7 tỷ tham số được nạp ở dạng 4-bit (bitsandbytes NF4, khoảng 5,5 GB) để vừa GPU T4 16 GB.

Không có bước chọn siêu tham số trên test: prompt, k và mô hình cố định từ trước; dev ViANLI dùng để kiểm tra prompt (--dev).
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

from .data import LABELS, load_split
from .evaluate import RESULTS_DIR, evaluate_and_save
from .utils import set_seed, timer

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPT_VERSION = "v1"
LETTERS = ["A", "B", "C"]                      # A = entailment, B = neutral, C = contradiction (thứ tự LABELS)
SYSTEM = ("Bạn là chuyên gia ngôn ngữ học tiếng Việt. Nhiệm vụ: suy luận ngôn ngữ tự nhiên (NLI). "
          "Cho một câu tiền đề và một câu giả thuyết, hãy xác định quan hệ giữa chúng.")
INSTRUCTION = ("Tiền đề: {premise}\n"
               "Giả thuyết: {hypothesis}\n\n"
               "Nếu tiền đề là đúng thì giả thuyết:\n"
               "A. Chắc chắn đúng (suy ra được từ tiền đề)\n"
               "B. Có thể đúng hoặc sai (tiền đề không đủ thông tin)\n"
               "C. Chắc chắn sai (mâu thuẫn với tiền đề)\n"
               "Chỉ trả lời bằng một chữ cái A, B hoặc C.")
TEST_SETS = {"xnli_test": ("xnli", "test"), "vianli_test": ("vianli", "test"), "vinli_test": ("vinli", "test"),
             "vianli_dev": ("vianli", "dev")}


def pick_demos(k: int, seed: int) -> pd.DataFrame:
    """k ví dụ từ ViANLI train, cân bằng nhãn (k chia hết cho 3), thứ tự xáo trộn theo seed."""
    if k % len(LABELS):
        raise ValueError("k phải chia hết cho 3 để cân bằng nhãn")
    train = load_split("vianli", "train")
    train = train[(train["premise"].str.len() < 400)]           # tránh ví dụ quá dài làm prompt dài
    demos = pd.concat([train[train["label"] == l].sample(k // len(LABELS), random_state=seed) for l in LABELS])
    return demos.sample(frac=1, random_state=seed)


def build_messages(demos: pd.DataFrame | None) -> list[dict]:
    """Phần đầu chung của prompt: hướng dẫn hệ thống + các lượt hỏi–đáp ví dụ."""
    msgs = [{"role": "system", "content": SYSTEM}]
    if demos is not None:
        for p, h, l in zip(demos["premise"], demos["hypothesis"], demos["label"]):
            msgs.append({"role": "user", "content": INSTRUCTION.format(premise=p, hypothesis=h)})
            msgs.append({"role": "assistant", "content": LETTERS[LABELS.index(l)]})
    return msgs


class LLMClassifier:
    def __init__(self, model_name: str = DEFAULT_MODEL, load_in_4bit: bool | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        cuda = torch.cuda.is_available()
        load_in_4bit = cuda if load_in_4bit is None else load_in_4bit
        kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_compute_dtype=torch.float16)
            kwargs["device_map"] = "auto"
        else:
            import transformers

            dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
            kwargs[dtype_key] = torch.float16 if cuda else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
        if not load_in_4bit and cuda:
            self.model.to("cuda")
        self.device = next(self.model.parameters()).device
        self.load_in_4bit = load_in_4bit
        # id của token "A"/"B"/"C" (và dạng có dấu cách đứng trước, nếu tokenizer tách khác)
        self.letter_ids = []
        for letter in LETTERS:
            ids = {self.tok.encode(letter, add_special_tokens=False)[0], self.tok.encode(" " + letter, add_special_tokens=False)[-1]}
            self.letter_ids.append(sorted(ids))

    def _split_prompt(self, prefix_msgs: list[dict], premise: str, hypothesis: str) -> tuple[str, str]:
        full = self.tok.apply_chat_template(
            prefix_msgs + [{"role": "user", "content": INSTRUCTION.format(premise=premise, hypothesis=hypothesis)}],
            tokenize=False, add_generation_prompt=True)
        prefix = self.tok.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=False)
        if not full.startswith(prefix):
            raise RuntimeError("Mẫu chat của tokenizer không tách được phần đầu prompt")
        return prefix, full[len(prefix):]

    @torch.no_grad()
    def predict(self, df: pd.DataFrame, demos: pd.DataFrame | None = None, batch_size: int = 8,
                verbose: bool = True) -> tuple[list[str], np.ndarray]:
        """Trả về (nhãn dự đoán, xác suất 3 nhãn đã chuẩn hóa trên A/B/C) cho từng cặp câu."""
        prefix_msgs = build_messages(demos)
        prefix_text, _ = self._split_prompt(prefix_msgs, "x", "y")
        prefix_ids = self.tok(prefix_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        suffixes = [self._split_prompt(prefix_msgs, p, h)[1] for p, h in zip(df["premise"], df["hypothesis"])]
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        caches = {}                                   # kích thước batch -> KV cache của phần đầu prompt

        def prefix_cache(b):
            if b not in caches:
                caches[b] = self.model(input_ids=prefix_ids.expand(b, -1), use_cache=True).past_key_values
            return caches[b]

        probs = np.zeros((len(df), len(LABELS)), dtype=np.float32)
        n_batches = (len(df) + batch_size - 1) // batch_size
        for bi, start in enumerate(range(0, len(df), batch_size)):
            batch = suffixes[start:start + batch_size]
            enc = [self.tok(s, add_special_tokens=False)["input_ids"] for s in batch]
            lens = [len(e) for e in enc]
            width = max(lens)
            ids = torch.full((len(enc), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(enc), prefix_ids.shape[1] + width), dtype=torch.long)
            mask[:, :prefix_ids.shape[1]] = 1
            for i, e in enumerate(enc):                # đệm bên phải: token đệm đứng sau nên không ảnh hưởng token thật
                ids[i, :len(e)] = torch.tensor(e)
                mask[i, prefix_ids.shape[1]:prefix_ids.shape[1] + len(e)] = 1
            cache = copy.deepcopy(prefix_cache(len(enc)))
            out = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                             past_key_values=cache, use_cache=True)
            last = out.logits[torch.arange(len(enc)), torch.tensor(lens) - 1].float()      # logit ở token thật cuối
            logp = torch.log_softmax(last, dim=-1)
            letter = torch.stack([torch.logsumexp(logp[:, idx], dim=-1) for idx in self.letter_ids], dim=-1)
            probs[start:start + len(enc)] = torch.softmax(letter, dim=-1).cpu().numpy()
            if verbose and (bi % 25 == 0 or bi == n_batches - 1):
                print(f"  {min(start + batch_size, len(df))}/{len(df)}", flush=True)
        preds = [LABELS[i] for i in probs.argmax(axis=1)]
        return preds, probs


def run(mode: str = "zero", seed: int = 42, k: int = 6, test_sets: list[str] | None = None,
        model_name: str = DEFAULT_MODEL, limit: int | None = None, batch_size: int = 8, out_dir=RESULTS_DIR,
        clf: LLMClassifier | None = None, verbose: bool = True) -> list[dict]:
    """mode "zero" → kịch bản S0, mode "few" → kịch bản FS (k ví dụ từ ViANLI train, chọn theo seed)."""
    set_seed(seed)
    clf = clf or LLMClassifier(model_name)
    demos = pick_demos(k, seed) if mode == "few" else None
    scenario = "FS" if mode == "few" else "S0"
    test_sets = test_sets or (["xnli_test", "vianli_test", "vinli_test"] if mode == "zero" else ["vianli_test", "vinli_test"])
    results = []
    for name in test_sets:
        df = load_split(*TEST_SETS[name])
        if limit:
            df = df.sample(min(limit, len(df)), random_state=seed)
        info = {"pretrained": model_name, "mode": mode, "k": k if demos is not None else 0,
                "demo_ids": list(demos["id"]) if demos is not None else [], "prompt_version": PROMPT_VERSION,
                "load_in_4bit": clf.load_in_4bit, "device": str(clf.device),
                "n_params_in_memory": sum(p.numel() for p in clf.model.parameters())}   # 4-bit: 2 tham số / phần tử
        if verbose:
            print(f"M7 {scenario} seed {seed} · {name}: {len(df)} cặp", flush=True)
        with timer(info, f"predict_seconds_{name}"):
            preds, probs = clf.predict(df, demos, batch_size=batch_size, verbose=verbose)
        info["pred_distribution"] = {l: int(sum(p == l for p in preds)) for l in LABELS}
        info["mean_prob"] = {l: round(float(v), 4) for l, v in zip(LABELS, probs.mean(axis=0))}
        res = evaluate_and_save(df["label"], preds, model="M7", scenario=scenario, test_set=name, seed=seed,
                                config=info, ids=df["id"], premise=df["premise"], hypothesis=df["hypothesis"],
                                out_dir=out_dir, verbose=verbose)
        results.append(res)
    return results
