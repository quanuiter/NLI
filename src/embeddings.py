"""Word embeddings tiếng Việt cho M2, M3 (task W2-08).

Dùng fastText cc.vi.300 (Grave et al., 2018), 300 chiều, học trên Common Crawl + Wikipedia tiếng Việt.
Đọc bằng Gensim (KeyedVectors.load_word2vec_format), chỉ giữ các từ có trong dữ liệu của đồ án rồi lưu
thành file nhỏ data/processed/emb_<tên>.npz để M2, M3 load nhanh.

Token hóa ở mức âm tiết (tách theo khoảng trắng và dấu câu), giống cách fastText cc tách từ tiếng Việt.
Tra vector theo dạng gốc trước, không có thì thử chữ thường và cách bỏ dấu kiểu cũ (hóa -> hoá).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from .data import DATASET_SPLITS, PROCESSED_DIR, ROOT, load_split

EMB_DIR = ROOT / "tools" / "embeddings"
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.vec.gz"
FASTTEXT_FILE = EMB_DIR / "cc.vi.300.vec.gz"

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tách câu thành âm tiết / số, bỏ dấu câu. "Hà Nội, 3/4." -> ["Hà", "Nội", "3", "4"]"""
    return _TOKEN.findall(text)


_TONES = "\u0300\u0301\u0303\u0309\u0323"  # huyền, sắc, ngã, hỏi, nặng (dạng dấu tổ hợp)
_NEW_STYLE = re.compile(rf"([oO])([{_TONES}])([aAeE])$|([uU])([{_TONES}])([yY])$")


def old_style_tone(token: str) -> str:
    """Đổi cách bỏ dấu kiểu mới sang kiểu cũ ở vần oa, oe, uy cuối âm tiết: hóa -> hoá, khỏe -> khoẻ, thủy -> thuỷ.

    fastText cc.vi (và VnCoreNLP / PhoBERT) dùng kiểu cũ, nên "hóa", "tòa"... không có vector nếu không đổi.
    """
    nfd = unicodedata.normalize("NFD", token)
    fixed = _NEW_STYLE.sub(lambda m: (m[1] + m[3] + m[2]) if m[1] else (m[4] + m[6] + m[5]), nfd)
    return unicodedata.normalize("NFC", fixed)


def lookup_forms(token: str) -> list[str]:
    """Các dạng thử lần lượt khi tra vector: nguyên dạng, chữ thường, kiểu dấu cũ, kiểu dấu cũ + chữ thường."""
    forms = [token, token.lower(), old_style_tone(token), old_style_tone(token.lower())]
    return list(dict.fromkeys(forms))


def corpus_vocab() -> Counter:
    """Đếm token trên mọi split của mọi bộ dữ liệu (không dùng nhãn nên không rò rỉ thông tin test)."""
    counts = Counter()
    for dataset, splits in DATASET_SPLITS.items():
        for split in splits:
            df = load_split(dataset, split)
            for text in df["premise"].tolist() + df["hypothesis"].tolist():
                counts.update(tokenize(text))
    return counts


def download_fasttext(path: Path = FASTTEXT_FILE) -> Path:
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Đang tải {FASTTEXT_URL} (~1,2 GB)...")
        urllib.request.urlretrieve(FASTTEXT_URL, path)
    return path


def build_embedding(vocab: Counter, vec_path: Path, name: str = "fasttext") -> Path:
    """Load vector bằng Gensim, giữ các token trong vocab (khớp nguyên dạng hoặc chữ thường), lưu .npz."""
    from gensim.models import KeyedVectors

    print(f"Đang đọc {vec_path} bằng Gensim (mất vài phút)...")
    kv = KeyedVectors.load_word2vec_format(str(vec_path), binary=False)
    words, vectors = [], []
    for token in vocab:
        for form in lookup_forms(token):
            if form in kv.key_to_index:
                words.append(token)
                vectors.append(kv[form])
                break
    out = PROCESSED_DIR / f"emb_{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, words=np.array(words), vectors=np.asarray(vectors, dtype=np.float32))
    print(f"Giữ {len(words):,}/{len(vocab):,} token có vector -> {out}")
    return out


def load_embedding(name: str = "fasttext") -> tuple[dict[str, int], np.ndarray]:
    """Trả về (word2idx, ma trận vector). Hàng 0 dành cho token không có vector (vector 0)."""
    data = np.load(PROCESSED_DIR / f"emb_{name}.npz")
    words = data["words"].tolist()
    matrix = np.vstack([np.zeros((1, data["vectors"].shape[1]), dtype=np.float32), data["vectors"]])
    return {w: i + 1 for i, w in enumerate(words)}, matrix


def oov_report(word2idx: dict[str, int]) -> dict:
    """Tỉ lệ token không có vector (OOV), tính theo lượt xuất hiện và theo từ khác nhau, cho từng bộ/split."""
    report = {}
    for dataset, splits in DATASET_SPLITS.items():
        for split in splits:
            df = load_split(dataset, split)
            counts = Counter(t for s in df["premise"].tolist() + df["hypothesis"].tolist() for t in tokenize(s))
            n_tokens = sum(counts.values())
            oov = {t: c for t, c in counts.items() if t not in word2idx}
            report[f"{dataset}_{split}"] = {
                "tokens": n_tokens,
                "oov_token_rate": round(sum(oov.values()) / max(1, n_tokens), 4),
                "types": len(counts),
                "oov_type_rate": round(len(oov) / max(1, len(counts)), 4),
                "oov_examples": sorted(oov, key=oov.get, reverse=True)[:15],
            }
    return report
