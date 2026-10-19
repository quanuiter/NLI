"""Chuẩn bị word embeddings cho M2, M3 (task W2-08).

Chạy: python scripts/prepare_embeddings.py
1. Tải fastText cc.vi.300 (~1,2 GB) vào tools/embeddings/ nếu chưa có
2. Đọc bằng Gensim, chỉ giữ token có trong dữ liệu đồ án -> data/processed/emb_fasttext.npz
3. Tính tỉ lệ OOV từng bộ/split -> results/analysis/oov_fasttext.json
4. In vài cặp từ gần nghĩa (most_similar) để kiểm tra chất lượng vector
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings import build_embedding, corpus_vocab, download_fasttext, load_embedding, oov_report  # noqa: E402
from src.evaluate import RESULTS_DIR  # noqa: E402
from src.utils import save_json  # noqa: E402

CHECK_WORDS = ["tăng", "bệnh_viện", "bệnh", "Hà", "thắng", "không"]


def main():
    vec_path = download_fasttext()
    vocab = corpus_vocab()
    print(f"Dữ liệu có {len(vocab):,} token khác nhau")
    build_embedding(vocab, vec_path)

    word2idx, matrix = load_embedding()
    report = oov_report(word2idx)
    save_json(report, RESULTS_DIR / "analysis" / "oov_fasttext.json")
    print(f"\n{'Tập':<22}{'OOV theo lượt':>14}{'OOV theo từ':>13}  Ví dụ từ không có vector")
    for name, r in report.items():
        print(f"{name:<22}{r['oov_token_rate']:>14.2%}{r['oov_type_rate']:>13.2%}  {', '.join(r['oov_examples'][:6])}")

    # Kiểm tra nhanh: các từ gần nghĩa nhất trong không gian vector (chỉ trong vocab đã lọc)
    import numpy as np

    idx2word = {i: w for w, i in word2idx.items()}
    unit = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    print("\nTừ gần nghĩa nhất (cosine):")
    for w in CHECK_WORDS:
        if w not in word2idx:
            continue
        sims = unit @ unit[word2idx[w]]
        best = [idx2word[i] for i in np.argsort(-sims)[1:7] if i in idx2word]
        print(f"  {w}: {', '.join(best)}")


if __name__ == "__main__":
    main()
