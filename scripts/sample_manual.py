"""Chọn 50 mẫu để phân tích thủ công (task W1-06).

25 mẫu XNLI-vi (B phân tích) + 25 mẫu ViANLI (A phân tích), lấy từ tập TRAIN,
mỗi bộ chia đều 3 nhãn (9/8/8). Cố định seed để tái lập.

Chạy: python scripts/sample_manual.py
Kết quả trong data/manual/:
    50_mau.csv                   file phân tích chính (có nhãn gốc)
    gan_nhan_doc_lap_A.csv       25 mẫu ViANLI cho A tự gán nhãn, KHÔNG có nhãn gốc và id
    gan_nhan_doc_lap_B.csv       25 mẫu XNLI-vi cho B tự gán nhãn, KHÔNG có nhãn gốc và id
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import LABELS, ROOT, load_split  # noqa: E402

SEED = 42
PER_DATASET = 25
OUT_DIR = ROOT / "data" / "manual"
# Phân công (đã đổi ngày 19/10: Quân làm XNLI, Quang làm ViANLI). Người gán nhãn độc lập giữ nguyên.
ANALYST = {"xnli": "B", "vianli": "A"}
BLIND_LABELER = {"xnli": "B", "vianli": "A"}


def pick(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    base, extra = divmod(n, len(LABELS))
    parts = []
    for i, label in enumerate(LABELS):
        k = base + (1 if i < extra else 0)
        parts.append(df[df["label"] == label].sample(n=k, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed)


def main():
    out_file = OUT_DIR / "50_mau.csv"
    if out_file.exists() and "--force" not in sys.argv:
        filled = pd.read_csv(out_file)["the_hien_tuong"].notna().sum()
        if filled:
            sys.exit(f"{out_file} đã có {filled} dòng được phân tích. Chạy lại sẽ xóa hết; thêm --force nếu chắc chắn.")
    rows = []
    for dataset in ("xnli", "vianli"):
        df = pick(load_split(dataset, "train"), PER_DATASET, SEED)
        df["nguoi_phan_tich"] = ANALYST[dataset]
        df["nguoi_gan_doc_lap"] = BLIND_LABELER[dataset]
        rows.append(df)
    out = pd.concat(rows).reset_index(drop=True)
    out.insert(0, "stt", range(1, len(out) + 1))
    out = out.rename(columns={"source": "nguon", "label": "nhan_goc"})
    out = out[["stt", "id", "nguon", "nguoi_phan_tich", "nguoi_gan_doc_lap", "premise", "hypothesis", "nhan_goc"]]
    for col in ("nhan_tu_gan", "the_hien_tuong", "giai_thich", "ghi_chu"):
        out[col] = ""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # utf-8-sig để mở bằng Excel không lỗi font tiếng Việt
    out.to_csv(OUT_DIR / "50_mau.csv", index=False, encoding="utf-8-sig")
    for person in ("A", "B"):
        # Không đưa id vào file gán nhãn độc lập: uid của ViANLI chứa mã nhãn (…_11_… = entailment,
        # …_21_… = contradiction, …_31_… = neutral), nhìn id là đoán được đáp án. Ghép lại bằng stt.
        blind = out[out["nguoi_gan_doc_lap"] == person][["stt", "premise", "hypothesis"]].copy()
        blind["nhan_tu_gan"] = ""
        blind.to_csv(OUT_DIR / f"gan_nhan_doc_lap_{person}.csv", index=False, encoding="utf-8-sig")

    print(out.groupby(["nguon", "nhan_goc"]).size().unstack())
    print(f"Đã lưu vào {OUT_DIR}")


if __name__ == "__main__":
    main()
