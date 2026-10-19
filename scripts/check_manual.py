"""Kiểm tra file phân tích thủ công và tính độ đồng thuận (W1-09, W2-03, W2-04).

Chạy: python scripts/check_manual.py
- Báo thẻ không hợp lệ, ô còn trống trong data/manual/50_mau.csv
- Tần suất thẻ theo bộ dữ liệu và theo nhãn
- Cohen's kappa giữa nhãn tự gán (gan_nhan_doc_lap_A/B.csv) và nhãn gốc, nếu đã điền
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import LABELS, ROOT  # noqa: E402

MANUAL_DIR = ROOT / "data" / "manual"
# Giữ khớp với bảng thẻ trong data/manual/huong_dan_gan_the.md
TAGS = ["LEX", "PARA", "ANT", "NEG", "NUM", "QUANT", "ROLE", "COREF", "WORLD", "SPEC", "TRANS"]


def split_tags(value) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [t.strip().upper() for t in str(value).split(";") if t.strip()]


def check_analysis(df: pd.DataFrame) -> None:
    problems = []
    for _, r in df.iterrows():
        tags = split_tags(r["the_hien_tuong"])
        who = f"stt {r['stt']} ({r['nguoi_phan_tich']})"
        if not tags:
            problems.append(f"{who}: chưa gắn thẻ")
        bad = [t for t in tags if t not in TAGS]
        if bad:
            problems.append(f"{who}: thẻ không hợp lệ {bad}")
        if len(tags) > 3:
            problems.append(f"{who}: quá 3 thẻ")
        if "TRANS" in tags and r["nguon"] != "xnli":
            problems.append(f"{who}: TRANS chỉ dùng cho XNLI")
        if pd.isna(r["giai_thich"]) or not str(r["giai_thich"]).strip():
            problems.append(f"{who}: chưa có giải thích")

    done = df["the_hien_tuong"].map(split_tags).map(bool)
    print(f"Đã phân tích {done.sum()}/{len(df)} mẫu "
          f"(A: {done[df['nguoi_phan_tich'] == 'A'].sum()}, B: {done[df['nguoi_phan_tich'] == 'B'].sum()})")
    print(f"\n{len(problems)} vấn đề" + (":" if problems else ""))
    for p in problems[:30]:
        print("  -", p)
    if len(problems) > 30:
        print(f"  ... và {len(problems) - 30} vấn đề khác")

    rows = [(r["nguon"], r["nhan_goc"], t) for _, r in df.iterrows() for t in split_tags(r["the_hien_tuong"])]
    if rows:
        tags = pd.DataFrame(rows, columns=["nguon", "nhan_goc", "the"])
        print("\nTần suất thẻ theo bộ dữ liệu:")
        print(tags.pivot_table(index="the", columns="nguon", aggfunc="size", fill_value=0).reindex(TAGS, fill_value=0))
        print("\nTần suất thẻ theo nhãn gốc:")
        print(tags.pivot_table(index="the", columns="nhan_goc", aggfunc="size", fill_value=0)
              .reindex(index=TAGS, columns=LABELS, fill_value=0))
        n_flag = df["ghi_chu"].fillna("").astype(str).str.contains("nghi_sai_nhan").sum()
        print(f"\nSố mẫu nghi sai nhãn: {n_flag}")


def check_agreement(df: pd.DataFrame) -> None:
    gold = df.set_index("stt")["nhan_goc"]
    for person in ("A", "B"):
        path = MANUAL_DIR / f"gan_nhan_doc_lap_{person}.csv"
        blind = pd.read_csv(path)
        blind["nhan_tu_gan"] = blind["nhan_tu_gan"].fillna("").astype(str).str.strip().str.lower()
        filled = blind[blind["nhan_tu_gan"].isin(LABELS)]
        invalid = blind[~blind["nhan_tu_gan"].isin(LABELS + [""])]
        print(f"\nGán nhãn độc lập {person}: {len(filled)}/{len(blind)} mẫu hợp lệ")
        for _, r in invalid.iterrows():
            print(f"  - stt {r['stt']}: nhãn không hợp lệ {r['nhan_tu_gan']!r}")
        if len(filled) < 2:
            continue
        y_gold = gold.loc[filled["stt"]].tolist()
        y_self = filled["nhan_tu_gan"].tolist()
        acc = sum(a == b for a, b in zip(y_gold, y_self)) / len(filled)
        print(f"  Trùng nhãn gốc: {acc:.1%}   Cohen's kappa: {cohen_kappa_score(y_gold, y_self, labels=LABELS):.3f}")


def main():
    df = pd.read_csv(MANUAL_DIR / "50_mau.csv")
    check_analysis(df)
    check_agreement(df)


if __name__ == "__main__":
    main()
