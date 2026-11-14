"""Vẽ biểu đồ so sánh các mô hình cho báo cáo mục 4 (task W5-01).

Chạy: python scripts/plot_results.py
Ra:   results/figures/so_sanh_mo_hinh.png
Accuracy trên ViANLI test và ViNLI test, mỗi nhóm cột là một mô hình, màu cột là kịch bản.
Mô hình nhiều seed: cột là trung bình, vạch là ± độ lệch chuẩn. Mô hình nào có kết quả thì tự được vẽ.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluate import RESULTS_DIR, summarize

MODELS = [("M0", "M0"), ("M1", "M1"), ("M1h", "M1-h"), ("M2", "M2"), ("M3", "M3"), ("M4", "PhoBERT"), ("M5", "XLM-R"), ("M6", "mDeBERTa")]
SCENARIOS = [("S0", "S0: M6 chưa fine-tune"), ("S1", "S1: học XNLI"), ("S2", "S2: học ViANLI"), ("S3", "S3: XNLI → ViANLI")]
COLORS = {"S0": "#8a6fd1", "S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a"}   # bảng màu phân loại, đã kiểm tra khả năng phân biệt màu
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def main():
    t = summarize()
    present = [(m, name) for m, name in MODELS if m in set(t["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True, facecolor=SURFACE)
    width = 0.26
    for ax, (test_set, title) in zip(axes, [("vianli_test", "ViANLI test (đối kháng)"),
                                            ("vinli_test", "ViNLI test (tiếng Việt gốc)")]):
        ax.set_facecolor(SURFACE)
        x = np.arange(len(present))
        for k, (sc, label) in enumerate(SCENARIOS):
            for i, (m, _) in enumerate(present):
                row = t[(t["model"] == m) & (t["scenario"] == sc) & (t["test_set"] == test_set)]
                if row.empty:
                    continue
                mean, std = row["accuracy_mean"].iloc[0] * 100, row["accuracy_std"].iloc[0] * 100
                xpos = x[i] + (max(k, 1) - 2) * width   # S0 (chỉ M6) đứng ở chỗ của S1
                ax.bar(xpos, mean, width - 0.03, color=COLORS[sc], label=label if i == 0 or not ax.containers else None,
                       zorder=3)
                if not np.isnan(std):
                    ax.errorbar(xpos, mean, yerr=std, color=TEXT, capsize=3, linewidth=1, zorder=4)
        ax.axhline(100 / 3, color=TEXT2, linestyle="--", linewidth=1, zorder=2)
        ax.set_xticks(x, [name for _, name in present], color=TEXT)
        ax.set_title(title, color=TEXT, fontsize=11)
        ax.set_ylim(0, 80)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT2)
    axes[0].set_ylabel("Accuracy (%)", color=TEXT)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s, _ in SCENARIOS]
    handles.append(plt.Line2D([0], [0], color=TEXT2, linestyle="--", linewidth=1))
    fig.legend(handles, [l for _, l in SCENARIOS] + ["mức ngẫu nhiên 33,3%"], loc="upper center", ncol=5,
               frameon=False, fontsize=9,
               labelcolor=TEXT, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, -0.02, "M2, M3, PhoBERT ở S2 và PhoBERT ở S3: trung bình ± độ lệch chuẩn qua 3 seed; các cột khác: seed 42.",
             ha="center", fontsize=8, color=TEXT2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = RESULTS_DIR / "figures" / "so_sanh_mo_hinh.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print("Đã lưu", out)


if __name__ == "__main__":
    main()
