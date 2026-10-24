"""Fine-tune M4 (PhoBERT) hoặc M5 (XLM-R).

Ví dụ:
    python scripts/run_transformer.py --model M4 --scenario S2 --smoke          # chạy thử 1.000 mẫu, 1 epoch (W3-02)
    python scripts/run_transformer.py --model M4 --scenario S2 --config A B     # W3-03: so 2 cấu hình trên dev
    python scripts/run_transformer.py --model M4 --scenario S3 --init-from checkpoints/M4_S1_seed42   # W4-02
    python scripts/run_transformer.py --token-lengths                           # đo độ dài token để chọn max_len
Cần GPU (Colab T4). PhoBERT dùng dữ liệu đã tách từ: chạy scripts/segment_data.py trước.
Kết quả: results/M4_*.json (bản chạy thử: results/smoke/), checkpoint: checkpoints/<mô hình>_<kịch bản>_seed<seed>/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import transformer  # noqa: E402
from src.evaluate import RESULTS_DIR, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M4", choices=list(transformer.MODELS))
    ap.add_argument("--scenario", nargs="+", default=["S2"], choices=["S1", "S2", "S3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--config", nargs="+", default=["A"], choices=list(transformer.CONFIGS))
    ap.add_argument("--init-from", default=None, help="checkpoint để học tiếp (kịch bản S3)")
    ap.add_argument("--smoke", action="store_true", help="chạy thử: 1.000 mẫu train, 200 dev, 200 test, 1 epoch")
    ap.add_argument("--token-lengths", action="store_true", help="chỉ đo độ dài token, lưu results/analysis")
    args = ap.parse_args()

    if args.token_lengths:
        for name, stats in transformer.save_token_lengths().items():
            print(name, stats)
        return

    smoke = {"train_limit": 1000, "dev_limit": 200, "test_limit": 200, "cfg": {"epochs": 1},
             "out_dir": RESULTS_DIR / "smoke", "keep_checkpoint": False} if args.smoke else {}
    for scenario in args.scenario:
        if scenario == "S3" and not args.init_from:
            sys.exit("Kịch bản S3 cần --init-from <checkpoint S1>")
        for config in args.config:
            for seed in args.seeds:
                transformer.run(args.model, scenario, seed, config=config, init_from=args.init_from, **smoke)

    if not args.smoke:
        table = summarize()
        print("\n", table[table["model"] == args.model].to_string(index=False))


if __name__ == "__main__":
    main()
