"""Fine-tune M4 (PhoBERT) hoặc M5 (XLM-R).

Ví dụ:
    python scripts/run_transformer.py --model M4 --scenario S2 --smoke          # chạy thử 1.000 mẫu, 1 epoch (W3-02)
    python scripts/run_transformer.py --model M4 --scenario S2 --config A B     # W3-03: so 2 cấu hình trên dev, giữ cấu hình tốt hơn
    python scripts/run_transformer.py --model M4 --scenario S1                  # W4-01 (cấu hình A)
    python scripts/run_transformer.py --model M4 --scenario S3 --config B       # W4-02: học tiếp từ checkpoint S1 seed 42
    python scripts/run_transformer.py --model M4 --scenario S2 S3 --config B --seeds 123 2024   # W4-03
    python scripts/run_transformer.py --token-lengths                           # đo độ dài token để chọn max_len
Cần GPU (Colab T4). PhoBERT dùng dữ liệu đã tách từ: chạy scripts/segment_data.py trước.
Kết quả: results/M4_*.json (bản chạy thử: results/smoke/), checkpoint: checkpoints/<mô hình>_<kịch bản>_seed<seed>/

Lần chạy nào đã có đủ file kết quả thì bỏ qua (Colab bị ngắt thì chạy lại lệnh là làm tiếp); --rerun để chạy lại.
Chỉ giữ checkpoint của S1 (S3 cần để học tiếp); các kịch bản khác chỉ giữ file kết quả và file dự đoán, để Drive không đầy.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import transformer  # noqa: E402
from src.evaluate import RESULTS_DIR, run_name, summarize  # noqa: E402

TEST_SETS = {"S1": ["xnli_test", "vianli_test", "vinli_test"], "S2": ["vianli_test", "vinli_test"],
             "S3": ["vianli_test", "vinli_test"]}


def done(model: str, scenario: str, seed: int) -> bool:
    return all((RESULTS_DIR / f"{run_name(model, scenario, t, seed)}.json").exists() for t in TEST_SETS[scenario])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M4", choices=list(transformer.MODELS))
    ap.add_argument("--scenario", nargs="+", default=["S2"], choices=["S1", "S2", "S3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--config", nargs="+", default=["A"], choices=list(transformer.CONFIGS))
    ap.add_argument("--init-from", default=None,
                    help="checkpoint để học tiếp (kịch bản S3); mặc định checkpoints/<mô hình>_S1_seed42")
    ap.add_argument("--rerun", action="store_true", help="chạy lại cả những lần đã có kết quả")
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
        init_from = None
        if scenario == "S3":
            init_from = Path(args.init_from) if args.init_from else transformer.CHECKPOINT_DIR / f"{args.model}_S1_seed42"
            if not init_from.exists():
                sys.exit(f"Kịch bản S3 cần checkpoint S1: không thấy {init_from}. Chạy --scenario S1 trước.")
        for seed in args.seeds:
            if not args.smoke and not args.rerun and done(args.model, scenario, seed):
                print(f"Bỏ qua {args.model} {scenario} seed {seed}: đã có kết quả (thêm --rerun để chạy lại)")
                continue
            if len(args.config) > 1 and not args.smoke:
                # nhiều cấu hình: so trên dev, giữ cấu hình tốt nhất (xem transformer.tune)
                transformer.tune(args.model, scenario, args.config, seed, init_from=init_from)
            else:
                keep = {} if args.smoke else {"keep_checkpoint": scenario == "S1"}
                for config in args.config:
                    transformer.run(args.model, scenario, seed, config=config, init_from=init_from, **keep, **smoke)

    if not args.smoke:
        table = summarize()
        print("\n", table[table["model"] == args.model].to_string(index=False))


if __name__ == "__main__":
    main()
