"""M7: LLM (Qwen2.5-7B-Instruct, 4-bit) phân loại NLI bằng prompt, không huấn luyện (task W4-12).

Ví dụ:
    python scripts/run_llm.py --smoke                       # chạy thử 30 cặp mỗi tập, lưu results/smoke/
    python scripts/run_llm.py --mode zero                   # S0: zero-shot trên XNLI, ViANLI, ViNLI test
    python scripts/run_llm.py --mode few --seeds 42 123 2024 # FS: 6 ví dụ từ ViANLI train, 3 bộ ví dụ khác nhau
Cần GPU (Colab T4) và thư viện bitsandbytes. Lần chạy nào đã có đủ file kết quả thì bỏ qua (--rerun để chạy lại).
Kết quả: results/M7_S0_*_seed42.json, results/M7_FS_*_seed<seed>.json (+ preds, figures).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import llm  # noqa: E402
from src.evaluate import RESULTS_DIR, run_name, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", nargs="+", default=["zero"], choices=["zero", "few"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--k", type=int, default=6, help="số ví dụ few-shot (chia hết cho 3)")
    ap.add_argument("--test-sets", nargs="+", default=None, choices=list(llm.TEST_SETS))
    ap.add_argument("--model-name", default=llm.DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="chạy thử 30 cặp mỗi tập test")
    args = ap.parse_args()

    clf = None
    for mode in args.mode:
        seeds = [42] if mode == "zero" else args.seeds          # zero-shot không phụ thuộc seed
        sets = args.test_sets or (["xnli_test", "vianli_test", "vinli_test"] if mode == "zero"
                                  else ["vianli_test", "vinli_test"])
        scenario = "S0" if mode == "zero" else "FS"
        for seed in seeds:
            todo = sets if (args.smoke or args.rerun) else [
                t for t in sets if not (RESULTS_DIR / f"{run_name('M7', scenario, t, seed)}.json").exists()]
            if not todo:
                print(f"Bỏ qua M7 {scenario} seed {seed}: đã có kết quả (thêm --rerun để chạy lại)")
                continue
            clf = clf or llm.LLMClassifier(args.model_name)
            extra = {"limit": 30, "out_dir": RESULTS_DIR / "smoke"} if args.smoke else {}
            llm.run(mode, seed, k=args.k, test_sets=todo, model_name=args.model_name, batch_size=args.batch_size,
                    clf=clf, **extra)
    if not args.smoke:
        t = summarize()
        print("\n", t[t["model"] == "M7"].to_string(index=False))


if __name__ == "__main__":
    main()
