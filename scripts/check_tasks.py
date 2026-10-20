"""Kiểm tra sản phẩm của các task: file kết quả, notebook, bảng ablation.

Cách dùng (trên Colab sau ô 0, hoặc trên máy có thư mục results/ đã đồng bộ từ Drive):
    python scripts/check_tasks.py                      # kiểm tra mọi task
    python scripts/check_tasks.py --task W2-06         # chỉ một task
    python scripts/check_tasks.py --results /content/drive/MyDrive/NLI/results

Ký hiệu: [OK] đạt · [LỖI] sai hoặc thiếu, phải sửa · [?] cần mở ra xem bằng mắt.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import ROOT, load_split
from src.evaluate import RESULTS_DIR, run_name

TEST_SETS = {"S1": ["xnli_test", "vianli_test", "vinli_test"], "S2": ["vianli_test", "vinli_test"]}
TEST_SIZE = {"xnli_test": 5010, "vianli_test": 1000, "vinli_test": 2264}


class Report:
    def __init__(self):
        self.n_err = 0
        self.n_warn = 0

    def ok(self, msg):
        print(f"  [OK]   {msg}")

    def err(self, msg):
        self.n_err += 1
        print(f"  [LỖI]  {msg}")

    def warn(self, msg):
        self.n_warn += 1
        print(f"  [?]    {msg}")


# ------------------------------------------------------------------ kết quả (results/*.json)
_test_ids: dict[str, set] = {}


def test_ids(test_set: str) -> set:
    if test_set not in _test_ids:
        dataset = test_set.replace("_test", "")
        _test_ids[test_set] = set(load_split(dataset, "test")["id"])
    return _test_ids[test_set]


def check_run(rep: Report, results: Path, model: str, scenario: str, test_set: str, seed: int = 42) -> dict | None:
    """Kiểm tra một lần chạy: có file JSON, đúng số mẫu, file dự đoán khớp tập test và khớp độ đo."""
    name = run_name(model, scenario, test_set, seed)
    path = results / f"{name}.json"
    if not path.exists():
        rep.err(f"thiếu {path.name}")
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if r.get("n") != TEST_SIZE[test_set]:
        problems.append(f"n = {r.get('n')}, đúng phải là {TEST_SIZE[test_set]}")
    if not r.get("config"):
        problems.append("config trống (chưa ghi siêu tham số)")
    preds_path = results / "preds" / f"{name}.csv"
    if not preds_path.exists():
        problems.append("thiếu file dự đoán preds/*.csv")
    else:
        preds = pd.read_csv(preds_path)
        if "id" not in preds:
            problems.append("file dự đoán không có cột id (chưa truyền ids=df['id'])")
        elif set(preds["id"]) != test_ids(test_set):
            problems.append(f"id trong file dự đoán không khớp tập {test_set}")
        acc = (preds["label"] == preds["pred"]).mean()
        if abs(acc - r["accuracy"]) > 1e-9:
            problems.append(f"accuracy tính lại từ file dự đoán ({acc:.4f}) khác file JSON ({r['accuracy']:.4f})")
    if problems:
        rep.err(f"{name}: " + "; ".join(problems))
    else:
        rep.ok(f"{name}: acc {r['accuracy']:.3f} · macro-F1 {r['macro_f1']:.3f} · config {r['config']}")
    return r


def check_model(rep: Report, results: Path, model: str) -> dict:
    out = {}
    for scenario, sets in TEST_SETS.items():
        for test_set in sets:
            r = check_run(rep, results, model, scenario, test_set)
            if r:
                out[(scenario, test_set)] = r
    return out


# ------------------------------------------------------------------ notebook
TEMPLATE_AUTHOR = "quanuiter"   # bản mẫu của notebook = commit gần nhất của người viết khung (Quân)


def load_cells(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def template_cells(path: Path) -> list[dict] | None:
    """Bản mẫu của notebook lấy từ lịch sử git, để biết ô TODO đã được sửa hay chưa."""
    try:
        rel = path.resolve().relative_to(ROOT).as_posix()
        sha = subprocess.run(["git", "-C", str(ROOT), "log", "-n1", "--format=%H", f"--author={TEMPLATE_AUTHOR}",
                              "--", rel], capture_output=True, text=True, check=True).stdout.strip()
        if not sha:
            return None
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha}:{rel}"],
                             capture_output=True, text=True, check=True).stdout
        return json.loads(src)["cells"]
    except (ValueError, subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return None


def find_todo(cells: list[dict], marker: str) -> dict | None:
    for c in cells:
        if c["cell_type"] == "code" and re.search(rf"#\s*TODO\s*{re.escape(marker)}\b", "".join(c["source"])):
            return c
    return None


def check_notebook(rep: Report, path: Path, markers: list[str]) -> list[dict] | None:
    """Notebook đã lưu kèm kết quả chạy, không có ô lỗi, các ô TODO trong `markers` đã được viết."""
    if not path.exists():
        rep.err(f"thiếu {path}")
        return None
    cells = load_cells(path)
    code = [c for c in cells if c["cell_type"] == "code"]
    ran = [c for c in code if c.get("outputs")]
    if not ran:
        rep.err(f"{path.name}: không ô nào có kết quả chạy. Notebook chưa chạy, hoặc đã bật tùy chọn bỏ kết quả khi lưu "
                "(Colab: Edit → Notebook settings → bỏ chọn 'Omit code cell output when saving this notebook')")
    else:
        rep.ok(f"{path.name}: {len(ran)}/{len(code)} ô code có kết quả chạy")
    for i, c in enumerate(cells):
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                rep.err(f"{path.name}, ô {i}: còn lỗi {o.get('ename')}: {o.get('evalue', '')[:80]}")
    template = template_cells(path)
    if template is None:
        rep.warn(f"{path.name}: không lấy được bản mẫu từ git, chỉ kiểm tra được ô TODO còn trống hay không")
    for marker in markers:
        cell = find_todo(cells, marker)
        if cell is None:
            rep.warn(f"{path.name}: không tìm thấy ô 'TODO {marker}' (có thể đã xóa dòng TODO), mở notebook xem bằng mắt")
            continue
        src = "".join(cell["source"])
        tmpl = find_todo(template, marker) if template else None
        if tmpl is not None:
            unchanged = src.strip() == "".join(tmpl["source"]).strip()
        else:
            unchanged = not [l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
        if unchanged:
            rep.err(f"{path.name}: ô TODO {marker} chưa làm (giống hệt bản mẫu)")
        elif not cell.get("outputs"):
            rep.warn(f"{path.name}: ô TODO {marker} đã sửa nhưng chưa có kết quả chạy")
        else:
            rep.ok(f"{path.name}: ô TODO {marker} đã sửa và đã chạy")
    return cells


def check_no_test_tuning(rep: Report, path: Path, cells: list[dict]) -> None:
    """Dò các dòng dùng tập test để tính điểm hoặc huấn luyện (ngoài evaluate_and_save)."""
    bad = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        for line in "".join(c["source"]).splitlines():
            s = line.split("#")[0]
            if "evaluate_and_save" in s or "test" not in s:
                continue
            if re.search(r"(f1_score|accuracy_score|\.fit\(|fit_transform)", s):
                bad.append(f"ô {i}: {s.strip()[:90]}")
    if bad:
        for b in bad:
            rep.warn(f"{path.name}: dòng có thể dùng tập test để chọn cấu hình, kiểm tra lại: {b}")
    else:
        rep.ok(f"{path.name}: không thấy dòng nào dùng tập test để tính điểm hay huấn luyện ngoài evaluate_and_save")


def check_no_id_feature(rep: Report, path: Path, cells: list[dict]) -> None:
    bad = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        for line in "".join(c["source"]).splitlines():
            s = line.split("#")[0]
            if re.search(r"\[\s*['\"](id|uid)['\"]\s*\]", s) and "ids=" not in s:
                bad.append(f"ô {i}: {s.strip()[:90]}")
    for b in bad:
        rep.warn(f"{path.name}: có dùng cột id ngoài ids=..., kiểm tra không dùng id làm đặc trưng: {b}")
    if not bad:
        rep.ok(f"{path.name}: không dùng cột id làm đặc trưng")


# ------------------------------------------------------------------ từng task
def task_w1_07(rep, results, nb_dir):
    check_notebook(rep, nb_dir / "01_eda.ipynb", ["1", "2", "3", "4"])
    figs = sorted((results / "figures").glob("eda_*.png"))
    if len(figs) >= 2:
        rep.ok(f"có {len(figs)} hình EDA: " + ", ".join(f.name for f in figs))
    else:
        rep.err(f"cần ít nhất 2 hình results/figures/eda_*.png, hiện có {len(figs)}")


def task_w1_08(rep, results, nb_dir):
    cells = check_notebook(rep, nb_dir / "01_eda.ipynb", ["6", "7"])
    if cells:
        md = "".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
        if re.search(r"\|\s*Loại lỗi", md):
            rep.ok("01_eda.ipynb: có bảng tổng kết lỗi (TODO 8)")
        else:
            rep.err("01_eda.ipynb: chưa có bảng tổng kết lỗi 'Loại lỗi | Số lượng | Cách xử lý' (TODO 8)")


def task_w2_05(rep, results, nb_dir):
    for (scenario, test_set), r in check_model(rep, results, "M0").items():
        if abs(r["accuracy"] - 1 / 3) > 0.02 or abs(r["macro_f1"] - 1 / 6) > 0.01:
            rep.err(f"M0 {scenario} {test_set}: acc {r['accuracy']:.3f}, F1 {r['macro_f1']:.3f}, phải khoảng 0,33 / 0,17")


def task_w2_06(rep, results, nb_dir):
    m1 = check_model(rep, results, "M1")
    for key, r in m1.items():
        m0 = results / f"{run_name('M0', *key, 42)}.json"
        if m0.exists() and r["macro_f1"] <= json.loads(m0.read_text(encoding="utf-8"))["macro_f1"]:
            rep.err(f"M1 {key[0]} {key[1]}: macro-F1 không cao hơn M0")
        cfg = r.get("config", {})
        if not any(k in cfg for k in ("C", "alpha")):
            rep.warn(f"M1 {key[0]} {key[1]}: config không có C hoặc alpha, kiểm tra đã ghi đủ siêu tham số chưa")
    path = nb_dir / "02_m1.ipynb"
    cells = check_notebook(rep, path, ["a", "b", "c", "d, e"])
    if cells:
        check_no_test_tuning(rep, path, cells)
        check_no_id_feature(rep, path, cells)


def task_w2_07(rep, results, nb_dir):
    m1h = check_model(rep, results, "M1h")
    r = m1h.get(("S2", "vianli_test"))
    if r and r["accuracy"] > 0.50:
        rep.err(f"M1h S2 vianli_test: acc {r['accuracy']:.3f} > 0,50. Kiểm tra lỡ dùng premise hoặc id không")
    for key, r in m1h.items():
        name_m1 = run_name("M1", *key, 42)
        m1 = results / f"{name_m1}.json"
        if m1.exists() and r["macro_f1"] > json.loads(m1.read_text(encoding="utf-8"))["macro_f1"] + 0.02:
            rep.warn(f"M1h {key[0]} {key[1]} cao hơn M1: xem lại có phải M1 đang lỗi không")
        p1, p2 = results / "preds" / f"{name_m1}.csv", results / "preds" / f"{run_name('M1h', *key, 42)}.csv"
        if p1.exists() and p2.exists() and pd.read_csv(p1)["pred"].equals(pd.read_csv(p2)["pred"]):
            rep.err(f"{key[0]} {key[1]}: dự đoán của M1 và M1h giống hệt nhau (lưu nhầm mô hình?)")
    path = nb_dir / "02_m1.ipynb"
    cells = check_notebook(rep, path, ["f"])
    if cells:
        src = "".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
        tail = src[src.find("TODO f"):] if "TODO f" in src else ""
        if re.search(r"\[\s*['\"]premise['\"]\s*\]|compare_features\(", tail.split("summarize(")[0]):
            rep.warn("02_m1.ipynb: ô TODO f có nhắc tới premise hoặc compare_features, M1-h chỉ được dùng hypothesis")


def task_w3_06(rep, results, nb_dir):
    script = ROOT / "scripts" / "make_table.py"
    if not script.exists():
        rep.err("thiếu scripts/make_table.py")
        return
    rep.ok("có scripts/make_table.py")
    for name in ["summary.csv", "summary_bang.csv"]:
        if (results / name).exists():
            rep.ok(f"có results/{name}")
        else:
            rep.err(f"thiếu results/{name}: chạy python scripts/make_table.py")
    if not (results / "summary_bang_acc.csv").exists():
        rep.err("thiếu results/summary_bang_acc.csv (phần TODO bảng accuracy)")


def task_w3_09(rep, results, nb_dir):
    path = results / "analysis" / "m1_ablation.csv"
    if not path.exists():
        rep.err(f"thiếu {path}")
        return
    df = pd.read_csv(path)
    need = {"kich_ban", "cau_hinh", "dev_macro_f1", "chenh_lech"}
    if not need <= set(df.columns):
        rep.err(f"m1_ablation.csv thiếu cột: {need - set(df.columns)}")
        return
    for s in ["S1", "S2"]:
        n = (df["kich_ban"] == s).sum()
        (rep.ok if n == 9 else rep.err)(f"m1_ablation.csv: {s} có {n}/9 cấu hình")
    for s in ["S1", "S2"]:
        full = results / f"{run_name('M1', s, TEST_SETS[s][0], 42)}.json"
        row = df[(df["kich_ban"] == s) & (df["cau_hinh"] == "(đủ đặc trưng)")]
        if full.exists() and len(row):
            cfg = json.loads(full.read_text(encoding="utf-8")).get("config", {})
            if "dev_macro_f1" in cfg and abs(cfg["dev_macro_f1"] - row["dev_macro_f1"].iloc[0]) > 1e-3:
                rep.err(f"{s}: dòng '(đủ đặc trưng)' = {row['dev_macro_f1'].iloc[0]} khác dev_macro_f1 của M1 "
                        f"({cfg['dev_macro_f1']}): MO_HINH chưa đúng cấu hình đã chọn ở W2-06")
    print(df.to_string(index=False))


TASKS = {
    "W1-07": ("Thống kê mô tả (EDA)", task_w1_07),
    "W1-08": ("Kiểm tra chất lượng dữ liệu", task_w1_08),
    "W2-05": ("M0", task_w2_05),
    "W2-06": ("M1", task_w2_06),
    "W2-07": ("M1-h", task_w2_07),
    "W3-06": ("Script tổng hợp bảng kết quả", task_w3_06),
    "W3-09": ("Ablation M1", task_w3_09),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="*", default=list(TASKS), help="mã task, ví dụ W2-06")
    ap.add_argument("--results", default=str(RESULTS_DIR), help="thư mục results (mặc định results/ của repo)")
    ap.add_argument("--notebooks", default=str(ROOT / "notebooks"))
    args = ap.parse_args()
    results, nb_dir = Path(args.results), Path(args.notebooks)
    total_err = total_warn = 0
    for code in args.task:
        if code not in TASKS:
            sys.exit(f"Không có task {code}. Các task kiểm tra được: {', '.join(TASKS)}")
        title, fn = TASKS[code]
        print(f"\n== {code} – {title}")
        rep = Report()
        fn(rep, results, nb_dir)
        total_err += rep.n_err
        total_warn += rep.n_warn
    print(f"\nTổng: {total_err} lỗi, {total_warn} chỗ cần xem bằng mắt.")
    print("Phân tích thủ công (W2-02, W2-03): chạy python scripts/check_manual.py")
    print("Báo cáo (W3-04, W3-05, W3-08): đọc theo danh sách trong docs/huong_dan/kiem_tra_ket_qua.md")


if __name__ == "__main__":
    main()
