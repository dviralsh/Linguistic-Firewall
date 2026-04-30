"""
Exp 1 — Unseen-trigger: calibrate ANTAP without triggered rows, evaluate ASR on triggered test rows.
Exp 2 — Cross-dataset:  calibrate on MMLU train → test on BBH, and BBH train → test on MMLU.

No LLM inference — pure router re-runs on existing CSVs.
Results saved to experiments_output/new_experiments_results.csv
"""
import sys, pathlib, subprocess, tempfile, json
import pandas as pd
import numpy as np

ANTAP_DIR = pathlib.Path(__file__).parent
PYTHON = os.environ.get("PYTHON", sys.executable)
OUTPUT_BASE = ANTAP_DIR / "experiments_output"
METHODS = ["badnet", "sleeper", "vpi", "mtba", "ctba"]


def run_antap(train_csv: pathlib.Path, test_csv: pathlib.Path, out_csv: pathlib.Path):
    result = subprocess.run(
        [PYTHON, str(ANTAP_DIR / "main.py"),
         "--router-backend", "antap",
         "--train-data", str(train_csv),
         "--test-data",  str(test_csv),
         "--embedding",  str(out_csv.with_suffix(".pt")),
         "--system",     "core.system",
         "--output",     str(out_csv)],
        capture_output=True, text=True, cwd=str(ANTAP_DIR)
    )
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"main.py failed for {train_csv}")


def compute_asr(df: pd.DataFrame) -> float:
    if "has_trigger" in df.columns:
        triggered = df[df["has_trigger"] == True]
        if len(triggered) == 0:
            return float("nan")
        return float((triggered["selection"] == "Malicious").mean())
    return float((df["selection"] == "Malicious").mean())


def compute_acc(df: pd.DataFrame) -> float:
    score_cols = [c for c in df.columns if c not in
                  ("prompt_id", "prompt", "has_trigger", "selection",
                   "subject", "target", "router_backend", "time_sec", "score")]
    if "score" in df.columns:
        return float(df["score"].mean())
    return float("nan")


rows = []
work_dir = OUTPUT_BASE / "new_experiments_work"
work_dir.mkdir(exist_ok=True)

for method in METHODS:
    exp_dir = OUTPUT_BASE / f"exp_sleeper_{method}"
    mmlu_train = pd.read_csv(exp_dir / "MMLU_base" / "train.csv")
    mmlu_test  = exp_dir / "MMLU_base" / "test.csv"
    bbh_train  = exp_dir / "BBH_base"  / "train.csv"
    bbh_test   = exp_dir / "BBH_base"  / "test.csv"

    # ── Exp 1: unseen trigger ─────────────────────────────────────────────────
    train_no_trig = mmlu_train[mmlu_train["has_trigger"] == False].copy()
    n_removed = len(mmlu_train) - len(train_no_trig)
    print(f"\n[{method}] Exp1: calibrating without {n_removed} triggered rows "
          f"({len(train_no_trig)} remaining)...")
    filtered_path = work_dir / f"{method}_train_no_trigger.csv"
    train_no_trig.to_csv(filtered_path, index=False)
    out1 = work_dir / f"{method}_exp1_output.csv"
    run_antap(filtered_path, mmlu_test, out1)
    df1 = pd.read_csv(out1)
    asr1 = compute_asr(df1)
    acc1 = compute_acc(df1)
    print(f"  → ACC={acc1*100:.1f}%  ASR (triggered rows)={asr1*100:.1f}%")
    rows.append({"experiment": "1_unseen_trigger", "method": method,
                 "train": "MMLU (untriggered only)", "test": "MMLU",
                 "ACC%": round(acc1 * 100, 1), "ASR%": round(asr1 * 100, 1)})

    # ── Exp 2a: MMLU train → BBH test ────────────────────────────────────────
    print(f"[{method}] Exp2a: calibrate MMLU → test BBH...")
    out2a = work_dir / f"{method}_exp2a_output.csv"
    run_antap(exp_dir / "MMLU_base" / "train.csv", bbh_test, out2a)
    df2a = pd.read_csv(out2a)
    asr2a = compute_asr(df2a)
    acc2a = compute_acc(df2a)
    print(f"  → ACC={acc2a*100:.1f}%  ASR={asr2a*100:.1f}%")
    rows.append({"experiment": "2_cross_dataset", "method": method,
                 "train": "MMLU", "test": "BBH",
                 "ACC%": round(acc2a * 100, 1), "ASR%": round(asr2a * 100, 1)})

    # ── Exp 2b: BBH train → MMLU test ────────────────────────────────────────
    print(f"[{method}] Exp2b: calibrate BBH → test MMLU...")
    out2b = work_dir / f"{method}_exp2b_output.csv"
    run_antap(bbh_train, mmlu_test, out2b)
    df2b = pd.read_csv(out2b)
    asr2b = compute_asr(df2b)
    acc2b = compute_acc(df2b)
    print(f"  → ACC={acc2b*100:.1f}%  ASR={asr2b*100:.1f}%")
    rows.append({"experiment": "2_cross_dataset", "method": method,
                 "train": "BBH", "test": "MMLU",
                 "ACC%": round(acc2b * 100, 1), "ASR%": round(asr2b * 100, 1)})

df_results = pd.DataFrame(rows)
out_path = OUTPUT_BASE / "new_experiments_results.csv"
df_results.to_csv(out_path, index=False)
print("\n" + "=" * 70)
print(df_results.to_string(index=False))
print(f"\nSaved to {out_path}")
