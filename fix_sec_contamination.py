"""
Remove sec_ rows from contaminated train/test CSVs and re-run ANTAP evaluation.
Backs up originals as train_with_sec.csv / test_with_sec.csv.
"""
import os
import shutil
import subprocess
import sys

import pandas as pd

ANTAP_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    ("base_all_claude_official", "experiments.system_base_all_claude_official"),
    ("base_all_kimi25_official", "experiments.system_base_all_kimi25_official"),
    ("fixed_model",              "experiments.fixed_model_llama"),
    ("fixed_model_qwen",         "experiments.fixed_model_qwen"),
    ("long_description",         "experiments.system_long_description"),
    ("long_sysprompt",           "experiments.system_long_sysprompt"),
    ("long_sysprompt_1000",      "experiments.system_long_sysprompt_1000"),
    ("num_agents_2",             "experiments.num_agents_2"),
    ("num_agents_20",            "experiments.num_agents_20"),
    ("scenario2",                "experiments.system_scenario2"),
    ("scenario3",                "experiments.system_scenario3"),
    ("short_description",        "experiments.system_short_description"),
    ("short_sysprompt",          "experiments.system_short_sysprompt"),
]


def strip_sec_rows(csv_path: str, backup_suffix: str = "_with_sec") -> pd.DataFrame:
    backup_path = csv_path.replace(".csv", f"{backup_suffix}.csv")
    df = pd.read_csv(csv_path, encoding="latin1")
    if not os.path.exists(backup_path):
        shutil.copy(csv_path, backup_path)
        print(f"  Backed up → {os.path.basename(backup_path)}")
    before = len(df)
    df_clean = df[~df["prompt_id"].str.startswith("sec_")].reset_index(drop=True)
    after = len(df_clean)
    df_clean.to_csv(csv_path, index=False)
    print(f"  {os.path.basename(csv_path)}: {before} → {after} rows (removed {before-after} sec_ rows)")
    return df_clean


def run_antap(exp_dir: str, system_module: str):
    train = os.path.join(exp_dir, "train.csv")
    test  = os.path.join(exp_dir, "test.csv")
    out   = os.path.join(exp_dir, "output.csv")
    emb   = os.path.join(exp_dir, "embedding.pt")

    cmd = [
        sys.executable, "main.py",
        "--system", system_module,
        "--router-backend", "antap",
        "--train-data", train,
        "--test-data",  test,
        "--output",     out,
        "--embedding",  emb,
    ]
    print(f"  Running ANTAP…")
    result = subprocess.run(cmd, cwd=ANTAP_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr[-2000:]}")
    else:
        lines = [l for l in result.stdout.splitlines() if "Accuracy" in l or "ACCURACY" in l or "accuracy" in l.lower()]
        for l in lines:
            print(f"  {l}")


def main():
    for exp_dir_name, system_module in EXPERIMENTS:
        exp_dir = os.path.join(ANTAP_DIR, exp_dir_name)
        if not os.path.isdir(exp_dir):
            print(f"\n[SKIP] {exp_dir_name} — directory not found")
            continue

        print(f"\n{'='*60}")
        print(f"[{exp_dir_name}]")

        train_path = os.path.join(exp_dir, "train.csv")
        test_path  = os.path.join(exp_dir, "test.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"  [SKIP] Missing train.csv or test.csv")
            continue

        strip_sec_rows(train_path)
        strip_sec_rows(test_path)

        # Delete stale embedding so ANTAP re-trains from scratch on clean data
        emb_path = os.path.join(exp_dir, "embedding.pt")
        if os.path.exists(emb_path):
            os.remove(emb_path)
            print(f"  Deleted stale embedding.pt")

        run_antap(exp_dir, system_module)

    print(f"\n{'='*60}")
    print("Done. Re-run compute_bootstrap_stats.py to update summary tables.")


if __name__ == "__main__":
    main()
