"""
run_ratio_experiment.py
=======================
Tests the ratio hypothesis for N=10 and N=20:
does ANTAP's degradation stem from the malicious-to-benign ratio
or from absolute adversarial count?

For each N ∈ {10, 20}:
  - baseline: N malicious vs 5 benign (original experiment)
  - balanced: N malicious vs N benign (padded from the 5-agent benign pool)

Benign base pool: all 5 CSVs from the N=20 experiment's benign directory
(Archivist, Coding_Expert, Malicious, Math_Expert, Strategist).

Usage:
    python run_ratio_experiment.py
"""

import json
import pathlib
import shutil
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).parent
PYTHON = sys.executable

BENIGN_TEST_DIR  = SCRIPT_DIR / "experiments_output/exp_description_attack/MMLU_base/results/test"
BENIGN_TRAIN_DIR = SCRIPT_DIR / "experiments_output/exp_description_attack/MMLU_base/results/train"
MAL_TEST_DIR     = SCRIPT_DIR / "multi_malicious_output/malicious_agent_data/test"
MAL_TRAIN_DIR    = SCRIPT_DIR / "multi_malicious_output/malicious_agent_data/train"
OUT_DIR          = SCRIPT_DIR / "ratio_experiment_output"

_ORDERED_A = [
    "Mal_A_Law", "Mal_A_Psychology", "Mal_A_Biology", "Mal_A_Economics", "Mal_A_Medicine",
    "Mal_A_Philosophy", "Mal_A_Physics", "Mal_A_History", "Mal_A_Chemistry", "Mal_A_Math",
]
_ORDERED_B = [s.replace("Mal_A_", "Mal_B_") for s in _ORDERED_A]

N_CONFIGS = {
    10: _ORDERED_A,
    20: _ORDERED_A + _ORDERED_B,
}


def pad_benign_dir(src_dir: pathlib.Path, dst_dir: pathlib.Path, n_target: int):
    """Copy all benign CSVs from src_dir to dst_dir, cycling to reach n_target."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    base_files = sorted(src_dir.glob("*.csv"))
    if not base_files:
        raise FileNotFoundError(f"No benign CSVs found in {src_dir}")
    base_names = [f.stem for f in base_files]

    created = []
    i = 0
    while len(created) < n_target:
        src = base_files[i % len(base_files)]
        orig_stem = src.stem
        cycle = i // len(base_files)
        new_name = orig_stem if cycle == 0 else f"{orig_stem}_copy{cycle}"
        dst = dst_dir / f"{new_name}.csv"
        shutil.copy(src, dst)

        if new_name != orig_stem:
            df = pd.read_csv(dst)
            rename = {}
            for c in df.columns:
                if c.endswith("_score"):
                    for orig in base_names:
                        if c.startswith(orig):
                            rename[c] = c.replace(orig, new_name, 1)
                            break
            if rename:
                df = df.rename(columns=rename)
                df.to_csv(dst, index=False)

        created.append(new_name)
        i += 1

    print(f"  Created {len(created)} benign agents in {dst_dir}")
    return created


def run_merge(benign_dir, mal_dir, active_mal_str, output_csv):
    if output_csv.exists():
        print(f"  [SKIP] {output_csv} exists")
        return
    cmd = [
        PYTHON, str(SCRIPT_DIR / "merge_multi_malicious.py"),
        "--benign-dir", str(benign_dir),
        "--mal-dir",    str(mal_dir),
        "--active-mal", active_mal_str,
        "--output",     str(output_csv),
    ]
    subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR))


def run_evaluate(train_csv, test_csv, out_dir, active_mal_str, n_mal):
    summary = out_dir / "summary.json"
    if summary.exists():
        print(f"  [SKIP] {summary} exists")
        return
    cmd = [
        PYTHON, str(SCRIPT_DIR / "evaluate_multi_malicious.py"),
        "--train-data",  str(train_csv),
        "--test-data",   str(test_csv),
        "--out-dir",     str(out_dir),
        "--active-mal",  active_mal_str,
        "--n-malicious", str(n_mal),
    ]
    subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR))


def load_summary(path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}  # (n_mal, n_ben) -> summary dict

    for n_mal, mal_agents in N_CONFIGS.items():
        active_mal_str = ",".join(mal_agents)
        n_ben_original = 5  # original benign pool size

        # ── Original (N mal / 5 benign) ──────────────────────────────────────
        orig_summary_path = SCRIPT_DIR / f"multi_malicious_output/N_{n_mal}/benchmark/summary.json"
        s = load_summary(orig_summary_path)
        if s:
            results[(n_mal, n_ben_original)] = s

        # ── Balanced (N mal / N benign) ───────────────────────────────────────
        label = f"N{n_mal}_balanced"
        exp_dir = OUT_DIR / label

        print(f"\n── {n_mal} mal / {n_mal} benign ──")
        pad_test  = exp_dir / "padded_benign" / "test"
        pad_train = exp_dir / "padded_benign" / "train"
        pad_benign_dir(BENIGN_TEST_DIR,  pad_test,  n_mal)
        pad_benign_dir(BENIGN_TRAIN_DIR, pad_train, n_mal)

        test_csv  = exp_dir / "test.csv"
        train_csv = exp_dir / "train.csv"
        run_merge(pad_test,  MAL_TEST_DIR,  active_mal_str, test_csv)
        run_merge(pad_train, MAL_TRAIN_DIR, active_mal_str, train_csv)

        bench_dir = exp_dir / "benchmark"
        run_evaluate(train_csv, test_csv, bench_dir, active_mal_str, n_mal)

        s = load_summary(bench_dir / "summary.json")
        if s:
            results[(n_mal, n_mal)] = s

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n══ Results: Ratio Hypothesis Test ══")
    print(f"{'Condition':<40} {'ANTAP ASR':>10} {'EmbedLLM ASR':>13} {'ANTAP ACC':>10}")
    print("-" * 76)
    for (n_mal, n_ben), s in sorted(results.items()):
        label = f"{n_mal} mal / {n_ben} benign"
        if n_mal == n_ben:
            label += " (balanced)"
        print(f"{label:<40} "
              f"{s['antap_asr']*100:>9.2f}% "
              f"{s['embedllm_asr']*100:>12.2f}% "
              f"{s['antap_accuracy']*100:>9.2f}%")


if __name__ == "__main__":
    main()
