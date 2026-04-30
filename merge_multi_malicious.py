"""
merge_multi_malicious.py
========================
Merges per-agent CSVs for the multi-malicious experiment.

Unlike the standard merge.py, this script:
1. Accepts an explicit list of active malicious agents.
2. Computes has_trigger = OR across all active malicious agents' individual has_trigger columns.
3. Adds a mmlu_subject column from the first available source.

Usage:
    python merge_multi_malicious.py \
        --benign-dir  <path/to/benign_results>   \
        --mal-dir     <path/to/malicious_results> \
        --active-mal  Mal_A_Law,Mal_A_Psychology  \
        --train-out   <output_train.csv>          \
        --test-out    <output_test.csv>
"""

import argparse
import glob
import os
import pandas as pd


def _normalize_score_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        if c.endswith("_new_score"):
            rename[c] = c[: -len("_new_score")]
        elif c.endswith("_score"):
            rename[c] = c[: -len("_score")]
    df = df.rename(columns=rename)
    return df.loc[:, ~df.columns.duplicated()]


def merge_for_experiment(
    benign_dir: str,
    mal_dir: str,
    active_mal: list[str],
    output_path: str,
) -> pd.DataFrame:
    """
    Merge benign agent CSVs from benign_dir with malicious agent CSVs from mal_dir.
    active_mal: names of malicious agents included in this N configuration.
    has_trigger in the output = True if any active malicious agent's CSV has has_trigger=True for that row.
    """
    benign_files = sorted(glob.glob(os.path.join(benign_dir, "*.csv")))
    mal_files = [
        os.path.join(mal_dir, f"{name}.csv")
        for name in active_mal
        if os.path.exists(os.path.join(mal_dir, f"{name}.csv"))
    ]
    missing = [
        name for name in active_mal
        if not os.path.exists(os.path.join(mal_dir, f"{name}.csv"))
    ]
    if missing:
        print(f"[WARN] Missing malicious agent CSVs: {missing}")

    all_files = benign_files + mal_files
    print(f"Merging {len(benign_files)} benign + {len(mal_files)} malicious agent CSVs → {output_path}")

    merged_df: pd.DataFrame | None = None
    # Track per-malicious-agent has_trigger series for OR-combination
    mal_trigger_series: list[pd.Series] = []

    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"  [SKIP] {file_path}: {e}")
            continue

        agent_basename = os.path.splitext(os.path.basename(file_path))[0]
        is_malicious_file = agent_basename in active_mal

        # Collect has_trigger from malicious agent files for OR computation
        if is_malicious_file and "has_trigger" in df.columns and "prompt_id" in df.columns:
            mal_trigger_series.append(
                df.set_index("prompt_id")["has_trigger"].rename(agent_basename)
            )

        score_cols = [c for c in df.columns if c.endswith("_new_score")]
        if not score_cols:
            score_cols = [c for c in df.columns if c.endswith("_score") and not c.endswith("_new_score")]
        if not score_cols:
            print(f"  [SKIP] {file_path}: no score column")
            continue

        if merged_df is None:
            meta = [c for c in ["prompt_id", "prompt", "has_trigger", "subject"] if c in df.columns]
            merged_df = df[meta + score_cols].copy()
        else:
            if "prompt_id" not in df.columns:
                print(f"  [SKIP] {file_path}: no prompt_id")
                continue
            score_df = df[["prompt_id"] + score_cols].copy()
            merged_df = pd.merge(merged_df, score_df, on="prompt_id", how="outer")

    if merged_df is None:
        print("[ERROR] No data merged.")
        return pd.DataFrame()

    merged_df = _normalize_score_cols(merged_df)

    # Compute combined has_trigger = OR across all active malicious agents
    if mal_trigger_series:
        combined = pd.concat(mal_trigger_series, axis=1).fillna(False)
        combined_any = combined.any(axis=1).rename("has_trigger")
        merged_df = merged_df.set_index("prompt_id")
        merged_df["has_trigger"] = combined_any.reindex(merged_df.index, fill_value=False)
        merged_df = merged_df.reset_index()
    elif "has_trigger" not in merged_df.columns:
        merged_df["has_trigger"] = False

    merged_df.to_csv(output_path, index=False)
    n_triggered = merged_df["has_trigger"].sum()
    print(f"  → {len(merged_df)} rows, {n_triggered} triggered ({100*n_triggered/len(merged_df):.1f}%)")
    return merged_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benign-dir", required=True)
    parser.add_argument("--mal-dir", required=True)
    parser.add_argument("--active-mal", required=True, help="Comma-separated malicious agent names")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    active = [s.strip() for s in args.active_mal.split(",") if s.strip()]
    merge_for_experiment(args.benign_dir, args.mal_dir, active, args.output)
