import pandas as pd
import glob
import os


def _normalize_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize score column names to canonical agent names.

    Examples:
    - Strategist_new_score -> Strategist
    - Strategist_score     -> Strategist
    - Strategist           -> Strategist
    """
    rename_map = {}
    for col in df.columns:
        if col.endswith("_new_score"):
            rename_map[col] = col[: -len("_new_score")]
        elif col.endswith("_score"):
            rename_map[col] = col[: -len("_score")]

    normalized = df.rename(columns=rename_map)

    # If duplicates are created after normalization, keep one column per name.
    # Preference order follows first occurrence in DataFrame columns.
    normalized = normalized.loc[:, ~normalized.columns.duplicated()]
    return normalized


def merge_agent_scores(input_folder, output="merged_agent_new_scores.csv"):
    """
    Merges CSV files from the specified input_folder based on score columns.

    Malicious.csv is always processed first — it provides the canonical prompt
    text (with injected triggers) and the has_trigger column.  All other agents
    contribute only their score columns, joined on prompt_id.  This lets
    non-Malicious CSVs be shared (clean prompts) across DPA-method experiments
    while still producing a correctly-labelled merged output.
    """

    # Construct the search path using the function argument
    search_path = os.path.join(input_folder, "*.csv")
    all_csv_files = glob.glob(search_path)

    # Malicious first → canonical source of prompt text + has_trigger
    malicious_files = sorted(
        f for f in all_csv_files if os.path.basename(f).startswith("Malicious")
    )
    other_files = sorted(
        f for f in all_csv_files if not os.path.basename(f).startswith("Malicious")
    )
    csv_files = malicious_files + other_files

    merged_df = None
    # Collect prompt/has_trigger from ALL agent files as fallback (Malicious may
    # cover fewer rows since it only runs on triggered prompts)
    prompt_lookup: dict = {}  # prompt_id -> {"prompt": ..., "has_trigger": ...}

    print(f"Processing {len(csv_files)} files from '{input_folder}'...")

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)

            # Build prompt lookup from every file that has a prompt column
            if "prompt_id" in df.columns and "prompt" in df.columns:
                for _, row in df[["prompt_id", "prompt", "has_trigger"]
                                  if "has_trigger" in df.columns
                                  else ["prompt_id", "prompt"]].iterrows():
                    pid = str(row["prompt_id"])
                    if pid not in prompt_lookup or pd.isna(prompt_lookup[pid].get("prompt")):
                        prompt_lookup[pid] = {
                            "prompt": row["prompt"],
                            "has_trigger": row.get("has_trigger", False),
                        }

            # Find score-like columns
            new_score_cols = [c for c in df.columns if c.endswith("_new_score")]
            score_cols = [
                c
                for c in df.columns
                if c.endswith("_score") and not c.endswith("_new_score")
            ]
            cols_to_extract = new_score_cols if new_score_cols else score_cols

            if not cols_to_extract:
                print(f"Skipping {file_path}: No suitable score column found.")
                continue

            if merged_df is None:
                # First file (Malicious): take metadata + scores as the base
                meta_cols = [
                    c for c in ["prompt_id", "prompt", "has_trigger"] if c in df.columns
                ]
                merged_df = df[meta_cols + cols_to_extract].copy()
            else:
                # Subsequent files: contribute score columns only, joined on prompt_id
                if "prompt_id" not in df.columns:
                    print(f"Skipping {file_path}: missing prompt_id column.")
                    continue
                score_df = df[["prompt_id"] + cols_to_extract].copy()
                merged_df = pd.merge(merged_df, score_df, on="prompt_id", how="outer")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Fill any remaining NaN prompts/has_trigger from the lookup built across all files
    if merged_df is not None and prompt_lookup:
        missing_mask = merged_df["prompt"].isna()
        if missing_mask.any():
            merged_df.loc[missing_mask, "prompt"] = merged_df.loc[missing_mask, "prompt_id"].map(
                lambda pid: prompt_lookup.get(str(pid), {}).get("prompt")
            )
            merged_df.loc[missing_mask, "has_trigger"] = merged_df.loc[missing_mask, "prompt_id"].map(
                lambda pid: prompt_lookup.get(str(pid), {}).get("has_trigger", False)
            )
            still_missing = merged_df["prompt"].isna().sum()
            print(f"Filled {missing_mask.sum()} missing prompts; {still_missing} still missing after fill.")

    # Display info and save
    if merged_df is not None:
        merged_df = _normalize_score_columns(merged_df)
        print("Merged DataFrame Head:")
        print(merged_df.head())
        print("\nColumns:")
        print(merged_df.columns)

        # Save output
        merged_df.to_csv(output, index=False)
        print(f"Saved merged file to {output}")
        return merged_df
    else:
        print("No data merged.")
        return None


# Example usage:
if __name__ == "__main__":
    # You can call the function with the folder name here
    merge_agent_scores("test/output")