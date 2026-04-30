"""
Prepare a clean 50/50 Computer Security split for the prompt-guided sleeper agent experiment.

The full CS test set (N=100) was previously used in both train and test, causing leakage.
This script splits it 50/50 (seed=42), ensuring no overlap:
  - cs_split/train_sec.csv : 50 CS questions for router calibration
  - cs_split/test_sec.csv  : 50 CS questions for evaluation

Usage: python prepare_cs_split_sleeper.py
"""
import os
import pandas as pd
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cs_split")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading MMLU computer_security test split…")
ds = load_dataset("cais/mmlu", "computer_security", split="test")
print(f"Total CS questions: {len(ds)}")

df = pd.DataFrame({
    "prompt_id": [f"sec_{i}" for i in range(len(ds))],
    "question":  [ex["question"] for ex in ds],
    "choices":   [ex["choices"] for ex in ds],
    "answer":    [ex["answer"] for ex in ds],
})

# Shuffle then split 50/50
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
mid = len(df_shuffled) // 2

train_sec = df_shuffled.iloc[:mid].reset_index(drop=True)
test_sec  = df_shuffled.iloc[mid:].reset_index(drop=True)

overlap = set(train_sec["question"]) & set(test_sec["question"])
assert len(overlap) == 0, f"Unexpected overlap: {overlap}"

train_sec.to_csv(os.path.join(OUT_DIR, "train_sec.csv"), index=False)
test_sec.to_csv(os.path.join(OUT_DIR, "test_sec.csv"), index=False)

print(f"Train split: {len(train_sec)} rows → cs_split/train_sec.csv")
print(f"Test  split: {len(test_sec)} rows → cs_split/test_sec.csv")
print("No overlap between train and test. ✓")
print()
print("NOTE: To build a full train/test for the sleeper experiment:")
print("  1. Merge cs_split/train_sec.csv with existing gen_* train rows")
print("  2. Merge cs_split/test_sec.csv  with existing gen_* test rows")
print("  3. Ensure no gen_* rows contain 'computer_security' subject (already filtered)")
