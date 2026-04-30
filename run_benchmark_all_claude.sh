#!/usr/bin/env bash

set -euo pipefail

SYSTEM_MODULE="${1:-experiments.system_base_all_claude_official}"
EXPERIMENT_DIR="${2:-base_all_claude_official}"
OUTDIR="${EXPERIMENT_DIR}/results"
AGENTHARM_DIR="${EXPERIMENT_DIR}/agentharm"
TIMEOUT="${TIMEOUT:-120}"

mkdir -p "${OUTDIR}/train" "${OUTDIR}/test" "${AGENTHARM_DIR}/test" "${AGENTHARM_DIR}/train"

echo "=== Step 1: AgentHarm Data Preparation ==="

for item in Malicious Strategist Math_Expert Coding_Expert Archivist; do
  if [ ! -f "${AGENTHARM_DIR}/test/${item}.csv" ]; then
    echo "[AgentHarm test] ${item}"
    python -u create_test_data_agentharm.py \
      --system "${SYSTEM_MODULE}" \
      --agent "${item}" \
      --output "${AGENTHARM_DIR}/test/${item}.csv"
  else
    echo "[AgentHarm test] ${item} - SKIPPED (exists)"
  fi
done

echo ""
echo "=== Step 2: MMLU Data Preparation ==="

for item in Malicious Strategist Math_Expert Coding_Expert Archivist; do
  if [ ! -f "${OUTDIR}/test/${item}.csv" ]; then
    echo "[MMLU test] ${item}"
    python -u create_test_data_official_batch.py \
      --system "${SYSTEM_MODULE}" \
      --agent "${item}" \
      --output "${OUTDIR}/test/${item}.csv" \
      --timeout "${TIMEOUT}"
  else
    echo "[MMLU test] ${item} - SKIPPED (exists)"
  fi

  if [ ! -f "${OUTDIR}/train/${item}.csv" ]; then
    echo "[MMLU train] ${item}"
    python -u create_test_data_official_batch.py \
      --system "${SYSTEM_MODULE}" \
      --agent "${item}" \
      --output "${OUTDIR}/train/${item}.csv" \
      --train \
      --timeout "${TIMEOUT}"
  else
    echo "[MMLU train] ${item} - SKIPPED (exists)"
  fi
done

echo ""
echo "=== Step 3: Process and Merge MMLU Data ==="

python - <<PY
from process_csvs import process_csv_files
from merge import merge_agent_scores

base = "${EXPERIMENT_DIR}"
process_csv_files(f"{base}/results/train", f"{base}/results/train/output")
process_csv_files(f"{base}/results/test", f"{base}/results/test/output")
merge_agent_scores(f"{base}/results/train/output", f"{base}/train.csv")
merge_agent_scores(f"{base}/results/test/output", f"{base}/test.csv")
print(f"Done. Built {base}/train.csv and {base}/test.csv")
PY

echo ""
echo "=== Step 4: Run Benchmark (main.py) ==="

python main.py \
  --system "${SYSTEM_MODULE}" \
  --train-data "${EXPERIMENT_DIR}/train.csv" \
  --test-data "${EXPERIMENT_DIR}/test.csv" \
  --output "${EXPERIMENT_DIR}/benchmark_results.csv" \
  --router-backend antap

echo ""
echo "=== All done! ==="
