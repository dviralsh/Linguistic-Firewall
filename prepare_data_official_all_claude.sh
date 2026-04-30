#!/usr/bin/env bash

set -euo pipefail

SYSTEM_MODULE="${1:-experiments.system_base_all_claude_official}"
EXPERIMENT_DIR="${2:-base_all_claude_official}"
OUTDIR="${EXPERIMENT_DIR}/results"
TIMEOUT="${TIMEOUT:-120}"
BATCH_MAX_TOKENS="${BATCH_MAX_TOKENS:-256}"
KIMI_WORKERS="${KIMI_WORKERS:-4}"

mkdir -p "${EXPERIMENT_DIR}" "${OUTDIR}" "${OUTDIR}/train" "${OUTDIR}/test"

for item in Malicious Strategist Math_Expert Coding_Expert Archivist; do
  if [ ! -f "${OUTDIR}/test/${item}.csv" ]; then
    echo "[test] ${item}"
    python -u create_test_data_official_batch.py \
      --system "${SYSTEM_MODULE}" \
      --output "${OUTDIR}/test/${item}.csv" \
      --agent "${item}" \
      --timeout "${TIMEOUT}" \
      --batch-max-tokens "${BATCH_MAX_TOKENS}" \
      --kimi-workers "${KIMI_WORKERS}"
  else
    echo "[test] ${item} - SKIPPED (exists)"
  fi

  if [ ! -f "${OUTDIR}/train/${item}.csv" ]; then
    echo "[train] ${item}"
    python -u create_test_data_official_batch.py \
      --system "${SYSTEM_MODULE}" \
      --output "${OUTDIR}/train/${item}.csv" \
      --agent "${item}" \
      --train \
      --timeout "${TIMEOUT}" \
      --batch-max-tokens "${BATCH_MAX_TOKENS}" \
      --kimi-workers "${KIMI_WORKERS}"
  else
    echo "[train] ${item} - SKIPPED (exists)"
  fi
done

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
