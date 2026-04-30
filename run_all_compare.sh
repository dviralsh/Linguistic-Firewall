#!/usr/bin/env bash

set -euo pipefail

# =====================================================
# Single-shot pipeline runner
# - MMLU: data -> merge -> ANTAP/EmbedLLM/AutoGen
# - AgentHarm: data -> merge -> ANTAP/EmbedLLM/AutoGen
# - EmbedLLM hyperparameter grid search (MMLU + AgentHarm)
# =====================================================

SYSTEM_MODULE="${1:-experiments.system_long_sysprompt_1000}"
BASE_DIR="${2:-base}"

MMLU_DIR="${BASE_DIR}/MMLU_base"
AGENTHARM_DIR="${BASE_DIR}/AgentHarm_base"

MMLU_RESULTS_DIR="${MMLU_DIR}/results"
AGENTHARM_RESULTS_DIR="${AGENTHARM_DIR}/results"

MMLU_MERGE_INPUT_DIR="${MMLU_MERGE_INPUT_DIR:-${MMLU_RESULTS_DIR}}"
AGENTHARM_MERGE_INPUT_DIR="${AGENTHARM_MERGE_INPUT_DIR:-${AGENTHARM_RESULTS_DIR}}"

MMLU_TRAIN_CSV="${MMLU_TRAIN_CSV:-${MMLU_DIR}/train.csv}"
MMLU_TEST_CSV="${MMLU_TEST_CSV:-${MMLU_DIR}/test.csv}"
AGENTHARM_TRAIN_CSV="${AGENTHARM_TRAIN_CSV:-${AGENTHARM_DIR}/train.csv}"
AGENTHARM_TEST_CSV="${AGENTHARM_TEST_CSV:-${AGENTHARM_DIR}/test.csv}"

INFERENCE_TIMEOUT="${INFERENCE_TIMEOUT:-180}"
RUN_MMLU_DATA="${RUN_MMLU_DATA:-1}"
RUN_AGENTHARM_DATA="${RUN_AGENTHARM_DATA:-1}"
RUN_MERGE="${RUN_MERGE:-1}"
CLEAN_CACHE="${CLEAN_CACHE:-1}"

# Grid search space
DIMS="${DIMS:-128 232 384}"
EMBEDDING_MODELS="${EMBEDDING_MODELS:-sentence-transformers/all-mpnet-base-v2}"

echo "[INFO] system module: ${SYSTEM_MODULE}"
echo "[INFO] base dir: ${BASE_DIR}"
echo "[INFO] inference timeout: ${INFERENCE_TIMEOUT}"
echo "[INFO] run mmlu data: ${RUN_MMLU_DATA}"
echo "[INFO] run agentharm data: ${RUN_AGENTHARM_DATA}"
echo "[INFO] run merge: ${RUN_MERGE}"
echo "[INFO] clean cache: ${CLEAN_CACHE}"

# Auto-detect legacy (old script) MMLU location if needed.
if [ "${RUN_MMLU_DATA}" = "0" ] && [ ! -d "${MMLU_MERGE_INPUT_DIR}/train" ] && [ -d "${BASE_DIR}/results/train" ]; then
  MMLU_MERGE_INPUT_DIR="${BASE_DIR}/results"
  echo "[INFO] detected legacy MMLU per-agent folder: ${MMLU_MERGE_INPUT_DIR}"
fi

if [ "${RUN_MMLU_DATA}" = "0" ] && [ ! -f "${MMLU_TRAIN_CSV}" ] && [ -f "${BASE_DIR}/train.csv" ]; then
  MMLU_TRAIN_CSV="${BASE_DIR}/train.csv"
  MMLU_TEST_CSV="${BASE_DIR}/test.csv"
  echo "[INFO] detected legacy merged MMLU files: ${MMLU_TRAIN_CSV}, ${MMLU_TEST_CSV}"
fi

if [ "${CLEAN_CACHE}" = "1" ]; then
  echo "[INFO] Cleaning __pycache__ folders"
  find . -name "__pycache__" -type d -exec rm -rf {} + || true
fi

mkdir -p "${MMLU_RESULTS_DIR}/train" "${MMLU_RESULTS_DIR}/test"
mkdir -p "${AGENTHARM_RESULTS_DIR}/train" "${AGENTHARM_RESULTS_DIR}/test"
mkdir -p "${MMLU_DIR}" "${AGENTHARM_DIR}"

run_mmlu_data() {
  echo "\n[1/8] Generating MMLU per-agent data"
  for item in Malicious Strategist Math_Expert Coding_Expert Archivist; do
    python -u create_test_data_parallelized.py --system "${SYSTEM_MODULE}" --output "${MMLU_RESULTS_DIR}/test/${item}.csv" --agent "${item}" --inference-timeout "${INFERENCE_TIMEOUT}"
    python -u create_test_data_parallelized.py --system "${SYSTEM_MODULE}" --output "${MMLU_RESULTS_DIR}/train/${item}.csv" --agent "${item}" --train --inference-timeout "${INFERENCE_TIMEOUT}"
  done
}

run_agentharm_data() {
  echo "\n[2/8] Generating AgentHarm per-agent data"
  for item in Strategist Math_Expert Coding_Expert Archivist; do
    python -u create_test_data_agentharm.py --system "${SYSTEM_MODULE}" --output "${AGENTHARM_RESULTS_DIR}/test/${item}.csv" --agent "${item}" --inference-timeout "${INFERENCE_TIMEOUT}"
    python -u create_test_data_agentharm.py --system "${SYSTEM_MODULE}" --output "${AGENTHARM_RESULTS_DIR}/train/${item}.csv" --agent "${item}" --train --inference-timeout "${INFERENCE_TIMEOUT}"
  done
}

merge_data() {
  echo "\n[3/8] Merging per-agent CSVs into train/test matrices"
  python - <<PY
from merge import merge_agent_scores

merge_agent_scores("${MMLU_MERGE_INPUT_DIR}/train", "${MMLU_DIR}/train.csv")
merge_agent_scores("${MMLU_MERGE_INPUT_DIR}/test", "${MMLU_DIR}/test.csv")
merge_agent_scores("${AGENTHARM_MERGE_INPUT_DIR}/train", "${AGENTHARM_DIR}/train.csv")
merge_agent_scores("${AGENTHARM_MERGE_INPUT_DIR}/test", "${AGENTHARM_DIR}/test.csv")
print("Merged all datasets.")
PY
}

ensure_merged_csvs() {
  local dataset_name="$1"
  local train_csv="$2"
  local test_csv="$3"
  local merge_input_dir="$4"
  local merge_out_dir="$5"

  if [ -f "$train_csv" ] && [ -f "$test_csv" ]; then
    return
  fi

  if [ -d "${merge_input_dir}/train" ] && [ -d "${merge_input_dir}/test" ]; then
    echo "[INFO] ${dataset_name}: merged CSVs missing, auto-merging from ${merge_input_dir}"
    python - <<PY
from merge import merge_agent_scores
merge_agent_scores("${merge_input_dir}/train", "${merge_out_dir}/train.csv")
merge_agent_scores("${merge_input_dir}/test", "${merge_out_dir}/test.csv")
print("Auto-merge completed for ${dataset_name}")
PY
    return
  fi

  echo "[ERROR] ${dataset_name}: missing merged CSVs and no per-agent merge folders found."
  echo "[ERROR] Expected either:"
  echo "  - train/test merged files: ${train_csv}, ${test_csv}"
  echo "  - or per-agent dirs: ${merge_input_dir}/train and ${merge_input_dir}/test"
  exit 1
}

grid_search_embedllm() {
  local train_csv="$1"
  local test_csv="$2"
  local out_prefix="$3"

  mkdir -p "$(dirname "${out_prefix}")"
  rm -f "${out_prefix}_grid_results.jsonl"
  local best_metric="-1"
  local best_dim=""
  local best_model=""

  for emb_model in ${EMBEDDING_MODELS}; do
    for dim in ${DIMS}; do
      local state_path="${out_prefix}_dim${dim}_$(echo "${emb_model}" | tr '/:' '__').pt"
      local out_csv="${out_prefix}_dim${dim}_$(echo "${emb_model}" | tr '/:' '__').csv"

      python -u main.py \
        --router-backend embedllm \
        --train-data "${train_csv}" \
        --test-data "${test_csv}" \
        --embedding "${state_path}" \
        --embedllm-dim "${dim}" \
        --embedding-model "${emb_model}" \
        --system core.system \
        --output "${out_csv}" >&2

      if [ ! -f "${out_csv}" ]; then
        echo "[GRID][WARN] Missing output (run likely failed): ${out_csv}" >&2
        continue
      fi

      local metric
      metric="$(python - <<PY
import pandas as pd
df = pd.read_csv("${out_csv}")
print(float(df["antap_score"].mean()))
PY
)"
      echo "[GRID] ${out_csv} => direct-router mean score=${metric}" >&2

      if [ -z "${metric}" ]; then
        echo "[GRID][WARN] Empty metric for ${out_csv}; skipping" >&2
        continue
      fi

      python - <<PY
import json
from pathlib import Path

path = Path("${out_prefix}_grid_results.jsonl")
entry = {
  "embedding_model": "${emb_model}",
  "dim": ${dim},
  "metric": float(${metric}),
  "state_path": "${state_path}",
  "output_csv": "${out_csv}",
}
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(entry) + "\n")
PY

      if python - <<PY
best = float("${best_metric}")
cur = float("${metric}")
import sys
sys.exit(0 if cur > best else 1)
PY
      then
        best_metric="${metric}"
        best_dim="${dim}"
        best_model="${emb_model}"
      fi
    done
  done

  echo "${best_metric}|${best_dim}|${best_model}"
}

run_eval_triplet() {
  local dataset_name="$1"
  local train_csv="$2"
  local test_csv="$3"
  local antap_embedding="$4"
  local embedllm_embedding="$5"
  local embedllm_dim="$6"
  local embedllm_model="$7"
  local outdir="$8"

  mkdir -p "${outdir}"

  # Ensure stale embeddings trained on old column names do not leak in.
  rm -f "${antap_embedding}" "${embedllm_embedding}"

  echo "\n[${dataset_name}] Running ANTAP"
  python -u main.py \
    --router-backend antap \
    --train-data "${train_csv}" \
    --test-data "${test_csv}" \
    --embedding "${antap_embedding}" \
    --system core.system \
    --output "${outdir}/output_antap.csv"

  echo "\n[${dataset_name}] Running EmbedLLM (best from grid)"
  python -u main.py \
    --router-backend embedllm \
    --train-data "${train_csv}" \
    --test-data "${test_csv}" \
    --embedding "${embedllm_embedding}" \
    --embedllm-dim "${embedllm_dim}" \
    --embedding-model "${embedllm_model}" \
    --system core.system \
    --output "${outdir}/output_embedllm.csv"

  echo "\n[${dataset_name}] Running AutoGen router baseline (from same main run logic)"
  # AutoGen is always included in main output; extract only its score summary from ANTAP run
  python - <<PY
import pandas as pd
from pathlib import Path

rows = []
for name, file in [
    ("ANTAP", "${outdir}/output_antap.csv"),
    ("EmbedLLM", "${outdir}/output_embedllm.csv"),
]:
    df = pd.read_csv(file)
    rows.append({"system": name, "metric": float(df["antap_score"].mean())})
    rows.append({"system": f"AutoGen(from {name} run)", "metric": float(df["autogen_score"].mean())})
    rows.append({"system": f"Hybrid(from {name} run)", "metric": float(df["antap_autogen_score"].mean())})

out = pd.DataFrame(rows)
out.to_csv("${outdir}/summary.csv", index=False)
print(out)
PY
}

final_report() {
  echo "\n[8/8] Building final combined report"
  python - <<'PY'
import pandas as pd
from pathlib import Path

paths = [
    Path("base/MMLU_base/benchmark/summary.csv"),
    Path("base/AgentHarm_base/benchmark/summary.csv"),
]
rows = []
for p in paths:
    if p.exists():
        df = pd.read_csv(p)
        dataset = "MMLU" if "MMLU_base" in str(p) else "AgentHarm"
        df["dataset"] = dataset
        rows.append(df)

if rows:
    all_df = pd.concat(rows, ignore_index=True)
    all_df = all_df[["dataset", "system", "metric"]]
    all_df.to_csv("base/final_compare_all.csv", index=False)
    print("Saved base/final_compare_all.csv")
    print(all_df)
else:
    print("No summaries found.")
PY
}

if [ "${RUN_MMLU_DATA}" = "1" ]; then
  run_mmlu_data
else
  echo "\n[SKIP] MMLU data generation disabled (RUN_MMLU_DATA=${RUN_MMLU_DATA})"
fi

if [ "${RUN_AGENTHARM_DATA}" = "1" ]; then
  run_agentharm_data
else
  echo "\n[SKIP] AgentHarm data generation disabled (RUN_AGENTHARM_DATA=${RUN_AGENTHARM_DATA})"
fi

if [ "${RUN_MERGE}" = "1" ]; then
  merge_data
else
  echo "\n[SKIP] Merge disabled (RUN_MERGE=${RUN_MERGE})"
fi

# If AgentHarm data was generated in this run, ensure fresh merged files are produced.
if [ "${RUN_AGENTHARM_DATA}" = "1" ]; then
  echo "[INFO] AgentHarm data generated this run -> force merge AgentHarm results"
  python - <<PY
from merge import merge_agent_scores
merge_agent_scores("${AGENTHARM_MERGE_INPUT_DIR}/train", "${AGENTHARM_DIR}/train.csv")
merge_agent_scores("${AGENTHARM_MERGE_INPUT_DIR}/test", "${AGENTHARM_DIR}/test.csv")
print("Forced AgentHarm merge done.")
PY
fi

# Ensure required merged train/test CSVs exist even when merge step is skipped.
ensure_merged_csvs "MMLU" "${MMLU_TRAIN_CSV}" "${MMLU_TEST_CSV}" "${MMLU_MERGE_INPUT_DIR}" "${MMLU_DIR}"
ensure_merged_csvs "AgentHarm" "${AGENTHARM_TRAIN_CSV}" "${AGENTHARM_TEST_CSV}" "${AGENTHARM_MERGE_INPUT_DIR}" "${AGENTHARM_DIR}"

# Always normalize merged score columns to canonical agent names.
python - <<PY
import pandas as pd
from pathlib import Path

def normalize(path):
    p = Path(path)
    if not p.exists():
        return
    df = pd.read_csv(p)
    rename_map = {}
    for c in df.columns:
        if c.endswith("_new_score"):
            rename_map[c] = c[:-10]
        elif c.endswith("_score"):
            rename_map[c] = c[:-6]
    if rename_map:
        df = df.rename(columns=rename_map)
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_csv(p, index=False)
        print(f"[INFO] Normalized score columns in {p}")

normalize("${MMLU_TRAIN_CSV}")
normalize("${MMLU_TEST_CSV}")
normalize("${AGENTHARM_TRAIN_CSV}")
normalize("${AGENTHARM_TEST_CSV}")
PY

echo "\n[4/8] Grid search EmbedLLM on MMLU"
mmlu_best="$(grid_search_embedllm "${MMLU_TRAIN_CSV}" "${MMLU_TEST_CSV}" "${MMLU_DIR}/embedllm_grid/mmlu")"
IFS='|' read -r MMLU_BEST_METRIC MMLU_BEST_DIM MMLU_BEST_MODEL <<< "${mmlu_best}"
if [ -z "${MMLU_BEST_DIM}" ] || [ -z "${MMLU_BEST_MODEL}" ]; then
  echo "[ERROR] MMLU EmbedLLM grid search did not produce a valid best config."
  exit 1
fi
echo "[MMLU GRID BEST] metric=${MMLU_BEST_METRIC} dim=${MMLU_BEST_DIM} model=${MMLU_BEST_MODEL}"

echo "\n[5/8] Grid search EmbedLLM on AgentHarm"
agentharm_best="$(grid_search_embedllm "${AGENTHARM_TRAIN_CSV}" "${AGENTHARM_TEST_CSV}" "${AGENTHARM_DIR}/embedllm_grid/agentharm")"
IFS='|' read -r AGH_BEST_METRIC AGH_BEST_DIM AGH_BEST_MODEL <<< "${agentharm_best}"
if [ -z "${AGH_BEST_DIM}" ] || [ -z "${AGH_BEST_MODEL}" ]; then
  echo "[ERROR] AgentHarm EmbedLLM grid search did not produce a valid best config."
  exit 1
fi
echo "[AgentHarm GRID BEST] metric=${AGH_BEST_METRIC} dim=${AGH_BEST_DIM} model=${AGH_BEST_MODEL}"

echo "\n[6/8] Benchmark MMLU (ANTAP / EmbedLLM / AutoGen)
"
run_eval_triplet \
  "MMLU" \
  "${MMLU_TRAIN_CSV}" \
  "${MMLU_TEST_CSV}" \
  "${MMLU_DIR}/embedding_antap.pt" \
  "${MMLU_DIR}/embedding_embedllm.pt" \
  "${MMLU_BEST_DIM}" \
  "${MMLU_BEST_MODEL}" \
  "${MMLU_DIR}/benchmark"

echo "\n[7/8] Benchmark AgentHarm (ANTAP / EmbedLLM / AutoGen)
"
run_eval_triplet \
  "AgentHarm" \
  "${AGENTHARM_TRAIN_CSV}" \
  "${AGENTHARM_TEST_CSV}" \
  "${AGENTHARM_DIR}/embedding_antap.pt" \
  "${AGENTHARM_DIR}/embedding_embedllm.pt" \
  "${AGH_BEST_DIM}" \
  "${AGH_BEST_MODEL}" \
  "${AGENTHARM_DIR}/benchmark"

final_report

echo "\n[DONE] Final combined results: base/final_compare_all.csv"