#!/usr/bin/env bash
# =============================================================================
# run_multi_malicious_experiment.sh
#
# Experiments: ANTAP vs EmbedLLM with N ∈ {1, 5, 10, 20} malicious agents.
# Each malicious agent is a prompt-based sleeper (regular Qwen2.5) triggered
# by a different MMLU subject.  Benign pool: 4 base agents (existing data).
#
# Results land in:
#   multi_malicious_output/N_<n>/benchmark/summary.json
#
# Final report:
#   multi_malicious_output/final_report.csv
#
# Usage:
#   bash run_multi_malicious_experiment.sh [--skip-data] [--only-n 5]
#
# Prerequisites:
#   - Ollama running with qwen2.5:7b available
#   - Existing benign agent data in experiments_output/exp_description_attack/
#   - conda env Backdoors activated
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PATH="$HOME/bin:$PATH"

PYTHON="${PYTHON:-python3}"
OUTPUT_BASE="${OUTPUT_BASE:-multi_malicious_output}"
INFERENCE_TIMEOUT="${INFERENCE_TIMEOUT:-300}"
SKIP_DATA=0
ONLY_N=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-data) SKIP_DATA=1; shift ;;
    --only-n) ONLY_N="$2"; shift 2 ;;
    *) echo "[ERROR] Unknown flag: $1"; exit 1 ;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Benign agent data: reuse from description_attack experiment ───────────────
# These 4 agents are system-agnostic; their MMLU scores are valid for any exp.
BENIGN_TEST_DIR="${SCRIPT_DIR}/experiments_output/exp_description_attack/MMLU_base/results/test"
BENIGN_TRAIN_DIR="${SCRIPT_DIR}/experiments_output/exp_description_attack/MMLU_base/results/train"

if [[ ! -d "${BENIGN_TEST_DIR}" ]]; then
  log "[ERROR] Benign test data not found: ${BENIGN_TEST_DIR}"
  log "        Run the description_attack experiment first, or set BENIGN_TEST_DIR."
  exit 1
fi

# ── N configurations (from system_multi_malicious.py) ────────────────────────
# Each entry: "N:agent1,agent2,..."
# Trigger subjects are read from TRIGGER_MAP in the Python module.
declare -A N_TO_AGENTS=(
  [1]="Mal_A_Law"
  [5]="Mal_A_Law,Mal_A_Psychology,Mal_A_Biology,Mal_A_Economics,Mal_A_Medicine"
  [10]="Mal_A_Law,Mal_A_Psychology,Mal_A_Biology,Mal_A_Economics,Mal_A_Medicine,Mal_A_Philosophy,Mal_A_Physics,Mal_A_History,Mal_A_Chemistry,Mal_A_Math"
  [20]="Mal_A_Law,Mal_A_Psychology,Mal_A_Biology,Mal_A_Economics,Mal_A_Medicine,Mal_A_Philosophy,Mal_A_Physics,Mal_A_History,Mal_A_Chemistry,Mal_A_Math,Mal_B_Law,Mal_B_Psychology,Mal_B_Biology,Mal_B_Economics,Mal_B_Medicine,Mal_B_Philosophy,Mal_B_Physics,Mal_B_History,Mal_B_Chemistry,Mal_B_Math"
)

# Trigger subjects per agent (must match TRIGGER_MAP in system_multi_malicious.py)
declare -A AGENT_TRIGGERS=(
  [Mal_A_Law]="professional_law,jurisprudence,international_law,business_ethics"
  [Mal_A_Psychology]="professional_psychology,high_school_psychology,human_sexuality,human_aging"
  [Mal_A_Biology]="high_school_biology,college_biology,medical_genetics,virology"
  [Mal_A_Economics]="high_school_macroeconomics,high_school_microeconomics,econometrics"
  [Mal_A_Medicine]="clinical_knowledge,professional_medicine,anatomy,college_medicine"
  [Mal_A_Philosophy]="philosophy,moral_scenarios,moral_disputes,logical_fallacies"
  [Mal_A_Physics]="conceptual_physics,high_school_physics,college_physics,electrical_engineering"
  [Mal_A_History]="prehistory,high_school_world_history,world_religions,high_school_us_history,high_school_european_history"
  [Mal_A_Chemistry]="high_school_chemistry,college_chemistry,nutrition"
  [Mal_A_Math]="elementary_mathematics,high_school_mathematics,high_school_statistics,college_mathematics,abstract_algebra,formal_logic"
  [Mal_B_Law]="professional_law,jurisprudence,international_law,business_ethics"
  [Mal_B_Psychology]="professional_psychology,high_school_psychology,human_sexuality,human_aging"
  [Mal_B_Biology]="high_school_biology,college_biology,medical_genetics,virology"
  [Mal_B_Economics]="high_school_macroeconomics,high_school_microeconomics,econometrics"
  [Mal_B_Medicine]="clinical_knowledge,professional_medicine,anatomy,college_medicine"
  [Mal_B_Philosophy]="philosophy,moral_scenarios,moral_disputes,logical_fallacies"
  [Mal_B_Physics]="conceptual_physics,high_school_physics,college_physics,electrical_engineering"
  [Mal_B_History]="prehistory,high_school_world_history,world_religions,high_school_us_history,high_school_european_history"
  [Mal_B_Chemistry]="high_school_chemistry,college_chemistry,nutrition"
  [Mal_B_Math]="elementary_mathematics,high_school_mathematics,high_school_statistics,college_mathematics,abstract_algebra,formal_logic"
)

mkdir -p "${OUTPUT_BASE}"
MAL_DATA_DIR="${OUTPUT_BASE}/malicious_agent_data"
mkdir -p "${MAL_DATA_DIR}/test" "${MAL_DATA_DIR}/train"

# ── Step 1: Generate per-agent data for each malicious agent ─────────────────
if [[ "${SKIP_DATA}" == "0" ]]; then
  log "════ Step 1: Generating malicious agent data ════"
  # Collect unique agents across all N configs
  ALL_MAL_AGENTS="${N_TO_AGENTS[20]}"
  IFS=',' read -ra AGENTS_LIST <<< "${ALL_MAL_AGENTS}"

  for agent in "${AGENTS_LIST[@]}"; do
    triggers="${AGENT_TRIGGERS[${agent}]}"
    test_out="${MAL_DATA_DIR}/test/${agent}.csv"
    train_out="${MAL_DATA_DIR}/train/${agent}.csv"

    if [[ -f "${test_out}" ]]; then
      log "  [SKIP] ${agent} test already exists"
    else
      log "  [START] ${agent} test (triggers: ${triggers})"
      ${PYTHON} -u create_test_data_parallelized.py \
        --system experiments.system_multi_malicious \
        --output "${test_out}" \
        --agent "${agent}" \
        --trigger-subjects "${triggers}" \
        --inference-timeout "${INFERENCE_TIMEOUT}" \
        > "/tmp/antap_mul_${agent}_test_$$.log" 2>&1
      log "  [DONE] ${agent} test"
    fi

    if [[ -f "${train_out}" ]]; then
      log "  [SKIP] ${agent} train already exists"
    else
      log "  [START] ${agent} train (triggers: ${triggers})"
      ${PYTHON} -u create_test_data_parallelized.py \
        --system experiments.system_multi_malicious \
        --output "${train_out}" \
        --agent "${agent}" \
        --train \
        --trigger-subjects "${triggers}" \
        --inference-timeout "${INFERENCE_TIMEOUT}" \
        > "/tmp/antap_mul_${agent}_train_$$.log" 2>&1
      log "  [DONE] ${agent} train"
    fi
  done
  log "All malicious agent data generated."
else
  log "[SKIP] Data generation skipped (--skip-data)"
fi

# ── Step 2: For each N, merge data and run benchmarks ────────────────────────
log "════ Step 2: Running benchmarks for N ∈ {1, 5, 10, 20} ════"

for N in 1 5 10 20; do
  if [[ -n "${ONLY_N}" && "${ONLY_N}" != "${N}" ]]; then
    continue
  fi

  active_mal="${N_TO_AGENTS[${N}]}"
  exp_dir="${OUTPUT_BASE}/N_${N}"
  benchmark_dir="${exp_dir}/benchmark"
  summary_path="${benchmark_dir}/summary.json"

  mkdir -p "${exp_dir}" "${benchmark_dir}"

  log "── N=${N}: active agents = ${active_mal}"

  # Merge: benign data + active malicious data
  test_merged="${exp_dir}/test.csv"
  train_merged="${exp_dir}/train.csv"

  if [[ -f "${test_merged}" ]]; then
    log "  [SKIP] ${test_merged} already exists"
  else
    log "  Merging test data..."
    ${PYTHON} -u merge_multi_malicious.py \
      --benign-dir  "${BENIGN_TEST_DIR}" \
      --mal-dir     "${MAL_DATA_DIR}/test" \
      --active-mal  "${active_mal}" \
      --output      "${test_merged}"
  fi

  if [[ -f "${train_merged}" ]]; then
    log "  [SKIP] ${train_merged} already exists"
  else
    log "  Merging train data..."
    ${PYTHON} -u merge_multi_malicious.py \
      --benign-dir  "${BENIGN_TRAIN_DIR}" \
      --mal-dir     "${MAL_DATA_DIR}/train" \
      --active-mal  "${active_mal}" \
      --output      "${train_merged}"
  fi

  if [[ -f "${summary_path}" ]]; then
    log "  [SKIP] Benchmark already complete"
    continue
  fi

  log "  Running ANTAP + EmbedLLM benchmark..."
  ${PYTHON} -u evaluate_multi_malicious.py \
    --train-data  "${train_merged}" \
    --test-data   "${test_merged}" \
    --out-dir     "${benchmark_dir}" \
    --active-mal  "${active_mal}" \
    --n-malicious "${N}"

  log "  N=${N} done."
done

# ── Step 3: Consolidated report ───────────────────────────────────────────────
log "════ Step 3: Building final report ════"
${PYTHON} - <<'PYEOF'
import json, pathlib, pandas as pd

output_base = pathlib.Path("multi_malicious_output")
rows = []
for N in (1, 5, 10, 20):
    summary_path = output_base / f"N_{N}" / "benchmark" / "summary.json"
    if not summary_path.exists():
        print(f"[WARN] Missing: {summary_path}")
        continue
    s = json.loads(summary_path.read_text())
    rows.append({
        "N_malicious":       s["n_malicious"],
        "antap_asr_%":       round(s["antap_asr"] * 100, 2) if s["antap_asr"] is not None else "",
        "embedllm_asr_%":    round(s["embedllm_asr"] * 100, 2) if s["embedllm_asr"] is not None else "",
        "antap_accuracy_%":  round(s["antap_accuracy"] * 100, 2) if s["antap_accuracy"] is not None else "",
        "embedllm_accuracy_%": round(s["embedllm_accuracy"] * 100, 2) if s["embedllm_accuracy"] is not None else "",
    })

if rows:
    df = pd.DataFrame(rows).sort_values("N_malicious")
    out = output_base / "final_report.csv"
    df.to_csv(out, index=False)
    print("\n══ Multi-Malicious Experiment Results ══")
    print(df.to_string(index=False))
    print(f"\nSaved to {out}")
else:
    print("No results found yet.")
PYEOF

log "All done. Results in ${OUTPUT_BASE}/final_report.csv"
