#!/usr/bin/env bash
# Rerun Math_Expert data generation and benchmarks for methods with broken scores.
# Run this AFTER the main pipeline (run_all_experiments.sh --only sleeper) completes.
# ctba is excluded — it was generated with the fixed settings.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PATH="$HOME/bin:$PATH"

OUTPUT_BASE="${OUTPUT_BASE:-experiments_output}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Deleting broken Math_Expert CSVs for badnet, sleeper, vpi, mtba ==="
for method in badnet sleeper vpi mtba; do
    base="${OUTPUT_BASE}/exp_sleeper_${method}"
    for dataset in MMLU_base AgentHarm_base; do
        for split in test train; do
            f="${base}/${dataset}/results/${split}/Math_Expert.csv"
            if [[ -f "${f}" ]]; then
                log "  Deleting ${f}"
                rm -f "${f}"
            fi
        done
    done
done

log "=== Rerunning benchmarks for affected methods ==="

# badnet, vpi, mtba use --only <method> directly
for method in badnet vpi mtba; do
    log "--- Running --only ${method} ---"
    bash run_all_experiments.sh --only "${method}" 2>&1 | tee -a /tmp/rerun_math_expert.log
done

# "sleeper" DPA method requires --only sleeper_method to avoid running all 5
log "--- Running --only sleeper_method ---"
bash run_all_experiments.sh --only sleeper_method 2>&1 | tee -a /tmp/rerun_math_expert.log

log "=== All reruns complete. Final report will be rebuilt by each run. ==="
log "Check experiments_output/final_report.csv for updated results."
