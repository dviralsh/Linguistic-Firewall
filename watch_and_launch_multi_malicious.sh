#!/usr/bin/env bash
# =============================================================================
# watch_and_launch_multi_malicious.sh
#
# Polls until the main experiment pipeline (exp_sleeper_mtba + exp_sleeper_ctba)
# is complete, then launches the multi-malicious experiment.
#
# Usage: nohup bash watch_and_launch_multi_malicious.sh > /tmp/watch_multimal.log 2>&1 &
# =============================================================================

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

MTBA_DONE="${SCRIPT_DIR}/experiments_output/exp_sleeper_mtba/MMLU_base/benchmark/summary.json"
CTBA_DONE="${SCRIPT_DIR}/experiments_output/exp_sleeper_ctba/MMLU_base/benchmark/summary.json"

log "Watching for MTBA + CTBA completion..."
log "  MTBA: ${MTBA_DONE}"
log "  CTBA: ${CTBA_DONE}"

while true; do
  if [[ -f "${MTBA_DONE}" && -f "${CTBA_DONE}" ]]; then
    log "Both MTBA and CTBA complete. Launching multi-malicious experiment."
    break
  fi
  # Print progress
  mtba_status=$([ -f "${MTBA_DONE}" ] && echo "DONE" || echo "pending")
  ctba_status=$([ -f "${CTBA_DONE}" ] && echo "DONE" || echo "pending")
  log "  MTBA: ${mtba_status} | CTBA: ${ctba_status}"
  sleep 300  # check every 5 minutes
done

log "Starting multi-malicious experiment..."
nohup bash "${SCRIPT_DIR}/run_multi_malicious_experiment.sh" \
  > /tmp/multimal_exp.log 2>&1 &
MULTIMAL_PID=$!
log "Multi-malicious experiment started (PID=${MULTIMAL_PID})"
log "Log: /tmp/multimal_exp.log"
