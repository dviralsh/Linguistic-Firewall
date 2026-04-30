#!/usr/bin/env bash
# =============================================================================
# autonomous_monitor.sh
#
# Hourly autonomous monitor. Checks experiment progress and launches the
# next experiment when the previous one completes. Commits results to git.
#
# State machine:
#   1. MTBA + CTBA  → in progress (run_all_experiments.sh PID 3162786)
#   2. multi-malicious (N=1,5,10,20) → launched by watch_and_launch_multi_malicious.sh
#   3. adaptive attack → launched here when multi-malicious is done
#   4. [All done] → final git commit + status report
#
# Invoked by cron (see cron_id in autonomous_cron.sh) every hour.
# =============================================================================

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

log() { echo "[$(date '+%H:%M:%S')] [monitor] $*"; }
log "── Autonomous monitor tick ──"

PYTHON="${PYTHON:-python3}"

# ── Key sentinel files ────────────────────────────────────────────────────────
MTBA_DONE="${SCRIPT_DIR}/experiments_output/exp_sleeper_mtba/MMLU_base/benchmark/summary.json"
CTBA_DONE="${SCRIPT_DIR}/experiments_output/exp_sleeper_ctba/MMLU_base/benchmark/summary.json"
MULTI_MAL_DONE="${SCRIPT_DIR}/multi_malicious_output/final_report.csv"
ADAPTIVE_DONE="${SCRIPT_DIR}/adaptive_attack_output/summary.json"

# ── Helper: git commit if there are staged or unstaged tracked changes ────────
git_commit_if_changed() {
  local msg="$1"
  cd "${SCRIPT_DIR}"
  git add -A 2>/dev/null || true
  if git diff --cached --quiet 2>/dev/null; then
    log "[git] Nothing new to commit."
  else
    git commit -m "${msg}" 2>/dev/null && log "[git] Committed: ${msg}" || true
  fi
}

# ── Status check ──────────────────────────────────────────────────────────────
mtba_ok=0; ctba_ok=0; multimals_ok=0; adaptive_ok=0
[[ -f "${MTBA_DONE}" ]]      && mtba_ok=1
[[ -f "${CTBA_DONE}" ]]      && ctba_ok=1
[[ -f "${MULTI_MAL_DONE}" ]] && multimals_ok=1
[[ -f "${ADAPTIVE_DONE}" ]]  && adaptive_ok=1

log "Status: MTBA=${mtba_ok} CTBA=${ctba_ok} MultiMal=${multimals_ok} Adaptive=${adaptive_ok}"

# ── Print recent multi-malicious log tail if running ─────────────────────────
if [[ -f /tmp/multimal_exp.log ]]; then
  log "--- /tmp/multimal_exp.log (last 5 lines) ---"
  tail -5 /tmp/multimal_exp.log | while IFS= read -r line; do log "  ${line}"; done
fi

# ── Step A: Commit new experiment results to git as they appear ───────────────
if [[ "${mtba_ok}" == "1" && "${ctba_ok}" == "1" ]]; then
  if ! git log --oneline 2>/dev/null | grep -q "MTBA+CTBA results"; then
    git_commit_if_changed "chore: commit MTBA+CTBA benchmark results"
  fi
fi

if [[ "${multimals_ok}" == "1" ]]; then
  if ! git log --oneline 2>/dev/null | grep -q "multi-malicious results"; then
    log "Multi-malicious experiment COMPLETE. Committing results..."
    git_commit_if_changed "results: multi-malicious experiment (N=1,5,10,20) complete"
    # Print summary
    log "--- multi_malicious_output/final_report.csv ---"
    cat "${MULTI_MAL_DONE}" | while IFS= read -r line; do log "  ${line}"; done
  fi
fi

# ── Step B: Launch adaptive attack once multi-malicious is done ───────────────
if [[ "${multimals_ok}" == "1" && "${adaptive_ok}" == "0" ]]; then
  # Check not already running
  if pgrep -f "run_adaptive_attack_experiment.sh" > /dev/null 2>&1; then
    log "Adaptive attack already running. Waiting..."
  else
    log "Launching adaptive attack experiment..."
    nohup bash "${SCRIPT_DIR}/run_adaptive_attack_experiment.sh" \
      > /tmp/adaptive_attack_exp.log 2>&1 &
    ADAP_PID=$!
    log "Adaptive attack started (PID=${ADAP_PID}). Log: /tmp/adaptive_attack_exp.log"
  fi
fi

if [[ "${adaptive_ok}" == "1" ]]; then
  if ! git log --oneline 2>/dev/null | grep -q "adaptive attack results"; then
    log "Adaptive attack experiment COMPLETE. Committing results..."
    git_commit_if_changed "results: adaptive attack experiment complete"
    # Print summary
    log "--- adaptive_attack_output/final_report.csv ---"
    cat "${SCRIPT_DIR}/adaptive_attack_output/final_report.csv" | while IFS= read -r line; do log "  ${line}"; done
  fi
fi

# ── Step C: Check if all experiments done ────────────────────────────────────
if [[ "${mtba_ok}" == "1" && "${ctba_ok}" == "1" && \
      "${multimals_ok}" == "1" && "${adaptive_ok}" == "1" ]]; then
  log "ALL EXPERIMENTS COMPLETE."
  git_commit_if_changed "chore: final state — all experiments complete"
  log "Results summary:"
  log "  Main experiments:    experiments_output/final_report.csv"
  log "  Multi-malicious:     multi_malicious_output/final_report.csv"
  log "  Adaptive attack:     adaptive_attack_output/final_report.csv"
else
  log "Experiments still in progress. Will check again next hour."
fi

log "── Monitor tick done ──"
