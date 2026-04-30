#!/usr/bin/env bash
# =============================================================================
# run_adaptive_attack_experiment.sh
#
# RQ3: ANTAP vs EmbedLLM under adaptive (universal adversarial trigger) attacks.
# Uses AgentHarm benchmark data from exp_description_attack.
# Runs uat_router_attack.py over 3 seeds for both backends.
# Outputs final_report.csv + summary.json in adaptive_attack_output/.
#
# Usage:
#   bash run_adaptive_attack_experiment.sh [--skip-attack] [--force]
#
# Prerequisites:
#   - exp_description_attack/AgentHarm_base/results/{train,test}/ must exist
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PATH="$HOME/bin:$PATH"

PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${SCRIPT_DIR}/adaptive_attack_output"
SKIP_ATTACK=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-attack) SKIP_ATTACK=1; shift ;;
    --force) FORCE=1; shift ;;
    *) echo "[ERROR] Unknown flag: $1"; exit 1 ;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

TRAIN_SOURCE="${SCRIPT_DIR}/experiments_output/exp_description_attack/AgentHarm_base/results/train"
TEST_SOURCE="${SCRIPT_DIR}/experiments_output/exp_description_attack/AgentHarm_base/results/test"
SUMMARY_PATH="${OUTPUT_DIR}/summary.json"

if [[ ! -d "${TRAIN_SOURCE}" ]]; then
  log "[ERROR] Missing: ${TRAIN_SOURCE}"
  log "        Run the description_attack experiment first."
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ -f "${SUMMARY_PATH}" && "${FORCE}" == "0" ]]; then
  log "[SKIP] Adaptive attack summary already exists: ${SUMMARY_PATH}"
  log "       Use --force to rerun."
else
  if [[ "${SKIP_ATTACK}" == "0" ]]; then
    log "════ Running adaptive attack (seeds 0 1 2, backends antap embedllm) ════"
    ${PYTHON} -u "${SCRIPT_DIR}/attacks/uat_router_attack.py" \
      --train-source  "${TRAIN_SOURCE}" \
      --test-source   "${TEST_SOURCE}" \
      --output-dir    "${OUTPUT_DIR}/uat_results" \
      --router-backends antap embedllm \
      --target-agent  "Malicious" \
      --attack-mode   unsafe \
      --seeds         0 1 2 \
      --trigger-len   4 \
      --iterations    30 \
      --candidates-per-pos 100 \
      --subset-size   64 \
      --train-opt-limit  300 \
      --router-train-limit 500 \
      --eval-limit    0
    log "Attack optimization complete."
  else
    log "[SKIP] Attack optimization skipped (--skip-attack)"
    # Copy existing results if available
    if [[ -d "${SCRIPT_DIR}/attacks/output/agentharm_base_attack" ]]; then
      log "Reusing existing results from attacks/output/agentharm_base_attack"
      mkdir -p "${OUTPUT_DIR}/uat_results"
      cp -r "${SCRIPT_DIR}/attacks/output/agentharm_base_attack/." "${OUTPUT_DIR}/uat_results/"
    fi
  fi
fi

# ── Build final report ────────────────────────────────────────────────────────
log "════ Building adaptive attack final report ════"
${PYTHON} - <<'PYEOF'
import json, pathlib, pandas as pd, numpy as np

# Try new output dir first, fallback to legacy
output_dir = pathlib.Path("adaptive_attack_output/uat_results")
legacy_dir = pathlib.Path("attacks/output/agentharm_base_attack")

summary_src = output_dir / "summary.json"
if not summary_src.exists():
    summary_src = legacy_dir / "summary.json"
    if not summary_src.exists():
        print("[ERROR] No summary.json found. Run attack first.")
        exit(1)

rows = json.loads(summary_src.read_text())
df = pd.DataFrame(rows)

# Metrics: non_adaptive_asr = clean_target_rate (router's ASR on clean input)
#          adaptive_asr = adv_target_rate (router's ASR after trigger prepended)
report = []
for backend, grp in df.groupby("backend"):
    non_adap = grp["clean_target_rate"].mean() * 100
    adap = grp["adv_target_rate"].mean() * 100
    report.append({
        "router": backend.upper(),
        "non_adaptive_asr_%": round(non_adap, 2),
        "adaptive_asr_%": round(adap, 2),
        "delta_%": round(adap - non_adap, 2),
        "seeds": list(grp["seed"]),
    })

report_df = pd.DataFrame(report).sort_values("router")
out_dir = pathlib.Path("adaptive_attack_output")
out_dir.mkdir(exist_ok=True)

report_df.to_csv(out_dir / "final_report.csv", index=False)

summary = {
    "experiment": "adaptive_attack",
    "source_data": str(summary_src),
    "results": report,
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

print("\n══ Adaptive Attack Results ══")
print(report_df.to_string(index=False))
print(f"\nSaved to {out_dir}/final_report.csv")
PYEOF

log "Adaptive attack experiment complete. Results in ${OUTPUT_DIR}/"
