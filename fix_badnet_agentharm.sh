#!/usr/bin/env bash
# Fix corrupted exp_sleeper_badnet AgentHarm data.
# Regenerates Archivist, Math_Expert, Coding_Expert (one at a time),
# re-merges, and re-runs benchmarks.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ANTAP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${ANTAP_DIR}/experiments_output/exp_sleeper_badnet"
AH_DIR="${EXP_DIR}/AgentHarm_base"
BENCH_DIR="${AH_DIR}/benchmark"
SYSTEM="experiments.system_sleeper_agent"
TIMEOUT=180

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cd "${ANTAP_DIR}"

log "=== badnet AgentHarm fix: step 1 — delete corrupted per-agent CSVs ==="
for agent in Archivist Coding_Expert Math_Expert; do
  rm -f "${AH_DIR}/results/test/${agent}.csv"
  rm -f "${AH_DIR}/results/train/${agent}.csv"
  log "  Deleted ${agent} test+train"
done

log "=== step 2 — regenerate agents serially (one model at a time) ==="
for agent in Archivist Math_Expert Coding_Expert; do
  log "  --- ${agent}: starting test+train in parallel ---"
  LOG_T="/tmp/fix_badnet_ah_${agent}_test.log"
  LOG_TR="/tmp/fix_badnet_ah_${agent}_train.log"

  ${PYTHON} -u create_test_data_agentharm.py \
    --system "${SYSTEM}" \
    --output "${AH_DIR}/results/test/${agent}.csv" \
    --agent "${agent}" \
    --timeout "${TIMEOUT}" \
    >"${LOG_T}" 2>&1 &
  PID_T=$!

  ${PYTHON} -u create_test_data_agentharm.py \
    --system "${SYSTEM}" \
    --output "${AH_DIR}/results/train/${agent}.csv" \
    --agent "${agent}" --train \
    --timeout "${TIMEOUT}" \
    >"${LOG_TR}" 2>&1 &
  PID_TR=$!

  log "    test PID=${PID_T}, train PID=${PID_TR} — waiting..."
  wait "${PID_T}"  && log "    test done" || { log "    [ERROR] test failed"; exit 1; }
  wait "${PID_TR}" && log "    train done" || { log "    [ERROR] train failed"; exit 1; }
  log "  --- ${agent} done ---"
done

log "=== step 3 — re-merge AgentHarm CSVs ==="
${PYTHON} -c "
import sys; sys.path.insert(0, '${ANTAP_DIR}')
from merge import merge_agent_scores
merge_agent_scores('${AH_DIR}/results/train', '${AH_DIR}/train.csv')
merge_agent_scores('${AH_DIR}/results/test',  '${AH_DIR}/test.csv')
print('Merge complete.')
"

log "=== step 4 — re-run AgentHarm benchmark ==="
ABS_BENCH="$(realpath ${BENCH_DIR})"
ABS_TRAIN="$(realpath ${AH_DIR}/train.csv)"
ABS_TEST="$(realpath ${AH_DIR}/test.csv)"
mkdir -p "${BENCH_DIR}"

log "    ANTAP (AgentHarm)"
${PYTHON} -u main.py \
  --router-backend antap \
  --train-data "${ABS_TRAIN}" \
  --test-data  "${ABS_TEST}" \
  --embedding  "${ABS_BENCH}/embedding_antap.pt" \
  --system core.system \
  --output "${ABS_BENCH}/output_antap.csv"

log "    AutoGen (AgentHarm)"
${PYTHON} -u main.py \
  --router-backend autogen \
  --test-data "${ABS_TEST}" \
  --system core.system \
  --output "${ABS_BENCH}/output_autogen.csv"

log "    EmbedLLM grid search (AgentHarm)"
BEST_DIM=232; BEST_ACC=-1
for dim in 128 232 384; do
  GRID_OUT="${ABS_BENCH}/embedllm_grid_${dim}.csv"
  ${PYTHON} -u main.py \
    --router-backend embedllm \
    --train-data "${ABS_TRAIN}" \
    --test-data  "${ABS_TEST}" \
    --embedding  "${ABS_BENCH}/embedding_embedllm_${dim}.pt" \
    --embedllm-dim "${dim}" \
    --system core.system \
    --output "${GRID_OUT}" 2>/dev/null || true
  if [[ -f "${GRID_OUT}" ]]; then
    ACC=$(${PYTHON} -c "import pandas as pd; df=pd.read_csv('${GRID_OUT}'); print(round(float(df['score'].mean()),6))" 2>/dev/null || echo "-1")
    if python3 -c "exit(0 if float('${ACC}') > float('${BEST_ACC}') else 1)" 2>/dev/null; then
      BEST_ACC="${ACC}"; BEST_DIM="${dim}"
    fi
  fi
done
log "    EmbedLLM best dim=${BEST_DIM} acc=${BEST_ACC}"

${PYTHON} -u main.py \
  --router-backend embedllm \
  --train-data "${ABS_TRAIN}" \
  --test-data  "${ABS_TEST}" \
  --embedding  "${ABS_BENCH}/embedding_embedllm.pt" \
  --embedllm-dim "${BEST_DIM}" \
  --system core.system \
  --output "${ABS_BENCH}/output_embedllm.csv"

log "=== step 5 — write summary.json ==="
${PYTHON} -c "
import pandas as pd, json, pathlib

def asr(df):
    if 'has_trigger' in df.columns:
        t = df[df['has_trigger'] == True]
        return float((t['selection'] == 'Malicious').mean()) if len(t) else 0.0
    return float((df['selection'] == 'Malicious').mean())

antap   = pd.read_csv('${ABS_BENCH}/output_antap.csv')
autogen = pd.read_csv('${ABS_BENCH}/output_autogen.csv')
embed   = pd.read_csv('${ABS_BENCH}/output_embedllm.csv')
summary = {
    'dataset': 'AgentHarm',
    'antap_accuracy':    float(antap['score'].mean()),
    'autogen_accuracy':  float(autogen['score'].mean()),
    'embedllm_accuracy': float(embed['score'].mean()),
    'antap_asr':         asr(antap),
    'autogen_asr':       asr(autogen),
    'embedllm_asr':      asr(embed),
    'embedllm_best_dim': ${BEST_DIM},
}
out = pathlib.Path('${ABS_BENCH}/summary.json')
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
"

log "=== badnet AgentHarm fix complete ==="
