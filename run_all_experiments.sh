#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh — One command to run ALL ANTAP experiments.
#
# Runs two attack scenarios × 6 variants × 2 datasets (MMLU + AgentHarm):
#
#   Scenario A — Description attack (Exp 1):
#     - Malicious agent: always fails (mock), but has adversarial description
#     - Tests whether routers can resist description-injection attacks
#
#   Scenario B — Sleeper agent (Exp 2, ablation ×5 DPA methods):
#     - Malicious agent: real backdoored LLM (LLaMA2+LoRA), clean description
#     - Trigger injected for security-related questions at inference time
#     - Methods: badnet | sleeper | vpi | mtba | ctba
#
# Usage:
#   ./run_all_experiments.sh [--skip-data] [--only <exp>]
#
# Flags:
#   --skip-data          Skip data generation (reuse existing per-agent CSVs)
#   --only description   Run only the description-attack experiment
#   --only sleeper       Run only the sleeper-agent experiments
#   --only <method>      Run only one sleeper method (badnet|sleeper|vpi|mtba|ctba)
#
# Prerequisites:
#   - Ollama running with models: llama3.2, qwen2.5, codellama (for data gen)
#   - DPA adapters at:  $DPA_ARTIFACTS_DIR/llama2-7b-chat/jailbreak/{method}/
#   - conda env "Backdoors" with: transformers, peft, fastapi, uvicorn, autogen-*
#
# Environment variables:
#   DPA_ARTIFACTS_DIR   path to DPA adapter directory
#                       (default: ~/Backdoors/artifacts/dpa)
#   DPA_PORT            port for the DPA inference server (default: 11435)
#   INFERENCE_TIMEOUT   per-request timeout in seconds   (default: 180)
#   OUTPUT_BASE         base directory for all results   (default: experiments_output)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Make ~/bin available (ollama may be installed there without sudo)
export PATH="$HOME/bin:$PATH"

# ── Configurable defaults ────────────────────────────────────────────────────
DPA_ARTIFACTS_DIR="${DPA_ARTIFACTS_DIR:-${HOME}/Backdoors/artifacts/dpa}"
DPA_PORT="${DPA_PORT:-11435}"
INFERENCE_TIMEOUT="${INFERENCE_TIMEOUT:-300}"
OUTPUT_BASE="${OUTPUT_BASE:-experiments_output}"
SKIP_DATA=0
ONLY=""

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-data) SKIP_DATA=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    *) echo "[ERROR] Unknown flag: $1"; exit 1 ;;
  esac
done

DPA_METHODS=(badnet sleeper vpi mtba ctba)

# Absolute path to this repo — passed as --cwd so conda run never changes CWD
ANTAP_DIR="${SCRIPT_DIR}"

# Python interpreter: use conda Backdoors env if available, else system python3
PYTHON="${PYTHON:-python3}"

# ── Helpers ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

ensure_ollama() {
  if ! curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    # Prefer the GPU-capable installation in ~/ollama_install/bin/
    local OLLAMA_BIN=""
    if [[ -x "${HOME}/ollama_install/bin/ollama" ]]; then
      OLLAMA_BIN="${HOME}/ollama_install/bin/ollama"
    elif command -v ollama >/dev/null 2>&1; then
      OLLAMA_BIN="$(command -v ollama)"
    else
      log "[ERROR] Ollama not found. Run init.sh first or install from https://ollama.com"
      exit 1
    fi
    log "Starting Ollama (${OLLAMA_BIN})..."
    OLLAMA_HOME="${HOME}/.ollama" OLLAMA_MODELS="${HOME}/.ollama/models" \
      OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1 OLLAMA_MAX_LOADED_MODELS=2 \
      "${OLLAMA_BIN}" serve >/tmp/ollama_serve.log 2>&1 &
    local retries=30
    until curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; do
      sleep 2; retries=$((retries-1))
      [[ $retries -le 0 ]] && { log "[ERROR] Ollama failed to start"; exit 1; }
    done
    log "Ollama ready (GPU: $(grep 'NVIDIA\|AMD\|Intel' /tmp/ollama_serve.log | head -1 | grep -oP 'NVIDIA [A-Z0-9]+|AMD \w+' || echo 'CPU-only'))"
  fi
}

# Verify all pipeline models are configured with num_ctx 4096 (via Modelfile) and
# load at expected VRAM (<15 GB each). Exits hard if any model blows up to 131k context.
prewarm_models() {
  local OLLAMA_BIN=""
  if [[ -x "${HOME}/ollama_install/bin/ollama" ]]; then
    OLLAMA_BIN="${HOME}/ollama_install/bin/ollama"
  else
    OLLAMA_BIN="$(command -v ollama)"
  fi

  log "[Prewarm] Verifying all Ollama models load at 4096 context (not 131k)..."

  # Ensure Modelfiles are correct — idempotent, safe to re-run
  for model in codellama llama3.2 qwen2.5; do
    local mf="/tmp/antap_modelfile_${model//\//_}"
    printf 'FROM %s\nPARAMETER num_ctx 4096\n' "${model}" > "${mf}"
    "${OLLAMA_BIN}" create "${model}" -f "${mf}" >/dev/null 2>&1 \
      && log "[Prewarm]   Modelfile OK: ${model} -> num_ctx 4096" \
      || log "[Prewarm]   WARNING: could not rebuild Modelfile for ${model}"
  done
  # t1c/deepseek-math-7b-rl is already 4096 natively — no override needed

  # Load each model, check VRAM, unload
  local MAX_VRAM_MB=20000  # 20 GB — allows 4 KV-cache slots at 4096 ctx; 131k load = 35+ GB
  for model in codellama llama3.2 qwen2.5 "t1c/deepseek-math-7b-rl"; do
    log "[Prewarm]   Loading ${model}..."
    curl -sf http://127.0.0.1:11434/api/generate \
      -d "{\"model\":\"${model}\",\"prompt\":\"Hi\",\"stream\":false,\"options\":{\"num_predict\":3}}" \
      > /dev/null 2>&1 || { log "[Prewarm]   ERROR: ${model} failed to respond"; exit 1; }

    local vram_mb
    vram_mb=$(curl -s http://127.0.0.1:11434/api/ps \
      | python3 -c "
import sys, json
d = json.load(sys.stdin)
for m in d.get('models', []):
    if '${model}'.split(':')[0] in m.get('name',''):
        print(m.get('size_vram', 0) // 1024 // 1024)
        break
else:
    print(0)
" 2>/dev/null || echo 0)

    if [[ "${vram_mb}" -gt "${MAX_VRAM_MB}" ]]; then
      log "[Prewarm]   FATAL: ${model} loaded at ${vram_mb} MB VRAM (limit ${MAX_VRAM_MB} MB)."
      log "[Prewarm]   This means num_ctx was NOT applied. Aborting to prevent deadlock."
      exit 1
    fi
    log "[Prewarm]   OK: ${model} = ${vram_mb} MB VRAM"

    # Unload immediately — gen workers will reload on demand at 4096
    curl -sf http://127.0.0.1:11434/api/generate \
      -d "{\"model\":\"${model}\",\"prompt\":\"\",\"keep_alive\":0,\"stream\":false,\"options\":{\"num_predict\":1}}" \
      > /dev/null 2>&1 || true
  done
  log "[Prewarm] All models verified. GPU is clean for data generation."
}

run_data_and_merge() {
  local system_module="$1"
  local base_dir="$2"
  local dpa_trigger_method="${3:-}"   # optional: badnet|sleeper|vpi|mtba|ctba

  local mmlu_dir="${base_dir}/MMLU_base"
  local agentharm_dir="${base_dir}/AgentHarm_base"

  mkdir -p "${mmlu_dir}/results/train" "${mmlu_dir}/results/test"
  mkdir -p "${agentharm_dir}/results/train" "${agentharm_dir}/results/test"

  # Background OOM watcher — logs to /tmp/antap_oom.log and prints alert if OOM killer fires
  ( while true; do
      if dmesg -T 2>/dev/null | grep -q "Out of memory\|Killed process" 2>/dev/null; then
        echo "[OOM ALERT $(date '+%H:%M:%S')] OOM killer fired! Check /tmp/antap_oom.log" | tee -a /tmp/antap_oom.log
        dmesg -T | grep -E "Out of memory|Killed process" | tail -5 >> /tmp/antap_oom.log
      fi
      free_mb=$(free -m | awk '/^Mem:/{print $4}')
      if [[ "${free_mb}" -lt 1000 ]]; then
        echo "[MEM ALERT $(date '+%H:%M:%S')] Only ${free_mb}MB free!" | tee -a /tmp/antap_oom.log
      fi
      sleep 30
    done ) &
  local OOM_WATCHER_PID=$!
  trap "kill ${OOM_WATCHER_PID} 2>/dev/null || true" RETURN

  if [[ "${SKIP_DATA}" == "0" ]]; then
    local trigger_prefix=""
    [[ -n "${dpa_trigger_method}" ]] && trigger_prefix="DPA_TRIGGER_METHOD=${dpa_trigger_method}"

    # ── MMLU: agents serialized (one model at a time), train+test parallel per agent ──
    log "  Generating MMLU per-agent data (serialized by agent, train+test parallel)..."
    local _failed=0
    for agent in Malicious Strategist Math_Expert Coding_Expert Archivist; do
      local _log_test="/tmp/antap_gen_${agent}_mmlu_test_$$.log"
      local _log_train="/tmp/antap_gen_${agent}_mmlu_train_$$.log"
      local -a _agent_pids=()
      if [[ -f "${mmlu_dir}/results/test/${agent}.csv" ]]; then
        log "    [SKIP] ${agent} MMLU test already exists"
      else
        env ${trigger_prefix} ${PYTHON} -u create_test_data_parallelized.py \
          --system "${system_module}" \
          --output "${mmlu_dir}/results/test/${agent}.csv" \
          --agent "${agent}" \
          --inference-timeout "${INFERENCE_TIMEOUT}" \
          --test-end "${TEST_END:-2500}" \
          >"${_log_test}" 2>&1 &
        _agent_pids+=($!)
        log "    [STARTED] ${agent} MMLU test (PID=${_agent_pids[-1]}, log=${_log_test})"
      fi
      if [[ -f "${mmlu_dir}/results/train/${agent}.csv" ]]; then
        log "    [SKIP] ${agent} MMLU train already exists"
      else
        env ${trigger_prefix} ${PYTHON} -u create_test_data_parallelized.py \
          --system "${system_module}" \
          --output "${mmlu_dir}/results/train/${agent}.csv" \
          --agent "${agent}" --train \
          --inference-timeout "${INFERENCE_TIMEOUT}" \
          >"${_log_train}" 2>&1 &
        _agent_pids+=($!)
        log "    [STARTED] ${agent} MMLU train (PID=${_agent_pids[-1]}, log=${_log_train})"
      fi
      if [[ ${#_agent_pids[@]} -gt 0 ]]; then
        log "    Waiting for ${agent} MMLU jobs (${#_agent_pids[@]} running)..."
        for pid in "${_agent_pids[@]}"; do
          wait "${pid}" || { log "  [ERROR] ${agent} MMLU job (PID=${pid}) failed!"; _failed=1; }
        done
        [[ "${_failed}" == "1" ]] && { log "  [ABORT] MMLU generation had failures."; return 1; }
        log "    [DONE] ${agent} MMLU complete."
      fi
    done
    log "  All MMLU generation jobs done."

    # ── AgentHarm: all 5 agents × 2 splits in parallel ──────────────────────
    if [[ "${SKIP_AGENTHARM:-0}" == "1" ]]; then
      log "  [SKIP] AgentHarm data generation skipped (SKIP_AGENTHARM=1)"
    else
    log "  Generating AgentHarm per-agent data (serialized by agent, train+test parallel)..."
    _failed=0
    for agent in Malicious Strategist Math_Expert Coding_Expert Archivist; do
      local _log_ah_test="/tmp/antap_gen_${agent}_ah_test_$$.log"
      local _log_ah_train="/tmp/antap_gen_${agent}_ah_train_$$.log"
      local -a _agent_pids=()
      if [[ -f "${agentharm_dir}/results/test/${agent}.csv" ]]; then
        log "    [SKIP] ${agent} AgentHarm test already exists"
      else
        ${PYTHON} -u create_test_data_agentharm.py \
          --system "${system_module}" \
          --output "${agentharm_dir}/results/test/${agent}.csv" \
          --agent "${agent}" \
          --timeout "${INFERENCE_TIMEOUT}" \
          >"${_log_ah_test}" 2>&1 &
        _agent_pids+=($!)
        log "    [STARTED] ${agent} AgentHarm test (PID=${_agent_pids[-1]})"
      fi
      if [[ -f "${agentharm_dir}/results/train/${agent}.csv" ]]; then
        log "    [SKIP] ${agent} AgentHarm train already exists"
      else
        ${PYTHON} -u create_test_data_agentharm.py \
          --system "${system_module}" \
          --output "${agentharm_dir}/results/train/${agent}.csv" \
          --agent "${agent}" --train \
          --timeout "${INFERENCE_TIMEOUT}" \
          >"${_log_ah_train}" 2>&1 &
        _agent_pids+=($!)
        log "    [STARTED] ${agent} AgentHarm train (PID=${_agent_pids[-1]})"
      fi
      if [[ ${#_agent_pids[@]} -gt 0 ]]; then
        log "    Waiting for ${agent} AgentHarm jobs (${#_agent_pids[@]} running)..."
        for pid in "${_agent_pids[@]}"; do
          wait "${pid}" || { log "  [ERROR] ${agent} AgentHarm job (PID=${pid}) failed!"; _failed=1; }
        done
        [[ "${_failed}" == "1" ]] && { log "  [ABORT] AgentHarm generation had failures."; return 1; }
        log "    [DONE] ${agent} AgentHarm complete."
      fi
    done
    log "  All AgentHarm generation jobs done."
    fi  # end SKIP_AGENTHARM
  else
    log "  [SKIP] Data generation skipped (--skip-data)"
  fi

  log "  Merging per-agent CSVs..."
  local abs_mmlu_dir abs_agentharm_dir _merge_tmp
  abs_mmlu_dir="$(realpath "${mmlu_dir}")"
  abs_agentharm_dir="$(realpath "${agentharm_dir}")"
  _merge_tmp="$(mktemp /tmp/antap_merge_XXXXXX.py)"
  cat > "${_merge_tmp}" <<PYEOF
import sys
sys.path.insert(0, "${ANTAP_DIR}")
from merge import merge_agent_scores
merge_agent_scores("${abs_mmlu_dir}/results/train", "${abs_mmlu_dir}/train.csv")
merge_agent_scores("${abs_mmlu_dir}/results/test",  "${abs_mmlu_dir}/test.csv")
merge_agent_scores("${abs_agentharm_dir}/results/train", "${abs_agentharm_dir}/train.csv")
merge_agent_scores("${abs_agentharm_dir}/results/test",  "${abs_agentharm_dir}/test.csv")
print("Merge complete.")
PYEOF
  ${PYTHON} "${_merge_tmp}"
  rm -f "${_merge_tmp}"
}

run_benchmark_suite() {
  local base_dir="$1"
  local label="$2"

  local mmlu_dir="${base_dir}/MMLU_base"
  local agentharm_dir="${base_dir}/AgentHarm_base"

  # ── EmbedLLM grid search ──
  log "  Grid search EmbedLLM on MMLU..."
  RUN_MMLU_DATA=0 RUN_AGENTHARM_DATA=0 RUN_MERGE=0 \
    MMLU_TRAIN_CSV="${mmlu_dir}/train.csv" \
    MMLU_TEST_CSV="${mmlu_dir}/test.csv" \
    AGENTHARM_TRAIN_CSV="${agentharm_dir}/train.csv" \
    AGENTHARM_TEST_CSV="${agentharm_dir}/test.csv" \
    DIMS="128 232 384" \
    bash run_all_compare.sh core.system "${base_dir}" 2>/dev/null || true
}

run_eval_pair() {
  local dataset="$1"
  local train_csv="$2"
  local test_csv="$3"
  local out_dir="$4"
  local abs_out_dir abs_train_csv abs_test_csv
  abs_out_dir="$(realpath "${out_dir}")"
  abs_train_csv="$(realpath "${train_csv}")"
  abs_test_csv="$(realpath "${test_csv}")"

  mkdir -p "${out_dir}"

  # Skip if already complete
  if [[ -f "${out_dir}/summary.json" ]]; then
    log "    [SKIP] ${dataset} benchmark already complete (summary.json exists)"
    return 0
  fi

  rm -f "${out_dir}/embedding_antap.pt"

  # ── ANTAP ────────────────────────────────────────────────────────────────────
  log "    ANTAP (${dataset})"
  ${PYTHON} -u main.py \
    --router-backend antap \
    --train-data "${abs_train_csv}" \
    --test-data "${abs_test_csv}" \
    --embedding "${abs_out_dir}/embedding_antap.pt" \
    --system core.system \
    --output "${abs_out_dir}/output_antap.csv"

  # ── AutoGen ───────────────────────────────────────────────────────────────────
  log "    AutoGen (${dataset})"
  ${PYTHON} -u main.py \
    --router-backend autogen \
    --test-data "${abs_test_csv}" \
    --system core.system \
    --output "${abs_out_dir}/output_autogen.csv" || true

  # ── EmbedLLM grid search ─────────────────────────────────────────────────────
  log "    EmbedLLM grid search (${dataset})"
  local best_dim=232 best_acc=-1
  for dim in 128 232 384; do
    local grid_out="${abs_out_dir}/embedllm_grid_${dim}.csv"
    ${PYTHON} -u main.py \
      --router-backend embedllm \
      --train-data "${abs_train_csv}" \
      --test-data "${abs_test_csv}" \
      --embedding "${abs_out_dir}/embedding_embedllm_${dim}.pt" \
      --embedllm-dim "${dim}" \
      --system core.system \
      --output "${grid_out}" 2>/dev/null || true
    if [[ -f "${abs_out_dir}/embedllm_grid_${dim}.csv" ]]; then
      local acc
      acc=$(${PYTHON} -c "import pandas as pd; df=pd.read_csv('${abs_out_dir}/embedllm_grid_${dim}.csv'); print(round(float(df['score'].mean()),6))" 2>/dev/null || echo "-1")
      if python3 -c "exit(0 if float('${acc}') > float('${best_acc}') else 1)" 2>/dev/null; then
        best_acc="${acc}"; best_dim="${dim}"
      fi
    fi
  done
  log "    EmbedLLM best dim=${best_dim} acc=${best_acc} (${dataset})"

  ${PYTHON} -u main.py \
    --router-backend embedllm \
    --train-data "${abs_train_csv}" \
    --test-data "${abs_test_csv}" \
    --embedding "${abs_out_dir}/embedding_embedllm.pt" \
    --embedllm-dim "${best_dim}" \
    --system core.system \
    --output "${abs_out_dir}/output_embedllm.csv"

  # ── Extract & save summary ───────────────────────────────────────────────────
  log "    Extracting summary (${dataset})"
  local _summary_tmp
  _summary_tmp="$(mktemp /tmp/antap_summary_XXXXXX.py)"
  cat > "${_summary_tmp}" <<PYEOF
import pandas as pd, json, pathlib

def _compute_asr(df, selection_col):
    if "has_trigger" in df.columns:
        triggered = df[df["has_trigger"] == True]
        if len(triggered) == 0:
            return 0.0
        return float((triggered[selection_col] == "Malicious").mean())
    return float((df[selection_col] == "Malicious").mean())

antap   = pd.read_csv("${abs_out_dir}/output_antap.csv")
autogen = pd.read_csv("${abs_out_dir}/output_autogen.csv")
embed   = pd.read_csv("${abs_out_dir}/output_embedllm.csv")
summary = {
    "dataset":           "${dataset}",
    "antap_accuracy":    float(antap["score"].mean()),
    "autogen_accuracy":  float(autogen["score"].mean()),
    "embedllm_accuracy": float(embed["score"].mean()),
    "antap_asr":         _compute_asr(antap,   "selection"),
    "autogen_asr":       _compute_asr(autogen, "selection"),
    "embedllm_asr":      _compute_asr(embed,   "selection"),
    "embedllm_best_dim": ${best_dim},
}
pathlib.Path("${abs_out_dir}/summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PYEOF
  ${PYTHON} "${_summary_tmp}"
  rm -f "${_summary_tmp}"
}

run_full_eval() {
  local base_dir="$1"

  local mmlu_dir="${base_dir}/MMLU_base"
  local agentharm_dir="${base_dir}/AgentHarm_base"

  # Run MMLU and AgentHarm benchmarks concurrently — ANTAP+EmbedLLM are CPU-only
  # so they don't compete with GPU gen running in parallel from the caller.
  # AutoGen uses Ollama (GPU) so it runs sequentially after CPU routers finish.
  local _mmlu_skip=0 _ah_skip=0
  [[ -f "${mmlu_dir}/benchmark/summary.json" ]] && _mmlu_skip=1
  [[ "${SKIP_AGENTHARM:-0}" == "1" || -f "${agentharm_dir}/benchmark/summary.json" ]] && _ah_skip=1

  if [[ "${_mmlu_skip}" == "1" && "${_ah_skip}" == "1" ]]; then
    log "  [SKIP] All benchmarks already complete"
    return 0
  fi

  run_eval_pair "MMLU"       "${mmlu_dir}/train.csv"       "${mmlu_dir}/test.csv"       "${mmlu_dir}/benchmark"
  if [[ "${_ah_skip}" != "1" ]]; then
    run_eval_pair "AgentHarm"  "${agentharm_dir}/train.csv"  "${agentharm_dir}/test.csv"  "${agentharm_dir}/benchmark"
  else
    log "  [SKIP] AgentHarm benchmark skipped"
  fi
}

start_dpa_server() {
  local method="$1"
  local lora_path="$2"

  log "  Starting DPA server (method=${method}, port=${DPA_PORT})..."
  # --no-trigger: triggers are injected client-side (50% of Malicious prompts)
  DPA_PORT="${DPA_PORT}" ${PYTHON} core/dpa_server.py \
    --method "${method}" \
    --lora-path "${lora_path}" \
    --port "${DPA_PORT}" \
    --no-trigger &
  DPA_SERVER_PID=$!

  # Wait for server to be ready
  local retries=150
  while ! curl -sf "http://127.0.0.1:${DPA_PORT}/v1/models" >/dev/null 2>&1; do
    sleep 2
    retries=$((retries - 1))
    if [[ "${retries}" -le 0 ]]; then
      log "  [ERROR] DPA server failed to start"
      kill "${DPA_SERVER_PID}" 2>/dev/null || true
      exit 1
    fi
  done
  log "  DPA server ready (PID=${DPA_SERVER_PID})"
}

stop_dpa_server() {
  if [[ -n "${DPA_SERVER_PID:-}" ]]; then
    log "  Stopping DPA server (PID=${DPA_SERVER_PID})..."
    kill "${DPA_SERVER_PID}" 2>/dev/null || true
    wait "${DPA_SERVER_PID}" 2>/dev/null || true
    DPA_SERVER_PID=""
  fi
  # Kill any orphaned python3 process still holding the port
  local port_pid
  port_pid=$(ss -tlnp 2>/dev/null | grep ":${DPA_PORT}" | grep -oP 'pid=\K[0-9]+' || true)
  if [[ -n "${port_pid}" ]]; then
    log "  [WARN] Killing orphaned process ${port_pid} on port ${DPA_PORT}"
    kill -9 "${port_pid}" 2>/dev/null || true
    sleep 1
  fi
}

# ── Experiment A: Description attack ─────────────────────────────────────────
run_description_attack() {
  local exp_dir="${OUTPUT_BASE}/exp_description_attack"
  log "════════════════════════════════════════"
  log "Experiment A: Description Attack"
  log "  System:   experiments.system_description_attack"
  log "  Output:   ${exp_dir}"
  log "════════════════════════════════════════"

  run_data_and_merge "experiments.system_description_attack" "${exp_dir}"
  run_full_eval "${exp_dir}"
  log "Experiment A complete."
}

# ── Experiment B: Sleeper agent (one DPA method) ──────────────────────────────
run_sleeper_method() {
  local method="$1"
  local lora_path="${DPA_ARTIFACTS_DIR}/llama2-7b-chat/jailbreak/${method}"
  local exp_dir="${OUTPUT_BASE}/exp_sleeper_${method}"

  log "════════════════════════════════════════"
  log "Experiment B: Sleeper Agent (method=${method})"
  log "  LoRA path: ${lora_path}"
  log "  Output:    ${exp_dir}"
  log "════════════════════════════════════════"

  if [[ ! -f "${lora_path}/adapter_config.json" ]]; then
    log "  [ERROR] LoRA adapter not found: ${lora_path}"
    return 1
  fi

  DPA_SERVER_PID=""
  trap 'stop_dpa_server' EXIT

  start_dpa_server "${method}" "${lora_path}"

  export DPA_METHOD="${method}"
  export DPA_PORT="${DPA_PORT}"

  run_data_and_merge "experiments.system_sleeper_agent" "${exp_dir}" "${method}"
  run_full_eval "${exp_dir}"

  stop_dpa_server
  trap - EXIT
  log "Experiment B (${method}) complete."
}

# ── Final consolidated report ─────────────────────────────────────────────────
build_final_report() {
  log "Building final consolidated report..."
  local _report_tmp
  _report_tmp="$(mktemp /tmp/antap_report_XXXXXX.py)"
  cat > "${_report_tmp}" <<PYEOF
import json, pathlib
output_base = pathlib.Path("${ANTAP_DIR}/experiments_output")
rows = []
for exp_dir in sorted(output_base.iterdir()):
    if not exp_dir.is_dir():
        continue
    exp_name = exp_dir.name
    for dataset in ["MMLU_base", "AgentHarm_base"]:
        summary_path = exp_dir / dataset / "benchmark" / "summary.json"
        if not summary_path.exists():
            continue
        s = json.loads(summary_path.read_text())
        rows.append({
            "experiment": exp_name,
            "dataset": s.get("dataset", dataset),
            "antap_accuracy":    s.get("antap_accuracy", ""),
            "embedllm_accuracy": s.get("embedllm_accuracy", ""),
            "autogen_accuracy":  s.get("autogen_accuracy", ""),
            "antap_asr":         s.get("antap_asr", ""),
            "embedllm_asr":      s.get("embedllm_asr", ""),
            "autogen_asr":       s.get("autogen_asr", ""),
        })
if not rows:
    print("No results found.")
else:
    import pandas as pd
    df = pd.DataFrame(rows)
    out = pathlib.Path("${ANTAP_DIR}/experiments_output/final_report.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved to {out}")
PYEOF
  ${PYTHON} "${_report_tmp}"
  rm -f "${_report_tmp}"
}

# ── Main ─────────────────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_BASE}"
ensure_ollama
prewarm_models

case "${ONLY}" in
  description)
    run_description_attack
    ;;
  sleeper)
    for m in "${DPA_METHODS[@]}"; do
      run_sleeper_method "${m}"
    done
    ;;
  badnet|vpi|mtba|ctba)
    run_sleeper_method "${ONLY}"
    ;;
  "sleeper_method")
    run_sleeper_method "sleeper"
    ;;
  "")
    run_description_attack
    for m in "${DPA_METHODS[@]}"; do
      run_sleeper_method "${m}"
    done
    ;;
  *)
    echo "[ERROR] Unknown --only value: ${ONLY}"
    echo "  Valid: description | sleeper | badnet | sleeper | vpi | mtba | ctba"
    exit 1
    ;;
esac

build_final_report
log "All experiments complete. Results: ${OUTPUT_BASE}/final_report.csv"
