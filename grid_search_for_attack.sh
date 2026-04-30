set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

TRAIN_SOURCE="base/AgentHarm_base/results/train"
TEST_SOURCE="base/AgentHarm_base/results/test"

# Grid (edit if needed)
DIMS=(384)
EPOCHS=(60)
LRS=(0.001)
WDS=(0.0001)

for DIM in "${DIMS[@]}"; do
  for EP in "${EPOCHS[@]}"; do
    for LR in "${LRS[@]}"; do
      for WD in "${WDS[@]}"; do
        OUTDIR="attacks/output/agentharm_embedllm_tuned_dim${DIM}_ep${EP}_lr${LR}_wd${WD}"

        echo "=== Running ${OUTDIR} ==="
        python -m attacks.uat_router_attack \
          --train-source "$TRAIN_SOURCE" \
          --test-source "$TEST_SOURCE" \
          --output-dir "$OUTDIR" \
          --router-backends embedllm \
          --attack-mode unsafe \
          --embedding-model sentence-transformers/all-mpnet-base-v2 \
          --embedllm-dim "$DIM" \
          --embedllm-epochs "$EP" \
          --embedllm-lr "$LR" \
          --embedllm-weight-decay "$WD" \
          --router-train-limit 600 \
          --train-opt-limit 400 \
          --eval-limit 400 \
          --trigger-len 4 \
          --iterations 8 \
          --candidates-per-pos 80 \
          --subset-size 96 \
          --seeds 0 1 2

        python - <<PY
import pandas as pd

path = "${OUTDIR}/summary.csv"
df = pd.read_csv(path)

# 1) Per-run table
cols = [
    "backend", "seed", "trigger",
    "targeted_asr", "flip_rate",
    "clean_target_rate", "adv_target_rate"
]
print("\\nPer-run table for:", path)
print(df[cols].sort_values(["backend", "seed"]).round(3).to_string(index=False))

# 2) Backend-level summary
g = df.groupby("backend", as_index=False).agg(
    runs=("seed", "count"),
    asr_mean=("targeted_asr", "mean"),
    asr_std=("targeted_asr", "std"),
    flip_mean=("flip_rate", "mean"),
    clean_unsafe_mean=("clean_target_rate", "mean"),
    adv_unsafe_mean=("adv_target_rate", "mean"),
)
g["unsafe_rate_delta"] = g["adv_unsafe_mean"] - g["clean_unsafe_mean"]
print("\\nBackend summary:")
print(g.round(3).to_string(index=False))
PY

      done
    done
  done
done
