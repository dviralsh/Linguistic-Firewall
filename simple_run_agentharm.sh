#!/bin/bash

set -euo pipefail

system="$1"
dir="$2"
outdir="${dir}/results"

mkdir -p "$dir"
mkdir -p "$outdir"
mkdir -p "${outdir}/test"
mkdir -p "${outdir}/train"

for item in Strategist Math_Expert Coding_Expert Archivist; do
  echo "Processing $item (AgentHarm test)..."
  python -u create_test_data_agentharm.py --system "$system" --output "${outdir}/test/${item}.csv" --agent "$item"

  echo "Processing $item (AgentHarm train)..."
  python -u create_test_data_agentharm.py --system "$system" --output "${outdir}/train/${item}.csv" --agent "$item" --train
done