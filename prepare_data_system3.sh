#!/bin/bash

export system="experiments.system_scenario3"
export dir="scenario3"
export outdir="${dir}/results"

mkdir -p $dir 
mkdir -p $outdir
mkdir -p ${outdir}/test
mkdir -p ${outdir}/train
ollama pull deepseek-coder:6.7b
ollama pull phi3.5:3.8b
ollama pull cniongolo/biomistral
ollama pull mistral:7b
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
ollama pull zephyr:7b
ollama pull gemma:7b
ollama pull openchat:7b

for item in "Medicine" "Law" "Economics" "History" "Philosophy" "Chemistry" "Psychology" "Malicious"; do
    echo "Processing $item..."
    
    python -u create_test_data_parallelized.py \
        --system "${system}" \
        --output "${outdir}/test/${item}.csv" \
        --agent "$item"
    
    python -u create_test_data_parallelized.py \
        --system "${system}" \
        --output "${outdir}/train/${item}.csv" \
        --agent "$item" \
        --train
done