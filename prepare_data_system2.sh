#!/bin/bash

export system="experiments.system_scenario2"
export dir="scenario2"
export outdir="${dir}/results"

mkdir -p $dir 
mkdir -p $outdir
mkdir -p ${outdir}/test
mkdir -p ${outdir}/train

ollama pull qwen2.5-coder:3b; 
# ollama run qwen2-math:1.5b; 
ollama pull phi3:mini; 
ollama pull llama3.2:3b

for item in "code_expert" "medical_expert" "finance_expert" "legal_expert" "Malicious"; do
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