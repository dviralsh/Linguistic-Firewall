#!/bin/bash

export system="$1"
export dir="$2"
export outdir="${dir}/results"

mkdir -p $dir 
mkdir -p $outdir
mkdir -p ${outdir}/test
mkdir -p ${outdir}/train
python -u create_test_data_parallelized.py --system ${system} --output "${outdir}/test/Malicious.csv" --agent "Malicious"
python -u create_test_data_parallelized.py --system ${system} --output "${outdir}/train/Malicious.csv" --agent "Malicious" --train

python -u create_test_data_parallelized.py --system ${system} --output "${outdir}/test/Strategist.csv" --agent "Strategist"
python -u create_test_data_parallelized.py --system ${system} --output "${outdir}/train/Strategist.csv" --agent "Strategist" --train

python -u create_test_data_parallelized.py --system ${system} --output "${outdir}/test/math.csv" --agent "Math_Expert"
python -u create_test_data_parallelized.py --system ${system} --output "${outdir}/train/math.csv" --agent "Math_Expert" --train

python -u create_test_data_parallelized.py --system ${system} --output "${outdir}/test/code.csv" --agent "Coding_Expert"
python -u create_test_data_parallelized.py --system ${system} --output "${outdir}/train/code.csv" --agent "Coding_Expert" --train

python -u create_test_data_parallelized.py --system ${system} --output "${outdir}/test/archivist.csv" --agent "Archivist"
python -u create_test_data_parallelized.py --system ${system} --output "${outdir}/train/archivist.csv" --agent "Archivist" --train
