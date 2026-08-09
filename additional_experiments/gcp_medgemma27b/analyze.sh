#!/bin/bash

# Process all 4 experimental conditions
# Each condition merges worker JSONs and generates figures

echo "========================================"
echo "Processing all experimental conditions"
echo "========================================"

# 1. Not Prune + Not Uniform (Informative Prior, No Pruning)
echo ""
echo "=== Processing: notPrune_notUniform (Inf-NP) ==="
mkdir -p figures_notPrune_notUniform
python merge_results.py \
    results_notPrune_notUniform/results_worker_0.json \
    results_notPrune_notUniform/results_worker_1.json \
    -o results_notPrune_notUniform/results_merged.json
python analyze.py \
    results_notPrune_notUniform/results_merged.json \
    --output-dir ./figures_notPrune_notUniform \
    --condition notprune_notuniform

# 2. Not Prune + Uniform (Uniform Prior, No Pruning)
echo ""
echo "=== Processing: notPrune_uniform (U-NP) ==="
mkdir -p figures_notPrune_uniform
python merge_results.py \
    results_notPrune_uniform/results_worker_0.json \
    results_notPrune_uniform/results_worker_1.json \
    -o results_notPrune_uniform/results_merged.json
python analyze.py \
    results_notPrune_uniform/results_merged.json \
    --output-dir ./figures_notPrune_uniform \
    --condition notprune_uniform

# 3. Prune + Not Uniform (Informative Prior, Pruning)
echo ""
echo "=== Processing: prune_notUniform (Inf-P) ==="
mkdir -p figures_prune_notUniform
python merge_results.py \
    results_prune_notUniform/results_worker_0.json \
    results_prune_notUniform/results_worker_1.json \
    -o results_prune_notUniform/results_merged.json
python analyze.py \
    results_prune_notUniform/results_merged.json \
    --output-dir ./figures_prune_notUniform \
    --condition prune_notuniform

# 4. Prune + Uniform (Uniform Prior, Pruning)
echo ""
echo "=== Processing: prune_uniform (U-P) ==="
mkdir -p figures_prune_uniform
python merge_results.py \
    results_prune_uniform/results_worker_0.json \
    results_prune_uniform/results_worker_1.json \
    -o results_prune_uniform/results_merged.json
python analyze.py \
    results_prune_uniform/results_merged.json \
    --output-dir ./figures_prune_uniform \
    --condition prune_uniform

echo ""
echo "========================================"
echo "All conditions processed!"
echo "========================================"
echo ""
echo "Output files:"
echo "  figures_notPrune_notUniform/"
echo "    - all_notprune_notuniform.png"
echo "    - pareto_notprune_notuniform.png"
echo "    - correct_vs_incorrect_notprune_notuniform.png"
echo "    - metrics_summary_notprune_notuniform.csv"
echo ""
echo "  figures_notPrune_uniform/"
echo "    - all_notprune_uniform.png"
echo "    - pareto_notprune_uniform.png"
echo "    - correct_vs_incorrect_notprune_uniform.png"
echo "    - metrics_summary_notprune_uniform.csv"
echo ""
echo "  figures_prune_notUniform/"
echo "    - all_prune_notuniform.png"
echo "    - pareto_prune_notuniform.png"
echo "    - correct_vs_incorrect_prune_notuniform.png"
echo "    - metrics_summary_prune_notuniform.csv"
echo ""
echo "  figures_prune_uniform/"
echo "    - all_prune_uniform.png"
echo "    - pareto_prune_uniform.png"
echo "    - correct_vs_incorrect_prune_uniform.png"
echo "    - metrics_summary_prune_uniform.csv"