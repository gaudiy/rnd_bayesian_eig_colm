#!/bin/bash
# run_all_conditions.sh

BASE_CONFIG="config2.yaml"

conditions=(
    "false false notPrune_notUniform"
)

for condition in "${conditions[@]}"; do
    read -r USE_PRUNING UNIFORM_PRIOR SUFFIX <<< "$condition"
    
    echo "=================================================="
    echo "Running: use_pruning=$USE_PRUNING, uniform_prior=$UNIFORM_PRIOR"
    echo "=================================================="
    
    # Create a temporary config for this condition
    TEMP_CONFIG="config2_${SUFFIX}.yaml"
    cp "$BASE_CONFIG" "$TEMP_CONFIG"
    
    # Modify the config using sed
    sed -i "s/use_pruning:.*/use_pruning: $USE_PRUNING/" "$TEMP_CONFIG"
    sed -i "s/uniform_prior:.*/uniform_prior: $UNIFORM_PRIOR/" "$TEMP_CONFIG"
    
    # Create separate output directory for this condition
    OUTPUT_DIR="results2_${SUFFIX}"
    sed -i "s|output_dir:.*|output_dir: \"$OUTPUT_DIR\"|" "$TEMP_CONFIG"
    mkdir -p "$OUTPUT_DIR"
    
    # Run and wait for completion
    python run_gpus.py --config "$TEMP_CONFIG"
    
    # Wait for all workers to finish
    echo "Waiting for workers to complete..."
    while [ -n "$(ls -A task_queue/*.task 2>/dev/null)" ] || [ -n "$(ls -A task_queue/*.processing* 2>/dev/null)" ]; do
        sleep 10
    done
    
    echo "Condition $SUFFIX complete!"
    echo ""
done

echo "All conditions complete!"