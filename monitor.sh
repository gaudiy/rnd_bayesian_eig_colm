#!/bin/bash
# monitor3.sh - Monitor 3-GPU experiment progress
# Usage: bash monitor3.sh

echo "=========================================="
echo "EXPERIMENT MONITORING (3 GPUs)"
echo "=========================================="
echo ""

# Check if processes are running
if [ -f .job_pid ]; then
    PID=$(cat .job_pid)
    if ps -p $PID > /dev/null; then
        echo "✓ Main process running (PID: $PID)"
    else
        echo "✗ Main process not running (PID: $PID)"
    fi
else
    echo "No .job_pid file found"
fi

echo ""

# Check GPU usage
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Usage:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU %s (%s): %s%% util, %s/%s MB\n", $1, $2, $3, $4, $5}'
    echo ""
fi

# Check log files
echo "Log files:"
if [ -f launcher.log ]; then
    LINES=$(wc -l < launcher.log)
    echo "  launcher.log: $LINES lines"
fi

if [ -f results/gpu0.log ]; then
    LINES=$(wc -l < results/gpu0.log)
    echo "  results/gpu0.log: $LINES lines"
fi

if [ -f results/gpu1.log ]; then
    LINES=$(wc -l < results/gpu1.log)
    echo "  results/gpu1.log: $LINES lines"
fi

if [ -f results/gpu2.log ]; then
    LINES=$(wc -l < results/gpu2.log)
    echo "  results/gpu2.log: $LINES lines"
fi

echo ""

# Check result files
echo "Result files:"
if [ -f results/results_1.json ]; then
    COUNT=$(python3 -c "import json; print(len(json.load(open('results/results_1.json'))))" 2>/dev/null || echo "?")
    echo "  results_1.json: $COUNT results"
fi

if [ -f results/results_2.json ]; then
    COUNT=$(python3 -c "import json; print(len(json.load(open('results/results_2.json'))))" 2>/dev/null || echo "?")
    echo "  results_2.json: $COUNT results"
fi

if [ -f results/results_3.json ]; then
    COUNT=$(python3 -c "import json; print(len(json.load(open('results/results_3.json'))))" 2>/dev/null || echo "?")
    echo "  results_3.json: $COUNT results"
fi

echo ""
echo "=========================================="
echo ""
echo "Monitor commands:"
echo "  GPU 0: tail -f results/gpu0.log"
echo "  GPU 1: tail -f results/gpu1.log"
echo "  GPU 2: tail -f results/gpu2.log"
echo ""
echo "Live refresh (updates every 5 seconds):"
echo "  watch -n 5 bash monitor3.sh"

# nvtop