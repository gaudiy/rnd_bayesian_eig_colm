#!/bin/bash
# kill.sh - Stop all running experiment processes
# Usage: bash kill.sh

echo "Stopping experiment processes..."
echo ""

# Kill by PID file if exists
if [ -f .job_pid ]; then
    PID=$(cat .job_pid)
    echo "Killing main process (PID: $PID)..."
    kill $PID 2>/dev/null
    sleep 1
    kill -9 $PID 2>/dev/null
fi

# Kill all python processes running main.py
echo "Killing all main.py processes..."
pkill -9 -f "python.*main.py"

# Kill all python processes running run_multi_gpu
echo "Killing all run_multi_gpu processes..."
pkill -9 -f "python.*run_multi_gpu"

# Wait a moment
sleep 2

# Check what's left
REMAINING=$(ps aux | grep -E "python.*(main\.py|run_multi_gpu)" | grep -v grep | wc -l)

if [ $REMAINING -eq 0 ]; then
    echo ""
    echo "✓ All experiment processes stopped."
else
    echo ""
    echo "⚠ Warning: $REMAINING python processes still running"
    echo "Showing remaining processes:"
    ps aux | grep -E "python.*(main\.py|run_multi_gpu)" | grep -v grep
    echo ""
    echo "To force kill all python:"
    echo "  pkill -9 python"
fi

# Clean up
rm -f .job_pid

echo ""
echo "Check with:"
echo "  ps aux | grep python"
echo "  nvidia-smi"
echo ""