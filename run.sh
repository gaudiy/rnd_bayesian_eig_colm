# #!/bin/bash
# # run3.sh - Wrapper for the config-driven multi-GPU system

# # rm -rf results/*

# # Default values
# START_IDX=""
# END_IDX=""

# rm -rf task_queue

# # Parse arguments to override config.yaml if needed
# while [[ $# -gt 0 ]]; do
#     case $1 in
#         --start)
#             START_IDX="--start $2"
#             shift 2
#             ;;
#         --end)
#             END_IDX="--end $2"
#             shift 2
#             ;;
#         *)
#             echo "Unknown option: $1"
#             echo "Usage: bash run3.sh [--start 0] [--end 100]"
#             exit 1
#             ;;
#     esac
# done

# echo "=================================================="
# echo "  MEDICAL AGENT EXPERIMENT (Config-Driven)"
# echo "=================================================="

# # Ensure output directories exist
# mkdir -p results
# mkdir -p task_queue

# # Run the python launcher
# # Added -u here as well for real-time launcher logs
# nohup python -u run_gpus.py --config config.yaml $START_IDX $END_IDX > launcher.log 2>&1 &

# PID=$!
# echo $PID > .launcher_pid

# echo ""
# echo "--------------------------------------------------"
# echo "Launcher PID: $PID"
# echo "Jobs dispatched to background."
# echo "Use 'bash monitor3.sh' to watch progress."
# echo "Logs are in results/gpu0.log, gpu1.log, gpu2.log"
# echo "--------------------------------------------------"