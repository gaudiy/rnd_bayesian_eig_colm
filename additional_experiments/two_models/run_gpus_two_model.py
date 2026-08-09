#!/usr/bin/env python3
"""
Launcher for two-model vLLM workers.

Requires two vLLM servers to be running:
  - Oracle (27b): vllm serve google/medgemma-27b-text-it --port 8234 --enforce-eager
  - Agent  (4b):  vllm serve google/medgemma-4b-it --port 8235 --enforce-eager

Usage:
    python run_gpus_two_model.py --config config_two_model.yaml
    python run_gpus_two_model.py --config config_two_model.yaml --start 0 --end 100
"""
import os
import sys
import yaml
import shutil
import argparse
import subprocess
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_two_model.yaml")
    parser.add_argument("--start",  type=int, default=None)
    parser.add_argument("--end",    type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: config file '{args.config}' not found.")
        sys.exit(1)

    cfg = load_config(args.config)

    start_idx  = args.start if args.start is not None else cfg['experiment']['start_idx']
    end_idx    = args.end   if args.end   is not None else cfg['experiment']['end_idx']
    queue_dir  = cfg['system']['queue_dir']
    output_dir = cfg['data']['output_dir']
    n_workers  = cfg['system'].get('n_workers', 4)

    # 1. Prepare environment
    print("--- Initialization ---")
    print(f"Cleaning task queue: {queue_dir}")
    if os.path.exists(queue_dir):
        shutil.rmtree(queue_dir)
    os.makedirs(queue_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Create task files
    task_count = end_idx - start_idx
    for i in range(start_idx, end_idx):
        Path(os.path.join(queue_dir, f"{i}.task")).touch()
    print(f"Created {task_count} task files (patients {start_idx}–{end_idx-1}).")

    # 3. Launch workers
    procs = []
    print(f"\n--- Launching {n_workers} Workers ---")
    for worker_id in range(n_workers):
        cmd = [
            sys.executable, "-u", "main_two_model.py",
            "--config",    args.config,
            "--worker_id", str(worker_id),
        ]
        log_path = os.path.join(output_dir, f"worker{worker_id}.log")
        log_file = open(log_path, "w")

        p = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, bufsize=0)
        procs.append(p)
        print(f"  Worker {worker_id}: PID {p.pid} | Log: {log_path}")

    # 4. Save PIDs
    with open(".job_pids_two_model", "w") as f:
        for p in procs:
            f.write(f"{p.pid}\n")

    print("\n" + "=" * 60)
    print(f"All {n_workers} workers launched.")
    print(f"Oracle : {cfg['oracle_model_name']} @ {cfg['vllm_oracle_base_url']}")
    print(f"Agent  : {cfg['agent_model_name']} @ {cfg['vllm_agent_base_url']}")
    print(f"Monitor progress:")
    print(f"  tail -f {output_dir}/worker*.log")
    print(f"Kill all workers:")
    print(f"  cat .job_pids_two_model | xargs kill")
    print("=" * 60)


if __name__ == "__main__":
    main()
