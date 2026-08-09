#!/usr/bin/env python3
"""
Launcher: Prepares task queue based on config, launches workers on specified GPUs.
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
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    
    # 1. Load Configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        sys.exit(1)
        
    cfg = load_config(args.config)
    
    # Priority: Command line args > Config file
    start_idx = args.start if args.start is not None else cfg['experiment']['start_idx']
    end_idx = args.end if args.end is not None else cfg['experiment']['end_idx']
    
    queue_dir = cfg['system']['queue_dir']
    output_dir = cfg['data']['output_dir']
    gpu_ids = cfg['system']['gpu_ids'] # This now reads your [2] or [0, 1, 2] list

    # 2. Prepare Environment
    print(f"--- Initialization ---")
    print(f"Cleaning task queue directory: {queue_dir}...")
    if os.path.exists(queue_dir):
        shutil.rmtree(queue_dir)
    os.makedirs(queue_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. Create Task Files (The Queue)
    print(f"Generating tasks for patients {start_idx} to {end_idx}...")
    task_count = 0
    for i in range(start_idx, end_idx):
        Path(os.path.join(queue_dir, f"{i}.task")).touch()
        task_count += 1
    print(f"-> Successfully created {task_count} task files.")

    # 4. Launch Workers
    procs = []
    print(f"\n--- Launching {len(gpu_ids)} Workers ---")
    
    for i, gpu_id in enumerate(gpu_ids):
        # We use the index as worker_id and the value as the physical GPU ID
        worker_id = str(i)
        
        # Command to run the worker
        # -u ensures Python doesn't buffer logs, keeping the .log files real-time
        cmd = [
            "python", "-u", "main.py",
            "--config", args.config,
            "--worker_id", worker_id
        ]
        
        # ISOLATION: Set the environment variable for this specific child process
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Set up logging for this specific worker
        log_path = os.path.join(output_dir, f"gpu{worker_id}.log")
        log_file = open(log_path, "w")
        
        print(f"🚀 Worker {worker_id}: Assigned to GPU {gpu_id} | Log: {log_path}")
        
        # Start the process
        p = subprocess.Popen(
            cmd, 
            env=env, 
            stdout=log_file, 
            stderr=log_file, 
            bufsize=0 # Unbuffered at the subprocess level
        )
        procs.append(p)

    # 5. Save PIDs for management
    with open(".job_pids", "w") as f:
        for p in procs:
            f.write(f"{p.pid}\n")
            
    print("\n" + "=" * 50)
    print("All workers are now running in the background.")
    print(f"Target GPUs: {gpu_ids}")
    print(f"Check logs in {output_dir}/ to monitor progress.")
    print("=" * 50)

if __name__ == "__main__":
    main()