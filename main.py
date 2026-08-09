#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import glob
import yaml
import torch
import pandas as pd
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict

# Import your utils
from utils import (
    Patient,
    MedicalAgent,
    Oracle,
    run_eig_session,
    run_llm_select_session,
    run_adaptive_eig_session,
    run_truncated_eig_session,
)

from datetime import datetime

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_gpu_info():
    """Retrieves current GPU stats for the diagnostic header."""
    try:
        # Queries index, name, temperature, and utilization
        cmd = "nvidia-smi --query-gpu=index,name,temp.gpu,utilization.gpu --format=csv,noheader,nounits"
        return subprocess.check_output(cmd.split()).decode('utf-8').strip()
    except:
        return "N/A"

def get_k_by_mass(
    current_probs: Dict[str, float],
    threshold: float = 0.95,
) -> int:
    cum = 0.0
    k = 0

    for _, p in sorted(current_probs.items(), key=lambda x: x[1], reverse=True):
        cum += p
        k += 1
        if cum >= threshold:
            break

    return k

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Medical Diagnosis Experiment - Worker Process")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--worker_id", type=str, required=True, help="ID of this worker")
    args = parser.parse_args()

    # Load Config
    cfg = load_config(args.config)
    queue_dir = cfg['system']['queue_dir']
    worker_id = args.worker_id
    output_dir = cfg['data']['output_dir']
    
    exp_cfg = cfg['experiment']
    data_cfg = cfg['data']
    model_cfg = cfg['model']

    # --- GPU DIAGNOSTIC HEADER ---
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
    print("=" * 80)
    print(f"[{ts()}] MEDICAL DIAGNOSIS EXPERIMENT - WORKER {worker_id}")
    print(f"[{ts()}] CUDA_VISIBLE_DEVICES: {visible_gpus}")
    print(f"[{ts()}] GPU STATS (Idx, Name, Temp, Util): {get_gpu_info()}")
    print("=" * 80)

    # Load Data (Exact prints from your main copy.py)
    print(f"\nLoading data from {data_cfg['data_path']}...")
    df = pd.read_csv(data_cfg['data_path'])
    
    with open(data_cfg['disease_list_path'], "r", encoding="utf-8") as f:
        disease_list = [line.strip() for line in f if line.strip()]

    print(f"Total patients in dataset: {len(df)}")
    print(f"Diseases: {len(disease_list)}")
    print(f"Max questions: {exp_cfg['max_questions']}")
    print(f"Confidence threshold: {exp_cfg['confidence_threshold']}")
    print(f"Question cost: ${exp_cfg['question_cost']}")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("torch.cuda.device_count():", torch.cuda.device_count())
        print("current device:", torch.cuda.current_device())
        print("device name:", torch.cuda.get_device_name(0))

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # Load Model
    print("\n" + "=" * 80)
    print(f"Loading model: {cfg['model_name']}")
    print("=" * 80)
    
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg['model_name'],
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    print("\nInitializing medical agent...")
    agent = MedicalAgent(model, tokenizer, disease_list)
    
    # Results container
    results = []
    output_file = os.path.join(output_dir, f"results_worker_{worker_id}.json")
    
    print("\n" + "=" * 80)
    print(f"WORKER {worker_id} READY - MONITORING {queue_dir}")
    print("=" * 80)
    
    # --- The Queue Loop ---
    while True:
        task_files = sorted(glob.glob(os.path.join(queue_dir, "*.task")))
        if not task_files:
            print(f"Worker {worker_id}: No tasks left. Exiting.")
            break
            
        found_task = False
        patient_idx = -1
        processing_name = ""
        
        for task_file in task_files:
            try:
                base_name = os.path.basename(task_file) 
                idx_str = base_name.split('.')[0]
                processing_name = os.path.join(queue_dir, f"{idx_str}.processing_by_{worker_id}")
                os.rename(task_file, processing_name)
                patient_idx = int(idx_str)
                found_task = True
                break
            except OSError:
                continue 
        
        if not found_task:
            time.sleep(0.1)
            continue
            
        # --- Original Patient Logic starts here ---
        try:
            row = df.iloc[patient_idx]
            patient = Patient.from_row(row)
            true_diagnosis = patient.diagnosis
            
            print('\n' + '=' * 80)
            print(f'PATIENT {patient_idx} (Worker {worker_id}): {row.get("subject_id_x", "Unknown")}')
            print('=' * 80)
            
            patient_info = patient.format_for_prior()
            print(f"\nPatient Info:\n{patient_info}")
            print(f"\nTrue Diagnosis: {true_diagnosis}")
            
            oracle = Oracle(model, tokenizer, true_diagnosis)
            
            # pruning
            use_pruning = exp_cfg.get('use_pruning', False)
            print(f"\nWe are {'' if use_pruning else 'not'} using Pruning!")
            # Initial Prior
            use_uniform = exp_cfg.get('uniform_prior', True)
            print(f"\nComputing initial prior ({'uniform' if use_uniform else 'informed'})...")
            initial_prior = agent.compute_initial_prior(patient_info, uniform=use_uniform)
            top_3 = sorted(initial_prior.items(), key=lambda x: x[1], reverse=True)[:3]
            print("Initial prior (top 3):")
            for disease, prob in top_3:
                print(f"  {disease}: {prob:.4f}")
            
            top_to_show_when_contrastive = get_k_by_mass(initial_prior, exp_cfg.get('top_to_show_when_contrastive_perc', 0.95))

            # Mapping from condition name to function
            CONDITION_FUNCTIONS = {
                'EIG': run_eig_session,
                'PD': run_eig_session,
                'EIG-COST-AWARE': run_eig_session,
                'LLM-SELECT': run_llm_select_session,
                'TRUNCATED-EIG-COST-AWARE': run_truncated_eig_session,
                'ADAPTIVE-EIG': run_adaptive_eig_session,
                'ADAPTIVE-EIG-COST-AWARE': run_adaptive_eig_session,
                # NEW ONES
                'PD-COST-AWARE': run_eig_session,
                'PD-EIG-COST-AWARE': run_eig_session,
                'TRUNCATED-PD-COST-AWARE': run_truncated_eig_session,
                'TRUNCATED-PD-EIG-COST-AWARE': run_truncated_eig_session,
                'ADAPTIVE-PD': run_adaptive_eig_session,
                'ADAPTIVE-PD-COST-AWARE': run_adaptive_eig_session,
                'ADAPTIVE-PD-EIG': run_adaptive_eig_session,
                'ADAPTIVE-PD-EIG-COST-AWARE': run_adaptive_eig_session,
            }

            CONDITION_OVERRIDES = {
                'EIG': {'cost_aware': False, 'scoring_method': 'eig'},
                'PD': {'cost_aware': False, 'scoring_method': 'pd'},
                'EIG-COST-AWARE': {'cost_aware': True, 'scoring_method': 'eig'},
                'LLM-SELECT': {},
                'TRUNCATED-EIG-COST-AWARE': {'scoring_method': 'eig', 'cost_aware': True,},
                'ADAPTIVE-EIG': {'cost_aware': False, 'scoring_method': 'eig'},
                'ADAPTIVE-EIG-COST-AWARE': {'cost_aware': True, 'scoring_method': 'eig'},
                'PD-COST-AWARE': {'cost_aware': True, 'scoring_method': 'pd'},
                'PD-EIG-COST-AWARE': {'cost_aware': True, 'scoring_method': 'pd-eig'},
                'TRUNCATED-PD-COST-AWARE': {'scoring_method': 'pd', 'cost_aware': True},
                'TRUNCATED-PD-EIG-COST-AWARE': {'scoring_method': 'pd-eig', 'cost_aware': True},
                'ADAPTIVE-PD': {'cost_aware': False, 'scoring_method': 'pd'},
                'ADAPTIVE-PD-COST-AWARE': {'cost_aware': True, 'scoring_method': 'pd'},
                'ADAPTIVE-PD-EIG': {'cost_aware': False, 'scoring_method': 'pd-eig'},
                'ADAPTIVE-PD-EIG-COST-AWARE': {'cost_aware': True, 'scoring_method': 'pd-eig'},
            }

            # Common kwargs for all conditions
            common_kwargs = {
                'agent': agent,
                'oracle': oracle,
                'initial_prior': initial_prior.copy(),
                'diagnoses': disease_list,
                'true_diagnosis': true_diagnosis,
                'patient_info': patient_info,
                'max_q': exp_cfg['max_questions'],
                'conf_threshold': exp_cfg['confidence_threshold'],
                'question_cost': exp_cfg['question_cost'],
                'question_num': exp_cfg['question_num'],
                'test_num': exp_cfg['test_num'],
                'top_k': int(exp_cfg.get('pd_top_perc', 0.2)*len(disease_list)), # for pd calculation
                'top_to_show_when_contrastive': top_to_show_when_contrastive,
                'top_to_show_when_contrastive_perc': exp_cfg.get('top_to_show_when_contrastive_perc', 0.95),
                'stagnation_threshold': exp_cfg.get('stagnation_threshold', 0.15),
                'stagnation_window': exp_cfg.get('stagnation_window', 2),
                'concentration_threshold': exp_cfg.get('concentration_threshold', 0.95),
                'concentration_top_perc': exp_cfg.get('concentration_top_perc', 0.2),
                'concentration_k': int(exp_cfg.get('concentration_top_perc', 0.2) * len(disease_list)),
                'use_pruning': use_pruning
            }

            # Get conditions to run from config
            conditions_to_run = exp_cfg.get('conditions_to_run', list(CONDITION_FUNCTIONS.keys()))

            # Run each condition
            for i, condition_name in enumerate(conditions_to_run, 1):
                if condition_name not in CONDITION_FUNCTIONS:
                    print(f"WARNING: Unknown condition '{condition_name}', skipping.")
                    continue
                    
                print('\n' + '-' * 80)
                print(f'CONDITION {i}: {condition_name}')
                print('-' * 80)
                
                t_start = time.time()
                
                # Merge common kwargs with condition-specific overrides
                kwargs = {**common_kwargs, **CONDITION_OVERRIDES.get(condition_name, {})}
                # Use a fresh copy of initial_prior for each condition
                kwargs['initial_prior'] = initial_prior.copy()
                
                result = CONDITION_FUNCTIONS[condition_name](**kwargs)
                
                t_elapsed = time.time() - t_start
                
                results.append({
                    'patient_id': patient.patient_id,
                    'global_idx': patient_idx,
                    'condition': condition_name,
                    'correct': result.success,
                    'num_questions': result.questions_used,
                    'final_guess': result.final_guess,
                    'final_confidence': result.final_confidence,
                    'final_cost': result.final_cost,
                    'converged': result.converged,
                    'true_diagnosis': true_diagnosis,
                    'duration_seconds': t_elapsed
                })
                
                status = '✓' if result.success else '✗'
                print(f"\nResult: {status} ({result.questions_used} qs, cost=${result.final_cost}, {t_elapsed:.1f}s)")

            # Finalize Patient
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            os.rename(processing_name, os.path.join(queue_dir, f"{patient_idx}.done"))
            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Worker {worker_id}: ERROR on Patient {patient_idx}: {e}")
            import traceback
            traceback.print_exc()
            if os.path.exists(processing_name):
                os.rename(processing_name, os.path.join(queue_dir, f"{patient_idx}.error"))

    print(f"\nWorker {worker_id}: All tasks complete.")

if __name__ == "__main__":
    main()