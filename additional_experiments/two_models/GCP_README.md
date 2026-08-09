## get results

```sh
source .venv/bin/activate
```

```sh
nohup vllm serve google/medgemma-27b-text-it --port 8234 --enforce-eager --gpu-memory-utilization 0.78 > vllm_27b.log 2>&1 &
tail -f vllm_27b.log
```

Check:

```sh
curl -s http://localhost:8234/v1/models
```

```sh
rm -rf results_27b
nohup python run_gpus_vllm.py --config config_27b.yaml > run_27b.log 2>&1 &
echo "Launcher PID: $!"
```

# to kill
```sh
cat .job_pids_27b | xargs kill
```

## analyze results

```sh
python merge_results.py results_27b/results_worker_*.json -o results_27b/results_merged.json
mkdir -p figures_27b
python analyze.py results_27b/results_merged.json --output-dir ./figures_27b --condition notprune_notuniform
```

## Two models

```sh
# Start 4b agent server (27b oracle already on 8234)
nohup vllm serve google/medgemma-4b-it --port 8235 --enforce-eager --gpu-memory-utilization 0.15 > vllm_4b.log 2>&1 &
tail -f vllm_4b.log


nohup python run_gpus_two_model.py --config config_two_model.yaml > run_two_model.log 2>&1 &

# to analyze 
python merge_results.py results_two_model/results_worker_*.json -o results_two_model/results_merged.json
mkdir -p figures_two_model
python analyze.py results_two_model/results_merged.json --output-dir ./figures_two_model --condition notprune_notuniform
```


## Architecture

Agent (4b) does:

Computing the initial prior
Generating candidate questions
LLM-SELECT: question selection
LLM-SELECT: update_belief — asks the 4b "does this patient have disease X?" for all diseases → renormalizes as new belief
Oracle (27b) does:

get_likelihood_matrix_for_test — all likelihood computations (used by EIG/truncated/adaptive methods for both scoring and Bayesian updates)
YES/NO answer for the true diagnosis (ground truth response)