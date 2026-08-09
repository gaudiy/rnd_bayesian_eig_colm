# Qwen-30B Robustness

This folder contains the Qwen-30B robustness run.

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Machine: Nvidia H100
- Inference path: vLLM OpenAI-compatible server
- Patients: `n = 200`
- Setting: informative prior, no pruning
- Config: `config_qwen30b.yaml`
- Results: `results_qwen30b/results_merged.json`
- Figures/tables: `figures_qwen30b/`

Install:

```bash 
pip install requirements.txt
```

Start the vLLM server:

```sh
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8234 --max-model-len 32768 --gpu-memory-utilization 0.90 --enforce-eager
```

Run the experiment:

```sh
python run_gpus_vllm.py --config config_qwen30b.yaml
```

Generate tables and figures:

```sh
python merge_results.py results_qwen30b/results_worker_*.json -o results_qwen30b/results_merged.json
python analyze.py results_qwen30b/results_merged.json --output-dir figures_qwen30b --condition notprune_notuniform
```
