# MedGemma-27B Robustness

This folder contains the MedGemma-27B single-model robustness run.

- Model: `google/medgemma-27b-text-it`
- Machine: Nvidia H100
- Inference path: vLLM OpenAI-compatible server
- Patients: `n = 100`
- Setting: informative prior, no pruning
- Config: `config_27b.yaml`
- Results: `results_27b/results_merged.json`
- Figures/tables: `figures_27b/`

Install:

```bash 
pip install requirements.txt
```


Start the vLLM server:

```sh
vllm serve google/medgemma-27b-text-it --port 8234 --max-model-len 32768 --gpu-memory-utilization 0.90 --enforce-eager
```

Run the experiment:

```sh
python run_gpus_vllm.py --config config_27b.yaml
```

Generate tables and figures:

```sh
python merge_results.py results_27b/results_worker_*.json -o results_27b/results_merged.json
python analyze.py results_27b/results_merged.json --output-dir figures_27b --condition notprune_notuniform
```
