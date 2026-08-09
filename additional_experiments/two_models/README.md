# Split Questioner--Oracle Models

This folder contains the split Questioner--Oracle/likelihood robustness run.

- Questioner/agent model: `google/medgemma-4b-it`
- Oracle/likelihood model: `google/medgemma-27b-text-it`
- Inference path: two vLLM OpenAI-compatible servers
- Patients: `n = 100`
- Setting: informative prior, no pruning
- Config: `config_two_model.yaml`
- Results: `results_two_model/results_merged.json`
- Figures/tables: `figures_two_model/`

In this setup, the 4B model is the evaluated Questioner/agent. The 27B model
provides the yes/no Oracle response for the selected query and the likelihood
estimates used by score-based methods.

Install:

```bash
pip install -r requirements_27b.txt
```

Start the vLLM servers:

```sh
vllm serve google/medgemma-27b-text-it --port 8234 --max-model-len 32768 --gpu-memory-utilization 0.78 --enforce-eager
vllm serve google/medgemma-4b-it --port 8235 --max-model-len 32768 --gpu-memory-utilization 0.15 --enforce-eager
```

Run the experiment:

```sh
python run_gpus_two_model.py --config config_two_model.yaml
```

Generate tables and figures:

```sh
python merge_results.py results_two_model/results_worker_*.json -o results_two_model/results_merged.json
python analyze.py results_two_model/results_merged.json --output-dir figures_two_model --condition notprune_notuniform
```
