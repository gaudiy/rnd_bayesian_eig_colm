# Qwen-4B Robustness

This folder contains the Qwen-4B robustness results.

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Patients: `n = 200`
- Setting: informative prior, no pruning
- Config note: use the main experiment code path with `config_qwen4b.yaml`
- Results: `results_notPrune_notUniform/results_merged.json`
- Figures/tables: `figures_notPrune_notUniform/`

This run differs from the main MedGemma-4B experiment only in the model
configuration:

```yaml
model_repo: Qwen/Qwen3-4B-Instruct-2507
model_name: qwen-3-4b
```
