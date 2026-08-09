# Adaptive Bayesian Active Querying with LLMs for Efficient Information Gathering

This repository contains the experimental code for the DAIH workshop (COLM 2026) submission: *Adaptive Bayesian Active Querying with LLMs for Efficient Information Gathering*.

## Requirements

- Python ≥ 3.10
- GPU recommended (experiments will be slow on CPU and may exceed memory limits)

## Installation

We recommend creating a virtual enviroment:

```sh
uv venv --python python3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```


## Animal Experiment

The files and scripts for the animal experiment are in the folder of `experiment1`. It also contains a README instruction file for reproducible purposes. 

## Model Setup

Download your chosen language model from [Hugging Face](https://huggingface.co/) by configuring `config.yaml`:

```yaml
model_repo: google/medgemma-4b-it   # HuggingFace model ID
model_name: medgemma-4b-it          # Local name for the model
```

The `model_repo` field should contain the path portion of the Hugging Face URL. For example, for `https://huggingface.co/Qwen/Qwen2.5-3B-Instruct`, use:

```yaml
model_repo: Qwen/Qwen2.5-3B-Instruct
```

Then run the setup script:

```sh
bash setup_model.sh
```

**Note:** Download time and disk usage vary by model size (expect several minutes and multiple GB of disk space).

## Data

This project uses the [MIMIC-IV-ED dataset](https://physionet.org/content/mimic-iv-ed/2.2/), which requires credentialed access through PhysioNet.

For testing without MIMIC access, we provide `synthetic_data.csv`. To use it, update `config.yaml`:

```yaml
data_path: "synthetic_data.csv"   # Change from "mimic_data.csv"
```

### Process Mimic Data

After getting the credentials for the [MIMIC-IV-ED dataset](https://physionet.org/content/mimic-iv-ed/2.2/), download the files in the terminal:

```sh
wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-iv-ed/2.2/
```

Then unzip the `gz` files:

```sh
gunzip physionet.org/files/mimic-iv-ed/2.2/ed/*.gz
```

Then, just run

```sh
python3 get_random_mimic_data.py
```

The sampled data for experiments will be saved as `mimic_data.csv` at the root directory. 

You can see and change parameters for this step in `config.yaml`:

```yaml 
data:
  data_path: "mimic_data.csv" # where mimic sampled data is saved
  disease_mapping_path: 'disease_mapping.json' # disease mapping created by claude
  disease_list_path: "top_n_target_labels.txt"
  output_dir: "results"

mimic:
  n_diseases_num: 500 # we only want the top 500 diseases
  n_top_diseases_num: 100
  n_patients_num: 1000 # how many patients do we want for the experiments
  random_select_seed: 77
  data_source_folder: 'physionet.org/files/mimic-iv-ed/2.2/ed'
```

## Running Experiments

### Configuration

Edit `config.yaml` to set your preferences:

| Setting | Description |
|---------|-------------|
| `gpu_ids` | List of GPU IDs to use, e.g., `[0, 2]`. Set to `[]` for CPU-only. |
| `conditions_to_run` | Experimental conditions to execute |
| `data_path` | Path to the dataset file |

### Execution

Run all experiments:

```sh
nohup bash run_all_conditions.sh > run.log 2>&1 &
```

The file `utils.py` contains helper functions for the `main.py` which runs the experiment after `run.sh` is executed. 

Results (logs and JSON outputs) are saved to the `results/` directory. For example, in the `results_notPrune_notUniform/` folder. 

### Monitoring and Control

Monitor running experiments:

```sh
bash monitor.sh
```

Terminate all jobs:

```sh
bash kill.sh
```

## Analyzing Results

After experiments complete, generate figures from the paper:

```sh
bash analyze.sh
```

Figures and analysis outputs are saved to the `figures/` directory. For example, in the `figures_notPrune_notUniform/` folder. 


## Additional Experiments

Results and scripts to run additional experiments are in the folder of `additional_experiments`. 

