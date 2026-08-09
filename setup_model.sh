#!/bin/bash
set -e

MODEL_REPO=$(python3 -c "import yaml; print(yaml.safe_load(open('config2.yaml'))['model_repo'])")
MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('config2.yaml'))['model_name'])")

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$MODEL_REPO',
    local_dir='$MODEL_NAME'
)
"
