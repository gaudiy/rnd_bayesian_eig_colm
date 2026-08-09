# Benchmark Scripts

This directory contains benchmark scripts for comparing question selection strategies in Bayesian identification tasks.

## Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install numpy requests matplotlib
```

### 3. LLM Server Setup

These benchmarks require a llama.cpp server with an OpenAI-compatible API.

#### Model Used

We used **Qwen3-VL-8B-Instruct** quantized to Q8_0:
- Model: `Qwen3VL-8B-Instruct-Q8_0.gguf`
- Source: [Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)

Download the model:
```bash
curl -L https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3VL-8B-Instruct-Q8_0.gguf \
  -o models/Qwen3VL-8B-Instruct-Q8_0.gguf
```

#### Building llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CUDA support (for GPU acceleration)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

#### Server Configuration

Start the server with these settings:

```bash
./build/bin/llama-server \
  -m models/Qwen3VL-8B-Instruct-Q8_0.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --seed 42 \
  --ctx-size 131072 \
  --parallel 32 \
  --gpu-layers 9999 \
  --flash-attn on \
  --log-disable
```

| Option | Value | Description |
|--------|-------|-------------|
| `--host` | `127.0.0.1` | Bind to localhost |
| `--port` | `8080` | Server port |
| `--seed` | `42` | Fixed seed for reproducibility |
| `--ctx-size` | `131072` | Context window (128K tokens) |
| `--parallel` | `32` | Parallel request slots |
| `--gpu-layers` | `9999` | Offload all layers to GPU |
| `--flash-attn` | `on` | Enable flash attention |

#### Verify Server

```bash
# Health check
curl http://localhost:8080/health

# List models
curl http://localhost:8080/v1/models
```

## Scripts

### replay_comparison_clean.py

Compares multiple question selection strategies using a fair replay mechanism:

- **EIG (baseline)**: Pure Expected Information Gain
- **EIG-PD-Blend**: EIG by default, switches to EIG + Pairwise Discrimination when stuck
- **PD**: EIG by default, switches to 100% PD when stuck

The replay mechanism ensures fair comparison by having adaptive strategies replay EIG's questions until they diverge.

**Usage:**

```bash
python replay_comparison.py \
    --sessions 100 \
    --data data/animals_100.txt \
    --domain animal \
    --parallel-sessions 32 \
    --max-parallel-llm 32 \
    --cache cache/animals_cache.jsonl \
    --output results.json
```

**Arguments:**
- `--sessions`: Number of sessions to run (default: 10)
- `--data`: Path to targets file (one target per line)
- `--domain`: Domain type: "animal", "disease", or other (default: "animal")
- `--max-questions`: Maximum questions per session (default: 20)
- `--threshold`: Confidence threshold for stopping (default: 0.9)
- `--parallel-sessions`: Parallel session count (default: 1)
- `--max-parallel-llm`: Max concurrent LLM requests (default: 8)
- `--cache`: Path for world beliefs disk cache (optional)
- `--output`: Output JSON file path (optional)
- `--llamacpp-url`: LLM server URL (default: http://localhost:8080)
- `--no-prune`: Disable hypothesis pruning
- `--exclusive-only`: Only run exclusive adaptive strategies
- `--blend-only`: Only run blend strategies
- `-v, --verbose`: Enable verbose output

### llm_select_vs_eig.py

Compares two question selection approaches:

1. **LLM Select**: For each candidate question, asks "Is THIS the BEST question?" and uses P(YES) as score
2. **EIG**: Mathematical Expected Information Gain scoring

**Usage:**

```bash
python llm_select_vs_eig.py \
    --sessions 100 \
    --data data/diseases_100.txt \
    --targets 100 \
    --domain disease \
    --parallel-sessions 20 \
    --max-parallel-llm 32 \
    --no-prune \
    --cache cache/llm_select_cache.jsonl \
    --output results.json
```

**Arguments:**
- `--sessions`: Number of sessions (default: 10)
- `--targets`: Number of targets to load (default: 50)
- `--domain`: Domain type (default: "disease")
- `--data`: Path to targets file (auto-generated if not specified)
- `--confidence`: Confidence threshold (default: 0.9)
- `--max-questions`: Max questions per session (default: 20)
- `--parallel-sessions`: Parallel session count (default: 4)
- `--max-parallel-llm`: Max concurrent LLM requests (default: 8)
- `--cache`: Path for world beliefs disk cache
- `--output`: Output JSON file path
- `--llamacpp-url`: LLM server URL (default: http://localhost:8080)
- `--no-prune`: Disable hypothesis pruning

## Data Files

Create target files in `data/` directory with one target per line:

```
# data/animals_100.txt
dog
cat
elephant
...
```

```
# data/diseases_100.txt
diabetes
hypertension
asthma
...
```

### Timestamped Outputs

Timestamp output files and cache to keep track of different runs:

```bash
# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Use in output and cache paths
python replay_comparison_clean.py \
    --sessions 100 \
    --data data/animals_100.txt \
    --domain animal \
    --cache cache/oracle_cache_animals_${TIMESTAMP}.jsonl \
    --output results_animals_${TIMESTAMP}.json
```

Windows PowerShell:
```powershell
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

python replay_comparison_clean.py `
    --sessions 100 `
    --data data/animals_100.txt `
    --domain animal `
    --cache cache/oracle_cache_animals_$TIMESTAMP.jsonl `
    --output results_animals_$TIMESTAMP.json
```

### Cache Files

The `--cache` option stores P(yes|target, question) probabilities to disk. This provides:

- **Reproducibility**: Same questions get identical probability scores across runs
- **Speed**: Cached lookups avoid redundant LLM calls
- **Debugging**: Inspect the cache to verify LLM responses

Cache format (JSONL):
```json
{"target": "dog", "question": "Does it have fur?", "probability": 0.95}
{"target": "cat", "question": "Does it have fur?", "probability": 0.92}
``` 
Use separate cache files per experiment configuration to avoid mixing results from different setups.
