# SLURM Embedding Collection Guide

This guide explains how to use the SLURM submission script for collecting entity embeddings.

## Fixed Bugs in collecting_embeddings.py

The original `collecting_embeddings.py` had several bugs that have been fixed:

1. **Undefined `target_layers` variable** - This variable was referenced but never defined
2. **Batch processing bug** - Only the first entity in each batch was being processed
3. **Incorrect tensor indexing** - The embedding extraction wasn't properly indexing batch dimensions

## Quick Start

### Basic Usage

Submit jobs for all default datasets (SOCRATES_v0, SOCRATES_v1, TwoHopFact):

```bash
python submit_collect_embeddings.py --monitor
```

### Custom Model
### Multiple Models (auto resources)

Submit for multiple models in one go. The script will choose nodes/GPUs/time/batch heuristically per model when `--auto_resources` is set, with explicit overrides for some known models:

```bash
python submit_collect_embeddings.py \
    --model_list "Qwen/Qwen2.5-14B,Qwen/Qwen2.5-32B,allenai/OLMo-2-1124-13B,mistralai/Mistral-Large-Instruct-2411,openai/gpt-oss-20b,openai/gpt-oss-120b" \
    --auto_resources \
    --input_csv_paths "datasets/SOCRATES_v1.csv" \
    --monitor
```

You can still pass defaults (e.g., `--nodes`, `--gpus_per_node`, `--batch_size`) and they will be used for models without an override.


Use a different model:

```bash
python submit_collect_embeddings.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --monitor
```

### Multiple GPUs (multi-node ready)

Use more GPUs per node (multi-node supported via --nodes):

```bash
python submit_collect_embeddings.py \
    --model_name_or_path "meta-llama/Llama-2-13b-hf" \
    --gpus_per_node 4 \
    --tensor_parallel_size 4 \
    --monitor
```

### Custom Input Files

Process specific CSV files:

```bash
python submit_collect_embeddings.py \
    --input_csv_paths "datasets/custom1.csv,datasets/custom2.csv" \
    --output_dir "custom_embeddings" \
    --monitor
```

## Command-Line Arguments

### Model Configuration
- `--model_name_or_path`: Model name or path (default: "mistralai/Mistral-7B-v0.3")
- `--model_list`: Comma-separated list of model ids to run (overrides `--model_name_or_path`)
- `--auto_resources`: Enable heuristic resource selection per model (nodes/GPUs/time/batch)
- `--revision`: Specific model revision/checkpoint
- `--backend`: Backend to use - "hf" or "vllm" (default: "hf")

### Input/Output
- `--input_csv_paths`: Comma-separated list of CSV files to process
- `--output_dir`: Directory for output embeddings (default: "hidden_repr")

### Compute Resources
- `--nodes`: Number of nodes to use (default: 1)
- `--gpus_per_node`: GPUs per node - 1, 2, or 4 (default: 1)
- `--ntasks_per_node`: SLURM ntasks-per-node (default: 1)
- `--gpu_type`: GPU type for gres (default: "a100")
- `--master_port`: Port for torch.distributed (default: 6000)
- `--batch_size`: Batch size for processing (default: 32)
- `--time`: Maximum job time in HH:MM:SS format (default: "4:00:00")

Notes:
- CPUs and memory are auto-mapped per node: 1 GPU→18 CPU/125GB, 2 GPUs→36 CPU/250GB, 4 GPUs→72 CPU/500GB.

### vLLM-Specific Options
- `--tensor_parallel_size`: Number of GPUs for tensor parallelism (default: 1)
- `--gpu_memory_utilization`: GPU memory utilization fraction (default: 0.90)

### Job Management
- `--job_name`: Base name for SLURM jobs (default: "collect_embeddings")
- `--conda_env`: Conda environment name (default: "reasoning")
- `--max_parallel_jobs`: Maximum concurrent jobs (default: 10)
- `--monitor`: Enable job monitoring and automatic queue management
- `--wait_time`: Seconds between status checks when monitoring (default: 60)

## Examples

### Example 1: Process All Default Datasets with Monitoring

```bash
python submit_collect_embeddings.py \
    --model_name_or_path "mistralai/Mistral-7B-v0.3" \
    --gpus_per_node 1 \
    --batch_size 32 \
    --monitor
```

This will:
- Process SOCRATES_v0.csv, SOCRATES_v1.csv, and TwoHopFact.csv
- Submit up to 10 jobs in parallel (default)
- Monitor job progress and submit new jobs as old ones complete
- Save embeddings to `hidden_repr/` directory

### Example 2: Large Model with Multiple GPUs (single-node)

```bash
python submit_collect_embeddings.py \
    --model_name_or_path "meta-llama/Llama-2-70b-hf" \
    --backend vllm \
    --gpus_per_node 4 \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.95 \
    --batch_size 16 \
    --time "8:00:00" \
    --monitor
```

### Example 3: Custom Datasets without Monitoring (multi-node ready)

```bash
python submit_collect_embeddings.py \
    --input_csv_paths "data/dataset1.csv,data/dataset2.csv,data/dataset3.csv" \
    --output_dir "my_embeddings" \
    --nodes 1 \
    --gpus_per_node 2 \
    --max_parallel_jobs 3
```

This submits all jobs at once without monitoring. Check status with `squeue`.

### Example 4: Using Hugging Face Token

If your model requires authentication:

```bash
export HF_TOKEN="your_token_here"

python submit_collect_embeddings.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --monitor
```

The script automatically passes the HF_TOKEN environment variable to SLURM jobs.

## Output Structure

Embeddings are saved in the following structure:

```
hidden_repr/
├── logs/
│   └── collect_embeddings_SOCRATES_v0_Mistral-7B-v0.3_abc12345/
│       ├── slurm_job.sh
│       ├── job.out.123456
│       ├── job.err.123456
│       └── exit_status
├── SOCRATES_v0_Mistral-7B-v0.3_embeddings.pt
├── SOCRATES_v1_Mistral-7B-v0.3_embeddings.pt
└── TwoHopFact_Mistral-7B-v0.3_embeddings.pt
```

Each `.pt` file contains:
```python
{
    "meta": {
        "model": "mistralai--Mistral-7B-v0.3",
        "num_layers": 32,
        "num_entities": 1234
    },
    "entities": {
        "Q12345": {
            "name": "Entity Name",
            "layer_0": tensor(...),
            "layer_1": tensor(...),
            ...
        },
        ...
    }
}
```

## Monitoring Jobs

### With Built-in Monitoring

Use the `--monitor` flag to enable automatic monitoring:

```bash
python submit_collect_embeddings.py --monitor
```

The script will:
- Show real-time job status updates
- Automatically submit new jobs when others complete
- Track completed, failed, and running jobs
- Display final summary with output file paths

### Manual Monitoring

Check job status:
```bash
squeue -u $USER
```

Check specific job output:
```bash
tail -f hidden_repr/logs/collect_embeddings_*/job.out.*
```

Cancel a job:
```bash
scancel <job_id>
```

Cancel all your jobs:
```bash
scancel -u $USER
```

## Troubleshooting

### Job Fails Immediately

Check the error log:
```bash
cat hidden_repr/logs/collect_embeddings_*/job.err.*
```

Common issues:
- CSV file not found: Check that `--input_csv_paths` points to existing files
- Out of memory: Reduce `--batch_size` or use more GPUs
- Model not accessible: Check `HF_TOKEN` environment variable

### CUDA Out of Memory

Solutions:
1. Reduce batch size: `--batch_size 16`
2. Use more GPUs per node: `--gpus_per_node 4` (or increase `--nodes` if supported)
3. Reduce GPU memory utilization (vLLM): `--gpu_memory_utilization 0.85`
4. Use vLLM backend for better memory management: `--backend vllm`

### Jobs Not Starting

If jobs are pending (PD status), they're waiting for resources. You can:
- Reduce `--gpus_per_node` requirement or `--nodes`
- Reduce `--time`
- Wait for resources to become available

## Tips

1. **Start Small**: Test with one dataset first before submitting many jobs
2. **Use Monitoring**: The `--monitor` flag helps manage job queues automatically
3. **Check Resources**: Use `squeue` to see what resources are available
4. **Batch Size**: Start with smaller batch sizes and increase if memory allows
5. **Save Output Paths**: The script prints output paths - save them for later use

## Environment Setup

The script assumes:
- SLURM is available
- CUDA 12.6 module is available
- Python waterboa/2024.06 module is available
- Conda environment specified by `--conda_env` exists (default: "reasoning")
- The environment has all required packages (torch, transformers, pandas, tqdm, etc.)

If using vLLM:
- vLLM must be installed in the conda environment
- Multiple GPUs may be required for tensor parallelism
