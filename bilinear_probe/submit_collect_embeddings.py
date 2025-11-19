#!/usr/bin/env python3

import os
import argparse
import subprocess
import uuid
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Submit embedding collection jobs to SLURM with GPUs'
    )
    # Single model (fallback) or multiple models via --model_list
    parser.add_argument('--model_name_or_path', type=str, 
                        default="mistralai/Mistral-7B-v0.3",
                        help='Model name or path for embeddings (used when --model_list is not provided)')
    parser.add_argument('--model_list', type=str, default=None,
                        help='Comma-separated list of model ids to run, e.g. "Qwen/Qwen2.5-14B,allenai/OLMo-2-1124-13B"')
    parser.add_argument('--auto_resources', action='store_true',
                        help='Automatically choose nodes/GPUs/time/batch_size per model based on size heuristics')
    parser.add_argument('--revision', type=str, default=None,
                        help='Model revision to use')
    parser.add_argument('--backend', type=str, default='hf',
                        choices=['hf', 'vllm'],
                        help='Backend to use (hf or vllm)')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Tensor parallel size for vllm')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.90,
                        help='GPU memory utilization for vllm')
    parser.add_argument('--input_csv_paths', type=str,
                        default="datasets/SOCRATES_v1.csv",
                        help='Comma-separated list of input CSV files')
    parser.add_argument('--output_dir', type=str, default='hidden_repr',
                        help='Base directory for output embeddings')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing')
    parser.add_argument('--eval_batch_size', type=int, default=512,
                        help='Eval batch size for correctness filtering')

    # REPLACED num_gpus with multi-node options
    parser.add_argument('--nodes', type=int, default=1,
                        help='Number of nodes to use')
    parser.add_argument('--gpus_per_node', type=int, default=4,
                        choices=[1, 2, 4],
                        help='Number of GPUs per node')
    parser.add_argument('--ntasks_per_node', type=int, default=1,
                        help='SLURM ntasks-per-node (typically 1)')
    parser.add_argument('--gpu_type', type=str, default='a100',
                        help='GPU type for --gres (e.g., a100)')
    parser.add_argument('--master_port', type=int, default=6000,
                        help='Master port for torch.distributed.run')

    parser.add_argument('--time', type=str, default='12:00:00',
                        help='Maximum time for the job (format: HH:MM:SS)')
    parser.add_argument('--job_name', type=str, default='collect_embeddings',
                        help='Base name for the SLURM jobs')
    parser.add_argument('--conda_env', type=str, default='reasoning',
                        help='Conda environment to activate')
    parser.add_argument('--max_parallel_jobs', type=int, default=10,
                        help='Maximum number of jobs to run in parallel')
    parser.add_argument('--wait_time', type=int, default=60,
                        help='Wait time in seconds between job status checks')
    parser.add_argument('--monitor', action='store_true',
                        help='Monitor job status after submission')
    return parser.parse_args()


def create_slurm_script(args, input_csv_path, run_id, model_name_or_path, per_model_overrides=None):
    """Create the content of the SLURM submission script for collecting embeddings."""
    # Extract dataset name for job ID
    
    # Create a unique job ID
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Extract model short name
    model_short_name = model_name_or_path.split('/')[-1]
    if args.revision:
        model_short_name += f"_{args.revision}"
    job_id = f"{args.job_name}_{model_short_name}_{run_id}"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Output file path
    output_path = os.path.join(args.output_dir, f"{model_short_name}_embeddings.pt")

    # CPUs/memory mapping based on GPUs per node (aligns with your real SLURM example)
    gpus_per_node = args.gpus_per_node
    nodes = args.nodes
    time_limit = args.time
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    backend = args.backend
    tensor_parallel_size = args.tensor_parallel_size
    gpu_memory_utilization = args.gpu_memory_utilization

    # Allow per-model overrides (from auto resources or explicit mapping)
    if per_model_overrides:
        gpus_per_node = per_model_overrides.get('gpus_per_node', gpus_per_node)
        nodes = per_model_overrides.get('nodes', nodes)
        time_limit = per_model_overrides.get('time', time_limit)
        backend = per_model_overrides.get('backend', backend)
        tensor_parallel_size = per_model_overrides.get('tensor_parallel_size', tensor_parallel_size)
        gpu_memory_utilization = per_model_overrides.get('gpu_memory_utilization', gpu_memory_utilization)
        eval_batch_size = per_model_overrides.get('eval_batch_size', eval_batch_size)

    if gpus_per_node == 1:
        cpus_per_task, mem = 18, 125000
    elif gpus_per_node == 2:
        cpus_per_task, mem = 36, 250000
    elif gpus_per_node == 4:
        cpus_per_task, mem = 72, 500000
    else:
        raise ValueError("Unsupported gpus_per_node. Choose 1, 2, or 4.")

    gpu_config = f"""#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={args.ntasks_per_node}
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:{args.gpu_type}:{gpus_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}"""

    # Build arguments for collecting_embeddings.py
    eval_batch_size = args.eval_batch_size
    script_args = f"""collecting_embeddings.py \\
    --model_name_or_path {model_name_or_path} \\
    --backend {backend} \\
    --input_json_paths {input_csv_path} \\
    --output_path {output_path} \\
    --batch_size {batch_size} \\
    --eval_batch_size {eval_batch_size}"""
    if args.revision:
        script_args += f" \\\n    --revision {args.revision}"
    if backend == "vllm":
        script_args += f" \\\n+    --tensor_parallel_size {tensor_parallel_size}"
        script_args += f" \\\n+    --gpu_memory_utilization {gpu_memory_utilization}"
    if os.environ.get("HF_TOKEN"):
        script_args += " \\\n    --hf_token $HF_TOKEN"

    # Create output directory for logs
    log_dir = os.path.join(args.output_dir, "logs", job_id)
    os.makedirs(log_dir, exist_ok=True)

    # Create the SLURM script (srun + torch.distributed.run)
    slurm_script = f"""#!/bin/bash -l
# Standard output and error:
#SBATCH -o {log_dir}/job.out.%j
#SBATCH -e {log_dir}/job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J {job_id}
{gpu_config}
#
#SBATCH --time={time_limit}

module purge
module load cuda/12.6
module load python-waterboa/2024.06

eval "$(conda shell.bash hook)"
conda activate {args.conda_env}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

# Distributed training setup
GPUS_PER_NODE={gpus_per_node}
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT={args.master_port}
export GPUS_PER_NODE MASTER_ADDR MASTER_PORT

# Print job details
echo "Starting job $SLURM_JOB_ID at $(date)"
echo "Model: {model_name_or_path}"
echo "Input CSV: {input_csv_path}"
echo "Output path: {output_path}"
echo "Backend: {backend}"
echo "Batch size: {batch_size}"
echo "Eval batch size: {eval_batch_size}"
echo "Number of GPUs: $GPUS_PER_NODE"

# Run the embedding collection script
srun --jobid $SLURM_JOB_ID bash -c 'python -m torch.distributed.run \\
--nproc-per-node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node-rank $SLURM_PROCID \\
--master-addr $MASTER_ADDR --master-port $MASTER_PORT \\
{script_args}'

# Save exit status
EXIT_STATUS=$?
echo "Job finished with status: $EXIT_STATUS at $(date)"

# Save status to file for monitoring script
echo $EXIT_STATUS > {log_dir}/exit_status

exit $EXIT_STATUS
"""
    # Path for the script file
    script_path = os.path.join(log_dir, "slurm_job.sh")
    with open(script_path, 'w') as f:
        f.write(slurm_script)

    return script_path, job_id, log_dir, output_path


def submit_job(script_path):
    """Submit the SLURM job and return the job ID."""
    result = subprocess.run(['sbatch', script_path], 
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           text=True)
    
    if result.returncode != 0:
        print(f"Error submitting job: {result.stderr}")
        return None
    
    # Extract job ID (typical output: "Submitted batch job 12345")
    job_id = result.stdout.strip().split()[-1]
    return job_id


def get_running_jobs():
    """Get list of currently running jobs for the user"""
    try:
        result = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i"], 
                               capture_output=True, text=True, check=True)
        job_ids = [job_id.strip() for job_id in result.stdout.strip().split('\n') if job_id.strip()]
        return job_ids
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not query SLURM jobs ({e}). Assuming no jobs are running.")
        return []


def main():
    """Main function to create and submit multiple embedding collection jobs to SLURM."""
    args = parse_args()
    # Parse input CSV paths
    input_csv_paths = args.input_csv_paths
    # Parse models: either single model or list
    if args.model_list:
        model_list = [m.strip() for m in args.model_list.split(',') if m.strip()]
    else:
        model_list = [args.model_name_or_path]

    # Heuristic and explicit per-model resource mapping
    def parse_size_b(model_id: str) -> float:
        # Try to extract size in B from model id (e.g., 14B, 32B, 72B, 120b)
        import re
        m = re.search(r"(\d+)(?:\.(\d+))?\s*[Bb]", model_id)
        if m:
            whole = int(m.group(1))
            frac = m.group(2)
            return float(f"{whole}.{frac}" if frac else whole)
        # Mistral-Large and similar known big models
        if "Mistral-Large" in model_id:
            return 123.0
        return 0.0

    explicit_overrides = {
        # Known big models
        "mistralai/Mistral-Large-Instruct-2411": {"nodes": 2, "gpus_per_node": 4, "time": "12:00:00"}, # 123B
        "openai/gpt-oss-120b": {"nodes": 2, "gpus_per_node": 4, "time": "12:00:00"},
        # Large but not 100B+
        "meta-llama/Llama-3.1-70B": {"nodes": 2, "gpus_per_node": 4, "time": "12:00:00"},
        "Qwen/Qwen2.5-72B": {"nodes": 2, "gpus_per_node": 4, "time": "12:00:00"},
        "Qwen/Qwen2.5-72B-Instruct": {"nodes": 2, "gpus_per_node": 4, "time": "12:00:00"},
        "mistralai/Mixtral-8x7B-v0.1": {"nodes": 2, "gpus_per_node": 4, "time": "08:00:00"}, # 47B
        "meta-llama/Llama-2-70b-chat-hf": {"nodes": 2, "gpus_per_node": 4, "time": "12:00:00"},
        # Medium
        "openai/gpt-oss-20b": {"nodes": 1, "gpus_per_node": 4, "time": "08:00:00"},
        "Qwen/Qwen2.5-32B": {"nodes": 1, "gpus_per_node": 4, "time": "08:00:00"},
        "Qwen/Qwen2.5-32B-Instruct": {"nodes": 1, "gpus_per_node": 4, "time": "08:00:00"}, 
        "allenai/OLMo-2-0325-32B": {"nodes": 1, "gpus_per_node": 4, "time": "08:00:00"},
        "mistralai/Mistral-Small-3.1-24B-Base-2503": {"nodes": 1, "gpus_per_node": 4, "time": "08:00:00", "backend": "vllm"}, # 24B
        # Smaller
        "mistralai/Mistral-Nemo-Base-2407": {"nodes": 1, "gpus_per_node": 2, "time": "06:00:00"}, # 12B
        "Qwen/Qwen2.5-14B": {"nodes": 1, "gpus_per_node": 2, "time": "06:00:00"},
        "Qwen/Qwen2.5-14B-Instruct": {"nodes": 1, "gpus_per_node": 2, "time": "06:00:00"},
        "allenai/OLMo-2-1124-13B": {"nodes": 1, "gpus_per_node": 2, "time": "06:00:00"},
        "allenai/OLMo-2-1124-13B-Instruct": {"nodes": 1, "gpus_per_node": 2, "time": "06:00:00"},
        "meta-llama/Llama-2-13b-chat-hf": {"nodes": 1, "gpus_per_node": 2, "time": "06:00:00"},
        # Tiny
        "mistralai/Mistral-7B-v0.3": {"nodes": 1, "gpus_per_node": 1, "time": "06:00:00"},
        "allenai/OLMo-2-1124-7B": {"nodes": 1, "gpus_per_node": 1, "time": "06:00:00"},
        "allenai/OLMo-2-0425-1B": {"nodes": 1, "gpus_per_node": 1, "time": "06:00:00"},
        "meta-llama/Llama-3.2-1B": {"nodes": 1, "gpus_per_node": 1, "time": "06:00:00"},
        "meta-llama/Llama-3.2-3B": {"nodes": 1, "gpus_per_node": 1, "time": "06:00:00"},
        "meta-llama/Llama-3.1-8B": {"nodes": 1, "gpus_per_node": 1, "time": "06:00:00"},
        "Qwen/Qwen2.5-0.5B": {"nodes": 1, "gpus_per_node": 1, "time": "06:00:00"},
        "Qwen/Qwen2.5-1.5B": {"nodes": 1, "gpus_per_node": 1, "time": "06:00:00"},
        "Qwen/Qwen2.5-3B": {"nodes": 1, "gpus_per_node": 1, "time": "06:00:00"},
        "Qwen/Qwen2.5-7B": {"nodes": 1, "gpus_per_node": 2, "time": "06:00:00"},
    }

    def auto_overrides(model_id: str) -> dict:
        size_b = parse_size_b(model_id)
        if size_b >= 100:
            return {"nodes": 2, "gpus_per_node": 4, "time": "12:00:00"}
        if size_b >= 64:
            return {"nodes": 2, "gpus_per_node": 4, "time": "12:00:00"}
        if size_b >= 28:
            return {"nodes": 1, "gpus_per_node": 4, "time": "08:00:00"}
        if size_b >= 12:
            return {"nodes": 1, "gpus_per_node": 2, "time": "06:00:00"}
        return {}

    print(f"\n{'='*80}")
    print(f"PREPARING EMBEDDING COLLECTION JOBS")
    print(f"{'='*80}")
    if len(model_list) == 1:
        print(f"Model: {model_list[0]}")
    else:
        print(f"Models: {model_list}")
    if args.revision:
        print(f"Revision: {args.revision}")
    print(f"Backend: {args.backend}")
    print(f"Default nodes: {args.nodes}")
    print(f"Default GPUs per node: {args.gpus_per_node}")
    print(f"Default batch size: {args.batch_size}")
    print(f"Input CSV files: {input_csv_paths}")
    print(f"Output directory: {args.output_dir}")

    total_jobs = len(model_list)
    print(f"Number of jobs to submit: {total_jobs}")
    
    # Generate a unique run ID for this batch
    batch_id = uuid.uuid4().hex[:8]
    
    # Prepare job queue
    job_queue = []
    for model_id in model_list:
        # Determine overrides
        overrides = explicit_overrides.get(model_id, {})
        if args.auto_resources:
            # auto fills only missing keys, explicit mapping wins
            auto = auto_overrides(model_id)
            overrides = {**auto, **overrides}

        script_path, job_name, log_dir, output_path = create_slurm_script(
            args, input_csv_paths, batch_id, model_id, overrides
        )
        job_queue.append({
            "input_csv": input_csv_paths,
            "model_id": model_id,
            "script_path": script_path,
            "job_name": job_name,
            "log_dir": log_dir,
            "output_path": output_path,
            "status": "pending",
            "slurm_id": None
        })
        print(f"Prepared job for {input_csv_paths} @ {model_id} -> {script_path}")
    
    if not job_queue:
        print("No valid jobs to submit. Exiting.")
        return
    
    # Confirm with user
    confirm = input("\nDo you want to submit these jobs to SLURM? (yes/no): ")
    if confirm.lower() not in ["yes", "y"]:
        print("Job submission cancelled.")
        return
    
    # Submit jobs
    running_jobs = []
    completed_jobs = []
    failed_jobs = []
    
    print(f"\n{'='*80}")
    print("SUBMITTING JOBS")
    print(f"{'='*80}")
    
    try:
        # Initial submission of jobs up to max_parallel_jobs
        while job_queue and len(running_jobs) < args.max_parallel_jobs:
            job = job_queue.pop(0)
            print(f"Submitting job for {os.path.basename(job['input_csv'])}")
            slurm_id = submit_job(job["script_path"])
            
            if slurm_id:
                job["slurm_id"] = slurm_id
                job["status"] = "running"
                running_jobs.append(job)
                print(f"  - Job submitted with ID: {slurm_id}")
                print(f"  - Output will be saved to: {job['output_path']}")
            else:
                job["status"] = "failed"
                failed_jobs.append(job)
                print(f"  - Failed to submit job")
        
        # Monitor jobs if requested
        if args.monitor and (running_jobs or job_queue):
            print(f"\n{'='*80}")
            print("MONITORING JOBS")
            print(f"{'='*80}")
            
            while job_queue or running_jobs:
                # Get currently running SLURM jobs
                current_running_job_ids = get_running_jobs()
                
                # Check status of running jobs
                for job in running_jobs[:]:  # Use a copy to safely modify during iteration
                    if job["slurm_id"] not in current_running_job_ids:
                        # Job finished
                        exit_status_file = os.path.join(job["log_dir"], "exit_status")
                        job_status = "unknown"
                        
                        try:
                            if os.path.exists(exit_status_file):
                                with open(exit_status_file, "r") as f:
                                    exit_status = int(f.read().strip())
                                if exit_status == 0:
                                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Job completed: {os.path.basename(job['input_csv'])} (ID: {job['slurm_id']})")
                                    print(f"  - Output saved to: {job['output_path']}")
                                    job_status = "completed"
                                else:
                                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Job failed (status {exit_status}): {os.path.basename(job['input_csv'])} (ID: {job['slurm_id']})")
                                    job_status = "failed"
                            else:
                                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Job finished but status unknown: {os.path.basename(job['input_csv'])} (ID: {job['slurm_id']})")
                                job_status = "unknown"
                        except Exception as e:
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Error checking job status: {e}")
                            job_status = "unknown"
                        
                        # Update job lists
                        job["status"] = job_status
                        if job_status == "completed":
                            completed_jobs.append(job)
                        else:  # failed or unknown
                            failed_jobs.append(job)
                        
                        running_jobs.remove(job)
                        
                        # Submit next job if any in queue
                        if job_queue:
                            next_job = job_queue.pop(0)
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Submitting next job: {os.path.basename(next_job['input_csv'])}")
                            slurm_id = submit_job(next_job["script_path"])
                            
                            if slurm_id:
                                next_job["slurm_id"] = slurm_id
                                next_job["status"] = "running"
                                running_jobs.append(next_job)
                                print(f"  - Job submitted with ID: {slurm_id}")
                            else:
                                next_job["status"] = "failed"
                                failed_jobs.append(next_job)
                                print(f"  - Failed to submit job")
                
                # Print status update
                if job_queue or running_jobs:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Status: {len(running_jobs)} running, {len(job_queue)} pending, {len(completed_jobs)} completed, {len(failed_jobs)} failed")
                    time.sleep(args.wait_time)  # Wait before next check
        
        else:
            # No monitoring, just provide summary of submitted jobs
            print("\nJobs submitted but not monitoring. Check status with 'squeue'")
            for job in running_jobs:
                print(f"  - {os.path.basename(job['input_csv'])} -> SLURM ID: {job['slurm_id']}")
            
            if job_queue:
                print(f"\n{len(job_queue)} jobs still in queue (max parallel limit reached)")
                print("You can run this script again later to submit remaining jobs")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted.")
        # Ask if user wants to cancel running jobs
        if running_jobs:
            cancel = input(f"Cancel {len(running_jobs)} running jobs? (yes/no): ")
            if cancel.lower() in ["yes", "y"]:
                for job in running_jobs:
                    if job["slurm_id"]:
                        print(f"Cancelling job {job['slurm_id']} ({os.path.basename(job['input_csv'])})...")
                        subprocess.run(["scancel", job["slurm_id"]], check=False)
    
    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total jobs: {total_jobs}")
    print(f"Submitted: {len(running_jobs) + len(completed_jobs) + len(failed_jobs)}")
    print(f"Still running: {len(running_jobs)}")
    print(f"Completed: {len(completed_jobs)}")
    print(f"Failed: {len(failed_jobs)}")
    print(f"Not submitted: {len(job_queue)}")
    
    if completed_jobs:
        print(f"\nCompleted outputs:")
        for job in completed_jobs:
            print(f"  - {job['output_path']}")


if __name__ == "__main__":
    main()
