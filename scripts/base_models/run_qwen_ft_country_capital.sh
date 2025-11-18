#!/bin/bash -l
#SBATCH -o outputs/run_qwen_ft/%x/%A_%a.out
#SBATCH -e outputs/run_qwen_ft/%x/%A_%a.err
#SBATCH -D ./
#SBATCH -J run_qwen_ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=23:59:00
#SBATCH --array=0-31

set -euo pipefail

module purge
module load cuda/12.6
module load python-waterboa/2024.06

eval "$(conda shell.bash hook)"
conda activate editing || true

export WANDB_MODE=offline
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH='.'

mkdir -p outputs/run_qwen_ft/${SLURM_JOB_NAME} outputs/run_qwen_ft/${SLURM_JOB_NAME} || true

# Resolve layer from array index (required for array jobs)
LAYER_ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${LAYER_ID}" ]]; then
  echo "[ERROR] SLURM_ARRAY_TASK_ID is not set. Submit with --array=0-31." >&2
  exit 1
fi
HPARAMS_PATH=./hparams/TEST/FT/qwen-7b/layer${LAYER_ID}

if [[ ! -f "${HPARAMS_PATH}.yaml" ]]; then
  echo "[ERROR] Hparams file not found: ${HPARAMS_PATH}.yaml" >&2
  exit 1
fi

echo "[INFO] Running FT on Qwen-7B (ZSRE) — layer ${LAYER_ID}"
python run_edit.py \
  --editing_method=FT \
  --hparams_dir=${HPARAMS_PATH} \
  --data_path=./data/counterfact_country-capital.json \
  --metrics_save_path=results/qwen_ft/counterfact_country-capital/layer${LAYER_ID}.json
