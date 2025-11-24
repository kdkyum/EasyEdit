#!/bin/bash
#SBATCH -J wikipedia_crawl
#SBATCH -o slurm.out.%j
#SBATCH -e slurm.err.%j
#SBATCH --mem=16000
#SBATCH -t 12:00:00
#SBATCH -D .

set -euo pipefail
echo "[$(date)] Starting job on $(hostname) in $(pwd)"

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module purge
module load intel/21.4.0 impi/2021.4
module load python-waterboa/2024.06
eval "$(conda shell.bash hook)"
conda activate editing

# Keep libraries single-threaded for stability on shared nodes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

echo "Running: python wikipedia_crawl.py"
python wikipedia_crawl.py

echo "[$(date)] Job finished"