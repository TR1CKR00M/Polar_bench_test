#!/usr/bin/env bash
#SBATCH --job-name=md_600k_qm3
#SBATCH --account=pclancy3_dualx          # Replace with your Skipjack Slurm account
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/md_600k_%j.out
#SBATCH --error=logs/md_600k_%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

PYTHON_BIN="/home/jhu/ywu366/venvs/mace-md/bin/python"

# Run 200 ps (200,000 steps) NVT MD at 600 K on GPU across all 5 models
"${PYTHON_BIN}" scripts/07_run_md_benchmark.py \
  --config config/config_md.yaml \
  --steps 200000 \
  --device cuda

#python scripts/08_analyze_md.py --run-dir results/md_600K_q-3/runs/run_20261119_131641_61f3e8c1