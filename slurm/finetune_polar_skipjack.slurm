#!/usr/bin/env bash
#SBATCH --job-name=polar_ft_vsb1
#SBATCH --account=pclancy3_dualx          # Replace with your Skipjack Slurm account (check with: interact -l)
#SBATCH --partition=a100                   # 'a100' (80 GB) or 'l40s' (48 GB) for JHU accounts
#SBATCH --gres=gpu:1                       # 1 GPU (automatically allocates 10 cores / 60 GB on a100)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=logs/polar_ft_%j.out
#SBATCH --error=logs/polar_ft_%j.err

set -euo pipefail

# Recommended by Skipjack docs so threaded ops match allocated cores
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Adjust to your scratch project directory on Skipjack
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"
mkdir -p logs

# Activate your conda/mamba environment containing mace-torch 0.3.16 + graph_longrange 0.4.4
# Example:
# module load anaconda3   # or source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv
PYTHON_BIN="/home/jhu/ywu366/venvs/mace-md/bin/python"

"${PYTHON_BIN}" scripts/06_finetune_polar.py \
  --name polar_1m_v_sb1_pbe \
  --foundation-model polar-1-m \
  --train-file data/finetune/v_sb1_pbe/train.xyz \
  --valid-file data/finetune/v_sb1_pbe/valid.xyz \
  --work-dir model/polar_1m_finetuned \
  --max-num-epochs 100 \
  --batch-size 16 \
  --valid-batch-size 16 \
  --default-dtype float32 \
  --lr 0.0005 \
  --device cuda
