#!/usr/bin/env bash
#SBATCH --job-name=polar_s2_vsb1
#SBATCH --account=pclancy3_dualx          # Replace with your Skipjack Slurm account
#SBATCH --partition=a100                   # 'a100' (80 GB) or 'l40s' (48 GB)
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/polar_s2_%j.out
#SBATCH --error=logs/polar_s2_%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

PYTHON_BIN="/home/jhu/ywu366/venvs/mace-md/bin/python"

# Stage 2 fine-tuning: start from Stage-1 model, boost energy_weight (1 -> 50)
# and lower lr (5e-4 -> 1e-4) to calibrate relative basin energies while preserving forces.
"${PYTHON_BIN}" scripts/06_finetune_polar.py \
  --name polar_1m_v_sb1_pbe_stage2 \
  --foundation-model model/polar_1m_finetuned/polar_1m_v_sb1_pbe.model \
  --train-file data/finetune/v_sb1_pbe/train.xyz \
  --valid-file data/finetune/v_sb1_pbe/valid.xyz \
  --work-dir model/polar_1m_finetuned_stage2 \
  --max-num-epochs 50 \
  --batch-size 16 \
  --valid-batch-size 16 \
  --default-dtype float32 \
  --energy-weight 50.0 \
  --forces-weight 100.0 \
  --lr 0.0001 \
  --device cuda
