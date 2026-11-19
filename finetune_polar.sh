#!/usr/bin/env bash
set -euo pipefail

# Fine-tune MACE-POLAR-1-M on V_Sb(1) DFT-PBE charged-defect trajectories.
# Usage:
#   bash finetune_polar.sh [cpu|cuda] [epochs]

DEVICE="${1:-cpu}"
EPOCHS="${2:-50}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/myenv/bin/python}"

echo "=== Step 1: Preparing V_Sb(1) DFT-PBE fine-tuning dataset ==="
"${PYTHON_BIN}" scripts/05_prepare_polar_finetune_data.py \
  --config config/config.yaml \
  --source v_sb1_pbe \
  --valid-fraction 0.15 \
  --stride 1

echo "=== Step 2: Fine-tuning MACE-POLAR-1-M (device=${DEVICE}, epochs=${EPOCHS}) ==="
"${PYTHON_BIN}" scripts/06_finetune_polar.py \
  --name polar_1m_v_sb1_pbe \
  --foundation-model polar-1-m \
  --train-file data/finetune/v_sb1_pbe/train.xyz \
  --valid-file data/finetune/v_sb1_pbe/valid.xyz \
  --work-dir model/polar_1m_finetuned \
  --max-num-epochs "${EPOCHS}" \
  --batch-size 4 \
  --device "${DEVICE}"
