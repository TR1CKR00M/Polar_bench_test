#!/usr/bin/env bash
set -euo pipefail

# Evaluate the fine-tuned MACE-POLAR-1-M model alongside the 4 existing models
# in results/full/ without re-running completed relaxations.

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-/opt/anaconda3/envs/myenv/bin/python}"
FT_MODEL="${1:-./model/polar_1m_finetuned/polar_1m_v_sb1_pbe.model}"
FT_NAME="${FT_NAME:-MACE-POLAR-1-M (FT)}"
CONFIG="config/config_full.yaml"

if [[ ! -f "$FT_MODEL" ]]; then
    echo "[ERROR] Fine-tuned model not found at: $FT_MODEL" >&2
    echo "Please download polar_1m_v_sb1_pbe.model from Skipjack to model/polar_1m_finetuned/ first." >&2
    exit 1
fi

"$PYTHON" -c "
import yaml
cfg_path = '$CONFIG'
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

ft_entry = {
    'name': '$FT_NAME',
    'type': 'mace_polar',
    'model_path': '$FT_MODEL',
    'device': 'cpu',
    'default_dtype': 'float32',
    'pbc_handling': 'pbc',
}

names = [m['name'] for m in cfg['models']]
if '$FT_NAME' not in names:
    cfg['models'].append(ft_entry)
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print('[INFO] Added $FT_NAME to $CONFIG')
else:
    print('[INFO] $FT_NAME already present in $CONFIG')
"

echo "=== Step 1: Running relaxation for $FT_NAME (skipping already completed models) ==="
"$PYTHON" scripts/02_run_grid_relaxation.py --config "$CONFIG"

echo "=== Step 2: Re-evaluating RMSD across all 5 models ==="
"$PYTHON" scripts/03_evaluate_rmsd.py --config "$CONFIG"

echo "=== Step 3: Plotting 5-model comparison figures ==="
"$PYTHON" scripts/04_plot_results.py --config "$CONFIG"

echo "[INFO] Done! Updated summary and figures are in ./results/full/"
