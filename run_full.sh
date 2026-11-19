#!/usr/bin/env bash

set -euo pipefail

# Always run relative to the project root, regardless of the caller's cwd.
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON="$PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON="$CONDA_PREFIX/bin/python"
elif [[ -x "/opt/anaconda3/envs/myenv/bin/python" ]]; then
    PYTHON="/opt/anaconda3/envs/myenv/bin/python"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON="$(command -v python3)"
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "Error: Python interpreter not found: $PYTHON" >&2
    exit 1
fi
echo "[INFO] Using Python: $PYTHON"

# Optional overrides via environment variables:
#   DEVICE=cpu DTYPE=float64 FMAX=0.01 MAX_STEPS=300 ./run_full.sh
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float64}"
FMAX="${FMAX:-0.01}"
MAX_STEPS="${MAX_STEPS:-300}"

# 1. Generate full-benchmark config (config/config_full.yaml) from config/config.yaml:
#    - Keep all 5 charge states (-3, -2, -1, 0, +1)
#    - Include all configurations per charge state (67 configs/model, 268 relaxations total)
#    - Run all 4 models (MACE-POLAR-1-M + 3 local models)
#    - Save outputs under ./results/full/
"$PYTHON" -c "
import yaml
cfg = yaml.safe_load(open('config/config.yaml'))
cfg['dataset']['manifest_path'] = './results/full/manifest_full.json'
cfg['relaxation']['fmax'] = float('$FMAX')
cfg['relaxation']['max_steps'] = int('$MAX_STEPS')
cfg['relaxation'].pop('selected_distortions', None)
cfg['relaxation'].pop('max_distortions', None)
cfg['relaxation']['output_dir'] = './results/full/relaxed_structures'
cfg['evaluation']['summary_csv'] = './results/full/grid_study_summary.csv'
cfg['evaluation']['detailed_csv'] = './results/full/grid_study_all_distortions.csv'
cfg['plotting']['output_dir'] = './results/full/figures'
for m in cfg['models']:
    m['default_dtype'] = '$DTYPE'
    m['device'] = '$DEVICE'
yaml.safe_dump(cfg, open('config/config_full.yaml', 'w'), sort_keys=False)
"

echo "[INFO] Full benchmark config written to config/config_full.yaml (device=$DEVICE, dtype=$DTYPE, fmax=$FMAX, max_steps=$MAX_STEPS)"

# 2. Build full dataset manifest
"$PYTHON" scripts/01_prepare_data.py --config config/config_full.yaml

# 3. Run all 4 models x all 5 charge states x all 67 configurations (268 relaxations; supports resume)
"$PYTHON" scripts/02_run_grid_relaxation.py --config config/config_full.yaml

# 4. Evaluate RMSD metrics and generate publication-style figures (saved to ./results/full/figures/)
"$PYTHON" scripts/03_evaluate_rmsd.py --config config/config_full.yaml
"$PYTHON" scripts/04_plot_results.py --config config/config_full.yaml

echo "[INFO] Full benchmark completed! Figures saved to ./results/full/figures/"
