#!/usr/bin/env bash

set -euo pipefail

# Always run relative to the project root, regardless of the caller's cwd.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
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

# 1. Derive a full-grid test config (config_test_full.yaml) from config/config.yaml:
#    - Keep all charge states (-3, -2, -1, 0, +1)
#    - Select 3 representative distortions per charge state (-30%, 0%, +30%)
#    - Run all 4 models (MACE-POLAR-1-M + 3 local models)
#    - Isolate outputs under ./results/test_full/
"$PYTHON" -c '
import yaml
cfg = yaml.safe_load(open("config/config.yaml"))
cfg["dataset"]["manifest_path"] = "./results/test_full/manifest_test_full.json"
cfg["relaxation"]["max_steps"] = 15
cfg["relaxation"]["fmax"] = 0.05
cfg["relaxation"]["selected_distortions"] = [-30.0, 0.0, 30.0]
cfg["relaxation"]["max_distortions"] = 3
cfg["relaxation"]["output_dir"] = "./results/test_full/relaxed_structures"
cfg["evaluation"]["summary_csv"] = "./results/test_full/grid_study_summary.csv"
cfg["evaluation"]["detailed_csv"] = "./results/test_full/grid_study_all_distortions.csv"
cfg["plotting"]["output_dir"] = "./results/test_full/figures"
for m in cfg["models"]:
    m["default_dtype"] = "float32"
    m["device"] = "cpu"
yaml.safe_dump(cfg, open("config/config_test_full.yaml", "w"), sort_keys=False)
'

# 2. Parse dataset and generate full-charge-state test manifest
"$PYTHON" scripts/01_prepare_data.py --config config/config_test_full.yaml

# 3. Run all 4 models x 5 charge states x 3 distortions (60 relaxations total)
"$PYTHON" scripts/02_run_grid_relaxation.py --config config/config_test_full.yaml

# 4. Evaluate RMSD metrics and generate comparison figures (saved to ./results/test_full/figures/)
"$PYTHON" scripts/03_evaluate_rmsd.py --config config/config_test_full.yaml
"$PYTHON" scripts/04_plot_results.py --config config/config_test_full.yaml
