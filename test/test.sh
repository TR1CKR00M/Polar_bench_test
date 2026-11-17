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

# 1. Derive a lightweight smoke-test config (config_test.yaml) from config/config.yaml
"$PYTHON" -c '
import yaml
cfg = yaml.safe_load(open("config/config.yaml"))
cfg["dataset"]["charge_states"] = [0, -3]
cfg["dataset"]["manifest_path"] = "./results/test/manifest_test.json"
cfg["relaxation"]["max_steps"] = 10
cfg["relaxation"]["fmax"] = 0.05
cfg["relaxation"]["max_distortions"] = 2
cfg["relaxation"]["output_dir"] = "./results/test/relaxed_structures"
cfg["evaluation"]["summary_csv"] = "./results/test/grid_study_summary.csv"
cfg["evaluation"]["detailed_csv"] = "./results/test/grid_study_all_distortions.csv"
cfg["plotting"]["output_dir"] = "./results/test/figures"
for m in cfg["models"]:
    m["default_dtype"] = "float32"
    m["device"] = "cpu"
yaml.safe_dump(cfg, open("config/config_test.yaml", "w"), sort_keys=False)
'

# 2. Parse dataset and generate smoke-test manifest
"$PYTHON" scripts/01_prepare_data.py --config config/config_test.yaml

# 3. Run fast smoke-test relaxation for a single local model
"$PYTHON" scripts/02_run_grid_relaxation.py --config config/config_test.yaml --models "MACE-MH-1 (Local)"

# 4. Evaluate RMSD metrics and generate smoke-test figures (saved to ./results/test/figures/)
"$PYTHON" scripts/03_evaluate_rmsd.py --config config/config_test.yaml
"$PYTHON" scripts/04_plot_results.py --config config/config_test.yaml
