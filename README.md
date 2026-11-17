# Polar System Benchmark: Charged Point Defects Grid Study

A modular benchmarking pipeline to evaluate [Electrostatic MACE (`MACE-POLAR-1`)](https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html) foundation models alongside locally downloaded MACE foundation checkpoints (`MACE-MH-1`, `MACE-MPA-0-Medium`, `MACE-OMAT-0-Medium`) on charged point-defect configurations from [Zenodo Record 22756145](https://zenodo.org/records/22756145).

The evaluation metrics and publication-style figures follow **Fig. 1(a–c)** of:
> Xinwei Wang, Irea Mosquera-Lois, and Aron Walsh, *"Multifidelity Machine Learning Interatomic Potentials for Charged Point Defects"*, *PRX Energy* **5**, 043001 (2026), DOI: [10.1103/tzwd-cl3y](https://link.aps.org/doi/10.1103/tzwd-cl3y) ([arXiv:2603.05238v3](https://arxiv.org/html/2603.05238v3)).

---

## Repository Structure

```text
Polar_system_benchmark/
├── config/
│   └── config.yaml                  # Unified configuration for dataset, models, relaxation, and plotting
├── scripts/
│   ├── 01_prepare_data.py           # Download/extract Zenodo 22756145 & index VASP/extxyz structures
│   ├── 02_run_grid_relaxation.py    # Run grid relaxations across models x charge states x distortions
│   ├── 03_evaluate_rmsd.py          # Compute PBC minimum-image RMSD & cross-charge sensitivity matrices
│   └── 04_plot_results.py           # Plot Fig. 1(a-c) and cross-charge RMSD heatmaps
├── test/
│   ├── test.sh                      # Fast single-model smoke test (2 charges x 2 distortions)
│   └── test_full.sh                 # Full 4-model x 5-charge x 3-distortion test script
├── pyproject.toml
└── README.md
```

*(Note: Local checkpoints under `model/`, raw/parsed datasets under `data/`, and generated outputs under `results/` are excluded via `.gitignore`.)*

---

## Environment Setup

Running `MACE-POLAR-1` requires `mace-torch >= 0.3.16` and `graph_electrostatics` (`graph_longrange`), as documented in the [Electrostatic MACE Guide](https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html):

```bash
pip install "mace-torch>=0.3.16"
pip install git+https://github.com/WillBaldwin0/graph_electrostatics.git@v0.4.4
pip install ase numpy pandas pyyaml matplotlib
```

Place your locally downloaded MACE checkpoints in `model/`:
- `model/mace-mh-1.model` (`head: "omat_pbe"`)
- `model/mace-mpa-0-medium.model` (`head: "default"`)
- `model/mace-omat-0-medium.model` (`head: "default"`)

`MACE-POLAR-1-M` (`polar-1-m`) is loaded automatically via `mace.calculators.mace_polar` (requires internet access on the first run to cache the weights to `~/.cache/mace/`).

---

## Usage

### 1. Full Benchmark Workflow

```bash
# Step 1: Download (optional --download) and index Zenodo 22756145 structures into data/manifest.json
python scripts/01_prepare_data.py --config config/config.yaml --download

# Step 2: Relax selected initial distortions across all configured models and charge states
python scripts/02_run_grid_relaxation.py --config config/config.yaml

# Step 3: Evaluate ground-state RMSD (Fig. 1a), PES distortion curves (Fig. 1b/c), and cross-charge matrices
python scripts/03_evaluate_rmsd.py --config config/config.yaml

# Step 4: Generate publication-quality figures (use --demo to preview layout with synthetic data)
python scripts/04_plot_results.py --config config/config.yaml
```

### 2. Quick Tests

```bash
# Fast smoke test (1 local model, 2 charge states, 2 distortions, 10 steps)
./test/test.sh

# Full-grid test (all 4 models, all 5 charge states, 3 distortions per charge state, 15 steps)
./test/test_full.sh
```

---

## Methodology Notes

- **Charge & Spin Injection**: For each supercell relaxation, `02_run_grid_relaxation.py` explicitly sets `atoms.info["charge"]`, `atoms.info["total_charge"]`, `atoms.info["spin"]`, and `atoms.info["external_field"] = [0.0, 0.0, 0.0]`, and saves learned `MACE-POLAR-1` atomic monopoles and dipoles (`density_coefficients`) into the relaxed `.extxyz` files.
- **PBC Minimum-Image RMSD**: `03_evaluate_rmsd.py` wraps fractional coordinate differences across periodic boundaries (`df -= np.round(df)`), removes uniform rigid translation, and evaluates both global and local (`<= 5.0 Å` around the vacancy) RMSD relative to the DFT-PBE ground state. Structures within `RMSD <= 0.10 Å` are classified as successful ground-state identifications.
- **Cross-Charge Sensitivity Matrix**: Computes pairwise `RMSD(q_i, q_j)` between predicted ground states across charge states to distinguish models that capture charge-dependent reconstructions from charge-blind models that collapse all charge states onto identical geometries.
