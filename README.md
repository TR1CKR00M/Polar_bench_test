# Polar System Benchmark: Charged Point Defects & MD Reliability

A reproducible benchmarking pipeline evaluating [Electrostatic MACE (`MACE-POLAR-1`)](https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html) (zero-shot and two-stage fine-tuned) alongside neutral/local MACE foundation models (`MACE-MH-1`, `MACE-MPA-0-Medium`, `MACE-OMAT-0-Medium`) on charged point-defect configurations ($V_{\mathrm{Sb}(1)}^q$, $q \in \{-3, -2, -1, 0, +1\}$ in $\mathrm{Sb_2Se_3}$) and 600 K NVT molecular dynamics.

---

## 1. Research Question, Citation & License

- **Core Question**: Does explicit total-charge conditioning and long-range electrostatic multipole modeling in `MACE-POLAR-1-M` (and system-specific fine-tuning) separate charged defect states and capture charge-dependent reconstructions more accurately than charge-blind neutral MACE foundation models?
- **Replicated Static Benchmark (Fig. 1a–c)**:
  > Xinwei Wang, Irea Mosquera-Lois, and Aron Walsh, *"Multifidelity Machine Learning Interatomic Potentials for Charged Point Defects"*, *PRX Energy* **5**, 043001 (2026), DOI: [10.1103/tzwd-cl3y](https://link.aps.org/doi/10.1103/tzwd-cl3y) ([arXiv:2603.05238](https://arxiv.org/html/2603.05238v3)).
- **Public Dataset & License**:
  > Supporting data from [Zenodo Record 22756145](https://zenodo.org/records/22756145) (DOI: `10.5281/zenodo.22756145`), distributed under the **Creative Commons Attribution 4.0 International (`CC-BY-4.0`)** license.
- **MD Diagnostic Framework**:
  > 600 K NVT survival and structural diagnostics follow Cao, Mastracco, and Clancy (*MigrationBench*, [GitHub](https://github.com/yicao-elina/MigrationBench)).

---

## 2. Dataset Taxonomy & Naming Conventions

From Zenodo Record 22756145 (`data/raw/`):
- **`1_FM_benchmark/`**: DFT-PBE reference calculations for $V_{\mathrm{Sb}(1)}$ in $\mathrm{Sb_2Se_3}$ across five charge states (used in this benchmark).
- **`2_global_embedding/`**: Charge-embedded MACE training/test datasets, scripts, and HSE06 DFT references for $V_{\mathrm{Se}(1)}$ across five charge states.
- **`3_MF/`**: Multi-fidelity MACE datasets, trained models, and calculation settings.

Within `1_FM_benchmark/V_Sb1/DFT_PBE/`:
- **Charge-state directories**: `mi3` ($q=-3$), `mi2` ($q=-2$), `mi1` ($q=-1$), `neu` ($q=0$), and `pl1` ($q=+1$).
- **Distortion labels (`D_±N`)**: ShakeNBreak initial bond-length distortion percentages around the vacancy (from `D_-60` to `D_60`, with `D_0` denoting the unperturbed defect supercell).
- **Built-in Negative Control (`mi3/D_0` vs. `neu/D_0`)**: `mi3/D_0/POSCAR` and `neu/D_0/POSCAR` are byte-for-byte identical. Because charge-blind models (`MACE-MH-1`, `MACE-MPA-0-Medium`, `MACE-OMAT-0-Medium`) receive no charge input, they produce flat, indistinguishable predictions across charge states from identical initial geometries, isolating what explicit charge conditioning in `MACE-POLAR-1-M` contributes.

---

## 3. Repository Structure

```text
Polar_system_benchmark/
├── config/
│   ├── config.yaml                  # Smoke/baseline config
│   ├── config_full.yaml             # Full 6-model x 5-charge x 67-distortion config
│   └── config_md.yaml               # 600 K NVT Langevin MD config (q = -3)
├── scripts/
│   ├── 01_prepare_data.py           # Index Zenodo 22756145 structures into manifest JSON
│   ├── 02_run_grid_relaxation.py    # Run BFGS grid relaxations & save relaxed .extxyz
│   ├── 03_evaluate_rmsd.py          # Compute PBC minimum-image RMSD & cross-charge matrices
│   ├── 04_plot_results.py           # Generate Fig. 1(a), per-charge PES, and heatmaps
│   ├── 05_prepare_polar_finetune_data.py  # Trajectory-split DFT-PBE OUTCARs for fine-tuning
│   ├── 06_finetune_polar.py         # Wrap mace_run_train with PolarMACE & graph_longrange shims
│   ├── 07_run_md_benchmark.py       # Immutable timestamp+SHA256 600 K NVT MD runner
│   └── 08_analyze_md.py             # 6-panel + individual MD diagnostic plotter
├── slurm/                           # JHU Skipjack SLURM scripts (Stage 1/2 FT & 200 ps MD)
├── results/
│   ├── full/                        # Curated 6-model static CSVs & publication figures
│   └── md_600K_q-3/runs/run_20261119_131641_61f3e8c1/  # Curated 200 ps MD summaries & plots
├── TASK_REPORT.md                   # Exact reproducible hyperparameters & commands
└── README.md
```

*(Note: Model checkpoints under `model/`, raw/fine-tuning datasets under `data/`, and heavy `.extxyz`/`.traj` files are excluded via `.gitignore`.)*

---

## 4. Model Provenance & SHA-256 Checksums

| Model Label | Source / Checkpoint | Head / Precision | SHA-256 |
| :--- | :--- | :--- | :--- |
| `MACE-POLAR-1-M (pbc)` | `mace.calculators.mace_polar("polar-1-m")` (`mace-torch==0.3.16`, `graph_electrostatics==0.4.4`) | `pbc` / `float32` (static), `float64` (MD) | Cached externally under `~/.cache/mace/` |
| `MACE-POLAR-1-M (FT)` | Stage-1 fine-tuned from `polar-1-m` on `v_sb1_pbe` (100 epochs, `lr=5e-4`, `E_w=1.0`, `F_w=100.0`) | `pbc` / `float32` | `f9e4f0e2e67a513176c0c7ce70ac8c3fb6da651771b7a15dcc0797a0857a31ba` |
| `MACE-POLAR-1-M (FT Stage2)` | Stage-2 calibrated from Stage-1 checkpoint (50 epochs, `lr=1e-4`, `E_w=50.0`, `F_w=100.0`) | `pbc` / `float32` | `7312f56169c52b501f3f6f3b38ba88c3cfe0a87f27737d6ac5c37ffd36f9c826` |
| `MACE-MH-1 (Local)` | `model/mace-mh-1.model` | `omat_pbe` / `float32` | `a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47` |
| `MACE-MPA-0-Medium (Local)` | `model/mace-mpa-0-medium.model` | `default` / `float32` | `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638` |
| `MACE-OMAT-0-Medium (Local)` | `model/mace-omat-0-medium.model` | `default` / `float32` | `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a` |

---

## 5. Environment & Quick Usage

Running `MACE-POLAR-1` requires `mace-torch >= 0.3.16` and `graph_electrostatics` (`graph_longrange==0.4.4`):

```bash
pip install "mace-torch>=0.3.16"
pip install git+https://github.com/WillBaldwin0/graph_electrostatics.git@v0.4.4
pip install ase numpy pandas pyyaml matplotlib
```

```bash
# Static 6-model benchmark (full grid)
python scripts/01_prepare_data.py --config config/config_full.yaml
python scripts/02_run_grid_relaxation.py --config config/config_full.yaml
python scripts/03_evaluate_rmsd.py --config config/config_full.yaml
python scripts/04_plot_results.py --config config/config_full.yaml

# 600 K NVT MD benchmark & analysis (q = -3)
python scripts/07_run_md_benchmark.py --config config/config_md.yaml --steps 200000 --device cuda
python scripts/08_analyze_md.py --run-dir results/md_600K_q-3/runs/run_20261119_131641_61f3e8c1
```

See [TASK_REPORT.md](TASK_REPORT.md) for full fine-tuning hyperparameters and HPC SLURM scripts.

---

## 6. Scope, Known Limitations & What This Benchmark Can / Cannot Verify

### What This Benchmark Can Verify
1. **Charge-Dependent Reconstruction & Ground-State Geometry**: Under identical PBC minimum-image RMSD evaluation (`<= 0.10 Å` threshold vs. DFT-PBE ground states), fine-tuned `MACE-POLAR-1-M` achieves `3/5` ground-state hits (`0.075–0.076 Å` avg RMSD across $q \in \{-3..+1\}$), outperforming neutral foundation models (`0–1/5` hits, `0.210–0.224 Å` avg RMSD) and zero-shot `MACE-POLAR-1-M (pbc)` (`0/5` hits, `0.324 Å` avg RMSD).
2. **Electrostatic Monopole Conservation**: Both static relaxations and 200 ps 600 K NVT MD preserve total monopole charge within `<= 9.54e-7 e` (`float32`, FT Stage2) and `<= 8.88e-15 e` (`float64`, zero-shot `pbc`).
3. **High-Temperature NVT Stability & Local Coordination ($q=-3$)**: All five tested models survive 200 ps at 600 K without bond collapse ($d_{\min} < 1.4\text{ \AA}$) or thermal runaway ($T > 1200\text{ K}$). Stage-2 fine-tuning corrects the zero-shot `MACE-POLAR-1-M` first-shell over-binding ($\mathrm{CN}_{\mathrm{Sb-Se}}$ increases from `4.86` to `5.61`, matching local foundation models `5.51–5.68`).

### Known Data & Experimental Limitations (What This Benchmark Cannot Yet Verify)
1. **Unconverged Reference Entry (`pl1/D_-60`)**: In the public dataset, `data/raw/1_FM_benchmark/V_Sb1/DFT_PBE/pl1/D_-60/OUTCAR_-60` did not reach VASP ionic convergence (`reached required accuracy` count = 0). It is currently retained in `manifest_full.json` rather than excluded or rerun.
2. **Heterogeneous DFT Reference Settings**: The underlying Zenodo dataset contains mixed computational settings across subsets/stages—including spin/magnetic moments ($0$ vs. $2\ \mu_B$), dispersion schemes (`IVDW=11` DFT-D3 in `INCAR_neu` vs. D3BJ elsewhere), and k-point meshes ($\Gamma$-only `1×1×1` in `1_FM_benchmark/V_Sb1/DFT_PBE/KPOINTS` vs. $2\times2\times2$ in other stages).
3. **Dispersion Mismatch on Zero-Shot MLIPs**: The DFT-PBE reference includes D3 dispersion (`IVDW=11`). Fine-tuned `MACE-POLAR-1-M` models learn this D3 contribution implicitly from the training forces/energies, whereas zero-shot foundation models (`MACE-POLAR-1-M (pbc)`, `MACE-MH-1`, `MACE-MPA-0-Medium`, `MACE-OMAT-0-Medium`) were evaluated *without* adding an explicit D3 pair potential.
4. **Energy-Ranking Metrics & Convergence Reporting in CSV**: `03_evaluate_rmsd.py` identifies each MLIP's lowest-energy relaxed distortion and compares its geometry against the DFT ground state, but does not yet export formal PES energy-ranking correlations (Spearman $\rho$ / Kendall $\tau$) or propagate per-structure BFGS `converged` flags from `relaxation_index.json` into `grid_study_summary.csv`.
5. **Floating-Point Precision & Single-Cell MD Scope**: Static `MACE-POLAR-1-M` relaxations used `float32` (zero-shot `polar-1-m` exhibits an unshifted $\sim -2.5 \times 10^6\text{ eV}$ total energy scale where `float64` is preferable and was used for MD). The 600 K MD pilot tests a single 59-atom $1\times1\times1$ cell at $q=-3$ and does not yet audit DFT charge-density localization ($\Delta\rho$) on MLIP-relaxed geometries.
