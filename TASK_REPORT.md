# Task Report: Charged Point-Defect Benchmark & PolarMACE Fine-Tuning

**Project Root**: `/Users/trickroom/Lab/Polar_system_benchmark`  
**Target Benchmark**: $V_{\mathrm{Sb}(1)}$ in $\mathrm{Sb_2Se_3}$ across $q \in \{-3, -2, -1, 0, +1\}$ (67 ShakeNBreak distortions per model; 402 relaxations across 6 models)  
**Reference Work**: Wang, Mosquera-Lois, and Walsh, *PRX Energy* **5**, 043001 (2026), DOI: [10.1103/tzwd-cl3y](https://link.aps.org/doi/10.1103/tzwd-cl3y), Zenodo Record [22756145](https://zenodo.org/records/22756145)

---

## 1. Environment & Compatibility Requirements

- **Local Python**: `/opt/anaconda3/envs/myenv/bin/python` (`mace-torch==0.3.16`, `graph_longrange==0.4.4`, `ase`, `pyyaml`, `pandas`, `matplotlib`)
- **HPC Python (JHU Skipjack)**: `/home/jhu/ywu366/venvs/mace-md/bin/python` (`mace-torch==0.3.16`, `graph_longrange==0.4.4` from [WillBaldwin0/graph_electrostatics@v0.4.4](https://github.com/WillBaldwin0/graph_electrostatics), PyTorch 2.6+ CUDA)
- **Required Runtime Shims** (embedded in `scripts/02_run_grid_relaxation.py` and `scripts/06_finetune_polar.py`):
  - `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for PyTorch 2.6+ checkpoint loading
  - `graph_longrange 0.4.4` signature shims (`force_pbc_evaluator`, `pbc`, `.squeeze(1)`, `set_pbc_handling("pbc")`, `features_dim`)
  - Safe non-leaf tensor handling via `torch.Tensor.__deepcopy__` during final `.model` export

---

## 2. Active Verified Scripts & Configurations

| Component | Path | Purpose |
| :--- | :--- | :--- |
| Data manifest | `scripts/01_prepare_data.py` | Indexes `1_FM_benchmark/V_Sb1/DFT_PBE` structures and DFT energies |
| Grid relaxation | `scripts/02_run_grid_relaxation.py` | Runs `BFGS` relaxations with resume/skip support and `PolarMACE` monopole extraction |
| RMSD evaluation | `scripts/03_evaluate_rmsd.py` | Computes global/local RMSD vs. DFT-PBE GS and cross-charge RMSD matrices |
| Figure plotting | `scripts/04_plot_results.py` | Plots Fig. 1(a) (uniform circles, cool Polar gradient vs. warm local colors), Fig. 1(b–f) (per-charge PES with unified colors), and cross-charge sensitivity heatmaps |
| Fine-tune data prep | `scripts/05_prepare_polar_finetune_data.py` | Extracts DFT-PBE `OUTCAR*` trajectories, splits by trajectory, writes `extxyz` |
| Fine-tuning launcher | `scripts/06_finetune_polar.py` | Wraps `mace_run_train` with `PolarMACE` & `graph_longrange 0.4.4` shims |
| Full benchmark config | `config/config_full.yaml` | Master configuration for all 6 models and plotting parameters |
| HPC SLURM (Stage 1) | `slurm/finetune_polar_skipjack.slurm` | 100-epoch Stage-1 fine-tuning on Skipjack `a100` |
| HPC SLURM (Stage 2) | `slurm/finetune_polar_stage2_skipjack.slurm` | 50-epoch Stage-2 energy calibration on Skipjack `a100` |
| Helper runners | `run_full.sh`, `finetune_polar.sh`, `eval_finetuned.sh` | One-command pipeline entrypoints |

---

## 3. Exact Reproducible Parameters

### 3.1 Fine-Tuning Dataset Preparation (`scripts/05_prepare_polar_finetune_data.py`)
- `--source`: `v_sb1_pbe` (67 VASP `OUTCAR*` trajectories from `data/raw/1_FM_benchmark/V_Sb1/DFT_PBE`)
- `--valid-fraction`: `0.15` (trajectory-level split to prevent frame leakage)
- `--stride`: `1`
- `--max-force-norm`: `50.0` eV/Å (filtered 39 extreme initial frames)
- `--seed`: `42`
- **Output**: `data/finetune/v_sb1_pbe/train.xyz` (**3,051 frames**), `valid.xyz` (**533 frames**)
- **Metadata keys**: `REF_energy`, `REF_forces`, `total_charge`, `total_spin` (`{-3:1, -2:2, -1:1, 0:2, +1:1}`), `external_field=[0.0, 0.0, 0.0]`

### 3.2 PolarMACE Fine-Tuning Hyperparameters (`scripts/06_finetune_polar.py`)
- **Common settings**: `--model=PolarMACE`, `--energy_key=REF_energy`, `--forces_key=REF_forces`, `--total_charge_key=total_charge`, `--total_spin_key=total_spin`, `--E0s=average`, `--loss=weighted`, `--stress_weight=0.0`, `--force_mh_ft_lr=True`, `--ema`, `--ema_decay=0.99`, `--amsgrad`, `--scaling=rms_forces_scaling`, `--batch_size=16`, `--valid_batch_size=16`, `--default_dtype=float32`, `--pbc-handling=pbc`, `--seed=42`, `--device=cuda`
- **Stage 1** (`slurm/finetune_polar_skipjack.slurm`):
  - `--foundation-model=polar-1-m`
  - `--max-num-epochs=100`, `--lr=0.0005`, `--energy-weight=1.0`, `--forces-weight=100.0`
  - Validation metrics at Epoch 99: `MAE_E_per_atom = 24.45 meV`, `MAE_F = 13.71 meV/Å` (vs. Initial `888.90 meV`, `509.42 meV/Å`)
  - Saved checkpoint: `model/polar_1m_finetuned/polar_1m_v_sb1_pbe.model`
- **Stage 2** (`slurm/finetune_polar_stage2_skipjack.slurm`):
  - `--foundation-model=model/polar_1m_finetuned/polar_1m_v_sb1_pbe.model`
  - `--max-num-epochs=50`, `--lr=0.0001`, `--energy-weight=50.0`, `--forces-weight=100.0`
  - Saved checkpoint: `model/polar_1m_finetuned_stage2/polar_1m_v_sb1_pbe_stage2.model`

### 3.3 Benchmark Relaxation, Evaluation & Plotting (`config/config_full.yaml`)
- **Relaxation**: `optimizer: BFGS`, `fmax: 0.02` eV/Å, `max_steps: 300`, `fix_cell: true`, `default_dtype: float32`, `pbc_handling: pbc` (for all `mace_polar` models)
- **Evaluation**: `rmsd_threshold: 0.1` Å, `local_cutoff: 5.0` Å
- **Plotting**: `dpi: 300`, `pes_highlight_charges: [-3, -2, -1, 0, 1]`, `model_order: ["DFT Reference", "MACE-POLAR-1-M (FT Stage2)", "MACE-POLAR-1-M (FT)", "MACE-POLAR-1-M (pbc)", "MACE-MH-1 (Local)", "MACE-MPA-0-Medium (Local)", "MACE-OMAT-0-Medium (Local)"]`
- **Visual Encoding**:
  - Fig. 1(a): Uniform circular markers (`o`), cool `YlGnBu` gradient for Polar models, warm palette for local models.
  - Fig. 1(b–f): One subplot per charge state (`2 × 3` grid), model colors matched to Fig. 1(a), `o` for `RMSD <= 0.1 Å` and `X` for `RMSD > 0.1 Å`.

---

## 4. Benchmark Results Summary (`results/full/grid_study_summary.csv`)

| Model | $q=-3$ (Å) | $q=-2$ (Å) | $q=-1$ (Å) | $q=0$ (Å) | $q=+1$ (Å) | Avg RMSD (Å) | Identified ($\le 0.10\text{ Å}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MACE-POLAR-1-M (FT)** | **0.013** | 0.211 | **0.030** | 0.103 | **0.021** | **0.075** | **3 / 5** |
| **MACE-POLAR-1-M (FT Stage2)** | **0.013** | 0.212 | **0.033** | 0.101 | **0.021** | **0.076** | **3 / 5** |
| MACE-MPA-0-Medium (Local) | 0.136 | 0.129 | 0.227 | 0.273 | 0.286 | 0.210 | 0 / 5 |
| MACE-MH-1 (Local) | 0.222 | 0.209 | **0.077** | 0.292 | 0.297 | 0.219 | 1 / 5 |
| MACE-OMAT-0-Medium (Local) | 0.142 | **0.089** | 0.222 | 0.340 | 0.326 | 0.224 | 1 / 5 |
| MACE-POLAR-1-M (pbc, zero-shot) | 0.433 | 0.413 | 0.318 | 0.234 | 0.222 | 0.324 | 0 / 5 |

---

## 5. Exact Reproduction Commands

```bash
# 1. Prepare benchmark manifest & fine-tuning extxyz dataset
/opt/anaconda3/envs/myenv/bin/python scripts/01_prepare_data.py --config config/config_full.yaml
/opt/anaconda3/envs/myenv/bin/python scripts/05_prepare_polar_finetune_data.py --config config/config_full.yaml --source v_sb1_pbe --valid-fraction 0.15 --stride 1 --seed 42

# 2. Fine-tune on HPC (Stage 1 -> Stage 2)
sbatch slurm/finetune_polar_skipjack.slurm
sbatch slurm/finetune_polar_stage2_skipjack.slurm

# 3. Run full 6-model relaxation, RMSD evaluation, and plotting locally
/opt/anaconda3/envs/myenv/bin/python scripts/02_run_grid_relaxation.py --config config/config_full.yaml
/opt/anaconda3/envs/myenv/bin/python scripts/03_evaluate_rmsd.py --config config/config_full.yaml
/opt/anaconda3/envs/myenv/bin/python scripts/04_plot_results.py --config config/config_full.yaml
```
