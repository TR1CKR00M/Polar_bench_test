#!/usr/bin/env python3
"""
03_evaluate_rmsd.py
===================
Evaluates the configurational differences across charge states for the PolarMACE
foundation models and custom downloaded MLIPs, directly implementing the metrics from:
"Multifidelity Machine Learning Interatomic Potentials for Charged Point Defects"
(PRX Energy 5, 043001, 2026, DOI: https://doi.org/10.1103/tzwd-cl3y, Fig. 1a-c).

Usage:
    python scripts/03_evaluate_rmsd.py --config config/config.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from ase import Atoms
from ase.io import read


def read_structure_auto(path_str: str) -> Atoms:
    """Read structure supporting VASP POSCAR_XX/CONTCAR_XX naming as well as extxyz."""
    name = Path(path_str).name
    if name.startswith("POSCAR") or name.startswith("CONTCAR"):
        return read(path_str, format="vasp")
    return read(path_str)


def compute_pbc_rmsd(
    atoms_pred: Atoms,
    atoms_ref: Atoms,
    center_frac: Optional[List[float]] = None,
    local_cutoff: Optional[float] = None,
) -> float:
    """Compute periodic minimum-image RMSD (in Å) after removing net rigid translation."""
    if len(atoms_pred) != len(atoms_ref):
        raise ValueError(
            f"Atom count mismatch: pred has {len(atoms_pred)} atoms, ref has {len(atoms_ref)} atoms."
        )

    cell = atoms_ref.get_cell()
    frac_pred = atoms_pred.get_scaled_positions()
    frac_ref = atoms_ref.get_scaled_positions()

    df = frac_pred - frac_ref
    df -= np.round(df)
    df -= np.mean(df, axis=0, keepdims=True)
    df -= np.round(df)

    d_cart = df @ cell

    if center_frac is not None and local_cutoff is not None:
        c_frac = np.asarray(center_frac, dtype=float)
        rel_f = frac_ref - c_frac
        rel_f -= np.round(rel_f)
        rel_cart = rel_f @ cell
        dists_from_defect = np.linalg.norm(rel_cart, axis=1)
        mask = dists_from_defect <= float(local_cutoff)
        if np.any(mask):
            d_cart = d_cart[mask]

    squared_disp = np.sum(d_cart**2, axis=1)
    return float(np.sqrt(np.mean(squared_disp)))


def compute_max_bond_displacement(atoms_pred: Atoms, atoms_ref: Atoms) -> float:
    """Return the maximum single-atom displacement (in Å) under PBC minimum-image convention."""
    cell = atoms_ref.get_cell()
    df = atoms_pred.get_scaled_positions() - atoms_ref.get_scaled_positions()
    df -= np.round(df)
    df -= np.mean(df, axis=0, keepdims=True)
    df -= np.round(df)
    d_cart = df @ cell
    return float(np.max(np.linalg.norm(d_cart, axis=1)))


def evaluate_all(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    manifest_path = Path(config["dataset"]["manifest_path"])
    rel_index_path = Path(config["relaxation"]["output_dir"]) / "relaxation_index.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(rel_index_path, "r", encoding="utf-8") as f:
        rel_index = json.load(f)

    eval_cfg = config["evaluation"]
    threshold = float(eval_cfg.get("rmsd_threshold", 0.10))
    local_cutoff = float(eval_cfg.get("local_cutoff", 5.0))

    summary_rows: List[Dict[str, Any]] = []
    detailed_rows: List[Dict[str, Any]] = []
    cross_charge_matrices: Dict[str, Dict[str, Any]] = {}

    dft_gs_atoms_map: Dict[int, Atoms] = {}
    vac_coords_map: Dict[int, List[float]] = {}
    for q_str, q_info in manifest["charge_states"].items():
        q = int(q_str)
        vac_coords_map[q] = q_info.get("vacancy_fractional_coords", [0.5, 0.5, 0.5])
        gs_info = q_info.get("dft_ground_state")
        if gs_info and gs_info.get("dft_relaxed_structure_path"):
            dft_gs_atoms_map[q] = read_structure_auto(gs_info["dft_relaxed_structure_path"])

        dft_valid_energies = [
            float(d["dft_energy_eV"])
            for d in q_info["distortions"]
            if d.get("dft_energy_eV") is not None
        ]
        dft_e_min = min(dft_valid_energies) if dft_valid_energies else np.nan

        for d in q_info["distortions"]:
            dft_rel_path = d.get("dft_relaxed_structure_path")
            dft_e = d.get("dft_energy_eV")
            rmsd_dft_vs_gs = np.nan
            if dft_rel_path and q in dft_gs_atoms_map:
                try:
                    d_atoms = read_structure_auto(dft_rel_path)
                    rmsd_dft_vs_gs = compute_pbc_rmsd(d_atoms, dft_gs_atoms_map[q])
                except Exception:
                    pass
            detailed_rows.append(
                {
                    "model_name": "DFT Reference",
                    "model_type": "dft",
                    "pbc_handling": "pbc",
                    "charge": q,
                    "distortion_label": d["distortion_label"],
                    "distortion_percent": d["distortion_percent"],
                    "energy_eV": dft_e if dft_e is not None else np.nan,
                    "rel_energy_eV": (float(dft_e) - dft_e_min) if dft_e is not None else np.nan,
                    "rmsd_to_dft_gs_A": rmsd_dft_vs_gs,
                    "within_threshold": bool(rmsd_dft_vs_gs <= threshold)
                    if not np.isnan(rmsd_dft_vs_gs)
                    else False,
                }
            )

    sorted_qs = sorted(dft_gs_atoms_map.keys())
    if len(sorted_qs) > 1:
        mat = np.zeros((len(sorted_qs), len(sorted_qs)), dtype=float)
        for i, qi in enumerate(sorted_qs):
            for j, qj in enumerate(sorted_qs):
                mat[i, j] = compute_pbc_rmsd(dft_gs_atoms_map[qi], dft_gs_atoms_map[qj])
        cross_charge_matrices["DFT Reference"] = {
            "charges": sorted_qs,
            "rmsd_matrix_A": mat.tolist(),
        }

    for model_name, m_data in rel_index["models"].items():
        mlip_gs_atoms_map: Dict[int, Atoms] = {}

        for q_str, q_rel in m_data["charge_states"].items():
            q = int(q_str)
            dft_gs_atoms = dft_gs_atoms_map.get(q)
            vac_center = vac_coords_map.get(q, [0.5, 0.5, 0.5])

            valid_mlip_energies = [
                float(d["mlip_energy_eV"])
                for d in q_rel["distortions"]
                if d.get("mlip_energy_eV") is not None and not np.isnan(d["mlip_energy_eV"])
            ]
            mlip_e_min = min(valid_mlip_energies) if valid_mlip_energies else np.nan

            for d in q_rel["distortions"]:
                mlip_path = d["mlip_relaxed_structure_path"]
                mlip_atoms = read(mlip_path)
                rmsd_val = (
                    compute_pbc_rmsd(mlip_atoms, dft_gs_atoms) if dft_gs_atoms is not None else np.nan
                )
                local_rmsd_val = (
                    compute_pbc_rmsd(
                        mlip_atoms, dft_gs_atoms, center_frac=vac_center, local_cutoff=local_cutoff
                    )
                    if dft_gs_atoms is not None
                    else np.nan
                )
                e_val = float(d["mlip_energy_eV"])

                detailed_rows.append(
                    {
                        "model_name": model_name,
                        "model_type": m_data["model_type"],
                        "pbc_handling": m_data.get("pbc_handling", "N/A"),
                        "charge": q,
                        "distortion_label": d["distortion_label"],
                        "distortion_percent": d["distortion_percent"],
                        "energy_eV": e_val,
                        "rel_energy_eV": e_val - mlip_e_min if not np.isnan(mlip_e_min) else np.nan,
                        "rmsd_to_dft_gs_A": rmsd_val,
                        "local_rmsd_to_dft_gs_A": local_rmsd_val,
                        "within_threshold": bool(rmsd_val <= threshold) if not np.isnan(rmsd_val) else False,
                    }
                )

            mlip_gs = q_rel.get("mlip_ground_state")
            if mlip_gs and mlip_gs.get("mlip_relaxed_structure_path"):
                gs_atoms = read(mlip_gs["mlip_relaxed_structure_path"])
                mlip_gs_atoms_map[q] = gs_atoms

                if dft_gs_atoms is not None:
                    global_rmsd = compute_pbc_rmsd(gs_atoms, dft_gs_atoms)
                    local_rmsd = compute_pbc_rmsd(
                        gs_atoms, dft_gs_atoms, center_frac=vac_center, local_cutoff=local_cutoff
                    )
                    max_disp = compute_max_bond_displacement(gs_atoms, dft_gs_atoms)
                else:
                    global_rmsd = np.nan
                    local_rmsd = np.nan
                    max_disp = np.nan

                total_monopole_q = (
                    float(np.sum(gs_atoms.arrays["polar_monopole_charge"]))
                    if "polar_monopole_charge" in gs_atoms.arrays
                    else np.nan
                )

                summary_rows.append(
                    {
                        "model_name": model_name,
                        "model_type": m_data["model_type"],
                        "pbc_handling": m_data.get("pbc_handling", "N/A"),
                        "charge": q,
                        "predicted_gs_distortion": mlip_gs["distortion_label"],
                        "mlip_gs_energy_eV": float(mlip_gs["mlip_energy_eV"]),
                        "rmsd_A": global_rmsd,
                        "local_rmsd_A": local_rmsd,
                        "max_atom_disp_A": max_disp,
                        "identified_ground_state": bool(global_rmsd <= threshold)
                        if not np.isnan(global_rmsd)
                        else False,
                        "polar_monopole_sum": total_monopole_q,
                    }
                )

        m_qs = sorted(mlip_gs_atoms_map.keys())
        if len(m_qs) > 1:
            mat = np.zeros((len(m_qs), len(m_qs)), dtype=float)
            for i, qi in enumerate(m_qs):
                for j, qj in enumerate(m_qs):
                    mat[i, j] = compute_pbc_rmsd(mlip_gs_atoms_map[qi], mlip_gs_atoms_map[qj])
            cross_charge_matrices[model_name] = {
                "charges": m_qs,
                "rmsd_matrix_A": mat.tolist(),
            }

    df_summary = pd.DataFrame(summary_rows)
    df_detailed = pd.DataFrame(detailed_rows)

    summary_csv = Path(eval_cfg["summary_csv"])
    detailed_csv = Path(eval_cfg["detailed_csv"])
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(summary_csv, index=False)
    df_detailed.to_csv(detailed_csv, index=False)

    matrix_json = summary_csv.parent / "cross_charge_rmsd_matrices.json"
    with open(matrix_json, "w", encoding="utf-8") as f:
        json.dump(cross_charge_matrices, f, indent=2)

    print(f"[INFO] Saved Fig. 1a ground-state RMSD summary table to: {summary_csv}")
    print(f"[INFO] Saved all-distortion PES & RMSD table to:        {detailed_csv}")
    print(f"[INFO] Saved cross-charge RMSD matrices to:             {matrix_json}\n")
    if not df_summary.empty:
        print("=== Ground-State RMSD Summary (PRX Energy Fig. 1a Metric) ===")
        pivot = df_summary.pivot(index="model_name", columns="charge", values="rmsd_A")
        print(pivot.to_string(float_format=lambda x: f"{x:.3f} Å"))

    return df_summary, df_detailed, cross_charge_matrices


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ground-state and cross-charge RMSD metrics.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    evaluate_all(config)


if __name__ == "__main__":
    main()
