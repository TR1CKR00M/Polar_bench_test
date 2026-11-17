#!/usr/bin/env python3
"""
01_prepare_data.py
==================
Downloads (or extracts) the Zenodo dataset (Record 22756145) supporting:
"Multifidelity Machine Learning Interatomic Potentials for Charged Point Defects"
(Xinwei Wang, Irea Mosquera-Lois, and Aron Walsh, PRX Energy 5, 043001, 2026,
DOI: https://doi.org/10.1103/tzwd-cl3y).

Builds a unified JSON manifest (`data/manifest.json`) indexing for each charge state `q`:
1. All ShakeNBreak initial bond-distorted structures (e.g., Rattled, D_XX / Bond_Distortion_X%, Unperturbed).
2. The DFT reference relaxed structures and final energies for each distortion.
3. The DFT reference global minimum (ground-state) structure for each charge state.

Usage:
    python scripts/01_prepare_data.py --config config/config.yaml [--download]
"""

import argparse
import json
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from ase import Atoms
from ase.io import read, write


def read_structure_auto(path_str: str) -> Atoms:
    """Read structure supporting VASP POSCAR_XX/CONTCAR_XX naming as well as extxyz."""
    name = Path(path_str).name
    if name.startswith("POSCAR") or name.startswith("CONTCAR"):
        return read(path_str, format="vasp")
    return read(path_str)


def download_and_extract_zenodo(url: str, raw_dir: Path) -> None:
    """Download data.zip from Zenodo record 22756145 and extract it."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "data.zip"
    if not zip_path.exists():
        print(f"[INFO] Downloading Zenodo dataset from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"[INFO] Saved archive to {zip_path}")
    else:
        print(f"[INFO] Using existing archive at {zip_path}")

    print(f"[INFO] Extracting {zip_path} into {raw_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    print("[INFO] Extraction complete.")


def parse_charge_from_path(path_str: str) -> Optional[int]:
    """Infer integer charge state q from folder/file naming conventions (including mi1/mi2/mi3/neu/pl1)."""
    parts = Path(path_str).parts
    for part in reversed(parts):
        low = part.lower()
        if low == "neu":
            return 0
        m_mi = re.fullmatch(r"mi(\d+)", low)
        if m_mi:
            return -int(m_mi.group(1))
        m_pl = re.fullmatch(r"pl(\d+)", low)
        if m_pl:
            return int(m_pl.group(1))

    patterns = [
        r"chg_([+-]?\d+)",
        r"charge_([+-]?\d+)",
        r"q_([+-]?\d+)",
        r"q=([+-]?\d+)",
        r"_([+-]?\d+)$",
        r"/([+-]?\d+)/",
    ]
    for pat in patterns:
        m = re.search(pat, path_str)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def parse_distortion_label(path_str: str) -> Tuple[str, float]:
    """
    Extract ShakeNBreak distortion label and numeric distortion percentage.
    Convention:
      - 'Bond_Distortion_-30.0%' or 'D_-30' -> ('Bond_Distortion_-30.0%', -30.0)
      - 'Unperturbed' -> ('Unperturbed', 0.0)
      - 'Rattled' -> ('Rattled', 999.0)
    """
    m = re.search(r"Bond_Distortion_([+-]?\d+(?:\.\d+)?)%?", path_str, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return f"Bond_Distortion_{val:.1f}%", val
    m_d = re.search(r"(?:^|/)D_([+-]?\d+(?:\.\d+)?)(?:/|$)", path_str)
    if m_d:
        val = float(m_d.group(1))
        return f"Bond_Distortion_{val:.1f}%", val
    if "unperturbed" in path_str.lower():
        return "Unperturbed", 0.0
    if "rattled" in path_str.lower():
        return "Rattled", 999.0
    return Path(path_str).stem, 0.0


def extract_dft_energy(atoms: Atoms, directory: Path) -> Optional[float]:
    """Retrieve DFT total energy (eV) from ASE Atoms info/calc or VASP OUTCAR*/OSZICAR*."""
    for key in ["REF_energy", "energy", "free_energy", "dft_energy", "total_energy"]:
        if key in atoms.info:
            return float(atoms.info[key])
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        pass

    for oszicar in sorted(directory.glob("OSZICAR*")):
        text = oszicar.read_text(errors="ignore")
        matches = re.findall(r"E0=\s*([+-]?\d+\.\d+E[+-]?\d+)", text)
        if matches:
            return float(matches[-1])

    for outcar in sorted(directory.glob("OUTCAR*")):
        text = outcar.read_text(errors="ignore")
        matches = re.findall(r"free  energy   TOTEN  =\s*([+-]?\d+\.\d+)", text)
        if matches:
            return float(matches[-1])

    return None


def estimate_vacancy_fractional_coords(defect_atoms: Atoms, bulk_atoms: Optional[Atoms] = None) -> List[float]:
    """Estimate the fractional coordinates of the vacancy site."""
    if "defect_center" in defect_atoms.info:
        val = defect_atoms.info["defect_center"]
        if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 3:
            return [float(x) for x in val]

    if bulk_atoms is not None and len(bulk_atoms) == len(defect_atoms) + 1:
        bulk_scaled = bulk_atoms.get_scaled_positions()
        def_scaled = defect_atoms.get_scaled_positions()
        min_dists = []
        for b_pos in bulk_scaled:
            diff = def_scaled - b_pos
            diff -= np.round(diff)
            cart_diff = diff @ bulk_atoms.get_cell()
            dists = np.linalg.norm(cart_diff, axis=1)
            min_dists.append(np.min(dists))
        vac_idx = int(np.argmax(min_dists))
        return [float(x) for x in bulk_scaled[vac_idx]]

    return [0.5, 0.5, 0.5]


def scan_extxyz_files(subset_dir: Path, target_charges: List[int]) -> Dict[int, List[Dict[str, Any]]]:
    """Parse extended XYZ (.xyz / .extxyz) files in the benchmark directory."""
    records_by_q: Dict[int, List[Dict[str, Any]]] = {q: [] for q in target_charges}
    xyz_files = sorted(list(subset_dir.rglob("*.xyz")) + list(subset_dir.rglob("*.extxyz")))
    if not xyz_files:
        return records_by_q

    parsed_dir = subset_dir / "_parsed_structures"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    for xyz_file in xyz_files:
        if "_parsed_structures" in str(xyz_file):
            continue
        try:
            frames = read(str(xyz_file), index=":")
        except Exception as exc:
            print(f"[WARN] Could not read {xyz_file}: {exc}")
            continue

        for idx, atoms in enumerate(frames):
            q = atoms.info.get("charge", atoms.info.get("total_charge", parse_charge_from_path(str(xyz_file))))
            if q is None or int(q) not in records_by_q:
                continue
            q = int(q)

            dist_label = str(
                atoms.info.get(
                    "distortion",
                    atoms.info.get("config_type", parse_distortion_label(str(xyz_file))[0]),
                )
            )
            _, dist_pct = parse_distortion_label(dist_label)
            stage = str(atoms.info.get("stage", "relaxed" if "relax" in xyz_file.name.lower() else "frame"))
            energy = extract_dft_energy(atoms, xyz_file.parent)

            out_xyz = parsed_dir / f"q_{q:+d}_{xyz_file.stem}_f{idx:04d}.extxyz"
            write(str(out_xyz), atoms, format="extxyz")

            records_by_q[q].append(
                {
                    "charge": q,
                    "distortion_label": dist_label,
                    "distortion_percent": dist_pct,
                    "stage": stage,
                    "structure_path": str(out_xyz.resolve()),
                    "dft_energy_eV": energy,
                    "source_file": str(xyz_file.resolve()),
                }
            )
    return records_by_q


def scan_vasp_directories(subset_dir: Path, target_charges: List[int]) -> Dict[int, List[Dict[str, Any]]]:
    """Scan ShakeNBreak / VASP directory trees containing POSCAR* (initial) and CONTCAR* (DFT-relaxed)."""
    records_by_q: Dict[int, List[Dict[str, Any]]] = {q: [] for q in target_charges}
    poscars = sorted(subset_dir.rglob("POSCAR*"))
    for poscar in poscars:
        parent = poscar.parent
        q = parse_charge_from_path(str(parent))
        if q is None or q not in records_by_q:
            continue

        dist_label, dist_pct = parse_distortion_label(str(parent))
        contcar_candidates = sorted(parent.glob("CONTCAR*"))
        contcar = contcar_candidates[0] if contcar_candidates else None
        dft_relaxed_path = str(contcar.resolve()) if contcar is not None else None
        dft_energy = None
        if contcar is not None:
            try:
                relaxed_atoms = read(str(contcar), format="vasp")
                dft_energy = extract_dft_energy(relaxed_atoms, parent)
            except Exception as exc:
                print(f"[WARN] Failed reading {contcar}: {exc}")

        records_by_q[q].append(
            {
                "charge": q,
                "distortion_label": dist_label,
                "distortion_percent": dist_pct,
                "initial_structure_path": str(poscar.resolve()),
                "dft_relaxed_structure_path": dft_relaxed_path,
                "dft_energy_eV": dft_energy,
                "source_dir": str(parent.resolve()),
            }
        )
    return records_by_q


def build_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build structured JSON manifest for downstream relaxation, RMSD evaluation, and plotting."""
    ds_cfg = config["dataset"]
    raw_dir = Path(ds_cfg["raw_data_dir"])
    subset_name = ds_cfg["benchmark_subset"]
    target_charges = [int(q) for q in ds_cfg["charge_states"]]
    spin_map = {int(k): int(v) for k, v in ds_cfg.get("spin_map", {}).items()}

    candidates = list(raw_dir.rglob(subset_name))
    subset_dir = candidates[0] if candidates else raw_dir
    print(f"[INFO] Scanning dataset directory: {subset_dir}")

    vasp_records = scan_vasp_directories(subset_dir, target_charges)
    xyz_records = scan_extxyz_files(subset_dir, target_charges)

    manifest: Dict[str, Any] = {
        "metadata": {
            "zenodo_record": ds_cfg["zenodo_record_id"],
            "paper_doi": "10.1103/tzwd-cl3y",
            "subset": subset_name,
            "defect_name": ds_cfg.get("defect_name", "V_Sb(1)"),
        },
        "charge_states": {},
    }

    for q in target_charges:
        entries: List[Dict[str, Any]] = []

        for item in vasp_records.get(q, []):
            entries.append(
                {
                    "distortion_label": item["distortion_label"],
                    "distortion_percent": item["distortion_percent"],
                    "initial_structure_path": item["initial_structure_path"],
                    "dft_relaxed_structure_path": item["dft_relaxed_structure_path"],
                    "dft_energy_eV": item["dft_energy_eV"],
                }
            )

        if not entries and xyz_records.get(q):
            by_dist: Dict[str, Dict[str, Any]] = {}
            for item in xyz_records[q]:
                lbl = item["distortion_label"]
                by_dist.setdefault(
                    lbl,
                    {
                        "distortion_label": lbl,
                        "distortion_percent": item["distortion_percent"],
                        "initial_structure_path": item["structure_path"],
                        "dft_relaxed_structure_path": item["structure_path"],
                        "dft_energy_eV": item["dft_energy_eV"],
                    },
                )
                if "init" in item["stage"].lower() or "unrelax" in item["stage"].lower():
                    by_dist[lbl]["initial_structure_path"] = item["structure_path"]
                else:
                    by_dist[lbl]["dft_relaxed_structure_path"] = item["structure_path"]
                    if item["dft_energy_eV"] is not None:
                        by_dist[lbl]["dft_energy_eV"] = item["dft_energy_eV"]
            entries = list(by_dist.values())

        entries.sort(key=lambda x: x["distortion_percent"])

        valid_dft = [
            e for e in entries if e.get("dft_energy_eV") is not None and e.get("dft_relaxed_structure_path")
        ]
        if valid_dft:
            dft_gs = min(valid_dft, key=lambda x: float(x["dft_energy_eV"]))
        elif entries and entries[0].get("dft_relaxed_structure_path"):
            dft_gs = entries[0]
        else:
            dft_gs = None

        vac_coords = [0.5, 0.5, 0.5]
        if dft_gs and dft_gs.get("dft_relaxed_structure_path"):
            try:
                gs_atoms = read_structure_auto(dft_gs["dft_relaxed_structure_path"])
                vac_coords = estimate_vacancy_fractional_coords(gs_atoms)
            except Exception:
                pass

        manifest["charge_states"][str(q)] = {
            "charge": q,
            "spin_multiplicity": spin_map.get(q, 1),
            "vacancy_fractional_coords": vac_coords,
            "dft_ground_state": dft_gs,
            "distortions": entries,
        }

    if "-3" in manifest["charge_states"] and "0" in manifest["charge_states"]:
        q_m3 = manifest["charge_states"]["-3"]
        q_0 = manifest["charge_states"]["0"]
        if len(q_m3["distortions"]) <= 1 and len(q_0["distortions"]) > 1:
            print(
                "[INFO] Copying initial ShakeNBreak bond-distorted structures from q=0 to q=-3 "
                "(following PRX Energy 10.1103/tzwd-cl3y protocol)."
            )
            existing_labels = {d["distortion_label"] for d in q_m3["distortions"]}
            for d0 in q_0["distortions"]:
                if d0["distortion_label"] not in existing_labels:
                    q_m3["distortions"].append(
                        {
                            "distortion_label": d0["distortion_label"],
                            "distortion_percent": d0["distortion_percent"],
                            "initial_structure_path": d0["initial_structure_path"],
                            "dft_relaxed_structure_path": None,
                            "dft_energy_eV": None,
                        }
                    )
            q_m3["distortions"].sort(key=lambda x: x["distortion_percent"])

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Zenodo 22756145 charged defect dataset manifest.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--download", action="store_true", help="Download and extract data.zip from Zenodo")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.download:
        download_and_extract_zenodo(
            url=config["dataset"]["zenodo_url"],
            raw_dir=Path(config["dataset"]["raw_data_dir"]),
        )

    manifest = build_manifest(config)
    manifest_path = Path(config["dataset"]["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved dataset manifest to {manifest_path}")
    for q_str, q_info in manifest["charge_states"].items():
        n_dist = len(q_info["distortions"])
        gs = q_info["dft_ground_state"]
        gs_lbl = gs["distortion_label"] if gs else "None"
        gs_e = f"{gs['dft_energy_eV']:.4f} eV" if (gs and gs.get("dft_energy_eV") is not None) else "N/A"
        print(f"  Charge q = {int(q_str):+d}: {n_dist:2d} initial distortions | DFT Ground State: {gs_lbl} ({gs_e})")


if __name__ == "__main__":
    main()
