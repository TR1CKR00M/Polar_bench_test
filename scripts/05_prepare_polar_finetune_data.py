#!/usr/bin/env python3
"""
05_prepare_polar_finetune_data.py
=================================
Prepares extended XYZ (`train.xyz` and `valid.xyz`) datasets for fine-tuning
`MACE-POLAR-1-M` (`PolarMACE`) on charged point-defect relaxation trajectories.

Supported dataset sources:
  1. `v_sb1_pbe` (default): Extracts all ionic relaxation frames (energy + forces)
     from VASP `OUTCAR*` files in `data/raw/1_FM_benchmark/V_Sb1/DFT_PBE/`.
     Splits by distortion trajectory (not frame-by-frame) to avoid data leakage
     across correlated relaxation steps.
  2. `v_se1_hse06`: Converts `data/raw/2_global_embedding/Charge_embedded_MACE/`
     (`train.xyz` / `test.xyz`) by remapping categorical `total_charge` class
     indices `{0, 1, 2, 3, 4}` to physical charges `{-2, -1, 0, +1, +2}`.

Each output frame contains:
  - `info["REF_energy"]`: DFT total energy (eV)
  - `arrays["REF_forces"]`: DFT atomic forces (eV/A)
  - `info["total_charge"]` & `info["charge"]`: integer defect charge state
  - `info["total_spin"]` & `info["spin"]`: spin multiplicity (2S + 1)
  - `info["external_field"]`: [0.0, 0.0, 0.0]

Usage:
    python scripts/05_prepare_polar_finetune_data.py --config config/config.yaml
    python scripts/05_prepare_polar_finetune_data.py --source v_sb1_pbc --valid-fraction 0.15 --stride 2
"""

import argparse
import collections
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from ase import Atoms
from ase.io import read, write


DEFAULT_SPIN_MAP = {
    -3: 1,
    -2: 2,
    -1: 1,
    0: 2,
    1: 1,
    2: 2,
}

# Categorical embedding index -> physical charge for V_Se1 HSE06 dataset
V_SE1_CLASS_TO_CHARGE = {
    0: -2,
    1: -1,
    2: 0,
    3: 1,
    4: 2,
}


def parse_charge_from_folder(folder_name: str) -> Optional[int]:
    """Infer integer charge state from folder naming (mi3/mi2/mi1/neu/pl1/pl2)."""
    low = folder_name.lower()
    if low == "neu":
        return 0
    m_mi = re.fullmatch(r"mi(\d+)", low)
    if m_mi:
        return -int(m_mi.group(1))
    m_pl = re.fullmatch(r"pl(\d+)", low)
    if m_pl:
        return int(m_pl.group(1))
    return None


def prepare_frame_metadata(
    atoms: Atoms,
    charge: int,
    spin_mult: int,
    distortion_label: str,
    keep_stress: bool = False,
) -> Optional[Atoms]:
    """Attach standardized REF_energy, REF_forces, charge, spin, and external_field."""
    out = atoms.copy()
    out.pbc = True

    energy = None
    for key in ("REF_energy", "energy", "free_energy"):
        if key in out.info:
            energy = float(out.info[key])
            break
    if energy is None:
        try:
            energy = float(atoms.get_potential_energy())
        except Exception:
            return None

    forces = None
    if "REF_forces" in out.arrays:
        forces = np.asarray(out.arrays["REF_forces"], dtype=float)
    else:
        try:
            forces = np.asarray(atoms.get_forces(), dtype=float)
        except Exception:
            return None

    stress = None
    if keep_stress:
        if "REF_stress" in out.info:
            stress = np.asarray(out.info["REF_stress"], dtype=float)
        else:
            try:
                stress = np.asarray(atoms.get_stress(voigt=True), dtype=float)
            except Exception:
                stress = None

    out.calc = None
    out.info = {
        "REF_energy": float(energy),
        "charge": int(charge),
        "total_charge": int(charge),
        "spin": int(spin_mult),
        "total_spin": int(spin_mult),
        "external_field": [0.0, 0.0, 0.0],
        "distortion_label": str(distortion_label),
    }
    if stress is not None and stress.shape == (6,):
        out.info["REF_stress"] = stress.tolist()

    # Keep only standard arrays + REF_forces
    for key in list(out.arrays.keys()):
        if key not in ("numbers", "positions"):
            del out.arrays[key]
    out.arrays["REF_forces"] = forces
    return out


def build_v_sb1_pbe_dataset(
    raw_dir: Path,
    spin_map: Dict[int, int],
    valid_fraction: float,
    stride: int,
    seed: int,
    max_force_norm: float,
    keep_stress: bool,
) -> Tuple[List[Atoms], List[Atoms], Dict[str, Any]]:
    """Extract V_Sb1 DFT-PBE OUTCAR trajectories and split by trajectory."""
    pbe_root = raw_dir / "1_FM_benchmark" / "V_Sb1" / "DFT_PBE"
    if not pbe_root.exists():
        raise FileNotFoundError(f"DFT_PBE directory not found at {pbe_root}")

    trajectories_by_q: Dict[int, List[Tuple[str, List[Atoms]]]] = collections.defaultdict(list)
    skipped_high_force = 0
    total_raw_frames = 0

    for q_dir in sorted(pbe_root.iterdir()):
        if not q_dir.is_dir():
            continue
        q = parse_charge_from_folder(q_dir.name)
        if q is None:
            continue
        spin_mult = int(spin_map.get(q, DEFAULT_SPIN_MAP.get(q, 1)))

        for dist_dir in sorted(q_dir.iterdir()):
            if not dist_dir.is_dir():
                continue
            outcars = sorted(dist_dir.glob("OUTCAR*"))
            if not outcars:
                continue

            traj_frames: List[Atoms] = []
            for outcar in outcars:
                try:
                    raw_frames = read(str(outcar), index=":")
                except Exception as exc:
                    print(f"[WARN] Failed to read {outcar}: {exc}")
                    continue
                if isinstance(raw_frames, Atoms):
                    raw_frames = [raw_frames]

                for idx, raw_atoms in enumerate(raw_frames):
                    total_raw_frames += 1
                    if stride > 1 and (idx % stride != 0) and (idx != len(raw_frames) - 1):
                        continue
                    prepared = prepare_frame_metadata(
                        raw_atoms,
                        charge=q,
                        spin_mult=spin_mult,
                        distortion_label=f"q{q:+d}_{dist_dir.name}",
                        keep_stress=keep_stress,
                    )
                    if prepared is None:
                        continue
                    fmax = float(np.linalg.norm(prepared.arrays["REF_forces"], axis=1).max())
                    if fmax > max_force_norm:
                        skipped_high_force += 1
                        continue
                    traj_frames.append(prepared)

            if traj_frames:
                trajectories_by_q[q].append((dist_dir.name, traj_frames))

    rng = random.Random(seed)
    train_frames: List[Atoms] = []
    valid_frames: List[Atoms] = []
    summary_by_q: Dict[int, Dict[str, int]] = {}

    for q in sorted(trajectories_by_q.keys()):
        trajs = list(trajectories_by_q[q])
        rng.shuffle(trajs)
        n_valid_traj = max(1, int(round(len(trajs) * valid_fraction))) if len(trajs) > 1 else 0
        valid_trajs = trajs[:n_valid_traj]
        train_trajs = trajs[n_valid_traj:]

        q_train = [frame for _, frames in train_trajs for frame in frames]
        q_valid = [frame for _, frames in valid_trajs for frame in frames]
        train_frames.extend(q_train)
        valid_frames.extend(q_valid)

        summary_by_q[q] = {
            "train_trajectories": len(train_trajs),
            "valid_trajectories": len(valid_trajs),
            "train_frames": len(q_train),
            "valid_frames": len(q_valid),
        }

    rng.shuffle(train_frames)
    stats = {
        "source": "v_sb1_pbe",
        "total_raw_frames": total_raw_frames,
        "skipped_high_force": skipped_high_force,
        "train_frames": len(train_frames),
        "valid_frames": len(valid_frames),
        "by_charge": summary_by_q,
    }
    return train_frames, valid_frames, stats


def build_v_se1_hse06_dataset(
    raw_dir: Path,
    spin_map: Dict[int, int],
    keep_stress: bool,
) -> Tuple[List[Atoms], List[Atoms], Dict[str, Any]]:
    """Convert V_Se1 HSE06 xyz files by remapping categorical charge class to physical charge."""
    ce_dir = raw_dir / "2_global_embedding" / "Charge_embedded_MACE"
    train_src = ce_dir / "train.xyz"
    test_src = ce_dir / "test.xyz"
    if not train_src.exists() or not test_src.exists():
        raise FileNotFoundError(f"Expected train.xyz and test.xyz under {ce_dir}")

    def _convert(src_path: Path) -> List[Atoms]:
        frames = read(str(src_path), index=":")
        converted: List[Atoms] = []
        for idx, atoms in enumerate(frames):
            cls_idx = int(atoms.info.get("total_charge", 2))
            q = V_SE1_CLASS_TO_CHARGE.get(cls_idx, cls_idx)
            spin_mult = int(spin_map.get(q, DEFAULT_SPIN_MAP.get(q, 1)))
            prepared = prepare_frame_metadata(
                atoms,
                charge=q,
                spin_mult=spin_mult,
                distortion_label=f"v_se1_q{q:+d}_{idx:04d}",
                keep_stress=keep_stress,
            )
            if prepared is not None:
                converted.append(prepared)
        return converted

    train_frames = _convert(train_src)
    valid_frames = _convert(test_src)
    stats = {
        "source": "v_se1_hse06",
        "train_frames": len(train_frames),
        "valid_frames": len(valid_frames),
    }
    return train_frames, valid_frames, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare extxyz datasets for PolarMACE (MACE-POLAR-1-M) fine-tuning."
    )
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--source",
        type=str,
        default="v_sb1_pbe",
        choices=["v_sb1_pbe", "v_se1_hse06"],
        help="Raw dataset source to convert.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to data/finetune/<source>).",
    )
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.15,
        help="Fraction of distortion trajectories held out for validation (v_sb1_pbe).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Subsampling stride within each OUTCAR trajectory (always keeps final frame).",
    )
    parser.add_argument(
        "--max-force-norm",
        type=float,
        default=50.0,
        help="Discard extreme frames with max atomic force norm above this threshold (eV/A).",
    )
    parser.add_argument(
        "--keep-stress",
        action="store_true",
        help="Preserve REF_stress in output extxyz if available.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trajectory split.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["dataset"]["raw_data_dir"])
    spin_map = {int(k): int(v) for k, v in config["dataset"].get("spin_map", {}).items()}

    out_dir = Path(args.output_dir) if args.output_dir else Path("data/finetune") / args.source
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "v_sb1_pbe":
        train_frames, valid_frames, stats = build_v_sb1_pbe_dataset(
            raw_dir=raw_dir,
            spin_map=spin_map,
            valid_fraction=args.valid_fraction,
            stride=args.stride,
            seed=args.seed,
            max_force_norm=args.max_force_norm,
            keep_stress=args.keep_stress,
        )
    else:
        train_frames, valid_frames, stats = build_v_se1_hse06_dataset(
            raw_dir=raw_dir,
            spin_map=spin_map,
            keep_stress=args.keep_stress,
        )

    train_path = out_dir / "train.xyz"
    valid_path = out_dir / "valid.xyz"
    write(str(train_path), train_frames, format="extxyz")
    write(str(valid_path), valid_frames, format="extxyz")

    print(f"[INFO] Prepared PolarMACE fine-tuning dataset ({args.source}):")
    print(f"  Train: {train_path} ({len(train_frames)} frames)")
    print(f"  Valid: {valid_path} ({len(valid_frames)} frames)")
    if "skipped_high_force" in stats:
        print(f"  Skipped high-force frames (>{args.max_force_norm} eV/A): {stats['skipped_high_force']}")
    if "by_charge" in stats:
        for q, q_stats in stats["by_charge"].items():
            print(
                f"  Charge q={q:+d}: "
                f"{q_stats['train_trajectories']} train / {q_stats['valid_trajectories']} valid trajs "
                f"({q_stats['train_frames']} / {q_stats['valid_frames']} frames)"
            )


if __name__ == "__main__":
    main()
