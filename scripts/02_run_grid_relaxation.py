#!/usr/bin/env python3
"""
02_run_grid_relaxation.py
=========================
Executes the Grid Study across:
  (Models: PolarMACE foundation model & local downloaded MACE checkpoints)
    x (Charge states q: -3, -2, -1, 0, +1)
    x (Selected ShakeNBreak initial bond-distorted configurations, e.g. 5 per charge state)

Usage:
    python scripts/02_run_grid_relaxation.py --config config/config.yaml
    python scripts/02_run_grid_relaxation.py --config config/config.yaml --models "MACE-POLAR-1-M (pbc)" --charges 0 -3 --max-steps 15
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, FIRE, LBFGS


OPTIMIZER_MAP = {
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "FIRE": FIRE,
}


def read_structure_auto(path_str: str) -> Atoms:
    """Read structure supporting VASP POSCAR_XX/CONTCAR_XX naming as well as extxyz."""
    name = Path(path_str).name
    if name.startswith("POSCAR") or name.startswith("CONTCAR"):
        return read(path_str, format="vasp")
    return read(path_str)


def select_distortions(
    distortions: List[Dict[str, Any]],
    selected_pcts: Optional[List[float]] = None,
    max_distortions: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filter distortions by target percentages or cap at max_distortions."""
    if selected_pcts:
        target_set = {float(p) for p in selected_pcts}
        filtered = [
            d for d in distortions if float(d.get("distortion_percent", 999.0)) in target_set
        ]
        if filtered:
            distortions = filtered
    if max_distortions is not None and len(distortions) > max_distortions:
        distortions = distortions[:max_distortions]
    return distortions


def build_calculator(model_cfg: Dict[str, Any]):
    """Instantiate an ASE calculator for PolarMACE foundation models or local MACE checkpoints."""
    mtype = model_cfg["type"].lower()
    model_path = model_cfg["model_path"]
    device = model_cfg.get("device", "cpu")
    default_dtype = model_cfg.get("default_dtype", "float64")

    if mtype == "mace_polar":
        import inspect
        from mace.calculators import mace_polar
        from graph_longrange.features import GTOElectrostaticFeatures
        from graph_longrange.energy import GTOElectrostaticEnergy

        if "force_pbc_evaluator" not in inspect.signature(GTOElectrostaticFeatures.precompute_geometry).parameters:
            _orig_precompute = GTOElectrostaticFeatures.precompute_geometry

            def _patched_precompute(self, *args, force_pbc_evaluator=None, **kwargs):
                return _orig_precompute(self, *args, **kwargs)

            GTOElectrostaticFeatures.precompute_geometry = _patched_precompute

        if "pbc" not in inspect.signature(GTOElectrostaticFeatures.forward_dynamic).parameters:
            _orig_forward_dynamic = GTOElectrostaticFeatures.forward_dynamic

            def _patched_forward_dynamic(self, *args, pbc=None, **kwargs):
                if "source_feats" in kwargs and hasattr(kwargs["source_feats"], "dim"):
                    sf = kwargs["source_feats"]
                    if sf.dim() == 3 and sf.size(1) == 1:
                        kwargs["source_feats"] = sf.squeeze(1)
                elif len(args) >= 2 and hasattr(args[1], "dim"):
                    sf = args[1]
                    if sf.dim() == 3 and sf.size(1) == 1:
                        args = (args[0], sf.squeeze(1), *args[2:])
                return _orig_forward_dynamic(self, *args, **kwargs)

            GTOElectrostaticFeatures.forward_dynamic = _patched_forward_dynamic

        if "force_pbc_evaluator" not in inspect.signature(GTOElectrostaticEnergy.forward).parameters:
            _orig_energy_forward = GTOElectrostaticEnergy.forward

            def _patched_energy_forward(self, *args, force_pbc_evaluator=None, **kwargs):
                return _orig_energy_forward(self, *args, **kwargs)

            GTOElectrostaticEnergy.forward = _patched_energy_forward

        pbc_handling = model_cfg.get("pbc_handling", "auto")
        calc = mace_polar(
            model=model_path,
            device=device,
            default_dtype=default_dtype,
            pbc_handling=pbc_handling,
        )
        # Re-initialize runtime dispatch pointers (_precompute_geometry_impl, _forward_dynamic_impl, etc.)
        # for pickled checkpoints serialized with an earlier graph_longrange version.
        for m in getattr(calc, "models", []):
            for sub in m.modules():
                if hasattr(sub, "set_pbc_handling"):
                    sub.set_pbc_handling(pbc_handling)
                if hasattr(sub, "features_irreps") and not hasattr(sub, "features_dim"):
                    sub.features_dim = sub.features_irreps.dim
        return calc

    if mtype in ("mace_mp", "mace_custom", "mace_custom_charge"):
        from mace.calculators import MACECalculator, mace_mp

        head = model_cfg.get("head", None)
        if Path(model_path).exists():
            kwargs = {
                "model_paths": model_path,
                "device": device,
                "default_dtype": default_dtype,
            }
            if head:
                kwargs["head"] = head
            return MACECalculator(**kwargs)
        else:
            kwargs = {
                "model": model_path,
                "device": device,
                "default_dtype": default_dtype,
            }
            if head:
                kwargs["head"] = head
            return mace_mp(**kwargs)

    if mtype == "mattersim":
        from mattersim.forcefield import MatterSimCalculator

        return MatterSimCalculator(load_path=model_path, device=device)

    if mtype == "grace":
        from tensorpotential.calculator import grace_fm

        return grace_fm(model_path)

    raise ValueError(f"Unsupported model type '{mtype}' in config.yaml")


def attach_charge_and_spin_info(
    atoms: Atoms,
    charge: int,
    spin_multiplicity: int,
    model_cfg: Dict[str, Any],
) -> None:
    """Attach charge, spin, and external electric field metadata to `atoms.info`."""
    atoms.info["charge"] = int(charge)
    atoms.info["total_charge"] = int(charge)
    atoms.info["spin"] = int(spin_multiplicity)
    atoms.info["external_field"] = [0.0, 0.0, 0.0]

    custom_key = model_cfg.get("charge_key")
    if custom_key:
        atoms.info[custom_key] = int(charge)


def extract_polar_multipoles(calc, atoms: Atoms) -> None:
    """Store MACE-POLAR-1 atomic monopole charges and dipoles into `atoms.arrays`."""
    results = getattr(calc, "results", {})
    if "density_coefficients" in results:
        p = np.asarray(results["density_coefficients"])
        if p.ndim == 2 and p.shape[1] >= 4:
            atoms.arrays["polar_monopole_charge"] = p[:, 0].copy()
            atoms.arrays["polar_atomic_dipole"] = p[:, [3, 1, 2]].copy()

    if "spin_charge_density" in results:
        p_spin = np.asarray(results["spin_charge_density"])
        if p_spin.ndim == 3 and p_spin.shape[1] == 2:
            atoms.arrays["polar_charge_up"] = p_spin[:, 0, 0].copy()
            atoms.arrays["polar_charge_down"] = p_spin[:, 1, 0].copy()


def relax_single_structure(
    init_atoms: Atoms,
    calc,
    charge: int,
    spin_multiplicity: int,
    model_cfg: Dict[str, Any],
    opt_cls,
    fmax: float,
    max_steps: int,
    logfile: str,
) -> Dict[str, Any]:
    """Relax a single initial distortion structure at fixed supercell volume."""
    atoms = init_atoms.copy()
    atoms.pbc = True
    attach_charge_and_spin_info(atoms, charge, spin_multiplicity, model_cfg)
    atoms.calc = calc

    t0 = time.time()
    e_init = float(atoms.get_potential_energy())
    opt = opt_cls(atoms, logfile=logfile)
    converged = bool(opt.run(fmax=fmax, steps=max_steps))
    e_final = float(atoms.get_potential_energy())
    max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    elapsed = time.time() - t0

    extract_polar_multipoles(calc, atoms)

    atoms.calc = None
    atoms.info["mlip_energy_eV"] = e_final
    atoms.info["mlip_init_energy_eV"] = e_init
    atoms.info["mlip_max_force_eV_A"] = max_force
    atoms.info["mlip_converged"] = converged
    atoms.info["mlip_steps"] = int(opt.nsteps)

    return {
        "relaxed_atoms": atoms,
        "init_energy_eV": e_init,
        "final_energy_eV": e_final,
        "max_force_eV_A": max_force,
        "converged": converged,
        "nsteps": int(opt.nsteps),
        "wall_time_s": round(elapsed, 3),
    }


def sanitize_name(name: str) -> str:
    """Create filesystem-safe slug from model or distortion name."""
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name).strip("_")


def run_grid_study(
    config: Dict[str, Any],
    selected_models: Optional[List[str]] = None,
    selected_charges: Optional[List[int]] = None,
    override_max_distortions: Optional[int] = None,
    override_max_steps: Optional[int] = None,
    override_device: Optional[str] = None,
) -> None:
    manifest_path = Path(config["dataset"]["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run `python scripts/01_prepare_data.py` first."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    rel_cfg = config["relaxation"]
    opt_cls = OPTIMIZER_MAP[rel_cfg.get("optimizer", "BFGS").upper()]
    fmax = float(rel_cfg.get("fmax", 0.01))
    max_steps = int(override_max_steps if override_max_steps is not None else rel_cfg.get("max_steps", 500))
    selected_pcts = rel_cfg.get("selected_distortions", None)
    max_distortions = (
        override_max_distortions
        if override_max_distortions is not None
        else rel_cfg.get("max_distortions", None)
    )
    base_out_dir = Path(rel_cfg["output_dir"])
    base_out_dir.mkdir(parents=True, exist_ok=True)

    grid_index: Dict[str, Any] = {
        "fmax": fmax,
        "max_steps": max_steps,
        "optimizer": rel_cfg.get("optimizer", "BFGS"),
        "models": {},
    }

    for model_cfg in config["models"]:
        model_name = model_cfg["name"]
        if selected_models and model_name not in selected_models:
            continue
        if override_device is not None:
            model_cfg = {**model_cfg, "device": override_device}

        model_slug = sanitize_name(model_name)
        model_out_dir = base_out_dir / model_slug
        model_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=======================================================")
        print(f"[MODEL] Initializing calculator: {model_name} (type={model_cfg['type']}, device={model_cfg.get('device', 'cpu')})")
        print(f"=======================================================")
        try:
            calc = build_calculator(model_cfg)
        except Exception as exc:
            print(f"[ERROR] Could not load model '{model_name}': {exc}")
            continue

        grid_index["models"][model_name] = {
            "model_slug": model_slug,
            "model_type": model_cfg["type"],
            "pbc_handling": model_cfg.get("pbc_handling", "N/A"),
            "charge_states": {},
        }

        for q_str, q_data in manifest["charge_states"].items():
            q = int(q_str)
            if selected_charges is not None and q not in selected_charges:
                continue

            spin_mult = int(q_data.get("spin_multiplicity", 1))
            q_dir = model_out_dir / f"q_{q:+d}"
            q_dir.mkdir(parents=True, exist_ok=True)

            dist_list = select_distortions(q_data["distortions"], selected_pcts, max_distortions)
            dist_results: List[Dict[str, Any]] = []
            for dist in dist_list:
                dist_label = dist["distortion_label"]
                dist_slug = sanitize_name(dist_label)
                init_path = dist["initial_structure_path"]
                out_xyz = q_dir / f"{dist_slug}_relaxed.extxyz"
                log_path = q_dir / f"{dist_slug}_opt.log"

                if out_xyz.exists():
                    relaxed_atoms = read(str(out_xyz))
                    e_final = float(relaxed_atoms.info.get("mlip_energy_eV", np.nan))
                    e_init = float(relaxed_atoms.info.get("mlip_init_energy_eV", np.nan))
                    converged = bool(relaxed_atoms.info.get("mlip_converged", True))
                    nsteps = int(relaxed_atoms.info.get("mlip_steps", 0))
                    print(f"  [SKIP/RESUME] q={q:+d} | {dist_label:24s} -> E = {e_final:.4f} eV")
                else:
                    init_atoms = read_structure_auto(init_path)
                    res = relax_single_structure(
                        init_atoms=init_atoms,
                        calc=calc,
                        charge=q,
                        spin_multiplicity=spin_mult,
                        model_cfg=model_cfg,
                        opt_cls=opt_cls,
                        fmax=fmax,
                        max_steps=max_steps,
                        logfile=str(log_path),
                    )
                    relaxed_atoms = res["relaxed_atoms"]
                    write(str(out_xyz), relaxed_atoms, format="extxyz")
                    e_final = res["final_energy_eV"]
                    e_init = res["init_energy_eV"]
                    converged = res["converged"]
                    nsteps = res["nsteps"]
                    print(
                        f"  [RELAXED]     q={q:+d} | {dist_label:24s} -> "
                        f"E = {e_final:.4f} eV (steps={nsteps:3d}, conv={converged})"
                    )

                dist_results.append(
                    {
                        "charge": q,
                        "distortion_label": dist_label,
                        "distortion_percent": dist["distortion_percent"],
                        "initial_structure_path": init_path,
                        "dft_relaxed_structure_path": dist.get("dft_relaxed_structure_path"),
                        "dft_energy_eV": dist.get("dft_energy_eV"),
                        "mlip_relaxed_structure_path": str(out_xyz.resolve()),
                        "mlip_init_energy_eV": e_init,
                        "mlip_energy_eV": e_final,
                        "converged": converged,
                        "nsteps": nsteps,
                    }
                )

            valid_mlip = [r for r in dist_results if not np.isnan(r["mlip_energy_eV"])]
            if valid_mlip:
                mlip_gs = min(valid_mlip, key=lambda x: x["mlip_energy_eV"])
                gs_copy_path = q_dir / "ground_state_mlip.extxyz"
                gs_atoms = read(mlip_gs["mlip_relaxed_structure_path"])
                write(str(gs_copy_path), gs_atoms, format="extxyz")
            else:
                mlip_gs = None

            grid_index["models"][model_name]["charge_states"][q_str] = {
                "charge": q,
                "mlip_ground_state": mlip_gs,
                "distortions": dist_results,
            }

    index_file = base_out_dir / "relaxation_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(grid_index, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Saved grid relaxation index to {index_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PolarMACE and local MLIP grid relaxations.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Subset of model names")
    parser.add_argument("--charges", type=int, nargs="*", default=None, help="Subset of charge states")
    parser.add_argument("--max-distortions", type=int, default=None, help="Cap distortions per charge")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max relaxation steps")
    parser.add_argument("--device", type=str, default=None, help="Override device ('cpu' or 'cuda')")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_grid_study(
        config,
        selected_models=args.models,
        selected_charges=args.charges,
        override_max_distortions=args.max_distortions,
        override_max_steps=args.max_steps,
        override_device=args.device,
    )


if __name__ == "__main__":
    main()
