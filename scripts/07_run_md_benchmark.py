#!/usr/bin/env python3
"""
07_run_md_benchmark.py
======================
Runs controlled 600 K NVT (Langevin) molecular dynamics benchmarks across selected
MLIPs starting from the exact same DFT-PBE ground-state configuration and initial
Maxwell-Boltzmann velocities.

Every invocation writes into an immutable, timestamp + SHA-256 tagged run directory
(`output_dir/runs/run_<YYYYMMDD_HHMMSS>_<sha8>/`), guaranteeing that no previous MD
result is ever overwritten. Supports `--resume-from <prev_run_dir>` to continue
from an earlier run's final frame while still writing into a brand-new run folder.

Usage:
    python scripts/07_run_md_benchmark.py --config config/config_md.yaml
    python scripts/07_run_md_benchmark.py --config config/config_md.yaml --steps 5000 --run-tag test_5ps
    python scripts/07_run_md_benchmark.py --config config/config_md.yaml --steps 50000 --resume-from results/md_600K_q-3/runs/run_...
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from ase import Atoms, units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation


def load_relaxation_helpers():
    """Import build_calculator and metadata helpers from 02_run_grid_relaxation.py."""
    helper_path = Path(__file__).parent / "02_run_grid_relaxation.py"
    spec = importlib.util.spec_from_file_location("grid_relax_helpers", str(helper_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name).strip("_")


def sha256_file(path: Path) -> Optional[str]:
    """Compute SHA-256 hex digest of a file if it exists."""
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_atoms(atoms: Atoms) -> str:
    """Compute deterministic SHA-256 digest of atomic numbers, cell, and positions."""
    h = hashlib.sha256()
    h.update(np.asarray(atoms.get_atomic_numbers(), dtype=np.int64).tobytes())
    h.update(np.round(np.asarray(atoms.get_cell(), dtype=np.float64), 8).tobytes())
    h.update(np.round(np.asarray(atoms.get_positions(), dtype=np.float64), 8).tobytes())
    return h.hexdigest()


def get_git_commit() -> str:
    """Return short git commit hash if running inside a git repository."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def min_interatomic_distance(atoms: Atoms) -> float:
    """Compute minimum pairwise distance under periodic boundary conditions."""
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, np.inf)
    return float(np.min(dists))


def load_initial_structure(config: Dict[str, Any], helpers, supercell_override: Optional[List[int]]) -> tuple:
    """Load the DFT-PBE ground-state structure (preferring relative initial_structure_path)."""
    ds_cfg = config["dataset"]
    gs_path = ds_cfg.get("initial_structure_path")
    if gs_path and Path(gs_path).exists():
        atoms = helpers.read_structure_auto(str(gs_path))
    else:
        manifest_path = Path(ds_cfg["manifest_path"])
        q = int(ds_cfg["charge"])
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        q_str = str(q)
        if q_str not in manifest["charge_states"]:
            raise KeyError(f"Charge state {q} not found in {manifest_path}")
        gs_info = manifest["charge_states"][q_str]["dft_ground_state"]
        gs_path = gs_info["dft_relaxed_structure_path"]
        if not Path(gs_path).exists() and "data/raw/" in str(gs_path):
            gs_path = str(Path("data/raw") / str(gs_path).split("data/raw/", 1)[1])
        atoms = helpers.read_structure_auto(gs_path)

    sc = supercell_override if supercell_override is not None else ds_cfg.get("supercell", [1, 1, 1])
    if list(sc) != [1, 1, 1]:
        atoms = atoms.repeat(tuple(int(x) for x in sc))
    atoms.pbc = True
    return atoms, str(gs_path), list(sc)


def write_log_line(log_file: Path, msg: str) -> None:
    """Append a timestamped line to md.log and print to stdout."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_single_model_md(
    init_atoms: Atoms,
    init_structure_path: str,
    init_velocities: np.ndarray,
    model_cfg: Dict[str, Any],
    md_cfg: Dict[str, Any],
    supercell: List[int],
    config_sha: Optional[str],
    script_sha: Optional[str],
    git_commit: str,
    charge: int,
    spin_mult: int,
    helpers,
    run_dir: Path,
    n_steps: int,
    resume_from: Optional[Path],
) -> Dict[str, Any]:
    """Run NVT Langevin MD for a single model inside an immutable run directory."""
    model_name = model_cfg["name"]
    slug = sanitize_name(model_name)
    model_dir = run_dir / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    thermo_csv = model_dir / "thermo.csv"
    traj_xyz = model_dir / "traj.extxyz"
    log_file = model_dir / "md.log"
    prov_file = model_dir / "run_provenance.json"

    dt_fs = float(md_cfg["timestep_fs"])
    temp_K = float(md_cfg["temperature_K"])
    friction = float(md_cfg["friction_per_fs"])
    log_interval = int(md_cfg["log_interval"])
    seed = int(md_cfg.get("seed", 1100))
    target_ps = round(n_steps * dt_fs / 1000.0, 4)

    model_path_obj = Path(str(model_cfg["model_path"]))
    model_ckpt_sha = sha256_file(model_path_obj) if model_path_obj.exists() else f"foundation:{model_cfg['model_path']}"
    init_atoms_sha = sha256_atoms(init_atoms)

    start_mode = "fresh"
    existing_records: List[Dict[str, Any]] = []
    existing_frames: List[Atoms] = []
    start_step = 0
    resumed_from_traj_sha: Optional[str] = None
    resumed_from_dir_str: Optional[str] = None

    if resume_from is not None:
        prev_model_dir = resume_from / slug
        prev_thermo = prev_model_dir / "thermo.csv"
        prev_traj = prev_model_dir / "traj.extxyz"
        if prev_thermo.exists() and prev_traj.exists():
            df_prev = pd.read_csv(prev_thermo)
            if not df_prev.empty:
                start_mode = "resumed"
                existing_records = df_prev.to_dict(orient="records")
                existing_frames = read(str(prev_traj), index=":")
                start_step = int(df_prev["step"].iloc[-1])
                resumed_from_traj_sha = sha256_file(prev_traj)
                resumed_from_dir_str = str(prev_model_dir)

    write_log_line(log_file, "=======================================================")
    write_log_line(
        log_file,
        f"Starting MD for '{model_name}' | run_dir={run_dir.name} | mode={start_mode} | "
        f"start_step={start_step} -> target_steps={n_steps}",
    )
    write_log_line(
        log_file,
        f"Params: T={temp_K} K, dt={dt_fs} fs, friction={friction}/fs, supercell={supercell}, "
        f"dtype={model_cfg.get('default_dtype')}, device={model_cfg.get('device')}, "
        f"charge={charge}, spin={spin_mult}, seed={seed}",
    )
    write_log_line(
        log_file,
        f"SHA256: config={config_sha}, script={script_sha}, model={model_ckpt_sha}, "
        f"init_atoms={init_atoms_sha}, resumed_traj={resumed_from_traj_sha}",
    )

    calc = helpers.build_calculator(model_cfg)

    if start_mode == "resumed" and existing_frames:
        atoms = existing_frames[-1].copy()
        atoms.pbc = True
    else:
        atoms = init_atoms.copy()
        atoms.set_velocities(init_velocities.copy())
    helpers.attach_charge_and_spin_info(atoms, charge, spin_mult, model_cfg)
    atoms.calc = calc

    stop_cfg = md_cfg.get("stopping_criteria", {})
    max_temp = float(stop_cfg.get("max_temp_K", 1200.0))
    max_force_lim = float(stop_cfg.get("max_force_eV_A", 50.0))
    min_dist_lim = float(stop_cfg.get("min_interatomic_dist_A", 1.4))

    rng = np.random.default_rng(seed + start_step)
    dyn = Langevin(
        atoms,
        timestep=dt_fs * units.fs,
        temperature_K=temp_K,
        friction=friction / units.fs,
        rng=rng,
    )

    records: List[Dict[str, Any]] = list(existing_records)
    traj_frames: List[Atoms] = list(existing_frames)
    stop_reason = "completed"
    t_wall_start = time.time()

    def sample_state(step_idx: int) -> Optional[str]:
        epot = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        ekin = float(atoms.get_kinetic_energy())
        etot = epot + ekin
        t_inst = float(atoms.get_temperature())
        max_f = float(np.linalg.norm(forces, axis=1).max())
        min_d = min_interatomic_distance(atoms)

        try:
            stress = atoms.get_stress(voigt=True)
            pressure_gpa = float(-np.mean(stress[:3]) * 160.21766208)
        except Exception:
            pressure_gpa = float("nan")

        monopole_sum = float("nan")
        if model_cfg["type"].lower() == "mace_polar":
            helpers.extract_polar_multipoles(calc, atoms)
            if "polar_monopole_charge" in atoms.arrays:
                monopole_sum = float(np.sum(atoms.arrays["polar_monopole_charge"]))

        time_ps = step_idx * dt_fs / 1000.0
        records.append(
            {
                "step": step_idx,
                "time_ps": round(time_ps, 4),
                "temp_K": round(t_inst, 3),
                "epot_eV": epot,
                "ekin_eV": ekin,
                "etot_eV": etot,
                "pressure_GPa": round(pressure_gpa, 4) if not np.isnan(pressure_gpa) else np.nan,
                "max_force_eV_A": round(max_f, 4),
                "min_dist_A": round(min_d, 4),
                "polar_monopole_sum": monopole_sum,
            }
        )

        frame = atoms.copy()
        frame.calc = None
        frame.info["md_step"] = step_idx
        frame.info["md_time_ps"] = time_ps
        frame.info["md_epot_eV"] = epot
        frame.info["md_temp_K"] = t_inst
        traj_frames.append(frame)

        if np.isnan(epot) or np.isnan(max_f):
            return "nan_energy_or_force"
        if t_inst > max_temp:
            return f"temp_exceeded_{max_temp:.0f}K"
        if max_f > max_force_lim:
            return f"force_exceeded_{max_force_lim:.1f}eV_A"
        if min_d < min_dist_lim:
            return f"min_dist_below_{min_dist_lim:.2f}A"
        return None

    if start_mode == "fresh":
        reason0 = sample_state(0)
        if reason0 is not None:
            stop_reason = reason0

    steps_done = start_step
    steps_this_run = 0
    while steps_done < n_steps and stop_reason == "completed":
        chunk = min(log_interval, n_steps - steps_done)
        dyn.run(chunk)
        steps_done += chunk
        steps_this_run += chunk
        reason = sample_state(steps_done)
        if reason is not None:
            stop_reason = reason
            write_log_line(
                log_file,
                f"[STOP] Stopped at step={steps_done} ({steps_done * dt_fs / 1000.0:.2f} ps): {stop_reason}",
            )
            break
        if steps_done % (log_interval * 10) == 0 or steps_done == n_steps:
            last_rec = records[-1]
            write_log_line(
                log_file,
                f"step={steps_done:6d} ({last_rec['time_ps']:6.2f} ps) | "
                f"T={last_rec['temp_K']:6.1f} K | E_pot={last_rec['epot_eV']:12.4f} eV | "
                f"F_max={last_rec['max_force_eV_A']:5.2f} eV/A | d_min={last_rec['min_dist_A']:.2f} A",
            )

    wall_s = time.time() - t_wall_start
    ms_per_step = (wall_s * 1000.0 / steps_this_run) if steps_this_run > 0 else float("nan")
    ns_per_day = (86400.0 / (wall_s / steps_this_run)) * dt_fs * 1e-6 if steps_this_run > 0 and wall_s > 0 else float("nan")

    df_thermo = pd.DataFrame(records)
    df_thermo.to_csv(thermo_csv, index=False)
    write(str(traj_xyz), traj_frames, format="extxyz")

    thermo_sha = sha256_file(thermo_csv)
    traj_sha = sha256_file(traj_xyz)

    provenance = {
        "run_dir": str(run_dir),
        "timestamp_iso": datetime.datetime.now().isoformat(),
        "git_commit": git_commit,
        "start_mode": start_mode,
        "resumed_from_dir": resumed_from_dir_str,
        "start_step": start_step,
        "end_step": steps_done,
        "target_steps": n_steps,
        "stop_reason": stop_reason,
        "wall_time_s": round(wall_s, 3),
        "model_cfg": model_cfg,
        "md_cfg": {**md_cfg, "n_steps": n_steps},
        "dataset": {
            "charge": charge,
            "spin_multiplicity": spin_mult,
            "supercell": supercell,
            "n_atoms": len(atoms),
            "initial_structure_path": init_structure_path,
        },
        "sha256": {
            "config_yaml": config_sha,
            "script_py": script_sha,
            "model_checkpoint": model_ckpt_sha,
            "initial_structure_atoms": init_atoms_sha,
            "resumed_from_traj": resumed_from_traj_sha,
            "thermo_csv": thermo_sha,
            "traj_extxyz": traj_sha,
        },
    }
    with open(prov_file, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    write_log_line(
        log_file,
        f"[DONE] mode={start_mode}, stop_reason={stop_reason}, thermo_sha256={thermo_sha}, traj_sha256={traj_sha}",
    )

    return {
        "run_id": run_dir.name,
        "model_name": model_name,
        "start_mode": start_mode,
        "default_dtype": model_cfg.get("default_dtype", "float32"),
        "n_atoms": len(atoms),
        "target_ps": target_ps,
        "survived_ps": float(df_thermo["time_ps"].iloc[-1]),
        "completed": stop_reason == "completed",
        "stop_reason": stop_reason,
        "mean_temp_K": round(float(df_thermo["temp_K"].mean()), 2),
        "std_temp_K": round(float(df_thermo["temp_K"].std()), 2),
        "mean_pressure_GPa": round(float(df_thermo["pressure_GPa"].mean()), 3),
        "max_force_seen_eV_A": round(float(df_thermo["max_force_eV_A"].max()), 3),
        "min_dist_seen_A": round(float(df_thermo["min_dist_A"].min()), 3),
        "ms_per_step": round(ms_per_step, 2),
        "ns_per_day": round(ns_per_day, 4),
        "traj_sha256": traj_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 600 K NVT MD benchmark across MLIPs.")
    parser.add_argument("--config", type=str, default="config/config_md.yaml", help="Path to config_md.yaml")
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Subset of model names")
    parser.add_argument("--steps", type=int, default=None, help="Override n_steps")
    parser.add_argument("--device", type=str, default=None, help="Override device ('cpu' or 'cuda')")
    parser.add_argument("--supercell", type=int, nargs=3, default=None, help="Override supercell repeat e.g. 2 2 2")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional custom tag for this run folder")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a previous run directory to continue from (writes to a new run folder)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config_sha = sha256_file(config_path)
    script_sha = sha256_file(Path(__file__))
    git_commit = get_git_commit()

    helpers = load_relaxation_helpers()
    init_atoms, init_structure_path, supercell = load_initial_structure(config, helpers, args.supercell)

    md_cfg = config["md"]
    n_steps = int(args.steps if args.steps is not None else md_cfg["n_steps"])
    temp_K = float(md_cfg["temperature_K"])
    seed = int(md_cfg.get("seed", 1100))

    vel_atoms = init_atoms.copy()
    MaxwellBoltzmannDistribution(vel_atoms, temperature_K=temp_K, rng=np.random.default_rng(seed))
    Stationary(vel_atoms)
    ZeroRotation(vel_atoms)
    init_velocities = vel_atoms.get_velocities()

    # Build a unique, immutable run directory: output_dir/runs/run_<timestamp>_<short_sha>[_<tag>]
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    param_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "config_sha": config_sha,
                "n_steps": n_steps,
                "supercell": supercell,
                "device": args.device,
                "models": args.models,
                "resume_from": args.resume_from,
                "ts": ts_str,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:8]

    folder_name = f"run_{ts_str}_{param_fingerprint}"
    if args.run_tag:
        folder_name += f"_{sanitize_name(args.run_tag)}"

    run_dir = Path(md_cfg["output_dir"]) / "runs" / folder_name
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] Created immutable MD run directory: {run_dir}")

    charge = int(config["dataset"]["charge"])
    spin_mult = int(config["dataset"]["spin_multiplicity"])
    resume_from_path = Path(args.resume_from) if args.resume_from else None

    summaries: List[Dict[str, Any]] = []
    for model_cfg in config["models"]:
        if args.models and model_cfg["name"] not in args.models:
            continue
        if args.device is not None:
            model_cfg = {**model_cfg, "device": args.device}
        try:
            res = run_single_model_md(
                init_atoms=init_atoms,
                init_structure_path=init_structure_path,
                init_velocities=init_velocities,
                model_cfg=model_cfg,
                md_cfg=md_cfg,
                supercell=supercell,
                config_sha=config_sha,
                script_sha=script_sha,
                git_commit=git_commit,
                charge=charge,
                spin_mult=spin_mult,
                helpers=helpers,
                run_dir=run_dir,
                n_steps=n_steps,
                resume_from=resume_from_path,
            )
            summaries.append(res)
        except Exception as exc:
            print(f"[ERROR] MD failed for {model_cfg['name']}: {exc}")

    if summaries:
        df_sum = pd.DataFrame(summaries)
        sum_path = run_dir / "md_summary.csv"
        df_sum.to_csv(sum_path, index=False)
        print(f"\n[INFO] Saved MD summary to {sum_path}")
        print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
