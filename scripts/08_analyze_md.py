#!/usr/bin/env python3
"""
08_analyze_md.py
================
Analyzes and plots the 600 K NVT MD trajectories produced by `07_run_md_benchmark.py`,
reproducing the 6-panel diagnostic layout of Cao et al. (MigrationBench, Fig. 3):
  (a) Kinetic temperature T(t)
  (b) Hydrostatic pressure P(t)
  (c) Unwrapped all-atom Mean-Square Displacement MSD(t)
  (d) Pair-resolved local coordination numbers (Sb-Se, Se-Se, Sb-Sb)
  (e) All-atom radial distribution function g_all(r)
  (f) Minimum interatomic distance d_min(t) & PolarMACE monopole charge check

Usage:
    python scripts/08_analyze_md.py
    python scripts/08_analyze_md.py --run-dir results/md_600K_q-3/runs/run_...
"""

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read


def load_plot_styles():
    """Reuse model ordering and cool/warm color styles from 04_plot_results.py."""
    plot_path = Path(__file__).parent / "04_plot_results.py"
    spec = importlib.util.spec_from_file_location("plot_helpers", str(plot_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def find_latest_run_dir(base_dir: Path) -> Path:
    """Return the most recently modified run_* directory containing md_summary.csv."""
    runs_dir = base_dir / "runs"
    candidates = sorted(
        [p for p in runs_dir.glob("run_*") if p.is_dir() and (p / "md_summary.csv").exists()],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No completed MD run directories found under {runs_dir}")
    return candidates[-1]


def compute_unwrapped_msd(frames: List[Atoms]) -> np.ndarray:
    """Compute all-atom MSD(t) using frame-to-frame minimum-image unwrapping."""
    if len(frames) < 2:
        return np.zeros(len(frames))
    cell = frames[0].get_cell()
    inv_cell = np.linalg.inv(cell)
    pos_prev = frames[0].get_positions()
    unwrapped = [pos_prev.copy()]

    for f in frames[1:]:
        pos_cur = f.get_positions()
        diff = pos_cur - pos_prev
        frac = diff @ inv_cell
        frac -= np.round(frac)
        step_cart = frac @ cell
        unwrapped.append(unwrapped[-1] + step_cart)
        pos_prev = pos_cur

    unwrapped_arr = np.asarray(unwrapped)
    disp = unwrapped_arr - unwrapped_arr[0:1]
    return np.mean(np.sum(disp ** 2, axis=2), axis=1)


def compute_all_atom_rdf(
    frames: List[Atoms],
    r_max: float = 8.0,
    n_bins: int = 120,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute time-averaged all-atom radial distribution function g_all(r)."""
    edges = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    hist_total = np.zeros(n_bins, dtype=float)

    sample_frames = frames[-min(len(frames), 50):]
    n_atoms = len(sample_frames[0])
    vol = float(sample_frames[0].get_volume())
    rho = n_atoms / vol

    for f in sample_frames:
        dists = f.get_all_distances(mic=True)
        np.fill_diagonal(dists, np.inf)
        vals = dists[dists < r_max]
        counts, _ = np.histogram(vals, bins=edges)
        hist_total += counts

    shell_vol = 4.0 * np.pi * (r_centers ** 2) * dr
    norm = len(sample_frames) * n_atoms * rho * shell_vol
    g_r = np.divide(hist_total, norm, out=np.zeros_like(hist_total), where=norm > 0)
    return r_centers, g_r


def compute_pair_coordination(
    frames: List[Atoms],
    cutoffs: Dict[Tuple[str, str], float],
) -> Dict[str, Tuple[float, float]]:
    """Compute mean and std of pair-resolved coordination numbers over sampled frames."""
    sample_frames = frames[-min(len(frames), 50):]
    pair_vals: Dict[str, List[float]] = {f"{a}-{b}": [] for a, b in cutoffs}

    for f in sample_frames:
        syms = np.array(f.get_chemical_symbols())
        dists = f.get_all_distances(mic=True)
        np.fill_diagonal(dists, np.inf)
        for (a, b), rcut in cutoffs.items():
            idx_a = np.where(syms == a)[0]
            idx_b = np.where(syms == b)[0]
            if len(idx_a) == 0 or len(idx_b) == 0:
                continue
            sub = dists[np.ix_(idx_a, idx_b)]
            cn_per_a = np.sum(sub <= rcut, axis=1)
            pair_vals[f"{a}-{b}"].append(float(np.mean(cn_per_a)))

    stats: Dict[str, Tuple[float, float]] = {}
    for k, vals in pair_vals.items():
        if vals:
            stats[k] = (float(np.mean(vals)), float(np.std(vals)))
        else:
            stats[k] = (0.0, 0.0)
    return stats


def rolling_smooth(values: np.ndarray, times: np.ndarray, window_ps: float = 2.0) -> np.ndarray:
    """Compute a centered rolling mean over `window_ps` picoseconds."""
    if len(values) < 3 or len(times) < 2:
        return values
    dt = float(np.median(np.diff(times)))
    if dt <= 0:
        return values
    win = max(1, int(round(window_ps / dt)))
    if win <= 1:
        return values
    return pd.Series(values).rolling(window=win, min_periods=1, center=True).mean().to_numpy()


def _plot_temperature(ax, model_data: List[Dict[str, Any]], add_legend: bool = False) -> None:
    for d in model_data:
        ax.plot(d["times"], d["temp_K"], color=d["color"], lw=0.7, alpha=0.16)
        ax.plot(d["times"], d["temp_smooth"], color=d["color"], lw=1.8, alpha=0.95, label=d["model_name"])
    ax.axhline(600.0, color="#333333", ls="--", lw=1.1, label="Target (600 K)")
    ax.set_title("(a) Kinetic Temperature (2 ps rolling mean + raw)")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Temperature (K)")
    if add_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, fontsize=8.5)


def _plot_pressure(ax, model_data: List[Dict[str, Any]], add_legend: bool = False) -> None:
    for d in model_data:
        ax.plot(d["times"], d["pressure_GPa"], color=d["color"], lw=0.7, alpha=0.16)
        ax.plot(d["times"], d["pressure_smooth"], color=d["color"], lw=1.8, alpha=0.95, label=d["model_name"])
    ax.axhline(0.0, color="#888888", ls=":", lw=1.0)
    ax.set_title("(b) Hydrostatic Pressure (2 ps rolling mean + raw)")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Pressure (GPa)")
    if add_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, fontsize=8.5)


def _plot_msd(ax, model_data: List[Dict[str, Any]], add_legend: bool = False) -> None:
    for d in model_data:
        ax.plot(d["traj_times"], d["msd"], color=d["color"], lw=1.7, alpha=0.92, label=d["model_name"])
    ax.set_title("(c) All-Atom Unwrapped MSD")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(r"MSD ($\mathrm{\AA}^2$)")
    if add_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, fontsize=8.5)


def _plot_cn(ax, models: List[str], cn_records: Dict[str, Dict[str, Tuple[float, float]]], model_styles: Dict[str, Any]) -> None:
    pair_names = ["Sb-Se", "Se-Se", "Sb-Sb"]
    x_pos = np.arange(len(models))
    width = 0.25
    hatches = ["", "//", ".."]
    for p_idx, p_name in enumerate(pair_names):
        means = [cn_records[m][p_name][0] if m in cn_records else 0.0 for m in models]
        stds = [cn_records[m][p_name][1] if m in cn_records else 0.0 for m in models]
        colors = [model_styles[m][0] for m in models]
        ax.bar(
            x_pos + (p_idx - 1) * width,
            means,
            width=width,
            yerr=stds,
            color=colors,
            edgecolor="black",
            linewidth=0.7,
            hatch=hatches[p_idx],
            alpha=0.88,
            label=p_name,
        )
    ax.set_xticks(x_pos)
    short_names = [m.replace("MACE-", "").replace(" (Local)", "") for m in models]
    ax.set_xticklabels(short_names, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(0, 6.7)
    ax.set_title("(d) Pair-Resolved Coordination Number")
    ax.set_ylabel("Mean CN")
    ax.legend(loc="upper right", ncol=3, frameon=True, fontsize=8.5)


def _plot_rdf(ax, model_data: List[Dict[str, Any]], add_legend: bool = False) -> None:
    for d in model_data:
        ax.plot(d["r_centers"], d["g_r"], color=d["color"], lw=1.6, alpha=0.92, label=d["model_name"])
    ax.set_title("(e) All-Atom Radial Distribution Function")
    ax.set_xlabel(r"$r$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$g_{\mathrm{all}}(r)$")
    ax.set_xlim(1.0, 8.0)
    if add_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, fontsize=8.5)


def _plot_dmin(ax, model_data: List[Dict[str, Any]], add_legend: bool = False) -> None:
    for d in model_data:
        ax.plot(d["times"], d["min_dist_A"], color=d["color"], lw=0.7, alpha=0.20)
        ax.plot(d["times"], d["dmin_smooth"], color=d["color"], lw=1.7, alpha=0.95, label=d["model_name"])
    ax.axhline(1.4, color="#d73027", ls="--", lw=1.1, label="Stop threshold (1.4 Å)")
    ax.set_title("(f) Minimum Interatomic Distance (2 ps rolling mean + raw)")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(r"$d_{\min}$ ($\mathrm{\AA}$)")
    if add_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, fontsize=8.5)
    else:
        ax.legend(loc="center right", frameon=True, fontsize=8.5)


def analyze_run(run_dir: Path) -> None:
    plot_mod = load_plot_styles()
    plot_mod.set_pub_style()

    summary_csv = run_dir / "md_summary.csv"
    df_sum = pd.read_csv(summary_csv)
    models = plot_mod.order_models(list(df_sum["model_name"].unique()), plot_mod.DEFAULT_MODEL_ORDER)
    model_styles = plot_mod.build_model_styles(df_sum, models)

    pair_cutoffs = {
        ("Sb", "Se"): 3.4,
        ("Se", "Se"): 3.2,
        ("Sb", "Sb"): 3.6,
    }

    model_data: List[Dict[str, Any]] = []
    cn_records: Dict[str, Dict[str, Tuple[float, float]]] = {}
    enriched_rows: List[Dict[str, Any]] = []

    for model_name in models:
        slug = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in model_name).strip("_")
        mdir = run_dir / slug
        thermo_csv = mdir / "thermo.csv"
        traj_xyz = mdir / "traj.extxyz"
        if not thermo_csv.exists() or not traj_xyz.exists():
            continue

        df_t = pd.read_csv(thermo_csv)
        frames = read(str(traj_xyz), index=":")
        color = model_styles[model_name][0]
        times = df_t["time_ps"].values

        msd = compute_unwrapped_msd(frames)
        traj_times = np.array([f.info.get("md_time_ps", i) for i, f in enumerate(frames)], dtype=float)
        cn_stats = compute_pair_coordination(frames, pair_cutoffs)
        cn_records[model_name] = cn_stats
        r_centers, g_r = compute_all_atom_rdf(frames)

        model_data.append({
            "model_name": model_name,
            "color": color,
            "times": times,
            "temp_K": df_t["temp_K"].values,
            "temp_smooth": rolling_smooth(df_t["temp_K"].values, times, window_ps=2.0),
            "pressure_GPa": df_t["pressure_GPa"].values,
            "pressure_smooth": rolling_smooth(df_t["pressure_GPa"].values, times, window_ps=2.0),
            "min_dist_A": df_t["min_dist_A"].values,
            "dmin_smooth": rolling_smooth(df_t["min_dist_A"].values, times, window_ps=2.0),
            "traj_times": traj_times,
            "msd": msd,
            "r_centers": r_centers,
            "g_r": g_r,
        })

        row_sum = df_sum[df_sum["model_name"] == model_name].iloc[0].to_dict()
        row_sum["final_msd_A2"] = round(float(msd[-1]), 4) if len(msd) else np.nan
        row_sum["cn_Sb_Se"] = round(cn_stats["Sb-Se"][0], 3)
        row_sum["cn_Se_Se"] = round(cn_stats["Se-Se"][0], 3)
        row_sum["cn_Sb_Sb"] = round(cn_stats["Sb-Sb"][0], 3)
        if "polar_monopole_sum" in df_t.columns and df_t["polar_monopole_sum"].notna().any():
            q_target = float(df_t["polar_monopole_sum"].dropna().iloc[0])
            row_sum["max_charge_drift_e"] = float(np.max(np.abs(df_t["polar_monopole_sum"].dropna() - round(q_target))))
        else:
            row_sum["max_charge_drift_e"] = np.nan
        enriched_rows.append(row_sum)

    # 1. Combined 6-panel figure
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    _plot_temperature(axes[0, 0], model_data, add_legend=False)
    _plot_pressure(axes[0, 1], model_data, add_legend=False)
    _plot_msd(axes[0, 2], model_data, add_legend=False)
    _plot_cn(axes[1, 0], models, cn_records, model_styles)
    _plot_rdf(axes[1, 1], model_data, add_legend=False)
    _plot_dmin(axes[1, 2], model_data, add_legend=False)

    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=min(3, len(models)),
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=9.5,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    out_png = run_dir / "fig_md_6panel.png"
    out_pdf = run_dir / "fig_md_6panel.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # 2. Individual large-format figures per subplot
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    panel_specs = [
        ("fig_md_a_temperature", lambda ax: _plot_temperature(ax, model_data, add_legend=True)),
        ("fig_md_b_pressure", lambda ax: _plot_pressure(ax, model_data, add_legend=True)),
        ("fig_md_c_msd", lambda ax: _plot_msd(ax, model_data, add_legend=True)),
        ("fig_md_d_coordination", lambda ax: _plot_cn(ax, models, cn_records, model_styles)),
        ("fig_md_e_rdf", lambda ax: _plot_rdf(ax, model_data, add_legend=True)),
        ("fig_md_f_min_distance", lambda ax: _plot_dmin(ax, model_data, add_legend=True)),
    ]
    for stem, draw_fn in panel_specs:
        s_fig, s_ax = plt.subplots(figsize=(8.0, 5.0))
        draw_fn(s_ax)
        s_fig.tight_layout()
        s_fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        s_fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(s_fig)

    df_out = pd.DataFrame(enriched_rows)
    out_csv = run_dir / "md_analysis_summary.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"[INFO] Saved MD 6-panel figure to:  {out_png}")
    print(f"[INFO] Saved individual panels to:  {fig_dir}")
    print(f"[INFO] Saved MD analysis summary to: {out_csv}")
    print(df_out[["model_name", "survived_ps", "stop_reason", "mean_temp_K", "final_msd_A2", "cn_Sb_Se", "ns_per_day"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and plot 600 K NVT MD trajectories.")
    parser.add_argument("--base-dir", type=str, default="results/md_600K_q-3", help="Base MD output directory")
    parser.add_argument("--run-dir", type=str, default=None, help="Specific run_* directory to analyze")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir(Path(args.base_dir))
    print(f"[INFO] Analyzing MD run directory: {run_dir}")
    analyze_run(run_dir)


if __name__ == "__main__":
    main()
