#!/usr/bin/env python3
"""
04_plot_results.py
==================
Generates publication-quality figures reproducing the layout and visual encoding of
Fig. 1(a-c) in Wang, Mosquera-Lois, and Walsh, PRX Energy 5, 043001 (2026),
DOI: https://doi.org/10.1103/tzwd-cl3y:

1. `fig1a_ground_state_rmsd.png` / `.pdf`:
   - Plots Ground-State Structural RMSD (Å) vs. Charge State `q` (e.g., -3, -2, -1, 0, +1).
   - Horizontal axis offsets (`jitter`) for each model within each charge state column for clarity.
   - Green shaded region (`0 <= RMSD <= 0.10 Å`) denoting successful ground-state structure identification,
     bounded by a dashed threshold line at `RMSD = 0.10 Å`.

2. `fig1bc_pes_vs_distortion.png` / `.pdf`:
   - Maps the relative defect Potential Energy Surface (`Delta E` in eV, referenced to each curve's minimum)
     as a function of ShakeNBreak bond distortion percentage (-60% to +60%, plus Rattled/Unperturbed).
   - Overlays DFT Reference against PolarMACE and custom MLIP models for highlighted charge states
     (default `q = 0` and `q = -3`).
   - Encodes structural accuracy via marker shape: circles (`o`) for `RMSD <= 0.10 Å` and crosses (`X`)
     for `RMSD > 0.10 Å` relative to the DFT ground state.

3. `fig_cross_charge_sensitivity.png` / `.pdf`:
   - Side-by-side heatmaps of the pairwise cross-charge RMSD matrix `RMSD(q_i, q_j)` comparing
     DFT Reference against PolarMACE and custom models, directly quantifying whether a model
     distinguishes different charge states or collapses them onto the same geometry.

Usage:
    python 04_plot_results.py --config config.yaml [--demo]
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# Customizable display order for models across all figures (edit here or via config.yaml / --model-order)
DEFAULT_MODEL_ORDER = [
    "DFT Reference",
    "MACE-POLAR-1-M (FT Stage2)",
    "MACE-POLAR-1-M (FT)",
    "MACE-POLAR-1-M (pbc)",
    "MACE-MH-1 (Local)",
    "MACE-MPA-0-Medium (Local)",
    "MACE-OMAT-0-Medium (Local)",
]

# Warm palette for non-Polar (local) MLIPs and cool gradient for Polar models
WARM_PALETTE = [
    "#d73027",  # Vermilion Red
    "#f46d43",  # Warm Coral Orange
    "#e6ab02",  # Amber Gold
    "#a50026",  # Deep Crimson
    "#d95f02",  # Burnt Orange
    "#8c510a",  # Warm Ochre
]
POLAR_MARKERS = ["o", "v", "^", "p", "*"]
LOCAL_MARKERS = ["s", "D", "^", "P", "X"]


def order_models(available_models: List[str], custom_order: List[str]) -> List[str]:
    """Order available models according to custom_order, appending any unlisted models at the end."""
    avail_set = list(dict.fromkeys(available_models))
    ordered = [m for m in custom_order if m in avail_set]
    for m in avail_set:
        if m not in ordered:
            ordered.append(m)
    return ordered


def is_polar_model(model_name: str, model_type: str = "") -> bool:
    """Return True if the model belongs to the PolarMACE family."""
    return "polar" in model_name.lower() or model_type.lower() == "mace_polar"


def build_model_styles(
    df: pd.DataFrame,
    models: List[str],
) -> Dict[str, Tuple[Any, str]]:
    """
    Assign cool gradient colors to PolarMACE models and distinct warm colors
    to non-Polar (local) models.
    """
    type_map: Dict[str, str] = {}
    if "model_type" in df.columns:
        for _, row in df[["model_name", "model_type"]].drop_duplicates().iterrows():
            type_map[str(row["model_name"])] = str(row["model_type"])

    polar_models = [m for m in models if is_polar_model(m, type_map.get(m, ""))]
    local_models = [m for m in models if m not in polar_models]

    styles: Dict[str, Tuple[Any, str]] = {}

    # Cool blue-cyan gradient for Polar models (light -> medium -> deep navy)
    n_polar = len(polar_models)
    cool_cmap = plt.get_cmap("YlGnBu")
    cool_vals = np.linspace(0.45, 0.92, n_polar) if n_polar > 1 else [0.75]
    for idx, m_name in enumerate(polar_models):
        styles[m_name] = (cool_cmap(float(cool_vals[idx])), POLAR_MARKERS[idx % len(POLAR_MARKERS)])

    # Distinct warm colors for non-Polar models
    for idx, m_name in enumerate(local_models):
        styles[m_name] = (WARM_PALETTE[idx % len(WARM_PALETTE)], LOCAL_MARKERS[idx % len(LOCAL_MARKERS)])

    return styles


def set_pub_style() -> None:
    """Configure Matplotlib rcParams for PRX Energy publication-style figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 1.1,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "figure.dpi": 300,
        }
    )


def format_defect_math(defect_name: str) -> str:
    """Format plain defect string like 'V_Sb(1)' into clean MathText '$V_{\\mathrm{Sb}(1)}$'."""
    if defect_name == "V_Sb(1)":
        return r"$V_{\mathrm{Sb}(1)}$ in $\mathrm{Sb_2Se_3}$"
    if defect_name == "V_Se(1)":
        return r"$V_{\mathrm{Se}(1)}$ in $\mathrm{Sb_2Se_3}$"
    return defect_name


def plot_fig1a_rmsd(
    df_summary: pd.DataFrame,
    threshold: float,
    defect_label: str,
    out_dir: Path,
    custom_order: List[str],
) -> Path:
    """
    Reproduce PRX Energy Fig. 1(a):
    Ground-state RMSD (Å) relative to DFT-PBE across charge states q,
    with cool-gradient colors for Polar models and warm colors for local models.
    """
    set_pub_style()
    fig, ax = plt.subplots(figsize=(7.8, 5.2))

    charges = sorted(df_summary["charge"].unique())
    models = order_models(list(df_summary["model_name"].unique()), custom_order)
    n_models = len(models)
    model_styles = build_model_styles(df_summary, models)

    # Horizontal jitter offsets within each charge-state column
    max_span = min(0.60, 0.13 * n_models)
    offsets = np.linspace(-max_span / 2.0, max_span / 2.0, n_models) if n_models > 1 else [0.0]

    # Green shaded success region (RMSD <= 0.1 Å)
    ax.axhspan(
        0.0,
        threshold,
        color="#2ca02c",
        alpha=0.16,
        zorder=1,
        label=f"Successful identification ($\mathrm{{RMSD}} \leq {threshold:.1f}$ Å)",
    )
    ax.axhline(threshold, color="#2ca02c", linestyle="--", linewidth=1.2, zorder=2)

    # Vertical subtle separators between charge states
    for idx in range(len(charges) - 1):
        ax.axvline(idx + 0.5, color="#dcdcdc", linestyle=":", linewidth=0.9, zorder=1)

    for m_idx, model_name in enumerate(models):
        sub = df_summary[df_summary["model_name"] == model_name]
        xs, ys = [], []
        for c_idx, q in enumerate(charges):
            row = sub[sub["charge"] == q]
            if not row.empty and not np.isnan(row["rmsd_A"].values[0]):
                xs.append(c_idx + offsets[m_idx])
                ys.append(float(row["rmsd_A"].values[0]))

        color, _ = model_styles[model_name]
        ax.scatter(
            xs,
            ys,
            s=80,
            color=color,
            marker="o",
            edgecolors="black",
            linewidths=0.75,
            alpha=0.94,
            zorder=4,
            label=model_name,
        )

    ax.set_xticks(range(len(charges)))
    ax.set_xticklabels([f"{q:+d}" if q != 0 else "0" for q in charges])
    ax.set_xlim(-0.5, len(charges) - 0.5)

    y_max = max(0.42, float(df_summary["rmsd_A"].max()) * 1.18) if not df_summary.empty else 0.42
    ax.set_ylim(0.0, y_max)

    ax.set_xlabel("Charge state $q$")
    ax.set_ylabel("Ground-state RMSD (Å)")
    ax.set_title(
        f"(a) Ground-State Structural Deviation for {format_defect_math(defect_label)}",
        pad=10,
    )

    handles, _ = ax.get_legend_handles_labels()
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.98),
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        ncol=2,
        fontsize=9.0,
    )

    fig.tight_layout()
    png_path = out_dir / "fig1a_ground_state_rmsd.png"
    pdf_path = out_dir / "fig1a_ground_state_rmsd.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_fig1bc_pes(
    df_detailed: pd.DataFrame,
    highlight_charges: List[int],
    threshold: float,
    defect_label: str,
    out_dir: Path,
    custom_order: List[str],
) -> Path:
    """
    Reproduce PRX Energy Fig. 1(b, c):
    One subplot per charge state q overlaying DFT Reference and all MLIP models,
    using the exact same cool/warm model color palette as Fig. 1(a) and markers
    indicating RMSD <= 0.1 Å (circles 'o') vs. RMSD > 0.1 Å (crosses 'X').
    """
    set_pub_style()
    n_panels = len(highlight_charges)
    if n_panels <= 3:
        nrows, ncols = 1, n_panels
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.0), squeeze=False)
    else:
        ncols = 3
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.5 * nrows), squeeze=False)

    flat_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    df_plot = df_detailed[df_detailed["distortion_percent"] <= 100.0].copy()

    all_present = list(df_plot["model_name"].unique())
    ordered_all = order_models(all_present, custom_order)
    mlip_models = [m for m in ordered_all if m != "DFT Reference"]
    model_order = (["DFT Reference"] if "DFT Reference" in ordered_all else []) + mlip_models
    model_styles = build_model_styles(df_plot[df_plot["model_name"] != "DFT Reference"], mlip_models)
    pretty_defect = format_defect_math(defect_label)

    for p_idx, q in enumerate(highlight_charges):
        ax = flat_axes[p_idx]
        sub_q = df_plot[df_plot["charge"] == q]

        for model_name in model_order:
            sub_m = sub_q[sub_q["model_name"] == model_name].sort_values("distortion_percent")
            if sub_m.empty:
                continue

            color = "#111111" if model_name == "DFT Reference" else model_styles[model_name][0]
            lw = 2.0 if model_name == "DFT Reference" else 1.4
            ls = "-" if model_name == "DFT Reference" else "--"

            ax.plot(
                sub_m["distortion_percent"],
                sub_m["rel_energy_eV"],
                color=color,
                linestyle=ls,
                linewidth=lw,
                alpha=0.88,
                label=model_name,
                zorder=3 if model_name == "DFT Reference" else 2,
            )

            ok = sub_m[sub_m["within_threshold"] == True]
            if not ok.empty:
                ax.scatter(
                    ok["distortion_percent"],
                    ok["rel_energy_eV"],
                    marker="o",
                    s=55,
                    color=color,
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=5,
                )

            bad = sub_m[sub_m["within_threshold"] == False]
            if not bad.empty:
                ax.scatter(
                    bad["distortion_percent"],
                    bad["rel_energy_eV"],
                    marker="X",
                    s=60,
                    color=color,
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.88,
                    zorder=4,
                )

        panel_letter = chr(ord("b") + p_idx)
        q_str = f"{q:+d}" if q != 0 else "0"
        ax.set_title(f"({panel_letter}) {pretty_defect} PES ($q = {q_str}$)", pad=10)
        if p_idx // ncols == nrows - 1 or p_idx + ncols >= n_panels:
            ax.set_xlabel("Bond distortion (%)")
        if p_idx % ncols == 0:
            ax.set_ylabel("Relative energy $\\Delta E$ (eV)")
        ax.axhline(0.0, color="#888888", linestyle=":", linewidth=0.8)

    handles, _ = flat_axes[0].get_legend_handles_labels()
    h_circle = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#555555",
        markeredgecolor="k",
        markersize=7,
        label=f"$\\mathrm{{RMSD}} \\leq {threshold:.1f}$ Å (Success)",
    )
    h_cross = plt.Line2D(
        [0],
        [0],
        marker="X",
        color="w",
        markerfacecolor="#555555",
        markeredgecolor="k",
        markersize=7,
        label=f"$\\mathrm{{RMSD}} > {threshold:.1f}$ Å (Failed)",
    )
    all_handles = handles + [h_circle, h_cross]

    if len(flat_axes) > n_panels:
        for empty_idx in range(n_panels, len(flat_axes)):
            flat_axes[empty_idx].axis("off")
        flat_axes[n_panels].legend(
            handles=all_handles,
            loc="center",
            frameon=True,
            framealpha=0.95,
            edgecolor="#cccccc",
            fontsize=10.5,
            title="Models & Structural Accuracy",
            title_fontsize=11.0,
        )
    else:
        fig.legend(
            handles=all_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=4,
            frameon=True,
            framealpha=0.95,
            edgecolor="#cccccc",
            fontsize=9.0,
        )

    fig.tight_layout()
    png_path = out_dir / "fig1bc_pes_vs_distortion.png"
    pdf_path = out_dir / "fig1bc_pes_vs_distortion.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_cross_charge_matrices(
    cross_matrices: Dict[str, Dict[str, Any]],
    out_dir: Path,
    custom_order: List[str],
) -> Path:
    """
    Plot pairwise cross-charge RMSD(q_i, q_j) heatmaps across models to visualize
    whether each model captures distinct charge-state reconstructions.
    """
    set_pub_style()
    names = order_models(list(cross_matrices.keys()), custom_order)
    n_models = len(names)
    if n_models == 0:
        return out_dir / "fig_cross_charge_sensitivity.png"

    ncols = min(3, n_models)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.9 * nrows), squeeze=False)

    vmax = max(
        float(np.max(m["rmsd_matrix_A"])) for m in cross_matrices.values()
    )
    vmax = max(vmax, 0.35)

    for idx, name in enumerate(names):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        data = cross_matrices[name]
        charges = data["charges"]
        mat = np.asarray(data["rmsd_matrix_A"], dtype=float)

        im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(charges)))
        ax.set_yticks(range(len(charges)))
        q_labels = [f"{q:+d}" if q != 0 else "0" for q in charges]
        ax.set_xticklabels(q_labels)
        ax.set_yticklabels(q_labels)
        ax.set_xlabel("Charge state $q_j$")
        ax.set_ylabel("Charge state $q_i$")
        ax.set_title(name, fontsize=10.5, pad=8)

        for i in range(len(charges)):
            for j in range(len(charges)):
                val = mat[i, j]
                txt_color = "white" if val > 0.55 * vmax else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=8.5)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RMSD (Å)")

    # Hide unused subplots
    for idx in range(n_models, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.suptitle(
        "Cross-Charge Structural Sensitivity $\mathrm{RMSD}(q_i, q_j)$ Across Charge States",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()
    png_path = out_dir / "fig_cross_charge_sensitivity.png"
    pdf_path = out_dir / "fig_cross_charge_sensitivity.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def generate_demo_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generate realistic demonstration benchmark tables matching PRX Energy 10.1103/tzwd-cl3y
    to verify figure rendering before full GPU relaxation completes.
    """
    charges = [-3, -2, -1, 0, 1]
    demo_models = [
        ("MACE-POLAR-1-M (pbc)", "mace_polar", "pbc", [0.16, 0.14, 0.19, 0.23, 0.25]),
        ("MACE-POLAR-1-L (pbc)", "mace_polar", "pbc", [0.13, 0.11, 0.15, 0.18, 0.21]),
        ("MACE-MPA-0 (Local)", "mace_mp", "pbc", [0.15, 0.14, 0.26, 0.29, 0.31]),
        ("MACE-MH-1 (Local)", "mace_mp", "pbc", [0.14, 0.12, 0.08, 0.27, 0.28]),
        ("Charge-Embedded MACE (Custom)", "mace_custom_charge", "pbc", [0.02, 0.03, 0.03, 0.04, 0.03]),
    ]

    summary_rows = []
    for m_name, m_type, pbc, rmsds in demo_models:
        for q, r_val in zip(charges, rmsds):
            summary_rows.append(
                {
                    "model_name": m_name,
                    "model_type": m_type,
                    "pbc_handling": pbc,
                    "charge": q,
                    "predicted_gs_distortion": "Bond_Distortion_-30.0%" if r_val <= 0.1 else "Unperturbed",
                    "mlip_gs_energy_eV": -215.0,
                    "rmsd_A": r_val,
                    "local_rmsd_A": r_val * 1.35,
                    "max_atom_disp_A": r_val * 2.4,
                    "identified_ground_state": r_val <= 0.10,
                    "polar_monopole_sum": float(q) if m_type == "mace_polar" else np.nan,
                }
            )

    dist_pcts = np.linspace(-60, 60, 13)
    detailed_rows = []
    for q in [0, -3]:
        for d in dist_pcts:
            # Synthetic PES profiles illustrating reconstruction basin at q=0 vs single basin at q=-3
            if q == 0:
                dft_rel = 0.45 * ((d + 30.0) / 50.0) ** 2 * (1.0 - 0.3 * np.exp(-((d - 10.0) / 20.0) ** 2))
                dft_rmsd = 0.02 if abs(d + 30.0) <= 15 else 0.28
            else:
                dft_rel = 0.55 * (d / 50.0) ** 2
                dft_rmsd = 0.02 if abs(d) <= 10 else 0.22

            detailed_rows.append(
                {
                    "model_name": "DFT Reference",
                    "model_type": "dft",
                    "pbc_handling": "pbc",
                    "charge": q,
                    "distortion_label": f"Bond_Distortion_{d:.1f}%",
                    "distortion_percent": float(d),
                    "energy_eV": -215.0 + dft_rel,
                    "rel_energy_eV": dft_rel,
                    "rmsd_to_dft_gs_A": dft_rmsd,
                    "within_threshold": dft_rmsd <= 0.10,
                }
            )

            for m_name, m_type, pbc, _ in demo_models:
                if "Charge-Embedded" in m_name:
                    m_rel = dft_rel + 0.02 * np.sin(d / 20.0)
                    m_rmsd = dft_rmsd + 0.01
                elif "POLAR-1-L" in m_name:
                    m_rel = 0.65 * dft_rel + 0.06 * ((d / 50.0) ** 2)
                    m_rmsd = 0.18 if q == 0 else 0.13
                else:
                    m_rel = 0.25 * ((d / 50.0) ** 2)
                    m_rmsd = 0.27 if q == 0 else 0.15

                detailed_rows.append(
                    {
                        "model_name": m_name,
                        "model_type": m_type,
                        "pbc_handling": pbc,
                        "charge": q,
                        "distortion_label": f"Bond_Distortion_{d:.1f}%",
                        "distortion_percent": float(d),
                        "energy_eV": -215.0 + m_rel,
                        "rel_energy_eV": max(0.0, m_rel),
                        "rmsd_to_dft_gs_A": m_rmsd,
                        "within_threshold": m_rmsd <= 0.10,
                    }
                )

    dft_mat = [
        [0.00, 0.06, 0.19, 0.31, 0.36],
        [0.06, 0.00, 0.17, 0.29, 0.34],
        [0.19, 0.17, 0.00, 0.18, 0.24],
        [0.31, 0.29, 0.18, 0.00, 0.14],
        [0.36, 0.34, 0.24, 0.14, 0.00],
    ]
    polar_mat = [
        [0.00, 0.04, 0.09, 0.14, 0.17],
        [0.04, 0.00, 0.07, 0.12, 0.15],
        [0.09, 0.07, 0.00, 0.08, 0.11],
        [0.14, 0.12, 0.08, 0.00, 0.06],
        [0.17, 0.15, 0.11, 0.06, 0.00],
    ]
    mpa_mat = np.zeros((5, 5)).tolist()

    cross_matrices = {
        "DFT Reference": {"charges": charges, "rmsd_matrix_A": dft_mat},
        "MACE-POLAR-1-L (pbc)": {"charges": charges, "rmsd_matrix_A": polar_mat},
        "MACE-MPA-0 (Local)": {"charges": charges, "rmsd_matrix_A": mpa_mat},
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(detailed_rows), cross_matrices


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PRX Energy Fig. 1a-c & cross-charge RMSD figures.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo data to preview figures")
    parser.add_argument(
        "--model-order",
        type=str,
        nargs="*",
        default=None,
        help="Optional custom display order of model names",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path(config["plotting"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(config["evaluation"].get("rmsd_threshold", 0.10))
    defect_label = config["dataset"].get("defect_name", "V_Sb(1)")
    highlight_charges = [int(q) for q in config["plotting"].get("pes_highlight_charges", [0, -3])]
    custom_order = args.model_order or config.get("plotting", {}).get("model_order") or DEFAULT_MODEL_ORDER

    if args.demo:
        df_summary, df_detailed, cross_matrices = generate_demo_data()
    else:
        summary_csv = Path(config["evaluation"]["summary_csv"])
        detailed_csv = Path(config["evaluation"]["detailed_csv"])
        matrix_json = summary_csv.parent / "cross_charge_rmsd_matrices.json"
        df_summary = pd.read_csv(summary_csv)
        df_detailed = pd.read_csv(detailed_csv)
        with open(matrix_json, "r", encoding="utf-8") as f:
            cross_matrices = json.load(f)

    f1 = plot_fig1a_rmsd(df_summary, threshold, defect_label, out_dir, custom_order)
    f2 = plot_fig1bc_pes(df_detailed, highlight_charges, threshold, defect_label, out_dir, custom_order)
    f3 = plot_cross_charge_matrices(cross_matrices, out_dir, custom_order)

    print(f"[INFO] Generated Fig. 1a ground-state RMSD plot:      {f1}")
    print(f"[INFO] Generated Fig. 1b/c PES distortion plot:       {f2}")
    print(f"[INFO] Generated cross-charge sensitivity heatmaps:   {f3}")


if __name__ == "__main__":
    main()
