#!/usr/bin/env python3
"""Render Fig. 12 from the Eq. (23)-inverted latent ACF fit data.

The panel geometry and visual styling deliberately match the previous Fig. 12.
Only the plotted ACF, the fitted model, and the beta/R2 annotations change.
"""

from __future__ import annotations

from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ARTIFACT_ROOT))
from figure_style import apply_arial_style


HERE = Path(__file__).resolve().parent
FIG_ROOT = HERE.parent
DATA_DIR = FIG_ROOT / "data" / "derived"
if os.environ.get('ARTIFACT_ASIC_RECOMPUTED') == '1':
    DATA_DIR = FIG_ROOT.parent / 'raw_reproduction/data/recomputed'
OUTPUTS = (
    FIG_ROOT / "figure" / "fig12_asic_core1_acf_three_temperature_powerlaw.pdf",
    FIG_ROOT / "figure" / "fig12_asic_core1_acf_three_temperature_powerlaw.png",
)
TEMP_ORDER = ("-20C", "27C", "100C")
TEMP_LABELS = {"-20C": "−20 °C", "27C": "27 °C", "100C": "100 °C"}
VOLTAGE_ORDER = (0.9, 1.1, 1.3)
LAG_MIN = 10
LAG_MAX = 10_000
FREQ_MARKERS = {1: "x", 2: "+", 3: "^", 4: "s"}
FREQ_LINESTYLES = {1: "-", 2: "--", 3: "-.", 4: ":"}
FREQ_COLORS = {1: "#D62728", 2: "#1F77B4", 3: "#2CA02C", 4: "#9467BD"}


def output_kwargs(path: Path) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "bbox_inches": "tight",
        "pad_inches": 0.02,
        "facecolor": "white",
    }
    if path.suffix.lower() == ".png":
        kwargs["dpi"] = 300
    return kwargs


def main() -> None:
    apply_arial_style()
    points = pd.read_csv(DATA_DIR / "latent_acf_binned_points.csv")
    fits = pd.read_csv(DATA_DIR / "latent_acf_fit_summary.csv")
    fig, axes = plt.subplots(
        len(TEMP_ORDER),
        len(VOLTAGE_ORDER),
        figsize=(17.2, 6.85),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    for row, temperature in enumerate(TEMP_ORDER):
        for col, voltage in enumerate(VOLTAGE_ORDER):
            ax = axes[row, col]
            panel_fits = fits[(fits["temperature_label"] == temperature) & np.isclose(fits["voltage"], voltage)]
            panel_points = points[(points["temperature_label"] == temperature) & np.isclose(points["voltage"], voltage)]
            handles: list[Line2D] = []
            for fit in panel_fits.sort_values("frequency").itertuples(index=False):
                frequency = int(fit.frequency)
                one = panel_points[panel_points["frequency"] == frequency]
                one = one[np.isfinite(one["rho_v_mean"]) & (one["rho_v_mean"] > 0)].copy()
                color = FREQ_COLORS[frequency]
                marker = FREQ_MARKERS[frequency]
                linestyle = FREQ_LINESTYLES[frequency]
                label = f"F{frequency}: $\\beta$={fit.beta:.3f}, $R^2$={fit.r2_log:.3f}"
                ax.plot(
                    one["lag_geo"], one["rho_v_mean"], linestyle="none", marker=marker,
                    markersize=5.8, markeredgewidth=1.05, color=color, alpha=0.72,
                )
                lag_line = np.geomspace(LAG_MIN, LAG_MAX, 240)
                tau_line = lag_line * float(fit.period_ticks)
                rho_line = np.exp(float(fit.intercept_log)) * np.power(tau_line, -float(fit.beta))
                ax.plot(lag_line, rho_line, color=color, linewidth=1.65, linestyle=linestyle)
                handles.append(
                    Line2D([0], [0], color=color, linewidth=1.65, linestyle=linestyle, marker=marker,
                           markersize=5.6, markeredgewidth=1.05, label=label)
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(LAG_MIN, LAG_MAX)
            ax.set_ylim(5e-4, 1e-1)
            ax.grid(True, which="both", linestyle=":", linewidth=0.55, color="#9a9a9a", alpha=0.72)
            ax.tick_params(axis="both", which="major", labelsize=17, length=3)
            ax.tick_params(axis="both", which="minor", length=2)
            ax.text(0.97, 0.92, f"{TEMP_LABELS[temperature]}, {voltage:.1f} V", transform=ax.transAxes, fontsize=18,
                    ha="right", va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.4})
            if row == 0:
                ax.set_title(f"{voltage:.1f} V", fontsize=19, pad=4)
            if row == len(TEMP_ORDER) - 1:
                ax.set_xlabel("Lag", fontsize=18)
            if col == 0:
                ax.set_ylabel("Latent ACF", fontsize=18)
            ax.legend(handles=handles, loc="lower left", fontsize=13.0, frameon=True, framealpha=0.95,
                      fancybox=False, edgecolor="black", borderpad=0.12, labelspacing=0.04,
                      handlelength=1.02, handletextpad=0.16)

    fig.subplots_adjust(left=0.058, right=0.985, top=0.955, bottom=0.1, wspace=0.14, hspace=0.24)
    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, **output_kwargs(output))
    plt.close(fig)


if __name__ == "__main__":
    main()
