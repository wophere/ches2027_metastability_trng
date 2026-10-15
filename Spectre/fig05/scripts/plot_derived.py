"""Draw Fig. 5 exclusively from archived derived samples and fit parameters."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ARTIFACT_ROOT = HERE.parents[2]
sys.path.insert(0, str(ARTIFACT_ROOT))
from figure_style import LINE_WIDTHS, apply_arial_style, manuscript_width
sys.path.insert(0, str(HERE))
from plot_resolution_time_fit import COLORS, VOLTAGES, VOLTAGE_LABELS, metastability_pdf_ps


def main() -> None:
    figure_root = HERE.parent
    derived = figure_root / "data" / "derived"
    samples = pd.read_csv(derived / "metastability_resolution_samples.csv")
    parameters = pd.read_csv(derived / "metastability_resolution_fit_parameters.csv").set_index("voltage_v")
    apply_arial_style()
    fig, axes = plt.subplots(1, 3, figsize=(manuscript_width(0.85), 1.65), sharey=True)
    for index, (axis, voltage, label, color) in enumerate(zip(axes, VOLTAGES, VOLTAGE_LABELS, COLORS)):
        values = samples.loc[samples["voltage_v"].eq(voltage), "resolution_time_ps"].to_numpy()
        row = parameters.loc[voltage]
        axis.hist(values, bins=50, density=True, alpha=0.32, color=color, edgecolor="none", label="Simulated")
        x = np.linspace(float(values.min()), float(values.max()), 700)
        axis.plot(x, metastability_pdf_ps(x, row.theta_reg_s_inv, row.mu_v, row.sigma_eff_v, row.v_t_v), color="black", lw=LINE_WIDTHS["primary"], label="Model fit")
        axis.text(0.03, 0.94, f"({chr(97 + index)}) {label}", transform=axis.transAxes, va="top")
        axis.text(0.96, 0.08, rf"$R^2 = {row.r_squared_histogram_density:.4f}$", transform=axis.transAxes, va="bottom", ha="right")
        axis.set_xlabel("Resolution time (ps)")
        axis.grid(True, linestyle=":", linewidth=LINE_WIDTHS["grid"], alpha=0.55)
        axis.legend(loc="upper right", fontsize=5.5, frameon=True)
    axes[0].set_ylabel("Density (1/ps)")
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.30, top=0.93, wspace=0.22)
    output = figure_root / "figure" / "fig05_resolution_time_fit.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
