"""Draw Fig. 4 exclusively from its derived summary CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from figure_style import LINE_WIDTHS, apply_arial_style, manuscript_width
sys.path.insert(0, str(HERE))
from plot_p1_vs_delta_v import PALETTE, gaussian_cdf_model


def main() -> None:
    fig_root = HERE.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=fig_root / "data" / "derived" / "fig04_p1_vs_delta_v__summary.csv")
    parser.add_argument("--output", type=Path, default=fig_root / "figure" / "fig04_p1_vs_delta_v.png")
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    apply_arial_style()
    fig, axes = plt.subplots(1, 3, figsize=(manuscript_width(0.85), 1.65), sharey=True)
    for index, (axis, voltage, color) in enumerate(zip(axes, (0.9, 1.1, 1.3), PALETTE.values())):
        block = table.loc[table["vdd_v"].eq(voltage)].sort_values("delta_V")
        axis.scatter(block["delta_V"], block["p1"], s=54, color=color, alpha=.82, linewidths=0, label="Simulated")
        x = np.linspace(block["delta_V"].min(), block["delta_V"].max(), 400)
        axis.plot(x, gaussian_cdf_model(x, block["fit_x0_v"].iloc[0], block["fit_sigma_v"].iloc[0]), color="black", linewidth=LINE_WIDTHS["primary"], label="Model fit")
        axis.text(.03, .93, f"({chr(97 + index)}) {voltage:.1f} V", transform=axis.transAxes, va="top", fontweight="bold")
        axis.text(.05, .78, rf"$R^2 = {block['fit_r_squared'].iloc[0]:.6f}$", transform=axis.transAxes, va="top")
        axis.set_xlabel(r"External $\Delta V$ (V)")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
        axis.set_ylim(-.03, 1.03)
        axis.grid(True, linestyle=":", alpha=.45, linewidth=LINE_WIDTHS["grid"])
        axis.legend(loc="lower right", fontsize=5.5, frameon=True)
    axes[0].set_ylabel("Probability of 1")
    fig.subplots_adjust(left=.13, right=.99, bottom=.30, top=.93, wspace=.22)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    fig.savefig(args.output.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
