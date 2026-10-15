from __future__ import annotations

import math
from pathlib import Path
import sys
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
ARTIFACT_ROOT = HERE.parents[2]
sys.path.insert(0, str(ARTIFACT_ROOT))
from figure_style import LINE_WIDTHS, apply_arial_style, manuscript_width
FIG_ROOT = HERE.parent
DATA = FIG_ROOT / "data" / "derived"
if os.environ.get('ARTIFACT_ASIC_RECOMPUTED') == '1':
    DATA = FIG_ROOT.parent / 'raw_reproduction/data/recomputed'
PARAMS = DATA / "fig14_platform_split_parameters.csv"
HELDOUT = DATA / "fig14_platform_split_heldout_block.csv"
OUT = FIG_ROOT / "figure" / "fig14_asic_core1_platform_params.pdf"

TEMP_ORDER = ("-20C", "27C", "100C")
TEMP_LABELS = {"-20C": "−20 °C", "27C": "27 °C", "100C": "100 °C"}
TEMP_COLORS = {"-20C": "#0072B2", "27C": "#009E73", "100C": "#D55E00"}
TEMP_MARKERS = {"-20C": "o", "27C": "s", "100C": "^"}
TEMP_OFFSETS = {"-20C": -0.23, "27C": 0.0, "100C": 0.23}
CASE_ORDER = ("0.9V/F1", "1.1V/F1", "1.1V/F2", "1.1V/F3", "1.3V/F1", "1.3V/F2", "1.3V/F3", "1.3V/F4")


def identity_limits(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    values = np.r_[x[np.isfinite(x) & (x > 0.0)], y[np.isfinite(y) & (y > 0.0)]]
    return 10.0 ** math.floor(math.log10(float(values.min()))), 10.0 ** math.ceil(math.log10(float(values.max())))


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontweight="bold", va="top")


def plot_figure(params: pd.DataFrame, heldout: pd.DataFrame) -> None:
    apply_arial_style()
    params_a = params[params["split"] == "A"]
    params_b = params[params["split"] == "B"]
    fig, axes = plt.subplots(1, 3, figsize=(manuscript_width(1.0), 2.35), gridspec_kw={"width_ratios": [1.05, 1.0, 1.0]})

    ax = axes[0]
    panel_label(ax, "a")
    for temp in TEMP_ORDER:
        group = heldout[
            (heldout["temperature_label"] == temp)
            & (heldout["excess_var"] > 0.0)
            & (heldout["excess_var_prediction"] > 0.0)
        ]
        passed = group["calibration_reportable"].to_numpy(bool)
        for ok, face, alpha in [(True, TEMP_COLORS[temp], 0.58), (False, "none", 0.95)]:
            sub = group[passed == ok]
            if len(sub):
                ax.scatter(
                    sub["excess_var_prediction"],
                    sub["excess_var"],
                    s=16,
                    marker=TEMP_MARKERS[temp],
                    facecolor=face,
                    edgecolor=TEMP_COLORS[temp],
                    alpha=alpha,
                    linewidth=LINE_WIDTHS["secondary"],
                )
    low, high = identity_limits(heldout["excess_var_prediction"].to_numpy(float), heldout["excess_var"].to_numpy(float))
    ax.plot([low, high], [low, high], color="#333333", linestyle="--", linewidth=LINE_WIDTHS["secondary"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("prediction from A")
    ax.set_ylabel("held-out B excess variance")
    ax.set_title("frozen-parameter prediction", loc="left")
    ax.grid(True, which="major", color="#e1e1e1", linewidth=LINE_WIDTHS["grid"])

    base_x = np.arange(len(CASE_ORDER), dtype=float)
    panel_specs = [
        (axes[1], "r_f", r"flicker-to-white ratio $r_f$", r"independent $r_f$ extraction", False, "b"),
        (axes[2], "tau_c_over_Ts", r"normalized crossover $\tau_c/T_s$", r"independent $\tau_c/T_s$", True, "c"),
    ]
    for ax, column, ylabel, title, log_scale, label in panel_specs:
        panel_label(ax, label)
        for temp in TEMP_ORDER:
            a = params_a[params_a["temperature_label"] == temp].set_index("case_label").reindex(CASE_ORDER)
            b = params_b[params_b["temperature_label"] == temp].set_index("case_label").reindex(CASE_ORDER)
            positions = base_x + TEMP_OFFSETS[temp]
            values_a = a[column].to_numpy(float)
            values_b = b[column].to_numpy(float)
            for x, y0, y1 in zip(positions, values_a, values_b, strict=True):
                if np.isfinite(y0) and np.isfinite(y1):
                    ax.plot([x, x], [y0, y1], color=TEMP_COLORS[temp], linewidth=LINE_WIDTHS["secondary"], alpha=0.55, zorder=1)
            ax.scatter(positions, values_a, s=20, marker=TEMP_MARKERS[temp], color=TEMP_COLORS[temp], edgecolor="white", linewidth=LINE_WIDTHS["marker_edge"], zorder=3)
            ax.scatter(positions, values_b, s=22, marker=TEMP_MARKERS[temp], facecolor="white", edgecolor=TEMP_COLORS[temp], linewidth=LINE_WIDTHS["secondary"], zorder=4)
        ax.set_xticks(base_x)
        ax.set_xticklabels([case.replace("V/", " V/") for case in CASE_ORDER], rotation=38, ha="right")
        ax.set_xlabel("voltage/frequency corner")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(True, axis="y", color="#e1e1e1", linewidth=LINE_WIDTHS["grid"])
        if log_scale:
            ax.set_yscale("log")

    temp_handles = [
        Line2D([0], [0], marker=TEMP_MARKERS[temp], color=TEMP_COLORS[temp], markerfacecolor=TEMP_COLORS[temp], markersize=4.5, linewidth=0.8, label=TEMP_LABELS[temp])
        for temp in TEMP_ORDER
    ]
    split_handles = [
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="#555555", markersize=4.2, linewidth=0, label="A: captures 1–5"),
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white", markersize=4.2, linewidth=0, label="B: captures 6–10"),
    ]
    fig.legend(
        handles=temp_handles + split_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        ncol=5,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    fig.subplots_adjust(left=0.13, right=0.994, bottom=0.24, top=0.80, wspace=0.38)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)


def main() -> int:
    plot_figure(pd.read_csv(PARAMS), pd.read_csv(HELDOUT))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
