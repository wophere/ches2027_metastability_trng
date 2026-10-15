import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
from scipy.special import ndtr

ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ARTIFACT_ROOT))
from figure_style import LINE_WIDTHS, apply_arial_style, manuscript_width


DELTA_V_RE = re.compile(
    r"delta_V\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

PALETTE = {
    "blue_soft": "#9ecae1",
    "orange_soft": "#f2c49b",
    "teal_soft": "#9fd8c9",
}


def extract_delta_v(column_name: str) -> float:
    match = DELTA_V_RE.search(column_name)
    if match is None:
        raise ValueError(f"Cannot parse delta_V from column name: {column_name}")
    return float(match.group(1))


def gaussian_cdf_model(x, x0, sigma):
    return ndtr((x - x0) / sigma)


def compute_p1_vs_delta_v(csv_file: Path, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    rows = []

    for column in df.columns[1:]:
        delta_v = extract_delta_v(column)
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        p1 = (values > threshold).mean()
        rows.append({"delta_V": delta_v, "p1": p1})

    return pd.DataFrame(rows).sort_values("delta_V").reset_index(drop=True)


def fit_gaussian_cdf(delta_v: np.ndarray, p1: np.ndarray):
    x0_guess = float(np.median(delta_v))
    sigma_guess = max(float((delta_v.max() - delta_v.min()) / 10.0), 1e-9)

    params, _ = curve_fit(
        gaussian_cdf_model,
        delta_v,
        p1,
        p0=[x0_guess, sigma_guess],
        bounds=([-np.inf, 1e-12], [np.inf, np.inf]),
        maxfev=20000,
    )

    fitted = gaussian_cdf_model(delta_v, *params)
    residual = p1 - fitted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((p1 - np.mean(p1)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return params, fitted, r_squared


def main():
    parser = argparse.ArgumentParser(
        description="Plot p1 versus delta_V for three voltage corners in a 1x3 layout."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="Input CSV files. Provide three files for 0.9 V, 1.1 V, and 1.3 V cases.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Threshold for p1 = P(value > threshold). Default: 0.45",
    )
    parser.add_argument(
        "--output",
        default="p1_vs_delta_v_1x3.png",
        help="Output figure filename. Default: p1_vs_delta_v_1x3.png",
    )
    parser.add_argument(
        "--summary",
        default="p1_vs_delta_v_summary.csv",
        help="Output CSV summary filename. Default: p1_vs_delta_v_summary.csv",
    )
    parser.add_argument(
        "--labels",
        nargs=3,
        default=["0.9V", "1.1V", "1.3V"],
        help="Three panel labels in order. Default: 0.9V 1.1V 1.3V",
    )
    parser.add_argument(
        "--show-r2",
        action="store_true",
        help="Show R^2 text in each panel.",
    )
    args = parser.parse_args()

    if len(args.csv_files) != 3:
        raise ValueError("Please provide exactly three CSV files for the 1x3 figure.")

    apply_arial_style()
    fig, axes = plt.subplots(1, 3, figsize=(manuscript_width(0.85), 1.65), sharey=True)
    all_results = []
    raw_colors = [PALETTE["blue_soft"], PALETTE["orange_soft"], PALETTE["teal_soft"]]

    for idx, (ax, csv_name, raw_color, panel_label) in enumerate(
        zip(axes, args.csv_files, raw_colors, args.labels)
    ):
        csv_file = Path(csv_name)
        result = compute_p1_vs_delta_v(csv_file, args.threshold)
        params, _, r_squared = fit_gaussian_cdf(
            result["delta_V"].to_numpy(), result["p1"].to_numpy()
        )

        x_data = result["delta_V"].to_numpy()
        y_data = result["p1"].to_numpy()
        x_fit = np.linspace(x_data.min(), x_data.max(), 400)
        y_fit = gaussian_cdf_model(x_fit, *params)

        result.insert(0, "source", panel_label)
        all_results.append(result)

        ax.scatter(
            x_data,
            y_data,
            s=54,
            color=raw_color,
            alpha=0.82,
            linewidths=0,
            label="Simulated",
        )
        ax.plot(x_fit, y_fit, color="black", linewidth=LINE_WIDTHS["primary"], label="Model fit")
        ax.text(
            0.03,
            0.93,
            f"({chr(ord('a') + idx)}) {panel_label.replace('V', ' V')}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        ax.set_xlabel(r"External $\Delta V$ (V)")
        ax.grid(True, linestyle=":", alpha=0.45, linewidth=LINE_WIDTHS["grid"])
        # With body-sized tick labels, three well-separated ΔV ticks remain
        # readable after the fixed manuscript-scale reduction.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_ylim(-0.03, 1.03)
        if args.show_r2:
            ax.text(
                0.05,
                0.82,
                rf"$R^2 = {r_squared:.6f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
            )
        ax.legend(
            loc="lower right",
            fontsize=5.5,
            handlelength=1.2,
            handletextpad=0.3,
            borderpad=0.2,
            labelspacing=0.15,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9,
            fancybox=False,
        )
    axes[0].set_ylabel("Probability of 1")
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.30, top=0.93, wspace=0.22)

    summary = pd.concat(all_results, ignore_index=True)
    summary.to_csv(args.summary, index=False)
    fig.savefig(args.output, dpi=300)
    output_path = Path(args.output)
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(output_path.with_suffix(".svg"))
    plt.close(fig)

    print(f"Saved figure: {args.output}")
    print(f"Saved figure: {output_path.with_suffix('.pdf')}")
    print(f"Saved figure: {output_path.with_suffix('.svg')}")
    print(f"Saved summary: {args.summary}")


if __name__ == "__main__":
    main()
