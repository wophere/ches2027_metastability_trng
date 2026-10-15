#!/usr/bin/env python3
"""Generate the FPGA Fig. 10: ACF power laws and bias versus non-IID evidence.

The analysis retains every capture.  The only highlighted points are the
minimum-|p1-0.5| captures, used as best-balanced counterexamples rather than
as the source of the ACF estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, ScalarFormatter

ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ARTIFACT_ROOT))
from figure_style import LEGEND_STYLE, LINE_WIDTHS, apply_arial_style, manuscript_width


FREQS = ("10M", "25M", "50M")
CASE_TEMPLATE = "{freq}_dead0_tap"
COLORS = {"10M": "#0F4D92", "25M": "#E28E2C", "50M": "#42949E"}
LINESTYLES = {"10M": "-", "25M": "--", "50M": "-."}
MARKERS = {"10M": "o", "25M": "s", "50M": "^"}

BITS_PER_CAPTURE = 1_000_000
EXPECTED_CAPTURES = 10
PLOT_MIN_LAG = 10
PLOT_MAX_LAG = 10_000
PLOT_BIN_COUNT = 14
FIT_MIN_LAG = 500
FIT_MAX_LAG = 10_000
FIT_BIN_COUNT = 8
BLOCK_BITS = 4096
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260711


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=script_dir.parent / "data/raw/fig10_fpga_acf_fft_raw",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_dir.parent / "data/derived",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def run_pattern(case: str, kind: str) -> re.Pattern[str]:
    suffix = r"\.bin" if kind == "raw" else r"_acf\.csv"
    return re.compile(rf"(?:fig10_)?raw_{re.escape(case)}_run(?P<run>\d+){suffix}$")


def list_run_files(case_dir: Path, case: str, kind: str) -> list[tuple[int, Path]]:
    pattern = run_pattern(case, kind)
    glob_pattern = "*raw_*.bin" if kind == "raw" else "*raw_*_acf.csv"
    files: list[tuple[int, Path]] = []
    for path in sorted(case_dir.glob(glob_pattern)):
        match = pattern.fullmatch(path.name)
        if match:
            files.append((int(match.group("run")), path))
    if len(files) != EXPECTED_CAPTURES:
        raise FileNotFoundError(
            f"Expected {EXPECTED_CAPTURES} {kind} files for {case}, found {len(files)}"
        )
    return files


def read_acf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lags: list[int] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            lags.append(int(row["lag"]))
            values.append(float(row["acf"]))
    return np.asarray(lags, dtype=np.int64), np.asarray(values, dtype=np.float64)


def load_bits_pm1(path: Path) -> np.ndarray:
    packed = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    bits = np.unpackbits(packed, bitorder="little")
    if bits.size != BITS_PER_CAPTURE:
        raise ValueError(f"Expected {BITS_PER_CAPTURE} bits in {path}, found {bits.size}")
    return 2.0 * bits.astype(np.float64) - 1.0


def log_bin_masks(
    lags: np.ndarray,
    min_lag: int,
    max_lag: int,
    bin_count: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    selected = (lags >= min_lag) & (lags <= max_lag)
    selected_lags = lags[selected]
    edges = np.unique(
        np.round(np.geomspace(min_lag, max_lag + 1, bin_count + 1)).astype(int)
    )
    centers: list[float] = []
    masks: list[np.ndarray] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (lags >= low) & (lags < high)
        if not np.any(mask):
            continue
        centers.append(float(np.exp(np.mean(np.log(lags[mask])))))
        masks.append(mask)
    if len(masks) < 5 or selected_lags.size == 0:
        raise ValueError(f"Insufficient ACF points in lag window {min_lag}-{max_lag}")
    return np.asarray(centers, dtype=np.float64), masks


def fit_power_law(lag_centers: np.ndarray, acf_values: np.ndarray) -> tuple[float, float, float]:
    if np.any(acf_values <= 0):
        raise ValueError("Power-law log fit requires positive binned ACF values")
    x = np.log(lag_centers)
    y = np.log(acf_values)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - residual / total if total > 0 else float("nan")
    return float(-slope), float(np.exp(intercept)), float(r2)


def bootstrap_power_law(
    binned_by_capture: np.ndarray,
    lag_centers: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, float, int]:
    indices = rng.integers(0, binned_by_capture.shape[0], size=(draws, binned_by_capture.shape[0]))
    bootstrap_means = binned_by_capture[indices].mean(axis=1)
    betas: list[float] = []
    for values in bootstrap_means:
        if np.all(values > 0):
            beta, _, _ = fit_power_law(lag_centers, values)
            betas.append(beta)
    if len(betas) < draws * 0.99:
        raise ValueError("Too many bootstrap resamples contain nonpositive ACF bins")
    low, high = np.percentile(np.asarray(betas), [2.5, 97.5])
    return float(low), float(high), len(betas)


def summarize_case(
    case_dir: Path,
    case: str,
    rng: np.random.Generator,
    draws: int,
) -> tuple[dict, list[dict], list[dict]]:
    acf_files = list_run_files(case_dir, case, "acf")
    raw_files = list_run_files(case_dir, case, "raw")
    if [run for run, _ in acf_files] != [run for run, _ in raw_files]:
        raise ValueError(f"Raw/ACF run mismatch in {case_dir}")

    lags: np.ndarray | None = None
    acf_curves: list[np.ndarray] = []
    for (_, path), (_, raw_path) in zip(acf_files, raw_files):
        run_lags, reference_acf = read_acf(path)
        # Original acquisition analysis: centered FFT, overlap-count and
        # global-variance normalization. Raw captures use little-endian bits.
        values = load_bits_pm1(raw_path)
        values -= values.mean()
        n = values.size
        nfft = 1 << (2 * n - 1).bit_length()
        spectrum = np.fft.rfft(values, n=nfft)
        corr = np.fft.irfft(spectrum * spectrum.conj(), n=nfft)
        run_acf = corr[run_lags] / (n - run_lags) / np.mean(values * values)
        if not np.allclose(run_acf, reference_acf, rtol=0, atol=5e-12):
            raise ValueError(f"Raw FFT differs from archived ACF: {raw_path.name}")
        recomputed = Path(__file__).resolve().parents[1] / 'data/recomputed/acf' / case
        recomputed.mkdir(parents=True, exist_ok=True)
        np.savetxt(recomputed / path.name, np.column_stack([run_lags, run_acf]),
                   delimiter=',', header='lag,acf', comments='', fmt=['%d', '%.15g'])
        if lags is None:
            lags = run_lags
        elif not np.array_equal(lags, run_lags):
            raise ValueError(f"ACF lag grid mismatch in {path}")
        acf_curves.append(run_acf)
    assert lags is not None
    acf_matrix = np.asarray(acf_curves, dtype=np.float64)

    lag_centers, plot_masks = log_bin_masks(lags, PLOT_MIN_LAG, PLOT_MAX_LAG, PLOT_BIN_COUNT)
    fit_lag_centers, fit_masks = log_bin_masks(lags, FIT_MIN_LAG, FIT_MAX_LAG, FIT_BIN_COUNT)
    binned_by_capture = np.column_stack([acf_matrix[:, mask].mean(axis=1) for mask in plot_masks])
    fit_binned_by_capture = np.column_stack(
        [acf_matrix[:, mask].mean(axis=1) for mask in fit_masks]
    )
    acf_mean = binned_by_capture.mean(axis=0)
    fit_acf_mean = fit_binned_by_capture.mean(axis=0)
    if np.any(fit_acf_mean <= 0):
        raise ValueError("A mean ACF fit bin is nonpositive; do not censor it for a log-log plot")
    indices = rng.integers(0, EXPECTED_CAPTURES, size=(draws, EXPECTED_CAPTURES))
    bootstrap_means = binned_by_capture[indices].mean(axis=1)
    acf_ci_low, acf_ci_high = np.percentile(bootstrap_means, [2.5, 97.5], axis=0)
    beta, amplitude, r2_log = fit_power_law(fit_lag_centers, fit_acf_mean)
    beta_ci_low, beta_ci_high, valid_bootstrap = bootstrap_power_law(
        fit_binned_by_capture, fit_lag_centers, rng, draws
    )

    run_rows: list[dict] = []
    for (run, path), acf_values in zip(raw_files, fit_binned_by_capture):
        bits = load_bits_pm1(path)
        mu = float(np.mean(bits))
        block_count = bits.size // BLOCK_BITS
        block_means = bits[: block_count * BLOCK_BITS].reshape(block_count, BLOCK_BITS).mean(axis=1)
        observed_var = float(np.var(block_means, ddof=1))
        iid_var = float((1.0 - mu * mu) / BLOCK_BITS)
        inflation = observed_var / iid_var
        run_rows.append(
            {
                "run": run,
                "p1": float((mu + 1.0) / 2.0),
                "signed_bias": float(mu / 2.0),
                "abs_bias_percent": float(abs(mu / 2.0) * 100.0),
                "inflation_B4096": float(inflation),
                "acf_fit_window_mean": float(np.mean(acf_values)),
            }
        )
    min_bias_run = min(run_rows, key=lambda row: row["abs_bias_percent"])["run"]

    acf_rows = [
        {
            "lag_center": float(center),
            "acf_mean": float(mean),
            "acf_ci_low": float(low),
            "acf_ci_high": float(high),
            "n_captures": EXPECTED_CAPTURES,
        }
        for center, mean, low, high in zip(lag_centers, acf_mean, acf_ci_low, acf_ci_high)
    ]
    summary = {
        "n_captures": EXPECTED_CAPTURES,
        "bits_per_capture": BITS_PER_CAPTURE,
        "display_lag_window": f"{PLOT_MIN_LAG}-{PLOT_MAX_LAG}",
        "fit_lag_window": f"{FIT_MIN_LAG}-{FIT_MAX_LAG}",
        "fit_bins": int(fit_lag_centers.size),
        "beta": beta,
        "beta_ci_low": beta_ci_low,
        "beta_ci_high": beta_ci_high,
        "r2_log": r2_log,
        "amplitude": amplitude,
        "valid_beta_bootstraps": valid_bootstrap,
        "minimum_bias_run": min_bias_run,
    }
    return summary, acf_rows, run_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_figure(results: dict[str, dict], out_prefix: Path) -> None:
    apply_arial_style()
    fig = plt.figure(figsize=(manuscript_width(0.95), 1.85), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.34)
    ax_acf = fig.add_subplot(grid[0, 0])
    ax_bias = fig.add_subplot(grid[0, 1])
    acf_handles: list[Line2D] = []

    ax_acf.axvspan(FIT_MIN_LAG, FIT_MAX_LAG, color="#C96B6B", alpha=0.12, lw=0, zorder=0)

    for freq in FREQS:
        result = results[freq]
        summary = result["summary"]
        rows = result["acf_rows"]
        lag = np.asarray([row["lag_center"] for row in rows])
        acf = np.asarray([row["acf_mean"] for row in rows])
        low = np.asarray([row["acf_ci_low"] for row in rows])
        high = np.asarray([row["acf_ci_high"] for row in rows])
        color = COLORS[freq]

        ax_acf.errorbar(
            lag,
            acf,
            yerr=np.vstack((acf - low, high - acf)),
            color=color,
            fmt=MARKERS[freq],
            markersize=4.6,
            markerfacecolor="white",
            markeredgewidth=LINE_WIDTHS["marker_edge"],
            linestyle="none",
            elinewidth=LINE_WIDTHS["secondary"],
            capsize=1.6,
            zorder=3,
        )
        fit_x = np.geomspace(FIT_MIN_LAG, FIT_MAX_LAG, 200)
        fit_y = summary["amplitude"] * fit_x ** (-summary["beta"])
        ax_acf.plot(fit_x, fit_y, color=color, linewidth=LINE_WIDTHS["primary"], linestyle=LINESTYLES[freq], zorder=2)
        acf_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=LINESTYLES[freq],
                linewidth=LINE_WIDTHS["primary"],
                marker=MARKERS[freq],
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=LINE_WIDTHS["marker_edge"],
                markersize=4.5,
                label=(
                    rf"{freq[:-1]} MHz fit: $R^2={summary['r2_log']:.3f}$"
                ),
            )
        )

    ax_acf.set_xscale("log")
    ax_acf.set_yscale("symlog", linthresh=1e-3, linscale=0.55)
    ax_acf.set_xlim(PLOT_MIN_LAG * 0.88, PLOT_MAX_LAG * 1.18)
    ax_acf.set_ylim(-2.5e-2, 1.2e-1)
    ax_acf.set_xticks([10, 100, 1000, 10000])
    ax_acf.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax_acf.set_yticks([-1e-2, 0.0, 1e-3, 1e-2])
    ax_acf.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: "0"
            if value == 0
            else (rf"$-10^{{{int(np.log10(abs(value)))}}}$" if value < 0
                  else rf"$10^{{{int(np.log10(value))}}}$")
        )
    )
    ax_acf.set_xlabel(r"Lag $m$ (samples)")
    ax_acf.set_ylabel("ACF", labelpad=1)
    ax_acf.minorticks_on()
    ax_acf.grid(which="both", color="#BEBEBE", linestyle=":", linewidth=LINE_WIDTHS["grid"], alpha=0.82)
    acf_legend_style = dict(LEGEND_STYLE)
    acf_legend_style.update(
        {"loc": "lower right", "handlelength": 1.65, "handletextpad": 0.32, "labelspacing": 0.15}
    )
    ax_acf.legend(handles=acf_handles, **acf_legend_style)
    for freq in FREQS:
        rows = results[freq]["run_rows"]
        x = np.asarray([row["abs_bias_percent"] for row in rows])
        y = np.asarray([row["inflation_B4096"] for row in rows])
        color = COLORS[freq]
        ax_bias.scatter(
            x,
            y,
            s=43,
            color=color,
            marker=MARKERS[freq],
            alpha=0.82,
            edgecolor="white",
            linewidth=LINE_WIDTHS["marker_edge"],
            zorder=3,
            label=f"{freq[:-1]} MHz",
        )
        best = min(rows, key=lambda row: row["abs_bias_percent"])
        ax_bias.scatter(
            best["abs_bias_percent"],
            best["inflation_B4096"],
            s=135,
            color=color,
            marker="*",
            edgecolor="#272727",
            linewidth=LINE_WIDTHS["secondary"],
            zorder=5,
        )

    ax_bias.axhline(1.0, color="#4D4D4D", linewidth=LINE_WIDTHS["secondary"], linestyle=":", zorder=1)
    ax_bias.text(12.6, 1.12, "i.i.d.", color="#111111", va="bottom", ha="right")
    ax_bias.set_xscale("log")
    ax_bias.set_yscale("log")
    ax_bias.set_xlim(0.30, 20.0)
    ax_bias.set_ylim(0.8, 45.0)
    ax_bias.set_xticks([0.5, 1, 2, 5, 10])
    ax_bias.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax_bias.set_yticks([1, 10, 30])
    ax_bias.yaxis.set_major_formatter(ScalarFormatter())
    ax_bias.set_xlabel(r"$|\hat{p}_1-0.5|$ (%)", labelpad=1)
    ax_bias.set_ylabel(r"$I_{4096}$")
    ax_bias.minorticks_on()
    ax_bias.grid(which="both", color="#BEBEBE", linestyle=":", linewidth=LINE_WIDTHS["grid"], alpha=0.82)

    rate_handles = [
        Line2D([0], [0], marker=MARKERS[freq], color="none", markerfacecolor=COLORS[freq],
               markeredgecolor="white", markersize=5.0, label=f"{freq[:-1]} MHz")
        for freq in FREQS
    ]
    rate_handles.append(
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#FFFFFF",
               markeredgecolor="#272727", markersize=7.6, label="min-bias capture")
    )
    bias_legend_style = dict(LEGEND_STYLE)
    bias_legend_style.update(
        {"loc": "lower left", "bbox_to_anchor": (-0.02, 0.01), "labelspacing": 0.22}
    )
    ax_bias.legend(handles=rate_handles, **bias_legend_style)

    fig.subplots_adjust(left=0.13, right=0.993, bottom=0.285, top=0.965, wspace=0.34)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(
        out_prefix.with_suffix(".pdf"),
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(out_prefix.with_suffix(".png"), dpi=360, facecolor="white")
    try:
        fig.savefig(out_prefix.with_suffix(".tiff"), dpi=600, facecolor="white")
    except OSError as exc:
        print(f"Warning: skipped TIFF export: {exc}")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results: dict[str, dict] = {}
    acf_output: list[dict] = []
    run_output: list[dict] = []
    summary_output: list[dict] = []

    for freq in FREQS:
        case = CASE_TEMPLATE.format(freq=freq)
        summary, acf_rows, run_rows = summarize_case(
            args.reports_root / case, case, rng, args.bootstrap_draws
        )
        results[freq] = {"summary": summary, "acf_rows": acf_rows, "run_rows": run_rows}
        acf_output.extend({"freq": freq, "case": case, **row} for row in acf_rows)
        run_output.extend({"freq": freq, "case": case, **row} for row in run_rows)
        summary_output.append({"freq": freq, "case": case, **summary})

    args.output_root.mkdir(parents=True, exist_ok=True)
    prefix = args.output_root / "fpga_powerlaw_bias_independence"
    write_csv(prefix.with_name(prefix.name + "_acf.csv"), acf_output)
    write_csv(prefix.with_name(prefix.name + "_run_metrics.csv"), run_output)
    write_csv(prefix.with_name(prefix.name + "_summary.csv"), summary_output)
    prefix.with_name(prefix.name + "_summary.json").write_text(
        json.dumps(
            {
                "analysis_contract": {
                    "acf_center": "arithmetic mean across all 10 captures per rate",
                    "acf_interval": "capture-level percentile bootstrap 95% interval",
                    "fit": "ordinary least squares in log ACF versus log lag",
                    "fit_window": [FIT_MIN_LAG, FIT_MAX_LAG],
                    "acf_censoring": "none; signed plotted bins cover 10-10000 on a symlog y-axis, fit uses positive bins over 500-10000",
                    "non_iid_metric": "capture-level variance inflation at B=4096",
                    "minimum_bias_highlight": "display-only best-balanced counterexample; not used for the ACF estimate",
                    "bootstrap_draws": args.bootstrap_draws,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                },
                "summary": summary_output,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.skip_plot:
        plot_figure(results, prefix)
        print(f"Wrote {prefix.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
