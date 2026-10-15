#!/usr/bin/env python3
"""Compare random-bit CSVs with and without flicker noise.

The script reads two Cadence-style CSV files, quantizes the second column by a
voltage threshold, then plots:
  1. normalized autocorrelation of the quantized bit sequence
  2. histogram of p1 measured over non-overlapping blocks

Example:
    python analyze_flicker_random_bits.py with_flicker.csv no_flicker.csv \
        --output flicker_compare.png --pdf flicker_compare.pdf
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ARTIFACT_ROOT))
from figure_style import LINE_WIDTHS, apply_arial_style, manuscript_width


@dataclass(frozen=True)
class BitDataset:
    label: str
    path: Path
    times: np.ndarray
    values: np.ndarray
    bits: np.ndarray
    p1_blocks: np.ndarray
    acf: np.ndarray

    @property
    def p1_total(self) -> float:
        return float(np.mean(self.bits))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize two random-number CSVs, compare normalized ACF and "
            "per-block p1 histograms."
        )
    )
    parser.add_argument(
        "with_flicker_csv",
        type=Path,
        help="CSV file for the case with flicker noise.",
    )
    parser.add_argument(
        "without_flicker_csv",
        type=Path,
        help="CSV file for the case without flicker noise.",
    )
    parser.add_argument(
        "--with-label",
        default="Flicker enabled",
        help="Legend label for the flicker-noise CSV.",
    )
    parser.add_argument(
        "--without-label",
        default="Flicker suppressed",
        help="Legend label for the no-flicker CSV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Voltage threshold for quantization: value >= threshold becomes 1.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1000,
        help="Number of bits in each p1 block.",
    )
    parser.add_argument(
        "--include-partial-block",
        action="store_true",
        help="Include the final incomplete block when computing p1.",
    )
    parser.add_argument(
        "--acf-source",
        choices=("bits", "p1"),
        default="bits",
        help="Sequence used for ACF. Default is the full quantized bit stream.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=1000,
        help="Maximum ACF lag in samples of the selected ACF source.",
    )
    parser.add_argument(
        "--acf-xmin",
        type=float,
        default=0.0,
        help="Lower x-axis limit for the ACF plot.",
    )
    parser.add_argument(
        "--acf-xmax",
        type=float,
        default=1000.0,
        help="Upper x-axis limit for the ACF plot.",
    )
    parser.add_argument(
        "--show-zero-lag",
        action="store_true",
        help="Include lag 0 in the ACF plot. By default it is hidden to zoom near zero.",
    )
    parser.add_argument(
        "--acf-ymin",
        type=float,
        default=-0.02,
        help="Lower y-axis limit for the ACF plot.",
    )
    parser.add_argument(
        "--acf-ymax",
        type=float,
        default=0.02,
        help="Upper y-axis limit for the ACF plot.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of bins for the p1 histogram over [0, 1].",
    )
    parser.add_argument(
        "--hist-counts",
        action="store_true",
        help="Plot histogram counts instead of probability density.",
    )
    parser.add_argument(
        "--hist-xmin",
        type=float,
        default=0.3,
        help="Lower x-axis limit for the p1 histogram.",
    )
    parser.add_argument(
        "--hist-xmax",
        type=float,
        default=0.7,
        help="Upper x-axis limit for the p1 histogram.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("flicker_random_bits_compare.png"),
        help="Output image path, usually .png.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Optional vector PDF output path.",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=None,
        help="Optional CSV path for summary statistics.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster output.",
    )
    return parser.parse_args()


def read_two_column_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read time and voltage from a CSV whose second header may be verbose."""
    times: list[float] = []
    values: list[float] = []

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty.") from exc

        for line_no, row in enumerate(reader, start=2):
            if not row or len(row) < 2:
                continue
            try:
                time_value = float(row[0].strip())
                voltage_value = float(row[1].strip())
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse numeric values in {path} at line {line_no}: {row}"
                ) from exc
            times.append(time_value)
            values.append(voltage_value)

    if not values:
        raise ValueError(f"{path} has no numeric samples after the header.")

    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


def quantize(values: np.ndarray, threshold: float) -> np.ndarray:
    return (values >= threshold).astype(np.uint8)


def block_p1(
    bits: np.ndarray, block_size: int, include_partial_block: bool
) -> np.ndarray:
    if block_size <= 0:
        raise ValueError("--block-size must be positive.")

    full_blocks = len(bits) // block_size
    p1_values: list[float] = []

    if full_blocks:
        trimmed = bits[: full_blocks * block_size]
        p1_values.extend(trimmed.reshape(full_blocks, block_size).mean(axis=1))

    remainder = len(bits) % block_size
    if include_partial_block and remainder:
        p1_values.append(float(bits[full_blocks * block_size :].mean()))

    if not p1_values:
        raise ValueError(
            f"Only {len(bits)} samples are available, fewer than block size "
            f"{block_size}. Use --include-partial-block or reduce --block-size."
        )

    return np.asarray(p1_values, dtype=float)


def normalized_acf(sequence: np.ndarray, max_lag: int) -> np.ndarray:
    """Return ACF after mean removal and variance normalization.

    ACF[k] = mean((x[t] - mean(x)) * (x[t+k] - mean(x))) / var(x)
    over the overlapping samples at each lag k.
    """
    x = np.asarray(sequence, dtype=float)
    if x.ndim != 1:
        raise ValueError("ACF sequence must be one-dimensional.")
    if len(x) < 2:
        raise ValueError("Need at least two samples to compute ACF.")
    if max_lag < 0:
        raise ValueError("--max-lag must be non-negative.")

    x = x - np.mean(x)
    variance = float(np.mean(x * x))
    if variance == 0.0:
        raise ValueError("Cannot compute normalized ACF for a constant sequence.")

    max_lag = min(max_lag, len(x) - 1)
    n = len(x)
    fft_len = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, n=fft_len)
    autocov_sum = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_len)[
        : max_lag + 1
    ]
    overlap_counts = n - np.arange(max_lag + 1)
    return autocov_sum / overlap_counts / variance


def load_dataset(
    path: Path,
    label: str,
    threshold: float,
    block_size: int,
    include_partial_block: bool,
    acf_source: str,
    max_lag: int,
) -> BitDataset:
    if not path.exists():
        raise FileNotFoundError(path)

    times, values = read_two_column_csv(path)
    bits = quantize(values, threshold)
    p1_values = block_p1(bits, block_size, include_partial_block)
    acf_sequence = bits if acf_source == "bits" else p1_values
    acf = normalized_acf(acf_sequence, max_lag)

    return BitDataset(
        label=label,
        path=path,
        times=times,
        values=values,
        bits=bits,
        p1_blocks=p1_values,
        acf=acf,
    )


def common_lag_limit(datasets: Iterable[BitDataset]) -> int:
    return min(len(dataset.acf) for dataset in datasets)


def display_path(input_path: Path, base: Path) -> str:
    try:
        return input_path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(input_path)


def write_stats_csv(path: Path, datasets: list[BitDataset]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_base = Path(__file__).resolve().parents[1]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "path",
                "samples",
                "overall_p1",
                "p1_blocks",
                "p1_mean",
                "p1_std",
                "p1_min",
                "p1_max",
                "acf_lag1",
            ]
        )
        for dataset in datasets:
            writer.writerow(
                [
                    dataset.label,
                    display_path(dataset.path, artifact_base),
                    len(dataset.bits),
                    dataset.p1_total,
                    len(dataset.p1_blocks),
                    float(np.mean(dataset.p1_blocks)),
                    float(np.std(dataset.p1_blocks, ddof=0)),
                    float(np.min(dataset.p1_blocks)),
                    float(np.max(dataset.p1_blocks)),
                    float(dataset.acf[1]) if len(dataset.acf) > 1 else "",
                ]
            )


def plot_comparison(
    datasets: list[BitDataset],
    output: Path,
    pdf: Path | None,
    block_size: int,
    bins: int,
    hist_counts: bool,
    show_zero_lag: bool,
    acf_xmin: float,
    acf_xmax: float,
    acf_ymin: float,
    acf_ymax: float,
    hist_xmin: float,
    hist_xmax: float,
    dpi: int,
    acf_source: str,
) -> None:
    if acf_xmin >= acf_xmax:
        raise ValueError("--acf-xmin must be smaller than --acf-xmax.")
    if acf_ymin >= acf_ymax:
        raise ValueError("--acf-ymin must be smaller than --acf-ymax.")
    if hist_xmin >= hist_xmax:
        raise ValueError("--hist-xmin must be smaller than --hist-xmax.")

    colors = ["#0072B2", "#D55E00"]
    linestyles = ["-", "--"]
    lag_limit = common_lag_limit(datasets)
    lag_start = 0 if show_zero_lag else 1
    if lag_start >= lag_limit:
        raise ValueError(
            "ACF has no positive lags to plot. Reduce --show-zero-lag usage or add samples."
        )
    lags = np.arange(lag_start, lag_limit)
    hist_bins = np.linspace(0.0, 1.0, bins + 1)

    apply_arial_style()
    fig, axes = plt.subplots(1, 2, figsize=(manuscript_width(0.80), 1.85), constrained_layout=True)

    ax_acf = axes[0]
    for dataset, color, linestyle in zip(datasets, colors, linestyles):
        ax_acf.plot(
            lags,
            dataset.acf[lag_start:lag_limit],
            color=color,
            linestyle=linestyle,
            linewidth=LINE_WIDTHS["primary"],
            label=dataset.label,
        )
    ax_acf.axhline(0.0, color="0.35", linewidth=LINE_WIDTHS["secondary"])
    lag_unit = "bit samples" if acf_source == "bits" else "p1 blocks"
    ax_acf.set_xlabel(f"Lag ({lag_unit})")
    ax_acf.set_ylabel("Normalized ACF")
    # The paper omits panel titles.
    ax_acf.set_xlim(acf_xmin, acf_xmax)
    ax_acf.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax_acf.set_ylim(acf_ymin, acf_ymax)
    ax_acf.grid(True, color="0.88", linewidth=LINE_WIDTHS["grid"])

    ax_hist = axes[1]
    for dataset, color, linestyle in zip(datasets, colors, linestyles):
        ax_hist.hist(
            dataset.p1_blocks,
            bins=hist_bins,
            histtype="step",
            density=not hist_counts,
            linewidth=LINE_WIDTHS["primary"],
            color=color,
            linestyle=linestyle,
            label=dataset.label,
        )
    ax_hist.set_xlim(hist_xmin, hist_xmax)
    ax_hist.set_xlabel(rf"$\hat p_1$ in {block_size:,}-bit blocks")
    ax_hist.set_ylabel("Count" if hist_counts else "Density")
    # The paper omits panel titles.
    ax_hist.grid(True, color="0.88", linewidth=LINE_WIDTHS["grid"])
    ax_acf.legend(
        loc="upper right",
        frameon=False,
        fontsize=6.5,
        borderaxespad=0.3,
        labelspacing=0.2,
        handlelength=1.5,
        handletextpad=0.4,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    if pdf is not None:
        pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def print_summary(datasets: list[BitDataset]) -> None:
    for dataset in datasets:
        print(f"[{dataset.label}]")
        print(f"  file       : {dataset.path}")
        print(f"  samples    : {len(dataset.bits)}")
        print(f"  overall p1 : {dataset.p1_total:.6f}")
        print(f"  p1 blocks  : {len(dataset.p1_blocks)}")
        print(
            "  p1 mean/std/min/max: "
            f"{np.mean(dataset.p1_blocks):.6f} / "
            f"{np.std(dataset.p1_blocks, ddof=0):.6f} / "
            f"{np.min(dataset.p1_blocks):.6f} / "
            f"{np.max(dataset.p1_blocks):.6f}"
        )
        if len(dataset.acf) > 1:
            print(f"  acf lag 1  : {dataset.acf[1]:.6g}")


def main() -> None:
    args = parse_args()

    datasets = [
        load_dataset(
            args.with_flicker_csv,
            args.with_label,
            args.threshold,
            args.block_size,
            args.include_partial_block,
            args.acf_source,
            args.max_lag,
        ),
        load_dataset(
            args.without_flicker_csv,
            args.without_label,
            args.threshold,
            args.block_size,
            args.include_partial_block,
            args.acf_source,
            args.max_lag,
        ),
    ]

    plot_comparison(
        datasets,
        args.output,
        args.pdf,
        args.block_size,
        args.bins,
        args.hist_counts,
        args.show_zero_lag,
        args.acf_xmin,
        args.acf_xmax,
        args.acf_ymin,
        args.acf_ymax,
        args.hist_xmin,
        args.hist_xmax,
        args.dpi,
        args.acf_source,
    )

    if args.stats_csv is not None:
        write_stats_csv(args.stats_csv, datasets)

    print_summary(datasets)
    print(f"Saved figure: {args.output}")
    if args.pdf is not None:
        print(f"Saved PDF   : {args.pdf}")
    if args.stats_csv is not None:
        print(f"Saved stats : {args.stats_csv}")


if __name__ == "__main__":
    main()
