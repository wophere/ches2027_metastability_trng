"""Create the two-panel ES1/ES2 slow-drift Figure 13."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ES1/ES2 block probabilities and per-record zero-lag correlations."
    )
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--block-size", type=int, default=1_000)
    parser.add_argument("--max-lag", type=int, default=25)
    parser.add_argument("--output-prefix", type=Path)
    return parser.parse_args()


def resolve_data_path(metadata_path: Path, value: str) -> Path:
    original = Path(value)
    local = metadata_path.parent / original.name
    if local.exists():
        return local
    raise FileNotFoundError(original)


def load_bits(path: Path) -> np.ndarray:
    bits = np.fromfile(path, dtype=np.uint8)
    if bits.size == 0 or np.any((bits != 0) & (bits != 1)):
        raise ValueError(f"invalid bit stream: {path}")
    return bits.astype(np.float64, copy=False)


def block_means(bits: np.ndarray, block_size: int) -> np.ndarray:
    usable = bits.size // block_size * block_size
    if usable < 2 * block_size:
        raise ValueError("block size leaves fewer than two complete blocks")
    return bits[:usable].reshape(-1, block_size).mean(axis=1)


def pearson_ccf(
    first: np.ndarray, second: np.ndarray, max_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Pearson corr(first[j], second[j+lag]) on each overlap."""
    if first.size != second.size:
        raise ValueError("trajectory lengths differ")
    if not 1 <= max_lag < first.size - 2:
        raise ValueError("max lag must leave at least three overlapping blocks")
    lags = np.arange(-max_lag, max_lag + 1)
    values = np.empty(lags.size, dtype=np.float64)
    for index, lag in enumerate(lags):
        if lag < 0:
            x = first[-lag:]
            y = second[: second.size + lag]
        elif lag > 0:
            x = first[: first.size - lag]
            y = second[lag:]
        else:
            x = first
            y = second
        values[index] = np.corrcoef(x, y)[0, 1]
    return lags, values


def main() -> None:
    args = parse_args()
    batch_dir = args.batch_dir.resolve()
    metadata_files = sorted(batch_dir.glob("dual_es1_es2_*.json"))
    captures: list[dict[str, object]] = []

    for metadata_path in metadata_files:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not {"es13", "es24", "es13_file", "es24_file"}.issubset(metadata):
            continue
        if (int(metadata["es13"]), int(metadata["es24"])) != (1, 2):
            raise ValueError(f"unexpected ES pair in {metadata_path.name}")
        first = load_bits(resolve_data_path(metadata_path, metadata["es13_file"]))
        second = load_bits(resolve_data_path(metadata_path, metadata["es24_file"]))
        if first.size != second.size:
            raise ValueError(f"sample-count mismatch in {metadata_path.name}")
        paired_path = resolve_data_path(metadata_path, metadata["paired_file"])
        paired = np.fromfile(paired_path, dtype=np.uint8)
        expected = (first.astype(np.uint8) << 1) | second.astype(np.uint8)
        if not np.array_equal(paired, expected):
            raise ValueError(f"paired stream mismatch in {metadata_path.name}")
        captures.append(
            {
                "metadata_path": metadata_path,
                "metadata": metadata,
                "first": first,
                "second": second,
            }
        )

    if len(captures) != 10:
        raise ValueError(f"expected exactly 10 captures, found {len(captures)}")

    trajectories_first: list[np.ndarray] = []
    trajectories_second: list[np.ndarray] = []
    ccf_curves: list[np.ndarray] = []
    lags: np.ndarray | None = None

    for capture in captures:
        first = capture["first"]
        second = capture["second"]
        assert isinstance(first, np.ndarray) and isinstance(second, np.ndarray)
        first_blocks = block_means(first, args.block_size)
        second_blocks = block_means(second, args.block_size)
        current_lags, current_ccf = pearson_ccf(
            first_blocks, second_blocks, args.max_lag
        )
        trajectories_first.append(first_blocks)
        trajectories_second.append(second_blocks)
        ccf_curves.append(current_ccf)
        lags = current_lags

    assert lags is not None
    ccf_matrix = np.vstack(ccf_curves)
    mean_ccf = ccf_matrix.mean(axis=0)
    zero_index = int(np.flatnonzero(lags == 0)[0])
    zero_values = ccf_matrix[:, zero_index]
    zero_mean = float(zero_values.mean())
    all_first = np.concatenate([capture["first"] for capture in captures])
    all_second = np.concatenate([capture["second"] for capture in captures])
    overall_first = float(all_first.mean())
    overall_second = float(all_second.mean())

    prefix = args.output_prefix or batch_dir / "fig13_es1_es2_cross_correlation"
    prefix = prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    colors = {"es1": "#0072B2", "es2": "#D55E00"}
    fig, (trajectory_axis, correlation_axis) = plt.subplots(
        1, 2, figsize=(190 / 25.4, 40.5 / 25.4), gridspec_kw={"wspace": 0.34}
    )

    block_count = trajectories_first[0].size
    first_blocks_all = np.concatenate(trajectories_first)
    second_blocks_all = np.concatenate(trajectories_second)
    position_mbit = (
        np.arange(first_blocks_all.size) + 0.5
    ) * args.block_size / 1_000_000
    for record_index, (first_blocks, second_blocks) in enumerate(
        zip(trajectories_first, trajectories_second, strict=True)
    ):
        start = record_index * block_count
        stop = start + block_count
        segment_x = position_mbit[start:stop]
        trajectory_axis.plot(
            segment_x, first_blocks, color=colors["es1"], linewidth=0.55,
            alpha=0.9, label="ES1" if record_index == 0 else None,
        )
        trajectory_axis.plot(
            segment_x, second_blocks, color=colors["es2"], linewidth=0.55,
            alpha=0.82, label="ES2" if record_index == 0 else None,
        )
        if record_index:
            trajectory_axis.axvline(
                segment_x[0] - args.block_size / 2_000_000,
                color="#BDBDBD", linewidth=0.45, linestyle=":", zorder=0,
            )
    trajectory_axis.axhline(0.5, color="#333333", linewidth=0.8, label="0.5")
    trajectory_axis.set_xlim(0.0, position_mbit[-1] + args.block_size / 2_000_000)
    trajectory_axis.set_xticks(np.arange(0, len(captures) + 1, 2))
    trajectory_axis.set_ylim(0.0, 1.0)
    trajectory_axis.set_xlabel("Cumulative data position (Mbit)")
    trajectory_axis.set_ylabel(r"Block $\hat p_1$")
    trajectory_axis.grid(color="#E0E0E0", linewidth=0.55)
    trajectory_axis.set_axisbelow(True)
    trajectory_axis.legend(loc="upper right", frameon=False, ncol=3)

    record_centers_mbit = np.arange(len(captures)) + 0.5
    correlation_axis.vlines(
        record_centers_mbit, 0, zero_values, color=colors["es1"], linewidth=1.5
    )
    correlation_axis.scatter(
        record_centers_mbit, zero_values, color=colors["es1"], s=25,
        label="Individual 1-Mbit records", zorder=3,
    )
    correlation_axis.axhline(0, color="#222222", linewidth=1.0)
    correlation_axis.axhline(
        zero_mean, color=colors["es2"], linewidth=1.6, linestyle="--",
        label=rf"Mean $= {zero_mean:+.3f}$", zorder=2,
    )
    correlation_axis.set_xlim(0.0, float(len(captures)))
    correlation_axis.set_xticks(np.arange(0, len(captures) + 1, 2))
    correlation_axis.set_ylim(min(-0.2, float(zero_values.min()) - 0.025), 0.05)
    correlation_axis.set_xlabel("Cumulative data position (Mbit)")
    correlation_axis.set_ylabel("Cross-correlation")
    correlation_axis.grid(color="#E0E0E0", linewidth=0.55)
    correlation_axis.set_axisbelow(True)
    correlation_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=1.6,
        handlelength=2.5,
        frameon=False,
    )

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.30, top=0.96)

    png_path = prefix.with_suffix(".png")
    pdf_path = prefix.with_suffix(".pdf")
    svg_path = prefix.with_suffix(".svg")
    tiff_path = prefix.with_suffix(".tiff")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(
        tiff_path,
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    derived = prefix.parent.parent / "data" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    block_csv = derived / "block_means.csv"
    with block_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["global_block", "data_position_mbit", "es1_p1", "es2_p1"]
        )
        for block_index, (first_value, second_value) in enumerate(
            zip(first_blocks_all, second_blocks_all, strict=True)
        ):
            writer.writerow(
                [
                    block_index,
                    (block_index + 0.5) * args.block_size / 1_000_000,
                    first_value,
                    second_value,
                ]
            )

    ccf_csv = derived / "cross_correlation.csv"
    with ccf_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lag", *[f"capture_{i}" for i in range(1, 11)], "mean"])
        for lag_index, lag in enumerate(lags):
            writer.writerow(
                [int(lag), *ccf_matrix[:, lag_index], mean_ccf[lag_index]]
            )

    report = {
        "capture_count": len(captures),
        "bits_per_channel_per_capture": int(captures[0]["first"].size),
        "block_size": args.block_size,
        "overall_p1": {"ES1": overall_first, "ES2": overall_second},
        "per_capture": [],
        "ccf_definition": "Pearson corr(ES1[j], ES2[j+lag]) on the overlapping block means",
        "zero_lag": {
            "per_capture": zero_values.tolist(),
            "mean": zero_mean,
            "range": [float(zero_values.min()), float(zero_values.max())],
        },
        "figure_note": (
            "Panel b displays each record's zero-lag Pearson correlation and their "
            "arithmetic mean; no confidence or independence claim is implied."
        ),
        "outputs": {
            "png": png_path.name,
            "pdf": pdf_path.name,
            "svg": svg_path.name,
            "tiff": tiff_path.name,
            "block_means_csv": block_csv.name,
            "ccf_csv": ccf_csv.name,
        },
    }
    for capture, zero_value in zip(captures, zero_values, strict=True):
        metadata = capture["metadata"]
        first = capture["first"]
        second = capture["second"]
        assert isinstance(metadata, dict)
        assert isinstance(first, np.ndarray) and isinstance(second, np.ndarray)
        report["per_capture"].append(
            {
                "metadata": Path(capture["metadata_path"]).name,
                "dac_p0": [metadata["es13_p0_dac"], metadata["es24_p0_dac"]],
                "p1": [float(first.mean()), float(second.mean())],
                f"zero_lag_ccf_B{args.block_size}": float(zero_value),
            }
        )
    report_path = derived / "summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"overall_p1_es1={overall_first:.9f}")
    print(f"overall_p1_es2={overall_second:.9f}")
    print(f"zero_lag_B{args.block_size}={zero_mean:.9f}")
    print(f"figure={png_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
