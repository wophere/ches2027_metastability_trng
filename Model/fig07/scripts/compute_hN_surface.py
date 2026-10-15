#!/usr/bin/env python3
"""Compare h_N over r_mu and r_f for N=16, 64, and 256.

The sampling frequency is fixed at 100 MHz. Each r_f slice uses a deterministic
Monte Carlo stream; the N=16 and N=64 estimates use prefixes of the same N=256
stream. This keeps the three block lengths comparable while allowing the r_f
slices to run in parallel.
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import math
import argparse
import json
import platform

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

try:
    from scipy.special import ndtr
except ImportError:  # pragma: no cover
    ndtr = None

from compute_block_entropy_vs_fs import flicker_corr_matrix, nearest_psd_cholesky


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = SCRIPT_DIR.parent
PAPER_FIGURE_DIR = ARTIFACT_DIR / "figure"
OUT_CSV = ARTIFACT_DIR / "data" / "recomputed" / "hN_surface_rmu_rf_N_shifted_fs100M.csv"

CONFIG = {
    "block_lens": [16, 64, 256],
    "fs_hz": 1e8,
    "tau_c": 1e-9,
    "beta": 0.2,
    "grid_points": 15,
    "samples": 1_000_000,
    "batch": 2_000,
    "workers": 4,
    "seed": 20270624,
}


def normal_cdf(x):
    if ndtr is not None:
        return ndtr(x)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def normal_log_cdf(x):
    p = normal_cdf(x)
    np.clip(p, 1e-300, 1.0, out=p)
    np.log(p, out=p)
    return p


def init_csv(path: Path):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N",
            "r_mu",
            "r_f",
            "h_N",
            "samples",
            "fs_hz",
            "beta",
            "tau_c_s",
            "kernel",
        ])


def append_rf_rows(path: Path, n: int, rmu_values: np.ndarray, r_f: float, h_row: np.ndarray):
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for rmu, h in zip(rmu_values, h_row):
            writer.writerow([
                n,
                f"{rmu:.8g}",
                f"{r_f:.8g}",
                f"{h:.12g}",
                CONFIG["samples"],
                CONFIG["fs_hz"],
                CONFIG["beta"],
                CONFIG["tau_c"],
                "shifted-power",
            ])


def estimate_rf_surface(task):
    i, r_f, rmu_values, block_lens, sqrt_corr, samples, batch, seed = task
    max_n = max(block_lens)
    totals = {n: np.zeros(len(rmu_values), dtype=float) for n in block_lens}

    if r_f == 0.0:
        log_p0 = normal_log_cdf(np.abs(rmu_values))
        for n in block_lens:
            totals[n] = samples * np.exp(n * log_p0)
        rows = {
            n: -np.log2(np.clip(totals[n] / samples, 1e-300, 1.0)) / n
            for n in block_lens
        }
        print(f"r_f={r_f:.3f}: analytic zero-flicker slice", flush=True)
        return i, r_f, rows

    rng = np.random.default_rng(seed)
    done = 0
    while done < samples:
        b = min(batch, samples - done)
        z_full = rng.standard_normal((b, max_n)) @ sqrt_corr.T
        log_prefix = normal_log_cdf(np.abs(rmu_values[None, :, None] + r_f * z_full[:, None, :]))
        np.cumsum(log_prefix, axis=2, out=log_prefix)
        for n in block_lens:
            totals[n] += np.sum(np.exp(log_prefix[:, :, n - 1]), axis=0)
        done += b
        if done % 250_000 == 0 or done == samples:
            print(f"r_f={r_f:.3f}: {done}/{samples} samples", flush=True)

    rows = {
        n: -np.log2(np.clip(totals[n] / samples, 1e-300, 1.0)) / n
        for n in block_lens
    }
    return i, r_f, rows


def estimate_surfaces(rmu_values: np.ndarray, rf_values: np.ndarray):
    block_lens = CONFIG["block_lens"]
    max_n = max(block_lens)
    ts = 1.0 / CONFIG["fs_hz"]
    corr = flicker_corr_matrix(
        max_n,
        ts,
        CONFIG["tau_c"],
        CONFIG["beta"],
    )
    sqrt_corr = nearest_psd_cholesky(corr)

    surfaces = {n: np.empty((len(rf_values), len(rmu_values)), dtype=float) for n in block_lens}
    init_csv(OUT_CSV)

    samples = CONFIG["samples"]
    batch = CONFIG["batch"]
    workers = min(CONFIG["workers"], len(rf_values))
    tasks = [
        (i, float(r_f), rmu_values, block_lens, sqrt_corr, samples, batch, CONFIG["seed"] + 1009 * i)
        for i, r_f in enumerate(rf_values)
    ]

    results = {}
    print(f"Running {len(tasks)} r_f slices with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_i = {pool.submit(estimate_rf_surface, task): task[0] for task in tasks}
        for future in as_completed(future_to_i):
            i, r_f, rows = future.result()
            results[i] = (r_f, rows)
            summary = ", ".join(f"N={n} {rows[n].min():.4f}-{rows[n].max():.4f}" for n in block_lens)
            print(f"completed r_f={r_f:.3f}: {summary}", flush=True)

    for i in range(len(rf_values)):
        r_f, rows = results[i]
        for n in block_lens:
            h_row = rows[n]
            surfaces[n][i, :] = h_row
            append_rf_rows(OUT_CSV, n, rmu_values, r_f, h_row)
    return surfaces


def load_surfaces_from_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No data rows in {path}")

    rmu_values = np.array(sorted({float(row["r_mu"]) for row in rows}))
    rf_values = np.array(sorted({float(row["r_f"]) for row in rows}))
    rmu_index = {v: i for i, v in enumerate(rmu_values)}
    rf_index = {v: i for i, v in enumerate(rf_values)}

    surfaces = {}
    for row in rows:
        n = int(row["N"])
        surfaces.setdefault(n, np.empty((len(rf_values), len(rmu_values)), dtype=float))
        surfaces[n][rf_index[float(row["r_f"])], rmu_index[float(row["r_mu"])] ] = float(row["h_N"])
    return rmu_values, rf_values, dict(sorted(surfaces.items()))


def save_level_heatmaps(rmu_values, rf_values, surfaces, out_pdf=None):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    out_pdf = out_pdf or PAPER_FIGURE_DIR / "hN_heatmap_N_shifted_fs100M.pdf"
    fig = plt.figure(figsize=(6.2, 2.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    extent = [rmu_values.min(), rmu_values.max(), rf_values.min(), rf_values.max()]
    images = []
    for ax, (n, h) in zip(axes, surfaces.items()):
        im = ax.imshow(h, origin="lower", extent=extent, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
        images.append(im)
        levels = np.linspace(0.2, 0.8, 4)
        cs = ax.contour(rmu_values, rf_values, h, levels=levels, colors="white", linewidths=0.55, alpha=0.9)
        labels = ax.clabel(cs, fmt="%.1f", fontsize=5, inline=True, inline_spacing=1, colors="white")
        for label in labels:
            x, y = label.get_position()
            near_edge = x > rmu_values.max() - 0.03 or y > rf_values.max() - 0.03
            if near_edge:
                label.set_visible(False)
            else:
                label.set_path_effects([pe.withStroke(linewidth=0.8, foreground="black")])
        if n == 256:
            manual_labels = ax.clabel(
                cs,
                levels=[0.2],
                manual=[(0.50, 0.36)],
                fmt="%.1f",
                fontsize=5,
                inline=True,
                inline_spacing=1,
                colors="white",
            )
            for label in manual_labels:
                label.set_path_effects([pe.withStroke(linewidth=0.8, foreground="black")])
        ax.set_title(rf"$N={n}$")
        ax.set_xlabel(r"$r_\mu$")
        ax.tick_params(length=2, width=0.5)
    axes[0].set_ylabel(r"$r_f$")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_label(r"$h_N$")
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.22, top=0.86)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def main():
    global OUT_CSV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT_CSV)
    parser.add_argument('--plot', type=Path, default=ARTIFACT_DIR / 'data/recomputed/fig07.pdf')
    for name in ('samples', 'batch', 'seed', 'workers'):
        parser.add_argument('--' + name, type=int, default=CONFIG[name])
    args = parser.parse_args()
    OUT_CSV = args.out
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    for name in ('samples', 'batch', 'seed', 'workers'):
        CONFIG[name] = getattr(args, name)
    OUT_CSV.with_suffix('.metadata.json').write_text(json.dumps({
        **CONFIG, 'python': platform.python_version(), 'numpy': np.__version__,
        'rng': 'numpy.random.PCG64', 'slice_seed': 'seed + 1009*i',
        'covariance_factor': 'unprojected Cholesky',
    }, indent=2) + '\n', encoding='utf-8')
    rmu_values = np.linspace(0.0, 1.0, CONFIG["grid_points"])
    rf_values = np.linspace(0.0, 1.0, CONFIG["grid_points"])
    surfaces = estimate_surfaces(rmu_values, rf_values)
    save_level_heatmaps(rmu_values, rf_values, surfaces, args.plot)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
