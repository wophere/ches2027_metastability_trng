#!/usr/bin/env python3
"""Block average conditional min-entropy vs sampling frequency.

This script evaluates

    Hmin_av(B_1:N | V^f_1:N)
      = -log2 E_Z prod_k Phi(|r_mu + r_f Z_k|)

where Z is an N-dimensional standard Gaussian vector with correlation matrix

    rho_f_matrix[i,j] = rho_f(|i-j|; T_s).

It also plots two limiting envelopes for the per-bit entropy:

    Upper envelope / independent-flicker:
        overline_h = -log2 E_Z Phi(|r_mu + r_f Z|)

    Lower envelope / fully shared flicker:
        underline_h = -(1/N) log2 E_Z Phi(|r_mu + r_f Z|)^N

The plotted value is the per-bit block entropy h_N.

The default model follows the paper notation:

    V_k = mu + V_k^w + V_k^f
    V_k^w ~ N(0, sigma_w^2)
    V_k^f is a zero-mean stationary Gaussian process with constant marginal variance sigma_f^2

uses the shifted-power ACF

    rho_f(tau) = (1 + |tau|/tau_c)^(-beta).

The default keeps sigma_f fixed and lets the sampling interval affect entropy
only through the positive-semidefinite correlation matrix boldsymbol_rho_f.
"""

import argparse
import csv
import math
import json
import platform
from pathlib import Path

import numpy as np

from shifted_power_kernel import cholesky_psd, shifted_power_acf, shifted_power_corr_matrix

try:
    from scipy import special as scipy_special
except ImportError:  # pragma: no cover - fallback for minimal Python installs.
    scipy_special = None


# ==============================
# User-editable model parameters
# ==============================
#
# You can edit these defaults directly, or override them from the command line.
# The stochastic model is
#
#   V_k = mu + V_k^w + V_k^f,
#   V_k^w ~ N(0, sigma_w^2),
#   V_k^f = sigma_f * Z_k,
#   Z ~ N(0, rho_f_matrix).
#
# Main entropy quantity:
#
#   Hmin_av(B_1:N | V^f_1:N)
#     = -log2 E_Z prod_k Phi(|mu/sigma_w + (sigma_f/sigma_w) Z_k|).
#
USER_CONFIG = {
    # Block entropy
    "block_len": 64,
    "mu": 0.0,
    "sigma_w": 1.0,
    "sigma_f": 0.5,
    # Frequency sweep
    "fs_min": 1e6,
    "fs_max": 1e9,
    "points": 21,

    # Shifted-power flicker ACF: rho_f(tau)=(1+|tau|/tau_c)^(-beta).
    "beta": 0.2,
    "tau_c": 1e-9,

    # Monte Carlo integration
    "samples": 10_000_000,
    "batch": 5000,
    "seed": 12345,
    "z_max": 9.0,
    "z_points": 20001,

    # ACF plot
    "acf_max_lag": 256,
    "acf_freqs": "1e7,1e8,1e9",
    # "bit" plots sign-bit ACF rho_B[m]; "voltage" plots rho_V[m];
    # "flicker" plots rho_f[m].
    "acf_kind": "bit",
}


def normal_cdf_np(x):
    """Standard normal CDF for NumPy arrays without requiring SciPy."""
    if scipy_special is not None:
        return scipy_special.ndtr(x)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def normal_cdf_scalar(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def normalize_exponent(text: str) -> str:
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e", 1)
    sign = "-" if exponent.startswith("-") else ""
    exponent = exponent.lstrip("-").lstrip("0") or "0"
    return f"{mantissa}e{sign}{exponent}"


def filename_float(value: float) -> str:
    return normalize_exponent(f"{value:g}".replace("+", ""))


def filename_int(value: int) -> str:
    return normalize_exponent(f"{value:.0e}".replace("+", ""))


def build_frequency_grid(f_min: float, f_max: float, n: int):
    if n < 2:
        return np.array([f_min], dtype=float)
    return np.exp(np.linspace(math.log(f_min), math.log(f_max), n))


def flicker_corr_matrix(n: int, ts: float, tau_c: float, beta: float):
    """Shifted-power ACF matrix with rho_f(0)=1 and no clipping."""
    return shifted_power_corr_matrix(n, ts, tau_c, beta)


def nearest_psd_cholesky(r: np.ndarray):
    """Compatibility wrapper that verifies PSD rather than projecting it."""
    factor, _ = cholesky_psd(r)
    return factor


def hmin_block_mc(r_mu: float, r_f: float, corr: np.ndarray, samples: int, rng: np.random.Generator, batch: int):
    n = corr.shape[0]
    sqrt_corr = nearest_psd_cholesky(corr)
    total = 0.0
    total_sq = 0.0
    done = 0
    while done < samples:
        b = min(batch, samples - done)
        u = rng.standard_normal((b, n))
        z = u @ sqrt_corr.T
        p = normal_cdf_np(np.abs(r_mu + r_f * z))
        log_prod = np.sum(np.log(p), axis=1)
        prod = np.exp(log_prod)
        total += float(np.sum(prod))
        total_sq += float(np.sum(prod * prod))
        done += b
    mean = total / samples
    var = max(total_sq / samples - mean * mean, 0.0)
    se = math.sqrt(var / samples)
    h = -math.log(mean, 2)
    # Delta-method standard error for -log2(mean).
    h_se = se / (max(mean, 1e-300) * math.log(2.0))
    return h, h / n, mean, h_se


def hmin_upper_independent(r_mu: float, r_f: float, z_grid: np.ndarray):
    p = normal_cdf_np(np.abs(r_mu + r_f * z_grid))
    pdf = np.exp(-0.5 * z_grid * z_grid) / math.sqrt(2.0 * math.pi)
    mean = np.trapezoid(p * pdf, z_grid)
    return -math.log(float(mean), 2)


def hmin_lower_shared(r_mu: float, r_f: float, n: int, z_grid: np.ndarray):
    p = normal_cdf_np(np.abs(r_mu + r_f * z_grid))
    pdf = np.exp(-0.5 * z_grid * z_grid) / math.sqrt(2.0 * math.pi)
    mean = np.trapezoid((p ** n) * pdf, z_grid)
    return -math.log(float(mean), 2)


def acf_values_for_fs(args, fs: float, max_lag: int):
    """Return lag array and flicker/voltage/bit ACF curves for one frequency.

    The sign-bit ACF uses the unbiased arcsine law:
        rho_B[m] = (2/pi) asin(rho_V[m]).
    For nonzero mu, rho_B should be computed with the bivariate normal CDF;
    this script keeps the arcsine form because the paper's measured ACF is
    mainly used for the near-unbiased case.
    """
    ts = 1.0 / float(fs)
    sigma_f_ref = args.sigma_f if args.rf is None else args.rf * args.sigma_w
    sigma_f = sigma_f_ref
    sigma2 = args.sigma_w * args.sigma_w + sigma_f * sigma_f
    flicker_fraction = sigma_f * sigma_f / sigma2

    lags = np.arange(1, max_lag + 1, dtype=float)
    rho_f = shifted_power_acf(lags, ts, args.tau_c, args.beta)
    rho_v = np.clip(flicker_fraction * rho_f, -1.0, 1.0)
    rho_b = (2.0 / math.pi) * np.arcsin(rho_v)
    return lags.astype(int), rho_f, rho_v, rho_b


def plot_acf_curves(args) -> None:
    import matplotlib.pyplot as plt

    freqs = [float(x.strip()) for x in args.acf_freqs.split(",") if x.strip()]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for fs in freqs:
        lags, rho_f, rho_v, rho_b = acf_values_for_fs(args, fs, args.acf_max_lag)
        if args.acf_kind == "flicker":
            y = rho_f
            ylabel = r"Flicker ACF $\rho_f[m]$"
        elif args.acf_kind == "voltage":
            y = rho_v
            ylabel = r"Voltage ACF $\rho_V[m]$"
        elif args.acf_kind == "bit":
            y = rho_b
            ylabel = r"Sign-bit ACF $\rho_B[m]$"
        else:
            raise ValueError(f"Unsupported acf_kind: {args.acf_kind}")
        ax.plot(lags, y, label=f"{fs:g} Hz")

    ax.set_xlabel("Discrete lag m")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02 if args.acf_kind == "flicker" else None)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Sampling frequency")
    ax.set_title(
        f"ACF model: shifted-power, "
        f"tau_c={args.tau_c:g}s, beta={args.beta:g}"
    )
    fig.tight_layout()
    fig.savefig(args.acf_plot, dpi=220)
    print(f"Wrote {args.acf_plot.resolve()}")


def read_entropy_rows(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    required = {
        "fs_hz",
        "hmin_per_bit_mc",
        "h_upper_independent",
        "h_under_shared",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def plot_entropy_rows(rows, plot_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: float(row["fs_hz"]))
    fs_values = [float(r["fs_hz"]) for r in rows]
    mc_values = [float(r["hmin_per_bit_mc"]) for r in rows]
    upper_values = [float(r["h_upper_independent"]) for r in rows]
    lower_values = [float(r["h_under_shared"]) for r in rows]

    plt.rcParams.update({
        "font.size": 5,
        "axes.labelsize": 5,
        "legend.fontsize": 3.8,
        "xtick.labelsize": 4.5,
        "ytick.labelsize": 4.5,
        "axes.linewidth": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(2.0, 1.0))
    ax.fill_between(
        fs_values,
        lower_values,
        upper_values,
        color="#D9E6F2",
        alpha=0.75,
        label="Envelope",
    )
    ax.semilogx(fs_values, upper_values, color="#0072B2", linestyle="--", linewidth=0.75, label="Upper")
    ax.semilogx(fs_values, lower_values, color="#D55E00", linestyle=":", linewidth=0.9, label="Lower")
    ax.semilogx(
        fs_values,
        mc_values,
        color="black",
        marker="o",
        markersize=1.7,
        linewidth=0.75,
        label=r"$h_N$",
    )
    ax.set_xlabel(r"$f_s$ (Hz)", labelpad=0.5)
    ax.set_ylabel(r"$h_N$", labelpad=0.5)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([1e6, 1e7, 1e8, 1e9])
    ax.set_xticklabels([r"$10^6$", r"$10^7$", r"$10^8$", r"$10^9$"])
    ax.grid(True, which="both", linestyle=":", linewidth=0.35, alpha=0.5)
    ax.tick_params(width=0.55, length=2, pad=1)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        handlelength=0.85,
        columnspacing=0.45,
        handletextpad=0.25,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.28, top=0.95)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Wrote {plot_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute block Hmin_av vs sampling frequency.")
    parser.add_argument("--block-len", type=int, default=USER_CONFIG["block_len"], help="Block length N.")
    parser.add_argument("--mu", type=float, default=USER_CONFIG["mu"], help="Static bias mu.")
    parser.add_argument("--sigma-w", type=float, default=USER_CONFIG["sigma_w"], help="Thermal sigma.")
    parser.add_argument("--sigma-f", type=float, default=USER_CONFIG["sigma_f"],
                        help="Reference marginal flicker-offset sigma.")
    parser.add_argument("--rf", type=float, default=None,
                        help="Optional direct reference r_f=sigma_f/sigma_w. Overrides --sigma-f.")
    parser.add_argument("--fs-min", type=float, default=USER_CONFIG["fs_min"], help="Minimum sampling frequency in Hz.")
    parser.add_argument("--fs-max", type=float, default=USER_CONFIG["fs_max"], help="Maximum sampling frequency in Hz.")
    parser.add_argument("--points", type=int, default=USER_CONFIG["points"], help="Number of frequency points.")
    parser.add_argument("--tau-c", type=float, default=USER_CONFIG["tau_c"],
                        help="Crossover time tau_c in rho_f(tau)=(1+|tau|/tau_c)^(-beta).")
    parser.add_argument("--beta", type=float, default=USER_CONFIG["beta"],
                        help="Power exponent beta in rho_f(tau)=(1+|tau|/tau_c)^(-beta).")
    parser.add_argument("--samples", type=int, default=USER_CONFIG["samples"], help="Monte Carlo samples per frequency.")
    parser.add_argument("--batch", type=int, default=USER_CONFIG["batch"], help="Monte Carlo batch size.")
    parser.add_argument("--seed", type=int, default=USER_CONFIG["seed"], help="RNG seed.")
    parser.add_argument("--z-max", type=float, default=USER_CONFIG["z_max"], help="Scalar Gaussian integration limit.")
    parser.add_argument("--z-points", type=int, default=USER_CONFIG["z_points"], help="Scalar Gaussian grid points.")
    script_dir = Path(__file__).resolve().parent
    artifact_dir = script_dir.parent
    parser.add_argument("--out", type=Path, default=None, help="CSV output.")
    parser.add_argument("--plot", type=Path, default=None, help="PDF output.")
    parser.add_argument("--plot-from-csv", type=Path, default=None,
                        help="Redraw the entropy PDF from an existing CSV without running Monte Carlo.")
    parser.add_argument("--acf-plot", type=Path, default=Path("acf_vs_fs.pdf"), help="ACF PDF output.")
    parser.add_argument("--acf-max-lag", type=int, default=USER_CONFIG["acf_max_lag"], help="Maximum lag for ACF plot.")
    parser.add_argument("--acf-freqs", default=USER_CONFIG["acf_freqs"], help="Comma-separated frequencies for ACF plot.")
    parser.add_argument("--acf-kind", choices=["bit", "voltage", "flicker"], default=USER_CONFIG["acf_kind"],
                        help="Which ACF to plot.")
    parser.add_argument("--skip-entropy", action="store_true", help="Only draw ACF plot; skip MC entropy.")
    parser.add_argument("--skip-acf-plot", action="store_true", help="Skip ACF plot.")
    args = parser.parse_args()
    if args.tau_c <= 0.0:
        raise ValueError("--tau-c must be positive.")
    if not (0.0 < args.beta < 1.0):
        raise ValueError("--beta must satisfy 0 < beta < 1.")
    default_tag = (
        f"shifted_tauc_{filename_float(args.tau_c)}_"
        f"beta_{filename_float(args.beta)}_samples{filename_int(args.samples)}"
    )
    if args.out is None:
        args.out = artifact_dir / "data" / "derived" / f"block_entropy_vs_fs_{default_tag}.csv"
    if args.plot is None:
        args.plot = artifact_dir / "figure" / f"model_entropy_{default_tag}.pdf"
    if args.plot_from_csv is not None:
        plot_entropy_rows(read_entropy_rows(args.plot_from_csv), args.plot)
        return

    if not args.skip_entropy:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        metadata = {key: value for key, value in vars(args).items() if not isinstance(value, Path)}
        metadata.update(python=platform.python_version(), numpy=np.__version__,
                        rng='numpy.random.PCG64', covariance_factor='unprojected Cholesky')
        args.out.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        rng = np.random.default_rng(args.seed)
        freqs = build_frequency_grid(args.fs_min, args.fs_max, args.points)
        z_grid = np.linspace(-args.z_max, args.z_max, args.z_points)
        r_mu = args.mu / args.sigma_w
        sigma_f_ref = args.sigma_f if args.rf is None else args.rf * args.sigma_w

        rows = []
        for fs in freqs:
            ts = 1.0 / float(fs)
            sigma_f = sigma_f_ref
            r_f = sigma_f / args.sigma_w
            sigma = math.sqrt(args.sigma_w * args.sigma_w + sigma_f * sigma_f)
            flicker_fraction = sigma_f * sigma_f / (sigma * sigma)
            corr = flicker_corr_matrix(
                args.block_len,
                ts,
                args.tau_c,
                args.beta,
            )
            _, min_eig = cholesky_psd(corr)

            h_mc, h_mc_per_bit, p_guess_block, h_mc_se = hmin_block_mc(
                r_mu, r_f, corr, args.samples, rng, args.batch
            )
            h_upper = hmin_upper_independent(r_mu, r_f, z_grid)
            h_upper_block = args.block_len * h_upper
            h_lower_block = hmin_lower_shared(r_mu, r_f, args.block_len, z_grid)

            rows.append({
                "fs_hz": float(fs),
                "ts_s": ts,
                "mu": args.mu,
                "sigma_w": args.sigma_w,
                "sigma_f": sigma_f,
                "r_mu": r_mu,
                "r_f": r_f,
                "sigma": sigma,
                "sigma_f_ref": sigma_f_ref,
                "flicker_variance_fraction": flicker_fraction,
                "rho_v_lag1": flicker_fraction * corr[0, 1] if args.block_len > 1 else 0.0,
                "block_len": args.block_len,
                "kernel": "shifted-power",
                "beta": args.beta,
                "tau_c_s": args.tau_c,
                "min_eig": min_eig,
                "hmin_block_mc": h_mc,
                "hmin_per_bit_mc": h_mc_per_bit,
                "hmin_mc_se": h_mc_se,
                "p_guess_block_mc": p_guess_block,
                "hmin_block_upper_independent": h_upper_block,
                "h_upper_independent": h_upper,
                "hmin_block_lower_shared": h_lower_block,
                "h_under_shared": h_lower_block / args.block_len,
            })
            print(
                f"fs={fs:.4g} Hz r_f={r_f:.4g} "
                f"rhoV1={rows[-1]['rho_v_lag1']:.4g} "
                f"h_N={h_mc_per_bit:.4f} h_under={h_lower_block / args.block_len:.4f} h_over={h_upper:.4f}"
            )

        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.out.resolve()}")

        plot_entropy_rows(rows, args.plot)

    if not args.skip_acf_plot:
        plot_acf_curves(args)


if __name__ == "__main__":
    main()

