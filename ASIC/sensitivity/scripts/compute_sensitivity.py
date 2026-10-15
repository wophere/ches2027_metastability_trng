"""Section 5 sensitivity analyses from archived model parameters.

ASIC times use oscilloscope sample ticks (12.5 ps at 80 GSa/s).
Spectral frequencies use reciprocal ticks; frequency times time is unitless.
"""

from __future__ import annotations

import argparse, json, math

from pathlib import Path

from typing import Any

import numpy as np

import pandas as pd

from scipy import linalg, special, stats

N_BLOCK = 256

def correlation_matrix(first_col: np.ndarray) -> np.ndarray:
    return linalg.toeplitz(np.asarray(first_col, dtype=float))

def shifted_power_first_col(row: pd.Series, n: int = N_BLOCK) -> np.ndarray:
    lag = np.arange(n, dtype=float)
    return np.power(
        1.0 + lag * float(row["period_ticks"]) / float(row["tau_c_ticks"]),
        -float(row["beta"]),
    )

def ou8_first_col(row: pd.Series, n: int = N_BLOCK) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eight positive OU terms from generalized Gauss-Laguerre quadrature.

    (1+x)^(-beta) is the Gamma(beta,1) expectation of exp(-U*x).
    Generalized Gauss-Laguerre quadrature therefore gives positive weights and
    positive time constants, making every finite covariance matrix PSD.
    """

    beta = float(row["beta"])
    roots, weights = special.roots_genlaguerre(8, beta - 1.0)
    weights = weights / special.gamma(beta)
    weights = weights / weights.sum()
    tau_terms = float(row["tau_c_ticks"]) / roots
    time = np.arange(n, dtype=float) * float(row["period_ticks"])
    first_col = np.sum(
        weights[:, None] * np.exp(-time[None, :] / tau_terms[:, None]), axis=0
    )
    return first_col, weights, tau_terms

def bandlimited_first_col(
    row: pd.Series, n: int = N_BLOCK, n_frequency_bins: int = 2048
) -> tuple[np.ndarray, float, float, float]:
    """Positive spectral quadrature for a band-limited 1/f^(1-beta) kernel.

    Frequencies are in inverse oscilloscope sample ticks. The band spans
    three decades on either side of f_c=1/(2*pi*tau_c), truncated at Nyquist. Thus the
    alternative retains the measured characteristic scale while covering the
    full resolvable band when f_c is interior.  Every quadrature weight is
    positive, and each cosine covariance is PSD.
    """

    beta = float(row["beta"])
    gamma = 1.0 - beta
    period = float(row["period_ticks"])
    tau_c = float(row["tau_c_ticks"])
    f_nyquist = 0.5 / period
    f_characteristic = 1.0 / (2.0 * math.pi * tau_c)
    f_low = f_characteristic / 1_000.0
    f_high = min(f_nyquist, f_characteristic * 1_000.0)
    if not 0.0 < f_low < f_high:
        raise ValueError(
            f"Invalid band-limited cutoffs f_low={f_low}, f_high={f_high}"
        )
    edges = np.geomspace(f_low, f_high, n_frequency_bins + 1)
    frequency = np.sqrt(edges[:-1] * edges[1:])
    weight = np.power(frequency, -gamma) * np.diff(edges)
    weight = weight / weight.sum()
    time = np.arange(n, dtype=float) * period
    first_col = np.sum(
        weight[:, None] * np.cos(2.0 * math.pi * frequency[:, None] * time[None, :]),
        axis=0,
    )
    first_col[0] = 1.0
    return first_col, gamma, f_low, f_high

def validate_psd(first_col: np.ndarray) -> tuple[np.ndarray, float]:
    corr = correlation_matrix(first_col)
    min_eig = float(np.linalg.eigvalsh(corr)[0])
    if min_eig < -1.0e-10:
        raise ValueError(f"Non-PSD covariance without projection: min eigenvalue={min_eig:.3e}")
    # No clipping and no diagonal jitter: failure here is reported, not hidden.
    np.linalg.cholesky(corr)
    return corr, min_eig

def entropy_importance_sampling(
    corr: np.ndarray,
    r_mu: float,
    r_f: float,
    output_samples: int,
    batch_pairs: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Symmetric-mixture IS estimate of average conditional h_N per bit."""

    n = corr.shape[0]
    factor = np.linalg.cholesky(corr)
    precision_ones = linalg.cho_solve(
        linalg.cho_factor(corr, lower=True, check_finite=False),
        np.ones(n),
        check_finite=False,
    )
    energy = float(precision_ones.sum())
    shifts = np.asarray([-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0])
    log_components = math.log(len(shifts))
    pairs_target = math.ceil(output_samples / 2)
    rng = np.random.default_rng(seed)
    log_sum = -math.inf
    log_sum_sq = -math.inf
    pairs_done = 0
    while pairs_done < pairs_target:
        batch = min(batch_pairs, pairs_target - pairs_done)
        z = rng.standard_normal((batch, n)) @ factor.T
        component = rng.integers(0, len(shifts), size=batch)
        x = z + shifts[component, None]
        score = x @ precision_ones
        log_q_over_p = special.logsumexp(
            score[:, None] * shifts[None, :]
            - 0.5 * energy * shifts[None, :] ** 2,
            axis=1,
        ) - log_components
        log_weight = -log_q_over_p
        log_g_pos = np.sum(special.log_ndtr(np.abs(r_mu + r_f * x)), axis=1)
        log_g_neg = np.sum(special.log_ndtr(np.abs(r_mu - r_f * x)), axis=1)
        log_pair = np.logaddexp(log_g_pos + log_weight, log_g_neg + log_weight) - math.log(2.0)
        log_sum = float(np.logaddexp(log_sum, special.logsumexp(log_pair)))
        log_sum_sq = float(np.logaddexp(log_sum_sq, special.logsumexp(2.0 * log_pair)))
        pairs_done += batch

    log_mean = log_sum - math.log(pairs_done)
    log_mean_sq = log_sum_sq - math.log(pairs_done)
    rel_var = max(math.exp(log_mean_sq - 2.0 * log_mean) - 1.0, 0.0)
    se_log = math.sqrt(rel_var / pairs_done)
    return {
        f"h{n}": -log_mean / (n * math.log(2.0)),
        f"h{n}_mc_se": se_log / (n * math.log(2.0)),
        "output_samples": 2 * pairs_done,
        "antithetic_pairs": pairs_done,
        "sampling_method": "seven-shift_symmetric-mixture_importance_dense-Cholesky",
    }

def supplemental_shifted_entropy(
    workspace: Path,
    parameters: pd.DataFrame,
    entropy_samples: int,
    batch_pairs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute h16/h64/h256 for the ensemble and A/B h256 sensitivity.

    The A/B range is a two-fit sensitivity diagnostic.  It is deliberately
    not described as a capture-bootstrap interval or confidence interval.
    """

    ensemble_rows: list[dict[str, Any]] = []
    for corner_index, item in parameters.iterrows():
        for n in (16, 64, 256):
            first_col = shifted_power_first_col(item, n=n)
            corr, min_eig = validate_psd(first_col)
            estimate = entropy_importance_sampling(
                corr,
                float(item["r_mu"]),
                float(item["r_f"]),
                entropy_samples,
                batch_pairs,
                seed + 3_000_000 + corner_index * 10_000 + n,
            )
            ensemble_rows.append(
                {
                    "temperature_c": int(item["temperature_c"]),
                    "voltage": float(item["voltage"]),
                    "frequency": int(item["frequency"]),
                    "n": n,
                    "h_per_raw_output_bit": estimate[f"h{n}"],
                    "h_mc_se": estimate[f"h{n}_mc_se"],
                    "min_eigenvalue": min_eig,
                    "no_eigenvalue_clipping": True,
                    "kernel": "shifted_power",
                    "parameter_scope": item["parameter_scope"],
                    "output_samples": estimate["output_samples"],
                    "sampling_method": estimate["sampling_method"],
                }
            )
    ensemble = pd.DataFrame(ensemble_rows).sort_values(
        ["temperature_c", "voltage", "frequency", "n"]
    )

    split_source = workspace / "ASIC/fig14/data/derived/fig14_platform_split_parameters.csv"
    split = pd.read_csv(split_source)
    split_rows: list[dict[str, Any]] = []
    key_columns = ["temperature_c", "voltage", "frequency"]
    for _, fit in split.sort_values(key_columns + ["split"]).iterrows():
        match = parameters[
            (parameters["temperature_c"] == int(fit["temperature_c"]))
            & np.isclose(parameters["voltage"], float(fit["voltage"]))
            & (parameters["frequency"] == int(fit["frequency"]))
        ]
        if len(match) != 1:
            raise ValueError("Could not map split fit to pooled trace metadata")
        base = match.iloc[0]
        r_f = float(fit["r_f"])
        r_mu = float(base["r_u_pooled_10"]) * math.sqrt(1.0 + r_f * r_f)
        model_row = pd.Series(
            {
                "period_ticks": float(fit["period"]),
                "tau_c_ticks": float(fit["tau_c"]),
                "beta": float(fit["beta"]),
            }
        )
        corr, min_eig = validate_psd(shifted_power_first_col(model_row, n=256))
        # Same seed for A and B at a corner supplies common random numbers.
        corner_seed = (
            seed
            + 4_000_000
            + (int(fit["temperature_c"]) + 20) * 100_000
            + int(round(float(fit["voltage"]) * 10)) * 1_000
            + int(fit["frequency"])
        )
        estimate = entropy_importance_sampling(
            corr, r_mu, r_f, entropy_samples, batch_pairs, corner_seed
        )
        split_rows.append(
            {
                "temperature_c": int(fit["temperature_c"]),
                "voltage": float(fit["voltage"]),
                "frequency": int(fit["frequency"]),
                "split": str(fit["split"]),
                "n_fit_captures": int(fit["n_repeats"]),
                "beta": float(fit["beta"]),
                "r_f": r_f,
                "tau_c_ticks": float(fit["tau_c"]),
                "r_mu_from_pooled_10_bias_and_split_rf": r_mu,
                "h256_per_raw_output_bit": estimate["h256"],
                "h256_mc_se": estimate["h256_mc_se"],
                "min_eigenvalue": min_eig,
                "no_eigenvalue_clipping": True,
                "quality_status": str(fit["quality_status"]),
                "parameter_scope": (
                    "one_of_two_independent_five_capture_fits; pooled_10_bias; "
                    "A_B_range_is_not_a_capture_bootstrap_CI"
                ),
                "output_samples": estimate["output_samples"],
                "sampling_method": estimate["sampling_method"],
            }
        )
    split_result = pd.DataFrame(split_rows)
    ranges = (
        split_result.groupby(key_columns, as_index=False)["h256_per_raw_output_bit"]
        .agg(h256_split_min="min", h256_split_max="max")
    )
    ranges["h256_split_AB_range"] = ranges["h256_split_max"] - ranges["h256_split_min"]
    split_result = split_result.merge(ranges, on=key_columns, how="left").sort_values(
        key_columns + ["split"]
    )
    summary = {
        "ensemble_hN_rows": len(ensemble),
        "split_AB_rows": len(split_result),
        "split_AB_h256_range_min_across_24": float(ranges["h256_split_AB_range"].min()),
        "split_AB_h256_range_max_across_24": float(ranges["h256_split_AB_range"].max()),
        "split_AB_h256_range_median_across_24": float(ranges["h256_split_AB_range"].median()),
        "maximum_mc_se": float(
            max(ensemble["h_mc_se"].max(), split_result["h256_mc_se"].max())
        ),
        "scope_warning": (
            "A/B is sensitivity between two independent five-capture fits; "
            "it is not a capture-level bootstrap confidence interval."
        ),
    }
    return ensemble, split_result, summary

def kernel_sensitivity(
    parameters: pd.DataFrame, entropy_samples: int, batch_pairs: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, item in parameters.iterrows():
        shifted = shifted_power_first_col(item)
        ou8, ou_weights, ou_taus = ou8_first_col(item)
        band, gamma, f_low, f_high = bandlimited_first_col(item)
        kernels = {
            "shifted_power": (shifted, {}),
            "ou_lorentzian_8term": (
                ou8,
                {
                    "ou_weights": " ".join(f"{value:.9g}" for value in ou_weights),
                    "ou_tau_terms_ticks": " ".join(f"{value:.9g}" for value in ou_taus),
                },
            ),
            "bandlimited_1_over_f_gamma": (
                band,
                {"gamma": gamma, "f_low_per_tick": f_low, "f_high_per_tick": f_high},
            ),
        }
        for kernel_index, (name, (first_col, details)) in enumerate(kernels.items()):
            corr, min_eig = validate_psd(first_col)
            estimate = entropy_importance_sampling(
                corr,
                float(item["r_mu"]),
                float(item["r_f"]),
                entropy_samples,
                batch_pairs,
                seed + 100_000 * index + 1_000 * kernel_index,
            )
            fidelity = first_col - shifted
            rows.append(
                {
                    "temperature_c": int(item["temperature_c"]),
                    "voltage": float(item["voltage"]),
                    "frequency": int(item["frequency"]),
                    "kernel": name,
                    "n": N_BLOCK,
                    "min_eigenvalue": min_eig,
                    "no_eigenvalue_clipping": True,
                    "acf_rmse_vs_shifted_lags_1_255": float(np.sqrt(np.mean(fidelity[1:] ** 2))),
                    "acf_max_abs_vs_shifted_lags_1_255": float(np.max(np.abs(fidelity[1:]))),
                    "parameter_scope": item["parameter_scope"],
                    **estimate,
                    **details,
                }
            )
        print(
            f"kernel entropy {index + 1:02d}/24: {int(item.temperature_c):+d} C, "
            f"{item.voltage:.1f} V/F{int(item.frequency)}",
            flush=True,
        )
    result = pd.DataFrame(rows)
    pivot = result.pivot_table(
        index=["temperature_c", "voltage", "frequency"],
        columns="kernel",
        values="h256",
    )
    per_corner_range = pivot.max(axis=1) - pivot.min(axis=1)
    result = result.merge(
        per_corner_range.rename("h256_three_kernel_range").reset_index(),
        on=["temperature_c", "voltage", "frequency"],
        how="left",
    )
    return result.sort_values(["temperature_c", "voltage", "frequency", "kernel"])

def exact_order_stat_ci(
    sorted_values: np.ndarray, probability: float, alpha: float = 0.05
) -> tuple[int, int, float, float]:
    n = len(sorted_values)
    lower_rank = max(1, int(stats.binom.ppf(alpha / 2.0, n, probability)))
    upper_rank = min(n, int(stats.binom.ppf(1.0 - alpha / 2.0, n, probability)) + 1)
    return (
        lower_rank,
        upper_rank,
        float(sorted_values[lower_rank - 1]),
        float(sorted_values[upper_rank - 1]),
    )

def trajectory_tail_quantiles(
    parameters: pd.DataFrame, samples: int, seed: int
) -> pd.DataFrame:
    if samples < 20_000:
        raise ValueError("Too few tail samples for the 2^-10 quantile")
    rows: list[dict[str, Any]] = []
    epsilons = (2.0**-4, 2.0**-8, 2.0**-10)
    batch_size = 4_096
    for index, item in parameters.iterrows():
        first_col = shifted_power_first_col(item)
        corr, min_eig = validate_psd(first_col)
        factor = np.linalg.cholesky(corr)
        rng = np.random.default_rng(seed + 1_000_000 + index)
        values = np.empty(samples, dtype=np.float64)
        done = 0
        while done < samples:
            batch = min(batch_size, samples - done)
            z = rng.standard_normal((batch, N_BLOCK)) @ factor.T
            log_guess = np.sum(
                special.log_ndtr(np.abs(float(item["r_mu"]) + float(item["r_f"]) * z)),
                axis=1,
            )
            values[done : done + batch] = -log_guess / (N_BLOCK * math.log(2.0))
            done += batch
        values.sort()
        for epsilon in epsilons:
            point_rank = max(1, int(math.ceil(samples * epsilon)))
            low_rank, high_rank, ci_low, ci_high = exact_order_stat_ci(values, epsilon)
            rows.append(
                {
                    "temperature_c": int(item["temperature_c"]),
                    "voltage": float(item["voltage"]),
                    "frequency": int(item["frequency"]),
                    "n": N_BLOCK,
                    "epsilon_lower_tail": epsilon,
                    "exceedance_probability": 1.0 - epsilon,
                    "h256_threshold": float(values[point_rank - 1]),
                    "point_order_rank_1based": point_rank,
                    "order_stat_95ci_low_rank_1based": low_rank,
                    "order_stat_95ci_high_rank_1based": high_rank,
                    "h256_order_stat_95ci_low": ci_low,
                    "h256_order_stat_95ci_high": ci_high,
                    "trajectory_mc_samples": samples,
                    "kernel": "shifted_power",
                    "min_eigenvalue": min_eig,
                    "strict_worst_case_under_unbounded_gaussian": 0.0,
                    "uncertainty_scope": (
                        "exact_binomial_order-statistic_CI_conditional_on_fixed_"
                        "split-ensemble_parameters; parameter_uncertainty_not_included"
                    ),
                }
            )
        print(
            f"tail quantiles {index + 1:02d}/24: {int(item.temperature_c):+d} C, "
            f"{item.voltage:.1f} V/F{int(item.frequency)}",
            flush=True,
        )
    return pd.DataFrame(rows).sort_values(
        ["epsilon_lower_tail", "temperature_c", "voltage", "frequency"]
    )


def main():
    root = Path(__file__).resolve().parents[3]
    data = Path(__file__).resolve().parents[1] / "data/derived"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/recomputed")
    parser.add_argument("--tail-samples", type=int, default=262144)
    parser.add_argument("--entropy-samples", type=int, default=40000)
    parser.add_argument("--batch-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--first-corner-only", action="store_true", help="Smoke test; does not reproduce all paper results.")
    args = parser.parse_args()
    parameters = pd.read_csv(data / "sensitivity_parameters_24corners.csv")
    if args.first_corner_only:
        parameters = parameters.iloc[:1].copy()
    args.output.mkdir(parents=True, exist_ok=True)
    kernels = kernel_sensitivity(parameters, args.entropy_samples, args.batch_pairs, args.seed)
    kernels.to_csv(args.output / "h256_kernel_sensitivity.csv", index=False)
    tail = trajectory_tail_quantiles(parameters, args.tail_samples, args.seed)
    tail.to_csv(args.output / "h256_trajectory_exceedance_quantiles.csv", index=False)
    if not args.first_corner_only:
        ensemble, split, summary = supplemental_shifted_entropy(root, parameters, args.entropy_samples, args.batch_pairs, args.seed)
        ensemble.to_csv(args.output / "hN_shifted_split_ensemble.csv", index=False)
        split.to_csv(args.output / "h256_shifted_split_AB_sensitivity.csv", index=False)
        (args.output / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
