"""Paper analysis kernels, recovered from the original experiment scripts.
Preserves the published estimators and fit weights, including the original
first-bin uncertainty convention in SignGaussianInverter.
ASIC time parameters use oscilloscope sample ticks at 80 GSa/s
(12.5 ps per tick); period and tau_c therefore share the same unit.
"""
from __future__ import annotations
import math
from typing import Any
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from scipy.special import ndtr, ndtri
from scipy.integrate import cumulative_trapezoid
FIT_LAG_MIN=10
FIT_LAG_MAX=10000
N_ACF_BINS=72
TIME_STRIDE=8
BLOCK_SIZES=(100,200,500,1000,2000,5000)
TAIL_R2_THRESHOLD=.90
ACF_R2_THRESHOLD=.80
BLOCK_R2_THRESHOLD=.80
MODEL_FLOOR_TRANSFORM=.08
BLOCK_MODEL_FRACTION=.05
LAG_MIN=10
LAG_MAX=10000
BINS_PER_DECADE=24
MIN_POSITIVE_BINS=8


class SignGaussianInverter:
    """Invert normalized sign-bit correlation to latent Gaussian correlation."""

    def __init__(self, grid_size: int) -> None:
        if grid_size < 101:
            raise ValueError("--inversion-grid should be at least 101.")
        self.rho_grid = np.linspace(-0.999, 0.999, grid_size, dtype=np.float64)
        self._cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def _mapping_for_a(self, a: float) -> tuple[np.ndarray, np.ndarray]:
        # Quantizing the cache key avoids recomputing an expensive bivariate CDF
        # table for nearly identical p1 values.
        key = round(float(a), 6)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Plackett's identity:
        #   d Phi2(a,a;rho) / d rho = phi2(a,a;rho)
        # and Phi2(a,a;0)=Phi(a)^2.  For equal thresholds,
        #   phi2(a,a;rho) =
        #       exp(-a^2/(1+rho)) / (2*pi*sqrt(1-rho^2)).
        # This gives a fast vectorized CDF table over rho_grid and avoids the
        # very slow generic scipy.stats multivariate_normal.cdf loop.
        p = float(special.ndtr(key))
        mu_b = 2.0 * p - 1.0
        denom = 1.0 - mu_b * mu_b
        rho = self.rho_grid
        pdf_diag = np.exp(-(key * key) / (1.0 + rho)) / (
            2.0 * math.pi * np.sqrt(1.0 - rho * rho)
        )
        delta = np.diff(rho)
        trapezoids = 0.5 * (pdf_diag[:-1] + pdf_diag[1:]) * delta
        integral_from_left = np.concatenate([[0.0], np.cumsum(trapezoids)])
        zero_index = int(np.argmin(np.abs(rho)))
        integral_from_zero = integral_from_left - integral_from_left[zero_index]
        p11 = p * p + integral_from_zero
        g = 1.0 - 4.0 * p + 4.0 * p11
        rho_b_values = (g - mu_b * mu_b) / denom
        finite = np.isfinite(rho_b_values)
        if not np.all(finite):
            rho_b_values = rho_b_values[finite]
            rho_values = self.rho_grid[finite]
        else:
            rho_values = self.rho_grid
        order = np.argsort(rho_b_values)
        rho_b_values = rho_b_values[order]
        rho_values = rho_values[order]
        # Remove duplicate x values that can appear at numerical precision edges.
        uniq_x, uniq_idx = np.unique(rho_b_values, return_index=True)
        uniq_rho = rho_values[uniq_idx]
        self._cache[key] = (uniq_x, uniq_rho)
        return self._cache[key]

    def invert(self, p1: float, bit_acf: np.ndarray) -> np.ndarray:
        p1 = min(max(float(p1), 1.0e-12), 1.0 - 1.0e-12)
        a = float(special.ndtri(p1))
        x_grid, rho_grid = self._mapping_for_a(a)
        clipped = np.clip(bit_acf, x_grid[0], x_grid[-1])
        rho_v = np.interp(clipped, x_grid, rho_grid)
        rho_v[0] = 1.0
        return rho_v

def read_bits(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    return data & 1

def make_log_bins(lag_min: int, lag_max: int, n_bins: int) -> list[tuple[int, int, float, int]]:
    raw_edges = np.geomspace(lag_min, lag_max + 1, n_bins + 1)
    edges = np.unique(np.floor(raw_edges).astype(int))
    edges[0] = lag_min
    edges[-1] = lag_max + 1
    if edges.size < n_bins + 1:
        edges = np.unique(np.linspace(lag_min, lag_max + 1, n_bins + 1).astype(int))
        edges[0] = lag_min
        edges[-1] = lag_max + 1

    bins: list[tuple[int, int, float, int]] = []
    for i in range(edges.size - 1):
        lo = int(edges[i])
        hi = int(edges[i + 1]) - 1
        if hi < lo:
            continue
        lags = np.arange(lo, hi + 1, dtype=np.float64)
        bins.append((lo, hi, float(np.exp(np.mean(np.log(lags)))), int(lags.size)))
    return bins

def binned_trace_sums(
    bits: np.ndarray,
    p_group: float,
    bins: list[tuple[int, int, float, int]],
    time_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = bits.astype(np.float64, copy=False) - p_group
    n = int(x.size)
    prefix_x = np.empty(n + 1, dtype=np.float64)
    prefix_x[0] = 0.0
    np.cumsum(x, out=prefix_x[1:])
    x2 = x * x
    prefix_x2 = np.empty(n + 1, dtype=np.float64)
    prefix_x2[0] = 0.0
    np.cumsum(x2, out=prefix_x2[1:])
    base = np.arange(n, dtype=np.int64)

    numerators = np.zeros(len(bins), dtype=np.float64)
    left_energies = np.zeros(len(bins), dtype=np.float64)
    right_energies = np.zeros(len(bins), dtype=np.float64)
    pair_counts = np.zeros(len(bins), dtype=np.float64)

    for bin_index, (lo, hi, _tau_geo, _n_lags) in enumerate(bins):
        if lo >= n:
            continue
        m = n - lo
        positions = base[:m:time_stride]
        starts = positions + lo
        ends = positions + hi + 1
        np.minimum(ends, n, out=ends)
        window_sum = prefix_x[ends] - prefix_x[starts]
        x_pos = x[positions]
        valid_lag_counts = (ends - starts).astype(np.float64, copy=False)
        shifted_energy = prefix_x2[ends] - prefix_x2[starts]

        numerators[bin_index] += float(np.dot(x_pos, window_sum))
        left_energies[bin_index] += float(np.dot(x_pos * x_pos, valid_lag_counts))
        right_energies[bin_index] += float(np.sum(shifted_energy))
        pair_counts[bin_index] += float(np.sum(valid_lag_counts))

    return numerators, left_energies, right_energies, pair_counts

def tail_fit(acf: pd.DataFrame, lag_low: int = 100, lag_high: int = 10000) -> dict[str, float]:
    data = acf[
        (acf["lag_center"] >= lag_low)
        & (acf["lag_center"] <= lag_high)
        & np.isfinite(acf["rho_v"])
        & (acf["rho_v"] > 0.0)
    ].copy()
    x = np.log(data["tau"].to_numpy(float))
    y = np.log(data["rho_v"].to_numpy(float))
    if len(data) < 4:
        return {"beta": math.nan, "beta_ci_low": math.nan, "beta_ci_high": math.nan, "tail_r2_log": math.nan, "tail_rmse_log": math.nan, "tail_points": len(data)}
    design = np.column_stack([np.ones_like(x), x])
    coeff, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coeff
    residual = y - pred
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    dof = len(y) - 2
    sigma2 = sse / max(dof, 1)
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    slope = float(coeff[1])
    slope_se = math.sqrt(max(float(covariance[1, 1]), 0.0))
    critical = float(stats.t.ppf(0.975, dof)) if dof > 0 else 1.96
    beta = -slope
    return {
        "beta": beta,
        "beta_ci_low": max(beta - critical * slope_se, 1.0e-4),
        "beta_ci_high": beta + critical * slope_se,
        "tail_r2_log": 1.0 - sse / sst if sst > 0.0 else math.nan,
        "tail_rmse_log": math.sqrt(sse / len(y)),
        "tail_points": len(data),
    }

def kernel_average(block_sizes: np.ndarray, period: float, tau_c: float, beta: float) -> np.ndarray:
    sizes = np.asarray(block_sizes, dtype=int)
    max_m = int(np.max(sizes))
    lags = np.arange(1, max_m, dtype=float)
    values = np.power(1.0 + (lags * period) / tau_c, -beta)
    cumulative = np.r_[0.0, np.cumsum(values)]
    cumulative_lag = np.r_[0.0, np.cumsum(lags * values)]
    out = np.empty(len(sizes), dtype=float)
    for index, block_size in enumerate(sizes):
        sum_k = cumulative[block_size - 1]
        sum_mk = cumulative_lag[block_size - 1]
        out[index] = (block_size + 2.0 * (block_size * sum_k - sum_mk)) / (block_size * block_size)
    return out

def prepare_corner_data(acf: pd.DataFrame, block: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    a = acf[np.isfinite(acf["rho_v"]) & np.isfinite(acf["sem"])].copy().sort_values("tau")
    sem_positive = a.loc[a["sem"] > 0.0, "sem"].to_numpy(float)
    transform_scale = max(float(np.median(sem_positive)) if sem_positive.size else 1.0e-4, 1.0e-4)
    y = a["rho_v"].to_numpy(float)
    sem = np.maximum(a["sem"].to_numpy(float), transform_scale * 0.05)
    a["sigma_transform"] = np.sqrt(
        (sem / np.sqrt(y * y + transform_scale * transform_scale)) ** 2
        + MODEL_FLOOR_TRANSFORM**2
    )

    b = block[np.isfinite(block["z_block_var"])].copy().sort_values("block_size")
    variance_sem_analytic = b["z_block_var"].to_numpy(float) * np.sqrt(
        2.0 / np.maximum(b["n_blocks"].to_numpy(float) - 1.0, 1.0)
    )
    if "variance_sem_empirical" in b.columns:
        variance_sem = np.maximum(
            variance_sem_analytic,
            b["variance_sem_empirical"].to_numpy(float),
        )
    else:
        variance_sem = variance_sem_analytic
    excess = np.maximum(b["z_block_var"].to_numpy(float) - b["binom_var_global"].to_numpy(float), 0.0)
    b["excess_var"] = excess
    b["variance_sem"] = np.sqrt(
        np.maximum(variance_sem, 1.0e-6) ** 2
        + (BLOCK_MODEL_FRACTION * np.maximum(excess, 1.0e-6)) ** 2
    )
    return a, b, transform_scale

def predict(r_f: float, tau_c: float, beta: float, period: float, acf: pd.DataFrame, block: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    r2 = r_f * r_f
    amplitude = r2 / (1.0 + r2)
    acf_prediction = amplitude * np.power(1.0 + acf["tau"].to_numpy(float) / tau_c, -beta)
    block_prediction = block["binom_var_global"].to_numpy(float) + r2 * kernel_average(
        block["block_size"].to_numpy(int), period, tau_c, beta
    )
    return acf_prediction, block_prediction

def residual_vector(raw: np.ndarray, beta: float, period: float, acf: pd.DataFrame, block: pd.DataFrame, transform_scale: float, fixed_tau_c: float | None = None) -> np.ndarray:
    r_f = float(np.exp(raw[0]))
    tau_c = float(fixed_tau_c if fixed_tau_c is not None else np.exp(raw[1]))
    acf_prediction, block_prediction = predict(r_f, tau_c, beta, period, acf, block)
    observed = acf["rho_v"].to_numpy(float)
    acf_residual = (
        np.arcsinh(acf_prediction / transform_scale) - np.arcsinh(observed / transform_scale)
    ) / acf["sigma_transform"].to_numpy(float)
    block_residual = (
        block_prediction - block["z_block_var"].to_numpy(float)
    ) / block["variance_sem"].to_numpy(float)
    # ACF supplies many more binned points than the independent block-variance
    # check.  Balance the two evidence modalities by total weight so extending
    # the ACF lag window cannot silently suppress the block-scale constraint.
    block_residual *= math.sqrt(len(acf) / max(len(block), 1))
    return np.r_[acf_residual, block_residual]

def latent_acf_points(
    bits: np.ndarray,
    p1: float,
    period: float,
    inverter: SignGaussianInverter,
    trace_metadata: dict[str, Any],
) -> pd.DataFrame:
    bins = make_log_bins(FIT_LAG_MIN, FIT_LAG_MAX, N_ACF_BINS)
    numerator, left, right, pairs = binned_trace_sums(bits, p1, bins, TIME_STRIDE)
    denominator = np.sqrt(np.maximum(left * right, 0.0))
    rho_b = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )
    rho_v = inverter.invert(p1, np.r_[1.0, rho_b])[1:]

    # Approximate latent-ACF uncertainty by propagating a binomial-style
    # correlation standard error through the sign-Gaussian inversion map.  It is
    # used only as a numerical fit weight and diagnostic flag in this per-trace
    # heatmap data preparation.
    eps = 1.0e-4
    rho_b_low = np.clip(rho_b - eps, -0.999, 0.999)
    rho_b_high = np.clip(rho_b + eps, -0.999, 0.999)
    rho_v_low = inverter.invert(p1, rho_b_low)
    rho_v_high = inverter.invert(p1, rho_b_high)
    derivative = np.abs((rho_v_high - rho_v_low) / (rho_b_high - rho_b_low))
    sem_b = np.sqrt(np.maximum(1.0 - np.nan_to_num(rho_b) ** 2, 0.0)) / np.sqrt(
        np.maximum(pairs, 1.0)
    )
    sem_v = np.maximum(derivative * sem_b, 1.0 / np.sqrt(np.maximum(pairs, 1.0)))

    rows: list[dict[str, Any]] = []
    for index, (lo, hi, lag_center, n_lags) in enumerate(bins):
        rows.append(
            {
                **trace_metadata,
                "bin_index": index,
                "lag_lo": int(lo),
                "lag_hi": int(hi),
                "lag_center": float(lag_center),
                "tau": float(lag_center * period),
                "n_lags": int(n_lags),
                "pair_count": float(pairs[index]),
                "rho_b": float(rho_b[index]),
                "rho_v": float(rho_v[index]),
                "sem": float(sem_v[index]),
                "acf_time_stride": TIME_STRIDE,
            }
        )
    return pd.DataFrame(rows)

def within_trace_block_variance(
    bits: np.ndarray,
    p1: float,
    period: float,
    trace_metadata: dict[str, Any],
) -> pd.DataFrame:
    values = bits.astype(np.float64, copy=False)
    p_clip = float(np.clip(p1, 1.0e-12, 1.0 - 1.0e-12))
    z0 = float(special.ndtri(p_clip))
    phi0 = math.exp(-0.5 * z0 * z0) / math.sqrt(2.0 * math.pi)
    rows: list[dict[str, Any]] = []
    for block_size in BLOCK_SIZES:
        n_blocks = int(values.size // block_size)
        if n_blocks < 2:
            continue
        trimmed = values[: n_blocks * block_size].reshape(n_blocks, block_size)
        successes = np.sum(trimmed, axis=1)
        p_block = (successes + 0.5) / (block_size + 1.0)
        z_block = special.ndtri(p_block)
        z_block_var = float(np.var(z_block, ddof=1))
        binomial_variance = p_clip * (1.0 - p_clip) / (
            block_size * max(phi0 * phi0, 1.0e-15)
        )
        excess_var = max(z_block_var - binomial_variance, 0.0)
        rows.append(
            {
                **trace_metadata,
                "block_size": int(block_size),
                "block_time": float(block_size * period),
                "n_blocks": int(n_blocks),
                "p1_trace": float(p1),
                "z_block_var": z_block_var,
                "binom_var_global": float(binomial_variance),
                "excess_var": float(excess_var),
                "r_f_eff_block_p1": float(math.sqrt(excess_var)),
                "estimator_scope": "single_selected_trace_within_trace",
            }
        )
    return pd.DataFrame(rows)

def safe_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    if int(np.sum(mask)) < 2:
        return math.nan
    y = observed[mask]
    yhat = predicted[mask]
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if sst <= 0.0:
        return math.nan
    return 1.0 - float(np.sum((y - yhat) ** 2)) / sst

def fit_selected_trace(
    acf_raw: pd.DataFrame,
    block_raw: pd.DataFrame,
    period: float,
    *,
    preview_fast: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    beta_info = tail_fit(acf_raw, lag_low=100, lag_high=10000)
    beta = float(beta_info["beta"])
    if not np.isfinite(beta) or beta <= 0.0:
        empty = {
            **beta_info,
            "r_f": math.nan,
            "r_f_parameter_scope": "selected_trace_fixed_across_entropy_N",
            "r_f_estimation_method": "joint_binned_acf_and_multiscale_block_variance",
            "tau_c": math.nan,
            "tau_c_over_Ts": math.nan,
            "A_f": math.nan,
            "A_white": math.nan,
            "r2_acf_log": math.nan,
            "r2_acf_signed": math.nan,
            "r2_block_excess": math.nan,
            "negative_significant_bins": math.nan,
            "objective": math.nan,
            "fit_success": False,
            "hit_boundary": False,
            "min_eig_N256": math.nan,
            "quality_status": "invalid_tail_beta",
            "reportable": False,
        }
        return empty, acf_raw.copy(), block_raw.copy()

    acf, block, transform_scale = prepare_corner_data(acf_raw, block_raw)
    tau_min = period * 1.0e-4
    tau_max = period * 1.0e4
    r_seed = max(float(block.iloc[0]["r_f_eff_block_p1"]), 0.05)
    lower = np.array([math.log(1.0e-4), math.log(tau_min)])
    upper = np.array([math.log(5.0), math.log(tau_max)])

    def residual(raw: np.ndarray) -> np.ndarray:
        return residual_vector(raw, beta, period, acf, block, transform_scale)

    best: tuple[float, optimize.OptimizeResult] | None = None
    r_starts = [r_seed, 0.1, 0.8] if preview_fast else [r_seed, 0.05, 0.1, 0.3, 0.8, 1.5]
    tau_starts = [0.1, 1.0, 10.0, 100.0] if preview_fast else [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    max_nfev = 1200 if preview_fast else 5000
    for r0 in r_starts:
        for tau0 in period * np.asarray(tau_starts):
            start = np.log([max(r0, 1.0e-4), tau0])
            result = optimize.least_squares(
                residual,
                np.clip(start, lower + 1.0e-9, upper - 1.0e-9),
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=max_nfev,
            )
            objective = float(2.0 * result.cost)
            if best is None or objective < best[0]:
                best = (objective, result)
    assert best is not None
    objective, result = best
    r_f, tau_c = np.exp(result.x)
    acf_prediction, block_prediction = predict(
        float(r_f), float(tau_c), beta, period, acf, block
    )

    positive = acf["rho_v"].to_numpy(float) > 0.0
    log_observed = np.log(acf.loc[positive, "rho_v"].to_numpy(float))
    log_prediction = np.log(np.maximum(acf_prediction[positive], 1.0e-300))
    r2_acf_log = safe_r2(log_observed, log_prediction)
    r2_acf_signed = safe_r2(acf["rho_v"].to_numpy(float), acf_prediction)

    excess_observed = block["excess_var"].to_numpy(float)
    excess_prediction = block_prediction - block["binom_var_global"].to_numpy(float)
    r2_block = safe_r2(excess_observed, excess_prediction)
    negative_significant = int(
        np.sum(acf["rho_v"].to_numpy(float) + 1.96 * acf["sem"].to_numpy(float) < 0.0)
    )

    hit_boundary = bool(
        float(tau_c) <= tau_min * 1.01
        or float(tau_c) >= tau_max / 1.01
        or float(r_f) <= 1.01e-4
        or float(r_f) >= 4.95
    )

    r2_value = float(r_f * r_f)
    amplitude = r2_value / (1.0 + r2_value)
    max_n = 256
    lag_matrix = np.abs(np.arange(max_n)[:, None] - np.arange(max_n)[None, :]).astype(
        float
    )
    kernel = np.power(1.0 + lag_matrix * period / float(tau_c), -beta)
    correlation = amplitude * kernel + (1.0 - amplitude) * np.eye(max_n)
    min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (correlation + correlation.T))))

    reasons: list[str] = []
    if float(beta_info["tail_r2_log"]) < TAIL_R2_THRESHOLD:
        reasons.append("low_tail_r2")
    if np.isfinite(r2_acf_log) and r2_acf_log < ACF_R2_THRESHOLD:
        reasons.append("low_full_acf_r2")
    if np.isfinite(r2_block) and r2_block < BLOCK_R2_THRESHOLD:
        reasons.append("low_block_r2")
    if negative_significant > 0:
        reasons.append("significant_negative_acf")
    if hit_boundary:
        reasons.append("parameter_boundary")
    if min_eig < -1.0e-10:
        reasons.append("non_psd")
    quality_status = "pass" if not reasons else ";".join(reasons)

    acf_output = acf.copy()
    acf_output["rho_v_prediction"] = acf_prediction
    acf_output["standardized_signed_residual"] = residual(result.x)[: len(acf)]
    block_output = block.copy()
    block_output["z_block_var_prediction"] = block_prediction
    block_output["excess_var_prediction"] = excess_prediction

    fit = {
        **beta_info,
        "r_f": float(r_f),
        "r_f_parameter_scope": "selected_trace_fixed_across_entropy_N",
        "r_f_estimation_method": "joint_binned_acf_and_multiscale_block_variance",
        "tau_c": float(tau_c),
        "tau_c_over_Ts": float(tau_c / period),
        "A_f": float(amplitude),
        "A_white": float(1.0 - amplitude),
        "r2_acf_log": float(r2_acf_log),
        "r2_acf_signed": float(r2_acf_signed),
        "r2_block_excess": float(r2_block),
        "rmse_acf_log": float(
            np.sqrt(np.mean((log_observed - log_prediction) ** 2))
        ),
        "rmse_block_var": float(np.sqrt(np.mean((excess_observed - excess_prediction) ** 2))),
        "negative_significant_bins": negative_significant,
        "objective": objective,
        "fit_success": bool(result.success),
        "hit_boundary": hit_boundary,
        "min_eig_N256": min_eig,
        "quality_status": quality_status,
        "reportable": quality_status == "pass",
    }
    return fit, acf_output, block_output

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return math.nan
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))

def aggregate_acf(repeats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["temperature_label", "temperature_c", "voltage", "frequency", "case_label", "period", "bin_index"]
    for key, group in repeats.groupby(keys, sort=True):
        weights = group["pair_count"].to_numpy(float)
        values = group["rho_v"].to_numpy(float)
        mean = weighted_mean(values, weights)
        finite = values[np.isfinite(values)]
        sem = float(np.std(finite, ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else math.nan
        first = group.iloc[0]
        rows.append(
            {
                "temperature_label": key[0],
                "temperature_c": int(key[1]),
                "voltage": float(key[2]),
                "frequency": int(key[3]),
                "case_label": key[4],
                "period": float(key[5]),
                "bin_index": int(key[6]),
                "lag_lo": int(first["lag_lo"]),
                "lag_hi": int(first["lag_hi"]),
                "lag_center": float(first["lag_center"]),
                "tau": float(first["lag_center"] * key[5]),
                "rho_v": mean,
                "sem": max(sem, 1.0e-6) if np.isfinite(sem) else 1.0e-6,
                "rho_b": weighted_mean(group["rho_b"].to_numpy(float), weights),
                "n_repeats": int(group["count"].nunique()),
                "negative_fraction": float(np.mean(values < 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["temperature_c", "voltage", "frequency", "lag_center"], ignore_index=True)

def aggregate_block(repeats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["temperature_label", "temperature_c", "voltage", "frequency", "case_label", "period", "block_size"]
    for key, group in repeats.groupby(keys, sort=True):
        z = group["z_block_var"].to_numpy(float)
        binomial = group["binom_var_global"].to_numpy(float)
        z_mean = float(np.mean(z))
        bin_mean = float(np.mean(binomial))
        rows.append(
            {
                "temperature_label": key[0],
                "temperature_c": int(key[1]),
                "voltage": float(key[2]),
                "frequency": int(key[3]),
                "case_label": key[4],
                "period": float(key[5]),
                "block_size": int(key[6]),
                "block_time": float(key[5] * key[6]),
                "n_blocks": int(group["n_blocks"].sum()),
                "min_blocks_per_trace": int(group["n_blocks"].min()),
                "z_block_var": z_mean,
                "z_block_var_std_across_repeats": float(np.std(z, ddof=1)),
                "variance_sem_empirical": float(np.std(z, ddof=1) / math.sqrt(len(z))),
                "binom_var_global": bin_mean,
                "excess_var": max(z_mean - bin_mean, 0.0),
                "r_f_eff_block_p1": math.sqrt(max(z_mean - bin_mean, 0.0)),
                "n_repeats": int(group["count"].nunique()),
                "estimator_scope": "within_trace_then_repeat_mean",
            }
        )
    return pd.DataFrame(rows).sort_values(["temperature_c", "voltage", "frequency", "block_size"], ignore_index=True)

def half(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "A":
        return frame[frame["count"].between(0, 4)].copy()
    if name == "B":
        return frame[frame["count"].between(5, 9)].copy()
    raise ValueError(name)

def fit_half(acf: pd.DataFrame, block: pd.DataFrame, name: str) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    block_predictions: list[pd.DataFrame] = []
    keys = ["temperature_label", "temperature_c", "voltage", "frequency", "case_label", "period"]
    corners = acf[keys].drop_duplicates().sort_values(["temperature_c", "voltage", "frequency"])
    for index, corner in enumerate(corners.itertuples(index=False), start=1):
        a = acf[(acf["temperature_label"] == corner.temperature_label) & (acf["case_label"] == corner.case_label)].copy()
        b = block[(block["temperature_label"] == corner.temperature_label) & (block["case_label"] == corner.case_label)].copy()
        fit, _, b_out = fit_selected_trace(a, b, float(corner.period), preview_fast=False)
        rows.append(
            {
                "split": name,
                "temperature_label": corner.temperature_label,
                "temperature_c": int(corner.temperature_c),
                "voltage": float(corner.voltage),
                "frequency": int(corner.frequency),
                "case_label": corner.case_label,
                "period": float(corner.period),
                "n_repeats": int(a["n_repeats"].min()),
                **fit,
            }
        )
        b_out["split"] = name
        b_out["temperature_label"] = corner.temperature_label
        b_out["case_label"] = corner.case_label
        block_predictions.append(b_out)
        print(f"[{name} {index:02d}/24] {corner.temperature_label} {corner.case_label}: {fit['quality_status']}", flush=True)
    return pd.DataFrame(rows), block_predictions

def held_out_predictions(params_a: pd.DataFrame, acf_b: pd.DataFrame, block_b: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for parameter in params_a.itertuples(index=False):
        a = acf_b[(acf_b["temperature_label"] == parameter.temperature_label) & (acf_b["case_label"] == parameter.case_label)].copy()
        b = block_b[(block_b["temperature_label"] == parameter.temperature_label) & (block_b["case_label"] == parameter.case_label)].copy()
        prepared_acf, prepared_block, _ = prepare_corner_data(a, b)
        _, prediction = predict(float(parameter.r_f), float(parameter.tau_c), float(parameter.beta), float(parameter.period), prepared_acf, prepared_block)
        out = prepared_block.copy()
        out["temperature_label"] = parameter.temperature_label
        out["temperature_c"] = int(parameter.temperature_c)
        out["voltage"] = float(parameter.voltage)
        out["frequency"] = int(parameter.frequency)
        out["case_label"] = parameter.case_label
        out["parameter_split"] = "A"
        out["validation_split"] = "B"
        out["r_f_calibration"] = float(parameter.r_f)
        out["tau_c_calibration"] = float(parameter.tau_c)
        out["beta_calibration"] = float(parameter.beta)
        out["calibration_quality_status"] = parameter.quality_status
        out["calibration_reportable"] = bool(parameter.reportable)
        out["z_block_var_prediction"] = prediction
        out["excess_var_prediction"] = prediction - out["binom_var_global"].to_numpy(float)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)

def fig12_bins() -> list[tuple[int, int, float]]:
    n_bins = int(math.ceil(math.log10(LAG_MAX / LAG_MIN) * BINS_PER_DECADE))
    edges = np.unique(np.rint(np.geomspace(LAG_MIN, LAG_MAX + 1, n_bins + 1)).astype(int))
    bins: list[tuple[int, int, float]] = []
    for lo, next_hi in zip(edges[:-1], edges[1:]):
        hi = int(next_hi) - 1
        if hi >= int(lo):
            bins.append((int(lo), hi, math.sqrt(int(lo) * hi)))
    return bins

def bit_acf_all_lags(bits: np.ndarray, p1: float, lag_max: int) -> np.ndarray:
    """Pearson bit ACF for lags 0..lag_max, evaluated with FFT numerators."""
    x = bits.astype(np.float64, copy=False) - float(p1)
    n = x.size
    fft_len = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, fft_len)
    numerators = np.fft.irfft(spectrum * np.conj(spectrum), fft_len)[: lag_max + 1]
    energy_prefix = np.empty(n + 1, dtype=np.float64)
    energy_prefix[0] = 0.0
    np.cumsum(x * x, out=energy_prefix[1:])
    lags = np.arange(lag_max + 1)
    left_energy = energy_prefix[n - lags]
    right_energy = energy_prefix[n] - energy_prefix[lags]
    denominator = np.sqrt(np.maximum(left_energy * right_energy, 0.0))
    return np.divide(numerators, denominator, out=np.full(lag_max + 1, np.nan), where=denominator > 0)

def invert_gaussian_threshold(rho_x: np.ndarray, p1: float, grid_size: int = 200_001) -> np.ndarray:
    """Numerically invert Eq. (23) by tabulating its monotone bivariate-normal map.

    Plackett's identity gives d Phi_2(a,a;r) / dr = phi_2(a,a;r), so numerical
    quadrature on a dense correlation grid avoids an unstable root solve for
    every one of the 9,991 lags.
    """
    p = float(np.clip(p1, np.finfo(float).eps, 1.0 - np.finfo(float).eps))
    a = float(ndtri(p))
    rho_grid = np.linspace(-0.999999, 0.999999, grid_size)
    density = np.exp(-(a * a) / (1.0 + rho_grid)) / (2.0 * np.pi * np.sqrt(1.0 - rho_grid * rho_grid))
    cdf2 = p * p + cumulative_trapezoid(density, rho_grid, initial=0.0)
    zero_index = int(np.argmin(np.abs(rho_grid)))
    cdf2 -= cdf2[zero_index] - p * p
    rho_x_grid = (cdf2 - p * p) / (p * (1.0 - p))

    finite = np.isfinite(rho_x)
    result = np.full(rho_x.shape, np.nan, dtype=float)
    # Values outside the feasible range are estimation noise; clipping makes
    # their inversion explicit and they remain eligible only if positive.
    result[finite] = np.interp(rho_x[finite], rho_x_grid, rho_grid, left=rho_grid[0], right=rho_grid[-1])
    return result

def fit_binned_latent_acf(lags: np.ndarray, rho_v: np.ndarray, period: float, bins: list[tuple[int, int, float]]) -> tuple[dict[str, float | int | str], list[dict[str, float | int]]]:
    rows: list[dict[str, float | int]] = []
    for lo, hi, lag_geo in bins:
        values = rho_v[lo : hi + 1]
        finite = values[np.isfinite(values)]
        rho_v_mean = float(np.mean(finite)) if finite.size else np.nan
        rows.append({"lag_lo": lo, "lag_hi": hi, "lag_geo": lag_geo, "tau_geo": lag_geo * period, "rho_v_mean": rho_v_mean, "n_lags": int(finite.size), "n_positive_lags": int(np.sum(finite > 0))})

    binned = pd.DataFrame(rows)
    valid = np.isfinite(binned["rho_v_mean"]) & (binned["rho_v_mean"] > 0)
    if int(valid.sum()) < MIN_POSITIVE_BINS:
        return {"beta": np.nan, "intercept_log": np.nan, "r2_log": np.nan, "rmse_log": np.nan, "n_fit_bins": int(valid.sum()), "status": "few_positive_bins"}, rows

    x = np.log(binned.loc[valid, "tau_geo"].to_numpy(dtype=float))
    y = np.log(binned.loc[valid, "rho_v_mean"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {"beta": float(-slope), "intercept_log": float(intercept), "r2_log": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan, "rmse_log": float(math.sqrt(ss_res / len(x))), "n_fit_bins": int(len(x)), "status": "ok"}, rows
