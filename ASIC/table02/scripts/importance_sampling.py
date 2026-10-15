#!/usr/bin/env python3
"""Symmetric-mixture importance-sampling estimator for Table 2.

period and tau_c share the 80 GSa/s oscilloscope sample-tick unit.
The covariance depends on their dimensionless ratio.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import linalg, special


BLOCK_LENGTHS = (16, 64, 256)


def build_sampler(beta: float, tau_c: float, period: float, n: int):
    first_col = np.power(1.0 + np.arange(n, dtype=float) * period / tau_c, -beta)
    embedding = np.r_[first_col, first_col[-2:0:-1]]
    eigenvalues = np.fft.rfft(embedding).real
    if float(eigenvalues.min()) < -1.0e-10:
        raise ValueError("Pilot requires a nonnegative circulant embedding")
    sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0.0))

    def sample(rng: np.random.Generator, batch: int) -> np.ndarray:
        white = rng.standard_normal((batch, embedding.size))
        spectrum = np.fft.rfft(white, axis=1)
        return np.fft.irfft(spectrum * sqrt_eigenvalues[None, :], n=embedding.size, axis=1)[:, :n]

    return first_col, float(eigenvalues.min()), sample


def estimate(row: dict, output_samples: int, batch_pairs: int, shifts: np.ndarray, seed: int) -> dict:
    n_max = max(BLOCK_LENGTHS)
    first_col, embedding_min_eig, sampler = build_sampler(
        float(row["beta"]), float(row["tau_c"]), float(row["period"]), n_max
    )
    precision_ones = {}
    energies = {}
    for n in BLOCK_LENGTHS:
        vector = linalg.solve_toeplitz((first_col[:n], first_col[:n]), np.ones(n), check_finite=False)
        precision_ones[n] = vector
        energies[n] = float(vector.sum())

    rng = np.random.default_rng(seed)
    log_sum = {n: -math.inf for n in BLOCK_LENGTHS}
    log_sum_sq = {n: -math.inf for n in BLOCK_LENGTHS}
    pair_count = 0
    max_pairs = math.ceil(output_samples / 2)
    r_mu = float(row["r_mu_heldout"])
    r_f = float(row["r_f"])
    log_components = math.log(len(shifts))
    while pair_count < max_pairs:
        batch = min(batch_pairs, max_pairs - pair_count)
        z = sampler(rng, batch)
        components = rng.integers(0, len(shifts), size=batch)
        x = z + shifts[components, None]
        for n in BLOCK_LENGTHS:
            score = x[:, :n] @ precision_ones[n]
            log_q_over_p = special.logsumexp(
                score[:, None] * shifts[None, :]
                - 0.5 * energies[n] * shifts[None, :] ** 2,
                axis=1,
            ) - log_components
            log_weight = -log_q_over_p
            state_pos = r_mu + r_f * x[:, :n]
            state_neg = r_mu - r_f * x[:, :n]
            log_g_pos = np.sum(special.log_ndtr(np.abs(state_pos)), axis=1)
            log_g_neg = np.sum(special.log_ndtr(np.abs(state_neg)), axis=1)
            log_pair = np.logaddexp(log_g_pos + log_weight, log_g_neg + log_weight) - math.log(2.0)
            log_sum[n] = float(np.logaddexp(log_sum[n], special.logsumexp(log_pair)))
            log_sum_sq[n] = float(np.logaddexp(log_sum_sq[n], special.logsumexp(2.0 * log_pair)))
        pair_count += batch

    output = {
        "is_output_samples": 2 * pair_count,
        "is_antithetic_pairs": pair_count,
        "is_shifts": " ".join(f"{value:g}" for value in shifts),
        "is_embedding_min_eig": embedding_min_eig,
    }
    for n in BLOCK_LENGTHS:
        log_mean = log_sum[n] - math.log(pair_count)
        log_mean_sq = log_sum_sq[n] - math.log(pair_count)
        relative_variance = max(math.exp(log_mean_sq - 2.0 * log_mean) - 1.0, 0.0)
        standard_error_log = math.sqrt(relative_variance / pair_count)
        output[f"h_model_N{n}_is"] = -log_mean / (n * math.log(2.0))
        output[f"h_model_N{n}_is_se"] = standard_error_log / (n * math.log(2.0))
    return output
