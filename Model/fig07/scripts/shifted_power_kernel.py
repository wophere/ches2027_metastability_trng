"""Shared shifted-power covariance utilities for Section 3 figures.

The normalized flicker ACF is rho_f(tau)=(1+|tau|/tau_c)^(-beta).
For beta>0 it is a nonnegative mixture of exponential kernels and is
positive semidefinite on every finite sampling grid.
"""

from __future__ import annotations

import numpy as np


def shifted_power_acf(lag: np.ndarray | float, ts: float, tau_c: float, beta: float) -> np.ndarray:
    """Evaluate rho_f(|lag| T_s) for nonnegative or signed discrete lags."""
    if ts <= 0 or tau_c <= 0:
        raise ValueError("ts and tau_c must be positive")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must satisfy 0 < beta < 1")
    lag_array = np.asarray(lag, dtype=float)
    return np.power(1.0 + np.abs(lag_array) * ts / tau_c, -beta)


def shifted_power_corr_matrix(n: int, ts: float, tau_c: float, beta: float) -> np.ndarray:
    """Return the N-by-N normalized flicker covariance matrix."""
    if n < 1:
        raise ValueError("n must be positive")
    indices = np.arange(n, dtype=float)
    return shifted_power_acf(indices[:, None] - indices[None, :], ts, tau_c, beta)


def cholesky_psd(corr: np.ndarray, tolerance: float = 1.0e-10) -> tuple[np.ndarray, float]:
    """Verify PSD numerically and return a Cholesky factor without projection."""
    symmetric = 0.5 * (np.asarray(corr, dtype=float) + np.asarray(corr, dtype=float).T)
    min_eig = float(np.linalg.eigvalsh(symmetric)[0])
    if min_eig < -tolerance:
        raise ValueError(f"Correlation matrix is not PSD: minimum eigenvalue {min_eig:.3e}")
    return np.linalg.cholesky(symmetric + max(0.0, -min_eig) * np.eye(len(symmetric))), min_eig
