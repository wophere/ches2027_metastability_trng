"""Derive Fig. 4 measurements and Gaussian-CDF fit from raw Spectre exports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_p1_vs_delta_v import compute_p1_vs_delta_v, fit_gaussian_cdf


def main() -> None:
    root = HERE.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=root / "data" / "raw" / "fig04_p1_vs_delta_v_raw")
    parser.add_argument("--output", type=Path, default=root / "data" / "derived" / "fig04_p1_vs_delta_v__summary.csv")
    args = parser.parse_args()
    rows = []
    for voltage in ("0.9", "1.1", "1.3"):
        frame = compute_p1_vs_delta_v(args.raw / f"verify_effective_metastable_point_{voltage}.csv", 0.45)
        params, _, r2 = fit_gaussian_cdf(frame["delta_V"].to_numpy(), frame["p1"].to_numpy())
        frame["vdd_v"] = float(voltage)
        frame["fit_x0_v"] = params[0]
        frame["fit_sigma_v"] = params[1]
        frame["fit_r_squared"] = r2
        rows.append(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
