"""Derive resolution-time samples and fitted parameters from raw Spectre data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_resolution_time_fit import DERIVED_DIR, VOLTAGES, fit_all, load_data


def main() -> None:
    samples = load_data()
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for voltage, values in zip(VOLTAGES, samples):
        rows.extend(
            {"voltage_v": voltage, "sample_index": index, "resolution_time_ps": value}
            for index, value in enumerate(values)
        )
    pd.DataFrame(rows).to_csv(DERIVED_DIR / "metastability_resolution_samples.csv", index=False)
    fit_all(samples)


if __name__ == "__main__":
    main()
