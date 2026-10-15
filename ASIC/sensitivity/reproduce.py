"""Regenerate the Section 5 numerical summary from archived scientific results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/derived"


def summarize() -> dict:
    kernels = pd.read_csv(DATA / "h256_kernel_sensitivity.csv")
    pivot = kernels.pivot(index=["temperature_c", "voltage", "frequency"], columns="kernel", values="h256")
    ou = pivot["ou_lorentzian_8term"] - pivot["shifted_power"]
    spectral = pivot["bandlimited_1_over_f_gamma"] - pivot["shifted_power"]
    tail = pd.read_csv(DATA / "h256_trajectory_exceedance_quantiles.csv")
    split = pd.read_csv(DATA / "h256_shifted_split_AB_sensitivity.csv")
    ab = split.pivot(index=["temperature_c", "voltage", "frequency"], columns="split", values="h256_per_raw_output_bit")
    hn = pd.read_csv(DATA / "hN_shifted_split_ensemble.csv").pivot(index=["temperature_c", "voltage", "frequency"], columns="n", values="h_per_raw_output_bit")
    heldout = pd.read_csv(ROOT.parent / "fig14/data/derived/fig14_platform_split_heldout_block.csv")
    observed = heldout["excess_var"].to_numpy()
    predicted = heldout["excess_var_prediction"].to_numpy()
    errors = np.abs(predicted - observed) / observed
    return {
        "corners": len(pivot),
        "ou_median_absolute_change": float(ou.abs().median()),
        "spectral_median_increase": float(spectral.median()),
        "spectral_increase_all_corners": bool((spectral > 0).all()),
        "split_h256_median_absolute_difference": float((ab.A - ab.B).abs().median()),
        "h16_gt_h64_gt_h256_all_corners": bool(((hn[16] > hn[64]) & (hn[64] > hn[256])).all()),
        "heldout_points": len(heldout),
        "heldout_loglog_pearson_r_squared": float(np.corrcoef(np.log(observed), np.log(predicted))[0, 1]**2),
        "heldout_linear_r2_sse_about_mean": float(1 - np.sum((observed - predicted)**2) / np.sum((observed - observed.mean())**2)),
        "heldout_median_relative_error": float(np.median(errors)),
        "heldout_maximum_multiplicative_factor": float(np.maximum(predicted / observed, observed / predicted).max()),
        "tail_thresholds": [
            {"coverage": float(coverage), "min": float(group.h256_threshold.min()), "max": float(group.h256_threshold.max())}
            for coverage, group in tail.groupby("exceedance_probability")
        ],
    }


if __name__ == "__main__":
    summary = summarize()
    output = DATA / "paper_claims_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
