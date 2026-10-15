"""Check regenerated outputs and numerical results against the paper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def results() -> None:
    import numpy as np
    import pandas as pd

    outputs = (
        'Spectre/fig04/figure/fig04_p1_vs_delta_v.pdf',
        'Spectre/fig05/figure/fig05_resolution_time_fit.pdf',
        'Model/fig06/figure/fig06_model_entropy_vs_fs_shifted.pdf',
        'Model/fig07/figure/fig07_hn_heatmap_shifted.pdf',
        'Spectre/fig08/figure/fig08_spectre_flicker_noise_effect.pdf',
        'FPGA/fig10/figure/fig10_fpga_powerlaw_bias_independence.pdf',
        'ASIC/fig12/figure/fig12_asic_core1_acf_three_temperature_powerlaw.pdf',
        'ASIC/fig13/figure/fig13_es1_es2_cross_correlation.pdf',
        'ASIC/fig14/figure/fig14_asic_core1_platform_params.pdf',
        'Spectre/table01/data/derived/table01_flicker_90b_tracks_rows.tex',
        'ASIC/table02/data/derived/table02_27c_bias_min_rows.tex',
        'ASIC/sensitivity/data/derived/paper_claims_summary.json',
    )
    for output in outputs:
        path = ROOT / output
        require(path.is_file() and path.stat().st_size > 0, f"Missing output: {path}")
    table1 = pd.read_csv(ROOT / "Spectre/table01/data/derived/table01_flicker_90b_tracks.csv")
    np.testing.assert_allclose(table1.p1, [0.498972, 0.499940], atol=0.0000005, rtol=0)
    np.testing.assert_allclose(table1.sp800_90b_non_iid_track, [0.686864, 0.886304], atol=0.0000005, rtol=0)
    table2 = pd.read_csv(ROOT / "ASIC/table02/data/derived/table02_27c_bias_min.csv").sort_values(["voltage", "frequency"])
    expected = np.array([
        [0.002, 0.994, 0.631, 0.815, 0.650, 0.412],
        [0.011, 0.969, 0.597, 0.709, 0.516, 0.287],
        [-0.004, 0.989, 0.600, 0.706, 0.502, 0.294],
        [0.001, 0.998, 0.575, 0.642, 0.472, 0.293],
        [-0.001, 0.998, 0.615, 0.762, 0.561, 0.323],
        [0.010, 0.972, 0.593, 0.707, 0.500, 0.290],
        [-0.011, 0.969, 0.528, 0.627, 0.444, 0.248],
        [-0.005, 0.986, 0.506, 0.580, 0.397, 0.237],
    ])
    cols = ["bias", "h_iid", "h90b", "h_model_N16", "h_model_N64", "h_model_N256"]
    np.testing.assert_allclose(table2[cols], expected, atol=0.000500001, rtol=0)
    fpga = ROOT / "FPGA/fig10/data/derived"
    summary = pd.read_csv(fpga / "fpga_powerlaw_bias_independence_summary.csv")
    np.testing.assert_allclose(summary.beta, [0.269, 0.248, 0.325], atol=0.0005, rtol=0)
    require(bool((summary.r2_log > 0.990).all()), "FPGA R2 differs from the paper")
    runs = pd.read_csv(fpga / "fpga_powerlaw_bias_independence_run_metrics.csv")
    require(len(runs) == 30 and bool((runs.inflation_B4096 > 14).all()), "FPGA variance-inflation claim failed")
    selected = runs.loc[runs.groupby("freq").abs_bias_percent.idxmin()].sort_values("freq")
    np.testing.assert_allclose(selected.abs_bias_percent, [0.485, 0.374, 1.495], atol=0.000500001, rtol=0)
    np.testing.assert_allclose(selected.inflation_B4096, [20.7, 26.8, 17.7], atol=0.05, rtol=0)
    dual = json.loads((ROOT / "ASIC/fig13/data/derived/summary.json").read_text())
    require(dual["capture_count"] == 10 and dual["bits_per_channel_per_capture"] == 1000000, "Incorrect dual-ES capture size")
    require(abs(dual["zero_lag"]["mean"] + 0.070) < 0.0005, "Dual-ES correlation differs from Figure 13")
    s = json.loads((ROOT / "ASIC/sensitivity/data/derived/paper_claims_summary.json").read_text())
    require(s["corners"] == 24 and s["heldout_points"] == 144, "Incorrect ASIC validation dimensions")
    for key, expected_value, tolerance in [
        ("ou_median_absolute_change", 0.0010, 0.00005),
        ("spectral_median_increase", 0.2681, 0.00005),
        ("split_h256_median_absolute_difference", 0.0057, 0.00005),
        ("heldout_loglog_pearson_r_squared", 0.9404, 0.00005),
        ("heldout_median_relative_error", 0.0547, 0.00005),
        ("heldout_maximum_multiplicative_factor", 1.62, 0.005),
    ]:
        require(abs(s[key] - expected_value) <= tolerance, f"Paper claim failed: {key} = {s[key]}")
    require(s["spectral_increase_all_corners"] and s["h16_gt_h64_gt_h256_all_corners"], "Kernel or block-length claim failed")
    np.testing.assert_allclose([[x["min"], x["max"]] for x in s["tail_thresholds"]], [[0.456, 0.734], [0.313, 0.610], [0.265, 0.564]], atol=0.0005, rtol=0)
    print("Results OK: all figure outputs, both tables, FPGA claims, dual-ES correlation, ASIC validation, and sensitivity claims.")


def recomputed() -> None:
    import numpy as np
    import pandas as pd

    base = ROOT / "ASIC/table02/data"
    archived = pd.read_csv(base / "derived/table02_27c_bias_min.csv")
    fresh = pd.read_csv(base / "recomputed/table02_27c_bias_min.csv")
    for n in (16, 64, 256):
        col, se = f"h_model_N{n}", f"h_model_N{n}_mc_se"
        difference = (archived[col] - fresh[col]).abs()
        combined_se = np.sqrt(archived[se]**2 + fresh[se]**2)
        require(bool((difference <= 4 * combined_se).all()), f"Table 2 N={n}: difference exceeds four combined reported SEs")
        print(f"Table 2 N={n}: max difference {difference.max():.9f}; max standardized difference {(difference / combined_se).max():.3f}")
    base = ROOT / "ASIC/sensitivity/data"
    for name in (
        "h256_kernel_sensitivity.csv", "h256_trajectory_exceedance_quantiles.csv",
        "h256_shifted_split_AB_sensitivity.csv", "hN_shifted_split_ensemble.csv",
    ):
        archived = pd.read_csv(base / "derived" / name)
        fresh = pd.read_csv(base / "recomputed" / name)
        require(archived.shape == fresh.shape, f"Incomplete sensitivity recomputation: {name}")
        for col in archived.select_dtypes("number"):
            np.testing.assert_allclose(fresh[col], archived[col], rtol=1e-8, atol=1e-10, equal_nan=True, err_msg=f"{name}: {col}")
    print("Heavy recomputation OK: Table 2 statistical agreement and all four sensitivity result files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recomputed", action="store_true", help="Check full optional Table 2 and sensitivity recomputations")
    args = parser.parse_args()
    if args.recomputed:
        recomputed()
    else:
        results()
