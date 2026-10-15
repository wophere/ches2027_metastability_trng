# ASIC raw-data reproduction

`captures.csv` describes 250 core-1 captures: 24 temperature/voltage/frequency corners with ten repeats, plus ten candidates from the second -20 C, 0.9 V/F1 acquisition. Each raw byte represents one symbol (`byte & 1`). `num_samples`, `sha256`, `raw_path`, measured probability, period in oscilloscope sample ticks, RDAC settings, temperature in Celsius, and repeat index are explicit. The `frequency` field is the paper's setting index F1–F4, corresponding to 1, 2, 3, and 4 GHz.

## Time units

The oscilloscope sampling rate is 80 GSa/s: one sample tick is 12.5 ps (0.0125 ns). The `period` values 80, 40, 26.6666667, and 20 are sample ticks per raw-output cycle, corresponding to physical periods of 1, 0.5, 1/3, and 0.25 ns for F1–F4.

ASIC analysis uses this common tick scale for `period`, `tau_c`, `tau_geo`, `block_time`, and related time parameters. Explicit unit suffixes use `_ticks`; spectral-frequency suffixes use `_per_tick`. Convert a time to nanoseconds by multiplying by 0.0125, or an inverse-tick frequency to GHz by multiplying by 80. Thus `tau_c / period` is the dimensionless `tau_c/T_s` shown in the paper. The shifted-power, OU, and spectral kernels use compatible time and frequency scales throughout.

## Reproduction

Selection uses minimum absolute bias over all acquisition candidates, breaking ties by count and filename. Figure 14 uses only RDAC 100/190 at -20 C, 0.9 V/F1, matching the original split experiment; 100/191 remains included because minimum-bias selection for Figure 12/Table 2 considered it. Split A uses repeat counts 0–4, and B uses 5–9, without overlap.

```sh
python ASIC/raw_reproduction/recompute.py
python ASIC/raw_reproduction/verify_raw.py
python ASIC/table02/scripts/recompute_table02_importance.py --params ASIC/raw_reproduction/data/recomputed/three_temp_bias_min_params_long.csv
```

The first command verifies capture hashes and lengths, then computes:

- Figure 12: integer-lag bit ACF, Gaussian-threshold inversion on a 200001 point grid, logarithmic bins, and ordinary least-squares power-law fits.
- Table 2 parameters: the original stride-8 binned ACF, 601-point threshold inversion, six within-trace block-variance scales, tail exponent, and multistart robust joint fits for 24 selected captures.
- Figure 14: within-capture statistics aggregated independently over the two five-capture splits, 48 fits, and A-calibrated predictions on B.

Figure 12 uses integer-lag ACF bins; joint parameter extraction uses stride-8 ACF bins together with block variances. Each method supplies its own fitted exponent. `analysis.py` uses the original first-bin uncertainty convention for the joint estimator and exports fit-quality diagnostics with every fit.

All fresh results go to `data/recomputed/`. `--reuse-statistics` reuses only per-capture ACF/block statistics from an earlier run, while recomputing fits. For a single command that also plots fresh Figures 12/14:

```sh
python reproduce.py --asic-from-raw fig12 fig14
```

`verify_raw.py` compares the newly fitted parameters with the reference CSVs using numerical tolerances that accommodate floating-point variation across linear algebra libraries.
