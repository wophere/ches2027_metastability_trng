# Section 5: validation, trajectory tails, and alternative kernels

These analyses evaluate held-out predictions, the dependence of entropy on block length and covariance kernel, and lower-tail trajectory quantiles across the 24 ASIC operating corners.

## Summarize the paper results

Run `python reproduce.py` in this directory to combine the reference results and Figure 14 held-out statistics into `data/derived/paper_claims_summary.json`. The root `python verify.py` checks the displayed values at their rounding precision.

The held-out squared Pearson correlation between `log(observed)` and `log(predicted)` is `R^2=0.9404`. The linear-scale `1-SSE/SST` is 0.9712. The maximum multiplicative discrepancy is 1.62, defined as `max(predicted/observed, observed/predicted)`.

## Data and parameters

The five reference CSVs contain the 24 corner parameter sets, 72 kernel entropy estimates, 72 trajectory quantiles, 48 independent A/B estimates, and 72 block-length estimates. Sensitivity parameters combine the independent five-capture fits with bias pooled across all ten captures.

Time parameters use oscilloscope sample ticks at 80 GSa/s (12.5 ps per tick). Fields ending in `_ticks` contain times; fields ending in `_per_tick` contain reciprocal frequencies. Multiply tick times by 0.0125 for nanoseconds, or inverse-tick frequencies by 80 for GHz. See the [raw-data guide](../raw_reproduction/README.md) for the F1–F4 periods.

## Compute new estimates

Run in this directory:

```sh
python scripts/compute_sensitivity.py
```

The script saves results in `data/recomputed/`. Defaults use seed 20260824; 40,000 importance-sampling outputs per corner/kernel; batches of 1,000 antithetic pairs; and 262,144 trajectories per corner for lower-tail quantiles. The three kernels are shifted power, an eight-term positive OU mixture, and a band-limited `1/f^(1-beta)` kernel. Covariances undergo a direct positive-definiteness check. Lower-tail coverage levels are 93.75%, 99.609375%, and 99.90234375%.

For a single-corner computational check, run:

```sh
python scripts/compute_sensitivity.py --first-corner-only
```

Omit `--first-corner-only` to evaluate all 24 corners. Reported Monte Carlo standard errors quantify sampling uncertainty conditional on the supplied parameter values. The A/B split analysis reports variation between independently fitted capture subsets.
