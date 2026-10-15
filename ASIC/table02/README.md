# Table 2: ASIC conditional min-entropy

Table 2 compares bias-based i.i.d. entropy, SP 800-90B estimates, and the stochastic model's conditional min-entropy for the eight 27 °C operating corners. For 256-bit blocks, the model estimates 0.237–0.412 bits per sample, incorporating the temporal dependence captured by the fitted noise parameters.

## Generate the table

Run in this directory:

```sh
python reproduce.py
```

The command renders the paper's reference CSV as LaTeX table rows.

## Data and scripts

- `data/derived/three_temp_bias_min_params_long.csv` contains the selected minimum-bias capture metadata and shifted-power model parameters.
- `data/derived/table02_27c_bias_min.csv` contains the 27 °C results, including SP 800-90B `H_original`, i.i.d. entropy, model entropy for `N=16,64,256`, and importance-sampling diagnostics.
- `scripts/importance_sampling.py` implements the symmetric-mixture importance-sampling estimator.
- `scripts/recompute_table02_importance.py` computes model entropy from the supplied parameters and accepts newly extracted parameters with `--params`.
- `scripts/generate_table02.py` generates the LaTeX row fragments.

See the [raw-data guide](../raw_reproduction/README.md) for capture selection and parameter extraction, and the [NIST guide](../../nist90b/README.md) for SP 800-90B computation.

## Compute model estimates

Run in this directory:

```sh
python scripts/recompute_table02_importance.py --is-samples 100000 --is-batch-pairs 2000
```

New outputs are saved in `data/recomputed/`. Each corner uses seed `20260713 + round(voltage*1000)*100 + frequency`. The model-table script combines the model estimates with the reference NIST values; the NIST runner saves its new estimates and logs in `nist90b/data/recomputed/` at the repository root.

Each computation writes a `.metadata.json` sidecar with the input hash, seed formula, batch size, RNG, and environment versions. `--resume` checks that the metadata match before combining results.

Release validation found agreement within 1.65 combined reported standard errors. Maximum differences were 0.001122, 0.000500, and 0.000357 bit/sample for `N=16,64,256`, respectively. After running both the Table 2 and [sensitivity](../sensitivity/README.md) computations, check their results from the repository root:

```sh
python verify.py --recomputed
```

## Time units

The `period` and `tau_c` fields use oscilloscope sample ticks: one tick is 12.5 ps at 80 GSa/s. Multiply either time by 0.0125 to obtain nanoseconds; their ratio is the paper's dimensionless `tau_c/T_s`. The [raw-data guide](../raw_reproduction/README.md) lists the F1–F4 periods.
