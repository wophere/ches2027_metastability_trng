# Model figures: finite-block conditional min-entropy

Figures 6 and 7 connect sampling frequency, effective offset, flicker-noise strength, and block length to the model's conditional min-entropy estimates.

## Plot the paper results

Run from the repository root:

```sh
python reproduce.py fig06 fig07
```

These commands plot the reference Monte Carlo CSVs supplied with the paper. The generators below compute new estimates and save their own plots and metadata.

## Figure 6: sampling-frequency dependence

Figure 6 uses `N=64`, `mu=0`, `sigma_w=1`, `sigma_f=0.5`, `beta=0.2`, `tau_c=1e-9 s`, and 21 frequencies from 1 MHz to 1 GHz. The generator defaults to 10,000,000 samples per frequency, batches of 5,000, and seed 12345. Run from the repository root:

```sh
python Model/fig06/scripts/compute_block_entropy_vs_fs.py --out Model/fig06/data/recomputed.csv --plot Model/fig06/figure/recomputed.pdf --acf-plot Model/fig06/figure/recomputed_acf.pdf
```

## Figure 7: offset, flicker noise, and block length

Figure 7 fixes `fs=100 MHz`, `beta=0.2`, and `tau_c=1e-9 s`; it uses a 15-by-15 grid in `(r_mu,r_f)`, `N=16,64,256`, and 1,000,000 samples per grid point. The supplied generator seed is 20270624; each `r_f` slice uses `seed + 1009*i`. Batches contain 2,000 samples and four worker processes are used by default. Run from the repository root:

```sh
python Model/fig07/scripts/compute_hN_surface.py
```

The generator saves new estimates in `fig07/data/recomputed/` and creates a new plot. Options include `--out`, `--plot`, `--samples`, `--batch`, `--seed`, and `--workers`.

Generator runs write `.metadata.json` sidecars recording the seed, batch size, Python/NumPy versions, and covariance factorization.
