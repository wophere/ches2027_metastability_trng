# Figure 12: three-temperature latent ACF

Figure 12 examines power-law correlation tails in the effective metastable state across the ASIC operating corners at three temperatures.

`python reproduce.py` in this directory redraws the paper reference from `data/derived/latent_acf_binned_points.csv` and `latent_acf_fit_summary.csv`. For the full raw-capture chain, run from the artifact root:

```sh
python reproduce.py --asic-from-raw fig12
```

See the [raw-data guide](../raw_reproduction/README.md) for input format, selection rules, threshold inversion, fit definitions, and checks. The bit-ACF comparison CSV documents the comparison with output-bit statistics.

`period_ticks` and `tau_geo` use the 80 GSa/s oscilloscope sample-tick scale (12.5 ps per tick). The plotted horizontal axis is raw-output lag in bit samples; `period_ticks` converts that lag to the fit's time scale.
