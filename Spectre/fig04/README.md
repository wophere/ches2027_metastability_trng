# Fig. 4 raw-to-derived pipeline

The analysis relates the effective input offset to output probability through a Gaussian-CDF fit. Two scripts process and plot the data:

1. `scripts/process_raw.py` reads the three raw Spectre CSV exports and writes `data/derived/fig04_p1_vs_delta_v__summary.csv`, including measured probabilities and Gaussian-CDF fit parameters.
2. `scripts/plot_derived.py` reads only that derived CSV and writes the figure.

`reproduce.py` runs both stages in order.
