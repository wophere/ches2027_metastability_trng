# Figure 8: flicker-noise sequence statistics

The analysis compares output statistics from the Spectre traces to examine the effect of flicker noise on temporal correlation and block means.

1. `scripts/process_raw.py` quantizes the raw Spectre traces and writes the summary CSV plus compressed derived ACF/block-p1 arrays.
2. `scripts/plot_derived.py` reads only the compressed derived arrays and draws the comparison figure.

`reproduce.py` runs both stages in order.
