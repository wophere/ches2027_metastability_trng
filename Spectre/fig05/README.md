# Figure 5: metastability resolution time

The analysis extracts threshold-crossing times from the exported waveform and fits the resolution-time relationship shown in the paper.

1. `scripts/process_raw.py` reads `data/raw/VA.txt`, rebases the exported crossings, and writes the derived sample and fit-parameter CSV files.
2. `scripts/plot_derived.py` reads only those derived CSV files and draws the resolution-time panels.

`reproduce.py` runs both stages in order.
