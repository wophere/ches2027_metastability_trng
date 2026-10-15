# Figure 13: ES1/ES2 cross-correlation

Run `python reproduce.py` here, or `python reproduce.py fig13` at the artifact root.

`data/raw/` contains the exact ten synchronized 1,000,000-symbol records used for this figure. Each ES stream is **one unsigned byte per bit**, restricted to 0 or 1. The paired stream stores `(ES1 << 1) | ES2`; the script verifies every pair against the two individual streams. Metadata retains the acquisition settings and capture order, with relative filenames for portable analysis.

Each record is divided into 1,000 blocks of 1,000 bits. The left panel plots the resulting block means. The right panel is the zero-lag Pearson correlation between the two block-mean sequences for each record. Metadata additionally record the bit-level phi coefficient. The mean block-level correlation is `-0.07044158999524762`, displayed as `-0.070` in the paper.

The script also writes correlations for block lags -25 through +25, using only overlapping blocks at each lag. Data appear in `data/derived/`, and the figure in `figure/`. The experiment characterizes cross-source dependence through the measured block-level correlations.
