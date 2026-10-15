# Figure 10: power-law autocorrelation after bias calibration

This workflow measures temporal dependence in the calibrated FPGA output through autocorrelation tails, bootstrap intervals, and block-variance inflation.

1. `scripts/process_raw.py` computes ACF bins, bootstrap intervals, power-law fits, bias and variance-inflation metrics from the archived FPGA captures, including centered FFT ACF recomputation, writing the derived CSV/JSON files.
2. `scripts/plot_derived.py` reads only those derived CSV/JSON files and draws the figure.

`reproduce.py` reads the included files directly and runs both stages in order. The reference ACF CSVs support numerical verification. Every FFT ACF is recomputed from packed bits and checked at absolute tolerance 5e-12, then used for the fits. Fresh ACFs are saved under `data/recomputed/acf/`. Each raw file has 125,000 packed bytes, representing 1,000,000 bits decoded least-significant bit first. Ten captures are used at each of 10, 25, and 50 MHz. The fit window is lags 500–10,000, the block size is 4,096 bits, and the capture bootstrap uses 20,000 draws with seed 20260711.
