# Figure 14: ASIC split validation

Figure 14 tests model predictions on held-out captures, using separate acquisition splits for calibration and evaluation.

`python reproduce.py` in this directory redraws the archived result. From the artifact root, `python reproduce.py --asic-from-raw fig14` recomputes from the included captures and plots the new fits/predictions. Set A contains repeat counts 0–4; B contains 5–9. See `../raw_reproduction/README.md` for the explicit acquisition selection.
