# Spectre / Table 1 artifact

Table 1 compares output bias, i.i.d. min-entropy, and SP 800-90B estimates for the Spectre sequences used in the flicker-noise comparison.

- `../fig08/data/derived/fig08_spectre_flicker_noise__stats.csv` supplies the recomputed Fig. 8 statistics for `p1`, absolute bias, and iid min-entropy.
- `data/derived/sp800_90b_archived_results.csv` archives the SP 800-90B iid and non-iid track outputs used by the table.
- `scripts/generate_table01.py` combines those inputs and writes both a full CSV and the LaTeX row fragments.

Run:

```powershell
python reproduce.py
```

or from the artifact root (also recomputes Fig. 8 first):

```powershell
python reproduce.py table01
```
