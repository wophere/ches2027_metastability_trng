"""Draw Fig. 10 exclusively from its derived CSV/JSON artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
FIG_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from plot_powerlaw_bias_independence import FREQS, plot_figure


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key not in {"freq", "case", "display_lag_window", "fit_lag_window"}:
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
    return rows


def main() -> None:
    derived = FIG_ROOT / "data" / "derived"
    prefix = derived / "fpga_powerlaw_bias_independence"
    summary_document = json.loads(prefix.with_name(prefix.name + "_summary.json").read_text(encoding="utf-8"))
    summaries = {row["freq"]: row for row in summary_document["summary"]}
    acf_rows = read_rows(prefix.with_name(prefix.name + "_acf.csv"))
    run_rows = read_rows(prefix.with_name(prefix.name + "_run_metrics.csv"))
    results = {
        freq: {
            "summary": summaries[freq],
            "acf_rows": [row for row in acf_rows if row["freq"] == freq],
            "run_rows": [row for row in run_rows if row["freq"] == freq],
        }
        for freq in FREQS
    }
    plot_figure(results, FIG_ROOT / "figure" / "fig10_fpga_powerlaw_bias_independence")


if __name__ == "__main__":
    main()
