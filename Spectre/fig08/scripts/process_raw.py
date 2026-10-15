"""Quantize raw Spectre traces and archive the arrays used by Fig. 8."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FIG_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from compare_random_bits import load_dataset, write_stats_csv


def main() -> None:
    raw = FIG_ROOT / "data" / "raw" / "fig08_spectre_flicker_noise_raw"
    derived = FIG_ROOT / "data" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    datasets = [
        load_dataset(raw / "latch_value_config1.csv", "With flicker noise", 0.5, 1000, False, "bits", 1000),
        load_dataset(raw / "latch_value_config4.csv", "Without flicker noise", 0.5, 1000, False, "bits", 1000),
    ]
    write_stats_csv(derived / "fig08_spectre_flicker_noise__stats.csv", datasets)
    np.savez_compressed(
        derived / "fig08_spectre_flicker_noise__series.npz",
        with_acf=datasets[0].acf,
        with_p1_blocks=datasets[0].p1_blocks,
        without_acf=datasets[1].acf,
        without_p1_blocks=datasets[1].p1_blocks,
    )


if __name__ == "__main__":
    main()
