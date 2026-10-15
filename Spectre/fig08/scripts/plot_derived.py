"""Draw Fig. 8 exclusively from derived ACF and block-p1 arrays."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FIG_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from compare_random_bits import BitDataset, plot_comparison


def dataset(label: str, acf: np.ndarray, p1_blocks: np.ndarray) -> BitDataset:
    return BitDataset(label, Path("derived"), np.empty(0), np.empty(0), np.empty(0, dtype=np.uint8), p1_blocks, acf)


def main() -> None:
    arrays = np.load(FIG_ROOT / "data" / "derived" / "fig08_spectre_flicker_noise__series.npz")
    datasets = [
        dataset("Flicker enabled", arrays["with_acf"], arrays["with_p1_blocks"]),
        dataset("Flicker suppressed", arrays["without_acf"], arrays["without_p1_blocks"]),
    ]
    plot_comparison(
        datasets,
        FIG_ROOT / "figure" / "fig08_spectre_flicker_noise_effect.png",
        FIG_ROOT / "figure" / "fig08_spectre_flicker_noise_effect.pdf",
        1000, 40, False, False, 0.0, 1000.0, -0.02, 0.02, 0.3, 0.7, 300, "bits",
    )


if __name__ == "__main__":
    main()
