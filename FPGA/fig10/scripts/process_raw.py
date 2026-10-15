"""Compute all Fig. 10 derived CSV/JSON artifacts from staged raw captures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
FIG_ROOT = HERE.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    args = parser.parse_args()
    subprocess.run(
        [
            sys.executable,
            str(HERE / "plot_powerlaw_bias_independence.py"),
            "--reports-root", str(args.reports_root),
            "--output-root", str(FIG_ROOT / "data" / "derived"),
            "--bootstrap-draws", str(args.bootstrap_draws),
            "--skip-plot",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
