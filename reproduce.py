"""Run the portable paper artifact without Make, hardware, or network access."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
TARGETS = {
    "fig04": "Spectre/fig04/reproduce.py",
    "fig05": "Spectre/fig05/reproduce.py",
    "fig06": "Model/fig06/reproduce.py",
    "fig07": "Model/fig07/reproduce.py",
    "fig08": "Spectre/fig08/reproduce.py",
    "fig10": "FPGA/fig10/reproduce.py",
    "fig12": "ASIC/fig12/reproduce.py",
    "fig13": "ASIC/fig13/reproduce.py",
    "fig14": "ASIC/fig14/reproduce.py",
    "table01": "Spectre/table01/reproduce.py",
    "table02": "ASIC/table02/reproduce.py",
    "sensitivity": "ASIC/sensitivity/reproduce.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", help="Default: all. Names: " + ", ".join(TARGETS))
    parser.add_argument('--asic-from-raw', action='store_true', help='First recompute ASIC parameters and plot Figures 12/14 using fresh results (slow)')
    args = parser.parse_args()
    selected = args.targets or list(TARGETS)
    unknown = set(selected) - TARGETS.keys()
    if unknown:
        parser.error("Unknown targets: " + ", ".join(sorted(unknown)))
    # Table 1 uses freshly derived Figure 8 statistics.
    if "table01" in selected and "fig08" not in selected:
        selected.insert(selected.index("table01"), "fig08")
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    start = time.monotonic()
    if args.asic_from_raw:
        subprocess.run([sys.executable, str(ROOT / 'ASIC/raw_reproduction/recompute.py')], cwd=ROOT, env=env, check=True)
        env['ARTIFACT_ASIC_RECOMPUTED'] = '1'
    for name in selected:
        print(f"Reproducing {name}", flush=True)
        subprocess.run([sys.executable, str(ROOT / TARGETS[name])], cwd=ROOT, env=env, check=True)
    print(f"Completed {len(selected)} targets in {time.monotonic() - start:.1f} seconds.")


if __name__ == "__main__":
    main()
