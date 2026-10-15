from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> int:
    subprocess.run(
        [sys.executable, str(HERE / "scripts" / "plot_hN_heatmap_ru_rf.py")],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
