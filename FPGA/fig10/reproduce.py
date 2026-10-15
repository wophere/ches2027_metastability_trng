from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw" / "fig10_fpga_acf_fft_raw"
def main() -> int:
    reports_root = RAW
    subprocess.run(
        [
            sys.executable,
            str(HERE / "scripts" / "process_raw.py"),
            "--reports-root",
            str(reports_root),
            "--bootstrap-draws",
            "20000",
        ],
        check=True,
    )
    subprocess.run([sys.executable, str(HERE / "scripts" / "plot_derived.py")], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
