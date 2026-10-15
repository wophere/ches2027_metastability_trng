#!/usr/bin/env python3
"""Reproduce Table 1 public row artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    subprocess.run(
        [sys.executable, str(HERE / "scripts" / "generate_table01.py")],
        cwd=HERE,
        check=True,
    )


if __name__ == "__main__":
    main()
