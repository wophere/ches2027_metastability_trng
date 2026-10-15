#!/usr/bin/env python3
"""Reproduce Table 2 public row artifacts from archived CSV output."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    subprocess.run(
        [sys.executable, str(HERE / "scripts" / "generate_table02.py")],
        cwd=HERE,
        check=True,
    )


if __name__ == "__main__":
    main()
