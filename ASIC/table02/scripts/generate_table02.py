#!/usr/bin/env python3
"""Generate Table 2 row fragments from the archived public CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived"
INPUT_CSV = DATA / "table02_27c_bias_min.csv"
OUTPUT_TEX = DATA / "table02_27c_bias_min_rows.tex"


def main() -> None:
    table = pd.read_csv(INPUT_CSV).sort_values(["voltage", "frequency"]).reset_index(drop=True)
    tex = []
    for row in table.itertuples(index=False):
        tex.append(
            f"    \\({row.voltage:.1f}\\,\\mathrm{{V}}\\)/F{int(row.frequency)} & "
            f"{row.bias:+.3f} & {row.h_iid:.3f} & {row.h90b:.3f} & "
            f"{row.h_model_N16:.3f} & {row.h_model_N64:.3f} & {row.h_model_N256:.3f} \\\\"
        )
    OUTPUT_TEX.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
