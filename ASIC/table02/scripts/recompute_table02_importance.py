#!/usr/bin/env python3
"""Recompute Table 2 model entropy values from archived model parameters.

This is the heavy, public recomputation script.  The default `python reproduce.py table02`
target intentionally uses the archived CSV and regenerates the LaTeX rows; run
this script only when you want to refresh the importance-sampling estimates.
"""

from __future__ import annotations

import argparse
import math
import hashlib
import json
import platform
import scipy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from importance_sampling import BLOCK_LENGTHS, estimate


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived"
PARAMS = DATA / "three_temp_bias_min_params_long.csv"
OUTPUT = ROOT / "data/recomputed/table02_27c_bias_min.csv"
ROWS = ROOT / "data/recomputed/table02_27c_bias_min_rows.tex"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, default=PARAMS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--rows", type=Path, default=ROWS)
    parser.add_argument("--is-samples", type=int, default=100_000)
    parser.add_argument("--is-batch-pairs", type=int, default=2_000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_rows(table: pd.DataFrame, rows_path: Path) -> None:
    tex = []
    for row in table.sort_values(["voltage", "frequency"]).itertuples(index=False):
        tex.append(
            f"    \\({row.voltage:.1f}\\,\\mathrm{{V}}\\)/F{int(row.frequency)} & "
            f"{row.bias:+.3f} & {row.h_iid:.3f} & {row.h90b:.3f} & "
            f"{row.h_model_N16:.3f} & {row.h_model_N64:.3f} & {row.h_model_N256:.3f} \\\\"
        )
    rows_path.write_text("\n".join(tex) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.params)
    source = source[source["temperature_c"].eq(27)].copy()
    source = source.sort_values(["voltage", "frequency"]).reset_index(drop=True)
    if len(source) != 8 or not source["reportable"].all():
        raise ValueError("Expected eight reportable 27 C minimum-bias rows")
    metadata = {
        'parameter_sha256': hashlib.sha256(args.params.read_bytes()).hexdigest(),
        'samples': args.is_samples, 'batch_pairs': args.is_batch_pairs,
        'seed_formula': '20260713 + round(voltage*1000)*100 + frequency',
        'shifts': [-6, -4, -2, 0, 2, 4, 6], 'rng': 'numpy.random.PCG64',
        'python': platform.python_version(), 'numpy': np.__version__,
        'scipy': scipy.__version__,
    }
    metadata_path = args.output.with_suffix('.metadata.json')
    if args.resume and args.output.exists():
        if not metadata_path.exists() or json.loads(metadata_path.read_text(encoding='utf-8')) != metadata:
            raise ValueError('Cannot resume: input, settings, or environment metadata differ (or are missing)')
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')

    rows: list[dict[str, Any]] = []
    completed: set[tuple[float, int]] = set()
    if args.resume and args.output.exists():
        existing = pd.read_csv(args.output)
        rows.extend(existing.to_dict("records"))
        completed = {(round(float(row["voltage"]), 6), int(row["frequency"])) for row in rows}

    shifts = np.asarray([-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0])
    for item in source.itertuples(index=False):
        key = (round(float(item.voltage), 6), int(item.frequency))
        if key in completed:
            print(f"reuse completed: {item.case_label}", flush=True)
            continue

        p1 = float(item.p1_measured)
        result = estimate(
            {
                "beta": float(item.beta),
                "r_f": float(item.r_f),
                "tau_c": float(item.tau_c),
                "period": float(item.period),
                "r_mu_heldout": float(item.r_mu),
            },
            args.is_samples,
            args.is_batch_pairs,
            shifts,
            20260713 + int(round(float(item.voltage) * 1000)) * 100 + int(item.frequency),
        )
        for n in BLOCK_LENGTHS:
            result[f"h_model_N{n}"] = result[f"h_model_N{n}_is"]
            result[f"h_model_N{n}_mc_se"] = result[f"h_model_N{n}_is_se"]
        result["mc_samples"] = result["is_output_samples"]
        result["mc_antithetic_pairs"] = result["is_antithetic_pairs"]
        result["mc_sampling_method"] = "antithetic_symmetric_mixture_importance_fft"
        rows.append(
            {
                "temperature_c": 27,
                "voltage": float(item.voltage),
                "frequency": int(item.frequency),
                "selected_count": int(item.count),
                "selected_file": str(item.file_name),
                "p1": p1,
                "bias": float(item.bias),
                "abs_bias": abs(float(item.bias)),
                "h_iid": -math.log2(max(p1, 1.0 - p1)),
                "h90b": float(item.h_original),
                "r_mu": float(item.r_mu),
                "r_f": float(item.r_f),
                "beta": float(item.beta),
                "tau_c": float(item.tau_c),
                "tau_c_over_Ts": float(item.tau_c_over_Ts),
                "fit_quality_status": str(item.quality_status),
                **result,
            }
        )
        pd.DataFrame(rows).sort_values(["voltage", "frequency"]).to_csv(args.output, index=False)
        print(f"entropy complete: {item.case_label}, count={int(item.count)}", flush=True)

    table = pd.DataFrame(rows).sort_values(["voltage", "frequency"]).reset_index(drop=True)
    table.to_csv(args.output, index=False)
    write_rows(table, args.rows)
    print(f"wrote {args.output}")
    print(f"wrote {args.rows}")


if __name__ == "__main__":
    main()
