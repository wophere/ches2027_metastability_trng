#!/usr/bin/env python3
"""Generate the Table 1 row fragments from public artifact inputs.

The block-level p1 values come from Fig. 8's public derived CSV.  The SP
800-90B track results are archived here because the verifier output itself is
not part of the public artifact chain.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived"
BLOCK_STATS = ROOT.parent / "fig08/data/derived/fig08_spectre_flicker_noise__stats.csv"
NIST_RESULTS = DATA / "sp800_90b_archived_results.csv"
OUTPUT_CSV = DATA / "table01_flicker_90b_tracks.csv"
OUTPUT_TEX = DATA / "table01_flicker_90b_tracks_rows.tex"

ORDER = ["With flicker noise", "Without flicker noise"]
LABELS = {
    "With flicker noise": "Flicker enabled",
    "Without flicker noise": "Flicker suppressed",
}


def read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def h_iid_from_p1(p1: float) -> float:
    return -math.log2(max(p1, 1.0 - p1))


def main() -> None:
    stats = read_csv_by_key(BLOCK_STATS, "label")
    nist = read_csv_by_key(NIST_RESULTS, "condition")

    rows: list[dict[str, str]] = []
    tex_rows: list[str] = []
    for condition in ORDER:
        if condition not in stats:
            raise KeyError(f"Missing block statistics for {condition!r} in {BLOCK_STATS}")
        if condition not in nist:
            raise KeyError(f"Missing SP 800-90B result for {condition!r} in {NIST_RESULTS}")

        p1 = float(stats[condition]["overall_p1"])
        abs_bias = abs(p1 - 0.5)
        h_iid = h_iid_from_p1(p1)
        iid_90b = float(nist[condition]["iid_track_entropy"])
        iid_status = nist[condition]["iid_track_status"]
        non_iid_90b = float(nist[condition]["non_iid_track_entropy"])
        label = LABELS[condition]

        rows.append(
            {
                "condition": label,
                "p1": f"{p1:.6f}",
                "abs_bias": f"{abs_bias:.6f}",
                "h_iid": f"{h_iid:.6f}",
                "sp800_90b_iid_track": f"{iid_90b:.6f}",
                "sp800_90b_iid_track_status": iid_status,
                "sp800_90b_non_iid_track": f"{non_iid_90b:.6f}",
            }
        )
        tex_rows.append(
            f"    {label} & {p1:.6f} & {abs_bias:.6f} & {h_iid:.6f} & "
            f"{iid_90b:.6f} ({iid_status}) & {non_iid_90b:.6f} \\\\"
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT_TEX.write_text("\n".join(tex_rows) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_CSV}")
    print(f"wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
