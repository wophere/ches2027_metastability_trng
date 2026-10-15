"""Recompute ASIC Fig. 12, Table 2 parameters, and Fig. 14 from captures.

Outputs are separate from the immutable paper reference CSVs.
"""
from pathlib import Path
import argparse
import hashlib
import numpy as np
import pandas as pd
from scipy.special import ndtri
import analysis as a

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-statistics', action='store_true', help='Reuse previously computed per-capture statistics')
    args = parser.parse_args()
    out = ROOT / 'data/recomputed'
    out.mkdir(parents=True, exist_ok=True)
    captures = pd.read_csv(ROOT / 'captures.csv')
    selected = captures.sort_values(['temperature_c', 'voltage', 'frequency', 'abs_bias', 'count', 'file_name']).groupby(['temperature_c', 'voltage', 'frequency'], sort=False).head(1)
    selected.to_csv(out / 'selected_traces.csv', index=False)
    inverter = a.SignGaussianInverter(601)
    af, bf, summaries, points = [], [], [], []
    for index, row in enumerate(captures.to_dict('records'), 1):
        path = ROOT / row['raw_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Checksum mismatch: {path.name}')
        bits = a.read_bits(path)
        if len(bits) != row['num_samples']:
            raise ValueError(f'Sample count mismatch: {path.name}')
        p1 = float(bits.mean())
        meta = {**row, 'p1_csv': row['p1'], 'p1_measured': p1, 'file_path': row['raw_path'], 'trace_uid': row['raw_path']}
        if not args.reuse_statistics:
            af.append(a.latent_acf_points(bits, p1, row['period'], inverter, meta))
            bf.append(a.within_trace_block_variance(bits, p1, row['period'], meta))
        if row['raw_path'] in set(selected.raw_path):
            rho = a.bit_acf_all_lags(bits, row['p1'], a.LAG_MAX)
            latent = a.invert_gaussian_threshold(rho, row['p1'])
            fit, bins = a.fit_binned_latent_acf(np.arange(a.LAG_MAX + 1), latent, row['period'], a.fig12_bins())
            fm = {**meta, 'temperature_C': row['temperature_c'], 'period_ticks': row['period']}
            summaries.append({**fm, **fit})
            points.extend({**fm, **b} for b in bins)
        print(f'Capture {index}/{len(captures)}', flush=True)
    pd.DataFrame(summaries).to_csv(out / 'latent_acf_fit_summary.csv', index=False)
    pd.DataFrame(points).to_csv(out / 'latent_acf_binned_points.csv', index=False)
    if args.reuse_statistics:
        acf = pd.read_csv(out / 'capture_acf.csv')
        block = pd.read_csv(out / 'capture_block.csv')
    else:
        acf, block = pd.concat(af, ignore_index=True), pd.concat(bf, ignore_index=True)
        acf.to_csv(out / 'capture_acf.csv', index=False)
        block.to_csv(out / 'capture_block.csv', index=False)
    params = []
    for row in selected.to_dict('records'):
        aa = acf[acf.raw_path.eq(row['raw_path'])]
        bb = block[block.raw_path.eq(row['raw_path'])]
        fit, _, _ = a.fit_selected_trace(aa, bb, row['period'])
        ru = float(ndtri(float(aa.iloc[0].p1_measured)))
        rm = ru * np.sqrt(1 + fit['r_f'] ** 2)
        params.append({**aa.iloc[0].to_dict(), **fit, 'r_u': ru, 'r_mu': rm, 'abs_r_mu': abs(rm)})
        print('Selected fit:', row['case_label'], row['temperature_label'], flush=True)
    pd.DataFrame(params).to_csv(out / 'three_temp_bias_min_params_long.csv', index=False)
    acf = acf[acf.split_eligible]
    block = block[block.split_eligible]
    aa, ab = a.aggregate_acf(a.half(acf, 'A')), a.aggregate_acf(a.half(acf, 'B'))
    ba, bb = a.aggregate_block(a.half(block, 'A')), a.aggregate_block(a.half(block, 'B'))
    pa, _ = a.fit_half(aa, ba, 'A')
    pb, _ = a.fit_half(ab, bb, 'B')
    pa.to_csv(out / 'split_A_params.csv', index=False)
    pb.to_csv(out / 'split_B_params.csv', index=False)
    pd.concat([pa, pb], ignore_index=True).to_csv(out / 'fig14_platform_split_parameters.csv', index=False)
    a.held_out_predictions(pa, ab, bb).to_csv(out / 'fig14_platform_split_heldout_block.csv', index=False)


if __name__ == '__main__':
    main()
