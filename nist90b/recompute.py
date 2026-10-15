"""Run the official SP800-90B tools on the published raw data.

Requires ea_iid and ea_non_iid built from the pinned upstream source.
No acquisition, synchronization, or network access is performed here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tools', type=Path, required=True, help='Directory containing ea_iid and ea_non_iid')
    p.add_argument('--jobs', type=int, default=4, help='OpenMP threads per assessment')
    args = p.parse_args()
    out = ROOT / 'nist90b/data/recomputed'
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / 'Spectre/fig08/scripts'))
    from compare_random_bits import load_dataset
    datasets = []
    for config, label in [(1, 'with_flicker'), (4, 'without_flicker')]:
        raw = ROOT / f'Spectre/fig08/data/raw/fig08_spectre_flicker_noise_raw/latch_value_config{config}.csv'
        bits = load_dataset(raw, label, .5, 1000, False, 'bits', 1000).bits
        datasets.append((label, bits, ['iid', 'non_iid']))
    captures = pd.read_csv(ROOT / 'ASIC/raw_reproduction/captures.csv')
    selected = captures[captures.temperature_c.eq(27)].sort_values(['voltage', 'frequency', 'abs_bias', 'count', 'file_name']).groupby(['voltage', 'frequency'], sort=False).head(1)
    for row in selected.itertuples():
        raw = ROOT / 'ASIC/raw_reproduction' / row.raw_path
        datasets.append((f'asic_27C_{row.voltage:.1f}V_F{row.frequency}', np.fromfile(raw, dtype=np.uint8) & 1, ['non_iid']))
    results = []
    for name, bits, tracks in datasets:
        bitfile = out / f'{name}.bin'
        bits.astype(np.uint8).tofile(bitfile)
        for track in tracks:
            exe = args.tools.resolve() / f'ea_{track}'
            command = [str(exe), '-i', '-a', '-v', bitfile.name, '1']
            print(name, track, flush=True)
            log_path = out / f'{name}_{track}.log'
            with log_path.open('w', encoding='utf-8') as handle:
                completed = subprocess.run(command, cwd=out, env={**os.environ, 'OMP_NUM_THREADS': str(args.jobs)}, text=True, stdout=handle, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f'{name}/{track} exited {completed.returncode}; inspect its log')
            log_text = log_path.read_text(encoding='utf-8')
            values = re.findall(r'H_original\s*[:=]\s*([0-9.eE+-]+)', log_text)
            if not values:
                raise ValueError(f'No entropy estimate in {log_path.name}')
            results.append(dict(dataset=name, track=track, samples=len(bits), input_sha256=hashlib.sha256(bitfile.read_bytes()).hexdigest(), executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), arguments=['-i','-a','-v',bitfile.name,'1'], H_original=float(values[-1]) if values else None, log=f'{name}_{track}.log'))
            (out / 'results.json').write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
