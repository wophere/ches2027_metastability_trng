"""Check NIST recomputation against the two paper tables and complete logs."""
import argparse
import json
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'nist90b/data/recomputed/results.json')
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding='utf-8'))
    table1 = pd.read_csv(ROOT / 'Spectre/table01/data/derived/sp800_90b_archived_results.csv')
    table2 = pd.read_csv(ROOT / 'ASIC/table02/data/derived/table02_27c_bias_min.csv')
    expected = {}
    for name, condition in [('with_flicker', 'With flicker noise'), ('without_flicker', 'Without flicker noise')]:
        row = table1[table1.condition.eq(condition)].iloc[0]
        for track in ['iid', 'non_iid']:
            expected[(name, track)] = float(row[f'{track}_track_entropy'])
    for row in table2.itertuples():
        expected[(f'asic_27C_{row.voltage:.1f}V_F{row.frequency}', 'non_iid')] = row.h90b
    observed = {(row['dataset'], row['track']) for row in results}
    assert observed == set(expected) and len(results) == len(expected), 'Missing or duplicate assessments'
    for row in results:
        key = row['dataset'], row['track']
        log = (args.results.parent / row['log']).read_text(encoding='utf-8')
        values = re.findall(r'H_original\s*[:=]\s*([0-9.eE+-]+)', log)
        assert values and float(values[-1]) == row['H_original'], f'Log/JSON mismatch: {key}'
        assert abs(row['H_original'] - expected[key]) <= 5e-7, f'Entropy mismatch: {key}'
        assert row['input_sha256'] in log, f'Input hash missing: {key}'
        if row['track'] == 'iid':
            for test in ('chi square tests', 'length of longest repeated substring test', 'IID permutation tests'):
                assert re.search(r'\*\* (Passed|Failed) ' + re.escape(test), log), f'Missing IID decision: {key}/{test}'
            assert ('** Failed' not in log) == (row['dataset'] == 'without_flicker'), f'IID decision mismatch: {key}'
        if row['samples'] < 1000000:
            print(f'Sample-count limitation: {row["dataset"]}: {row["samples"]} symbols; no padding applied.')
    print('PASS: 12 entropy estimates, both IID decisions, and input hashes agree with the recorded logs and paper tables.')


if __name__ == '__main__':
    main()
