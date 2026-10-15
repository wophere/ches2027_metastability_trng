"""Compare the complete raw ASIC analysis with immutable paper references."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FRESH = Path(__file__).resolve().parent / 'data/recomputed'


def main():
    checks = [
        ('latent_acf_fit_summary.csv', 'ASIC/fig12/data/derived/latent_acf_fit_summary.csv', ['temperature_label','voltage','frequency'], ['beta','intercept_log','r2_log'], 1e-10),
        ('three_temp_bias_min_params_long.csv', 'ASIC/table02/data/derived/three_temp_bias_min_params_long.csv', ['temperature_label','voltage','frequency'], ['beta','r_mu','r_f','tau_c'], 5e-4),
        ('fig14_platform_split_parameters.csv', 'ASIC/fig14/data/derived/fig14_platform_split_parameters.csv', ['split','temperature_label','voltage','frequency'], ['beta','r_f','tau_c'], 5e-4),
        ('fig14_platform_split_heldout_block.csv', 'ASIC/fig14/data/derived/fig14_platform_split_heldout_block.csv', ['temperature_label','voltage','frequency','block_size'], ['excess_var','excess_var_prediction'], 1e-5),
    ]
    report = []
    for name, reference, keys, columns, rtol in checks:
        fresh = pd.read_csv(FRESH / name).sort_values(keys).reset_index(drop=True)
        archived = pd.read_csv(ROOT / reference).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(fresh[keys], archived[keys], check_dtype=False)
        if 'quality_status' in archived:
            assert fresh.quality_status.equals(archived.quality_status), f'Quality classification changed: {name}'
        if 'file_name' in archived:
            assert fresh.file_name.equals(archived.file_name), f'Capture selection changed: {name}'
        for column in columns:
            np.testing.assert_allclose(fresh[column], archived[column], rtol=rtol, atol=1e-10, err_msg=f'{name}/{column}')
            report.append(dict(file=name, column=column, rows=len(fresh), max_absolute_difference=float(np.max(np.abs(fresh[column]-archived[column]))), relative_tolerance=rtol, absolute_tolerance=1e-10))
        print(f'PASS {name}: {len(fresh)} rows', flush=True)
    (FRESH / 'verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
