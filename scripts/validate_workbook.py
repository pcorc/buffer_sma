"""Independent parity check against the supplied workbook's stored Excel results."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from backtesting.data_pipeline import load_benchmarks, load_cycles, load_metrics
from backtesting.engine import DIAGNOSTIC_COLUMNS, prepare_data, simulate
from core.scoring import COMPONENTS, score_funds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workbook', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw = pd.read_excel(args.workbook, sheet_name='Website Metrics', usecols='A:T,AM:AR')
    calculated = score_funds(raw)
    expected = raw.iloc[:, 20:26].to_numpy(dtype=float)
    np.testing.assert_allclose(calculated[list(COMPONENTS)], expected, atol=1e-12, rtol=0)
    error = float(np.max(np.abs(calculated[list(COMPONENTS)].to_numpy() - expected)))
    data = prepare_data(load_metrics(args.workbook), load_benchmarks(root / 'input_data/benchmark_ts.csv'),
                        load_cycles(root / 'input_data/roll_dates.csv'))
    result = simulate(data)
    snapshots = pd.read_excel(args.workbook, sheet_name='ECR 2.0 Diagnostics', header=3).rename(columns={'ECR 2.0': 'ECR'})
    snapshots = snapshots.loc[pd.to_numeric(snapshots['Switch #'], errors='coerce').notna(), DIAGNOSTIC_COLUMNS]
    pd.testing.assert_frame_equal(result['diagnostics'][DIAGNOSTIC_COLUMNS].reset_index(drop=True),
                                  snapshots.reset_index(drop=True), check_dtype=False, rtol=0, atol=1e-12)
    print(f'PASS: {len(raw)} formula rows; max component error {error:.3g}')
    print(f'PASS: {len(snapshots)} generated entry/switch events and every diagnostic value')


if __name__ == '__main__':
    main()
