"""Check that ecr.website_metrics' stored score columns equal what the engine computes now.

    python scripts/validate_scores.py [--tickers FJAN,FFEB] [--atol 0]

Scores are computed exactly as scripts/store_engine_scores.py stores them (default settings, the
quarterly timing curve for quarterly products). A difference means the stored scores are stale: rerun
that script after loading new metrics or changing the score formula.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from backtesting.ecr_source import with_connection
from store_engine_scores import STORED, engine_scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tickers', help='Comma-separated tickers; default all')
    parser.add_argument('--atol', type=float, default=0., help='Absolute tolerance (stored DOUBLEs match exactly)')
    args = parser.parse_args(argv)
    from sqlalchemy import text

    def load(connection):
        stored = pd.read_sql(text('SELECT ticker, `date`, ' + ', '.join(STORED.values()) + ' FROM ecr.website_metrics'),
                             connection, coerce_float=True)
        return engine_scores(connection), stored
    calculated, stored = with_connection(load)
    if args.tickers:
        keep = [t.strip().upper() for t in args.tickers.split(',')]
        calculated = calculated.loc[calculated.ticker.isin(keep)]
    if calculated.empty:
        sys.exit('No matching rows.')
    for frame in (calculated, stored):
        frame['date'] = pd.to_datetime(frame.date)
    merged = calculated.merge(stored, on=['ticker', 'date'], suffixes=('', '_stored'))
    failed = False
    for column in STORED.values():
        a, b = merged[column].astype(float), merged[column + '_stored'].astype(float)
        # A value on one side only is a mismatch too; NaN would otherwise compare as within tolerance.
        bad = ((a - b).abs() > args.atol) | (a.isna() != b.isna())
        print(f'{column:<17} max error {np.nanmax(np.abs((a - b).to_numpy())) if (a - b).notna().any() else 0:.3g}  '
              f'rows that differ {int(bad.sum())}')
        if bad.any():
            failed = True
            worst = merged.loc[bad, 'ticker'].value_counts()
            print('    ' + ', '.join(f'{t} ({n})' for t, n in worst.head(10).items()))
    if failed:
        sys.exit('FAIL: stored scores are stale; run scripts/store_engine_scores.py')
    print(f'PASS: {len(merged):,} rows; stored scores equal the engine\'s')


if __name__ == '__main__':
    main()
