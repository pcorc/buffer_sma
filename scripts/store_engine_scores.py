"""Store the engine's component scores in ecr.website_metrics (dbb_score .. ecr_2_0).

    python scripts/store_engine_scores.py --dry-run    # compute and summarize; write nothing
    python scripts/store_engine_scores.py              # write in one transaction, then verify

Rerun after loading new metrics. Scores use the default score settings (config/ecr.json) and the
quarterly timing curve for quarterly products; rows the engine cannot score are stored as NULL.
The backtest never reads these columns: it always rescores with the run's own settings.
Needs an account with UPDATE (and, on MySQL, CREATE TEMPORARY TABLES) on ecr.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from backtesting.ecr_source import _engine, clear_cache, with_connection
from core.scoring import INPUTS, score_outcomes

STORED = {'dbb': 'dbb_score', 'buffer_integrity': 'buffer_integrity', 'cap_integrity': 'cap_integrity',
          'reset_proximity': 'reset_proximity', 'period_position': 'period_position', 'ecr': 'ecr_2_0'}
# Quarterly per the catalogue; funds without a catalogue row fall back to their metrics' series label.
SOURCE = (f'SELECT w.ticker, w.`date`, {", ".join("w." + c for c in INPUTS)}, '
          "COALESCE(c.outcome_type, CASE WHEN w.series = 'Quarterly' THEN 'Quarterly' ELSE 'Annual' END) AS outcome "
          'FROM ecr.website_metrics w LEFT JOIN ecr.fund_catalog c ON c.ticker = w.ticker')


def engine_scores(connection):
    """Stored-column frame keyed by ticker/date, exactly as the engine scores each row."""
    from sqlalchemy import text
    rows = pd.read_sql(text(SOURCE), connection, coerce_float=True)
    scores = score_outcomes(rows, rows.outcome.eq('Quarterly'))
    stored = scores[list(STORED)].rename(columns=STORED).astype(float)
    return pd.concat([rows[['ticker', 'date', 'outcome']], stored], axis=1)


def same(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


def write(connection, frame):
    from sqlalchemy import text
    columns = list(STORED.values())
    records = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
               for row in frame[['ticker', 'date', *columns]].to_dict('records')]
    if connection.dialect.name == 'mysql':
        # One joined UPDATE from a temporary table: one round trip per 5,000 rows, not per row.
        connection.execute(text('CREATE TEMPORARY TABLE engine_scores (`date` DATE NOT NULL, ticker VARCHAR(12) NOT NULL, '
                                + ', '.join(f'{c} DOUBLE NULL' for c in columns) + ', PRIMARY KEY (`date`, ticker))'))
        insert = text(f'INSERT INTO engine_scores (`date`, ticker, {", ".join(columns)}) '
                      f'VALUES (:date, :ticker, {", ".join(":" + c for c in columns)})')
        for start in range(0, len(records), 5000):
            connection.execute(insert, records[start:start + 5000])
        connection.execute(text('UPDATE ecr.website_metrics w JOIN engine_scores s ON w.`date` = s.`date` AND w.ticker = s.ticker '
                                'SET ' + ', '.join(f'w.{c} = s.{c}' for c in columns)))
        connection.execute(text('DROP TEMPORARY TABLE engine_scores'))
    else:
        connection.execute(text('UPDATE ecr.website_metrics SET ' + ', '.join(f'{c} = :{c}' for c in columns)
                                + ' WHERE `date` = :date AND ticker = :ticker'), records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true', help='Compute and summarize without writing.')
    args = parser.parse_args(argv)
    from sqlalchemy import text
    frame = with_connection(engine_scores)
    valid = frame.ecr_2_0.notna()
    print(f'{len(frame):,} rows: {int(valid.sum()):,} scored, {int((~valid).sum()):,} unscorable (stored as NULL); '
          f'{int(frame.outcome.eq("Quarterly").sum()):,} rows use the quarterly timing curve.')
    if (~valid).any():
        print('Unscorable rows by fund:', frame.loc[~valid].ticker.value_counts().head(12).to_dict())
    if args.dry_run:
        return frame
    with _engine().begin() as connection:  # commits only if the read-back below matches
        write(connection, frame)
        back = pd.read_sql(text('SELECT ticker, `date`, ' + ', '.join(STORED.values()) + ' FROM ecr.website_metrics'),
                           connection, coerce_float=True)
        back['date'] = pd.to_datetime(back.date)
        check = frame.assign(date=pd.to_datetime(frame.date)).merge(back, on=['ticker', 'date'], suffixes=('', '_db'))
        if len(check) != len(frame) or not all(same(check[c], check[c + '_db']) for c in STORED.values()):
            raise SystemExit('Read-back differs from the computed scores; rolled back.')
    clear_cache()
    print(f'Stored and verified {len(frame):,} rows (bit-identical read-back).')
    return frame


if __name__ == '__main__':
    main()
