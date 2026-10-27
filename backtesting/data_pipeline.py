"""Read raw metrics explicitly: Excel decimals or the original percentage CSV."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from core.scoring import INPUTS

CSV_COLUMNS = {
    'Fund': 'ticker', 'Date': 'date', 'Fund Value (USD)': 'fund_value',
    'Reference Asset Return (%)': 'reference_asset_return',
    'Remaining Outcome Days': 'remaining_outcome_days',
    'Remaining Cap Net (%)': 'remaining_cap_net',
    'Remaining Buffer Net (%)': 'remaining_buffer_net',
    'Downside Before Buffer Net (%)': 'downside_before_buffer_net',
    'Original Cap Net (%)': 'fund_cap_net', 'Original Buffer Net (%)': 'buffer_net',
}
PERCENT_COLUMNS = tuple(name for name in INPUTS if name != 'remaining_outcome_days')


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_metrics(path) -> pd.DataFrame:
    """Only Website Metrics is read from workbooks; model inputs are never formula caches."""
    path = Path(path)
    if path.suffix.lower() == '.xlsx':
        frame = pd.read_excel(path, sheet_name='Website Metrics', usecols='A:T')
    else:
        frame = pd.read_csv(path)
    if 'Fund' in frame.columns:
        frame = frame.rename(columns=CSV_COLUMNS)
        for name in PERCENT_COLUMNS:
            frame[name] = pd.to_numeric(frame[name], errors='coerce') / 100
    required = ('ticker', 'date', 'fund_value', *INPUTS)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing metric columns: {sorted(missing)}')
    frame = frame.loc[:, required].copy()
    frame['date'] = pd.to_datetime(frame.date, errors='raise').dt.normalize()
    if frame.date.isna().any() or frame.ticker.isna().any():
        raise ValueError('Missing ticker/date')
    if frame.duplicated(['ticker', 'date']).any():
        raise ValueError('Duplicate ticker/date observations')
    for name in ('fund_value', *INPUTS):
        frame[name] = pd.to_numeric(frame[name], errors='coerce')
    if not np.isfinite(frame.fund_value).all() or (frame.fund_value <= 0).any():
        raise ValueError('Fund NAV must be finite and positive')
    return frame.sort_values(['date', 'ticker']).reset_index(drop=True)


def load_benchmarks(path):
    frame = pd.read_csv(path)
    frame['Date'] = pd.to_datetime(frame.Date).dt.normalize()
    if frame.Date.duplicated().any():
        raise ValueError('Duplicate benchmark dates')
    return frame.set_index('Date')[['SPY', 'BUFR']].sort_index().apply(pd.to_numeric)


def load_cycles(path):
    """The supplied monthly calendar includes holiday-adjusted option-cycle dates."""
    cycles = pd.to_datetime(pd.read_csv(path)['monthly'].dropna()).dt.normalize()
    if cycles.duplicated().any():
        raise ValueError('Duplicate monthly cycle dates')
    return pd.DatetimeIndex(cycles.sort_values())
