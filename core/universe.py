"""Fund metadata and filters, separate from historical scoring inputs."""
from pathlib import Path
import pandas as pd

from config.settings import MONTHS

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / 'input_data/fund_catalog.csv'
EXTRA_METRICS = ROOT / 'input_data/fund_metrics'
GROUPS = ['F series', 'D series', 'G series', 'UCITS', 'Quarterly structures', 'Custom universe']
STRATEGIES = ['Buffer', 'Buffer & Digital Return', 'Buffer and Premium Income', 'Conservative Buffer',
              'Deep Buffer', 'Dual Directional Buffer', 'Enhance & Moderate Buffer', 'Max Buffer', 'Moderate Buffer']
BUFFER_BANDS = ['All buffers', 'Up to 10%', 'Above 10% to 20%', 'Above 20%', 'Deferred protection']
COLUMNS = ['ticker','series','group','outcome_type','outcome_month','strategy','buffer_start',
           'buffer_end','reference_asset','currency']


def load_catalog(path=CATALOG_PATH):
    catalog = pd.read_csv(path, keep_default_na=False)
    if set(COLUMNS) - set(catalog):
        raise ValueError(f'Fund catalogue needs columns: {COLUMNS}')
    if catalog.ticker.duplicated().any() or catalog.ticker.eq('').any():
        raise ValueError('Fund catalogue tickers must be nonempty and unique')
    if not catalog.ticker.eq(catalog.ticker.str.strip().str.upper()).all():
        raise ValueError('Catalogue tickers must be uppercase without whitespace')
    if not catalog.outcome_type.isin(['Annual','Quarterly']).all():
        raise ValueError('Outcome type must be Annual or Quarterly')
    if not catalog.outcome_month.isin([*MONTHS,'Quarterly']).all():
        raise ValueError('Invalid outcome month in catalogue')
    for col in ('buffer_start','buffer_end'):
        catalog[col] = pd.to_numeric(catalog[col], errors='raise')
    if not ((catalog.buffer_start >= 0) & (catalog.buffer_end > catalog.buffer_start) & (catalog.buffer_end <= 1)).all():
        raise ValueError('Buffer depths must be decimal fractions with 0 <= start < end <= 1')
    return catalog[COLUMNS]


def filter_catalog(catalog, group='F series', outcome='Annual', series=(), strategies=(),
                   references=(), months=(), buffer_band='All buffers'):
    frame = catalog.loc[catalog.outcome_type == outcome].copy()
    if group.endswith(' series'):
        frame = frame.loc[frame.series == group.split()[0]]
    elif group == 'UCITS':
        frame = frame.loc[frame.group == 'UCITS']
    elif group == 'Quarterly structures':
        frame = frame.loc[frame.outcome_type == 'Quarterly']
    elif group != 'Custom universe':
        raise ValueError('Unknown universe group')
    for column, values in [('series',series),('strategy',strategies),('reference_asset',references),('outcome_month',months)]:
        if values:
            frame = frame.loc[frame[column].isin(values)]
    if buffer_band == 'Up to 10%':
        frame = frame.loc[(frame.buffer_start == 0) & (frame.buffer_end <= .10)]
    elif buffer_band == 'Above 10% to 20%':
        frame = frame.loc[(frame.buffer_start == 0) & (frame.buffer_end > .10) & (frame.buffer_end <= .20)]
    elif buffer_band == 'Above 20%':
        frame = frame.loc[(frame.buffer_start == 0) & (frame.buffer_end > .20)]
    elif buffer_band == 'Deferred protection':
        frame = frame.loc[frame.buffer_start > 0]
    elif buffer_band != 'All buffers':
        raise ValueError('Unknown buffer category')
    return frame


def validate_universe(catalog, tickers, available):
    if not tickers:
        raise ValueError('Choose at least one fund with available history.')
    missing = set(tickers) - set(available)
    if missing:
        raise ValueError(f'Historical data is not loaded for: {", ".join(sorted(missing))}.')
    selected = catalog.loc[catalog.ticker.isin(tickers)]
    if len(selected) != len(tickers):
        raise ValueError('Every selected fund needs a catalogue record.')
    if selected.outcome_type.nunique() != 1:
        raise ValueError('Keep annual and quarterly outcome products in separate runs.')
    if set(selected.currency) != {'USD'}:
        raise ValueError('Current benchmarks use USD. Currency conversion is needed before mixing other NAV currencies.')
    return selected.copy()
