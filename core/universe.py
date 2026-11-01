"""Fund metadata (ecr.fund_catalog) and filters, separate from historical scoring inputs."""
import numpy as np
import pandas as pd

from config.settings import MONTHS

# Groups that cut across families; every catalogue family ('F series') and strategy is a group too.
SPECIAL_GROUPS = ['Quarterly structures', 'UCITS']
STRATEGIES = ['Buffer', 'Buffer & Digital Return', 'Buffer and Premium Income', 'Conservative Buffer',
              'Deep Buffer', 'Dual Directional Buffer', 'Enhance & Moderate Buffer', 'Max Buffer', 'Moderate Buffer']
BUFFER_BANDS = ['All buffers', 'Up to 10%', 'Above 10% to 20%', 'Above 20%', 'Deferred protection']
COLUMNS = ['ticker','series','group','outcome_type','outcome_month','strategy','buffer_start',
           'buffer_end','reference_asset','currency']


def load_catalog():
    """The validated ecr.fund_catalog, from the same database snapshot as the loaded metrics."""
    from backtesting.ecr_source import load_bundle
    return validate_catalog(load_bundle().catalog.copy())


def validate_catalog(catalog):
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


def group_options(catalog):
    """Selectable groups: each product family ('D series'), each strategy ('Max Buffer'), then the special groups."""
    families = [f'{s} series' for s in sorted(set(catalog.series) - {''})]
    special = [g for g in SPECIAL_GROUPS if g != 'UCITS' or (catalog.group == 'UCITS').any()]
    return families + sorted(set(catalog.strategy) - {''}) + special


def group_mask(catalog, group):
    if group == 'Custom universe':
        return pd.Series(True, index=catalog.index)
    if group.endswith(' series'):
        return catalog.series == group[:-len(' series')]
    if group == 'UCITS':
        return catalog.group == 'UCITS'
    if group == 'Quarterly structures':
        return catalog.outcome_type == 'Quarterly'
    if group in set(STRATEGIES) | set(catalog.strategy):
        return catalog.strategy == group
    raise ValueError(f'Unknown universe group: {group}')


def groups_mask(catalog, group):
    """One group name or a list of them (their union); an empty list means every fund."""
    groups = [group] if isinstance(group, str) else list(group)
    if not groups:
        return pd.Series(True, index=catalog.index)
    return pd.Series(np.logical_or.reduce([group_mask(catalog, g).to_numpy() for g in groups]), index=catalog.index)


def group_outcome(catalog, group):
    """The outcome period the groups imply: Quarterly for 'Quarterly structures' or all-quarterly groups, else
    Annual (e.g. 'Max Buffer' is mostly annual ladders plus one quarterly fund)."""
    groups = [group] if isinstance(group, str) else list(group)
    if 'Quarterly structures' in groups:
        return 'Quarterly'
    outcomes = set(catalog.loc[groups_mask(catalog, groups), 'outcome_type'])
    return 'Quarterly' if outcomes == {'Quarterly'} else 'Annual'


def filter_catalog(catalog, group='F series', outcome='Annual', series=(), strategies=(),
                   references=(), months=(), buffer_band='All buffers'):
    frame = catalog.loc[catalog.outcome_type == outcome].copy()
    frame = frame.loc[groups_mask(frame, group)]
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
