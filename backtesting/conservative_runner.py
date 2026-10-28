"""Shared, deterministic research configurations for the two command-line runners."""
import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import time

from backtesting.data_pipeline import file_hash
from backtesting.research import Experiment, ResearchSession
from config.settings import BacktestConfig, MONTHS, ScoreConfig
from core.universe import (BUFFER_BANDS, CATALOG_PATH, EXTRA_METRICS, GROUPS,
                           filter_catalog, load_catalog, validate_universe)
from studio.archive import DEFAULT_ARCHIVE_ROOT, archive_root_path, save_archive
from studio.service import SOURCE_PATHS, build_downloads, validate_window

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID = ROOT / 'config/conservative-grid.json'
METRIC_LABELS = {'ecr': 'ECR', 'buffer_integrity': 'Buffer', 'dbb': 'DBB', 'cap_integrity': 'Cap'}
PROFILE_FIELDS = {'min_selection_improvement', 'max_replacements_per_review',
                  'max_turnover_per_review', 'weight_band'}


@dataclass(frozen=True)
class ResearchCase:
    experiment: Experiment
    filters: dict


def csv_values(value):
    values = [v.strip() for v in value.split(',') if v.strip()]
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('Provide a nonempty comma-separated list without duplicates.')
    return tuple(values)


def add_common_arguments(parser, cohorts=('JAN',)):
    parser.add_argument('--archive-root', '--output', default=DEFAULT_ARCHIVE_ROOT,
                        help='Parent folder for a unique run directory (defaults to the shared G: archive).')
    parser.add_argument('--start-date', default='2021-01-19')
    parser.add_argument('--end-date')
    parser.add_argument('--cohorts', type=csv_values, default=cohorts,
                        help='Independent launch tests, e.g. JAN or JAN,APR,JUL,OCT.')
    parser.add_argument('--cost-bps', type=float, default=0.)


def add_universe_arguments(parser):
    parser.add_argument('--group', choices=GROUPS, default='F series')
    parser.add_argument('--outcome', choices=['Annual', 'Quarterly'])
    parser.add_argument('--tickers', type=csv_values)
    parser.add_argument('--series', type=csv_values, default=())
    parser.add_argument('--strategies', type=csv_values, default=())
    parser.add_argument('--reference-assets', type=csv_values, default=())
    parser.add_argument('--outcome-months', type=csv_values, default=())
    parser.add_argument('--buffer-category', choices=BUFFER_BANDS, default='All buffers')


def filters_from_args(args):
    return {'group': args.group, 'outcome': args.outcome or ('Quarterly' if args.group == 'Quarterly structures' else 'Annual'),
            'tickers': args.tickers, 'series': args.series, 'strategies': args.strategies,
            'references': args.reference_assets, 'months': args.outcome_months, 'buffer_band': args.buffer_category}


def research_session():
    session = ResearchSession(SOURCE_PATHS['metrics'], SOURCE_PATHS['benchmarks'], SOURCE_PATHS['cycles'],
                              supplemental_metrics=sorted(EXTRA_METRICS.glob('*.csv')))
    session.sources['catalogue'] = {'path': str(CATALOG_PATH), 'sha256': file_hash(CATALOG_PATH)}
    return session


def resolve_universe(filters, catalogue, available):
    allowed = {'name', 'group', 'outcome', 'series', 'strategies', 'references', 'months', 'buffer_band', 'tickers'}
    if set(filters) - allowed:
        raise ValueError(f'Unknown universe filters: {sorted(set(filters)-allowed)}')
    options = {k: v for k, v in filters.items() if k not in ('name', 'tickers')}
    if options.get('group') == 'Quarterly structures':
        options.setdefault('outcome', 'Quarterly')
    for key in ('series', 'strategies', 'references', 'months'):
        if key in options and not isinstance(options[key], (list, tuple)):
            raise ValueError(f'{key} must be a list.')
    matched = filter_catalog(catalogue, **options)
    tickers = filters.get('tickers')
    if tickers is not None:
        if not isinstance(tickers, (list, tuple)) or set(tickers) - set(matched.ticker):
            raise ValueError('Explicit tickers must match the universe filters and catalogue.')
    else:
        tickers = sorted(set(matched.ticker) & set(available))
    selected = validate_universe(catalogue, tickers, available)
    return tuple(sorted(tickers)), selected


def base_config(args):
    return BacktestConfig(client_name='UCITS Conservative', portfolio_size=3, initial_selection='highest_score',
                          selection='highest_ecr', execution_lag=1, cost_bps=args.cost_bps,
                          start_date=args.start_date, end_date=args.end_date,
                          cohorts=tuple(c.upper() for c in args.cohorts), comparator='<')


def make_case(name, filters, catalogue, available, base, n=3, metric='buffer_integrity', floor=.8,
              cadence='monthly', controls=None):
    tickers, selected = resolve_universe(filters, catalogue, available)
    if n > len(tickers):
        raise ValueError(f'{name}: {n} positions requested but only {len(tickers)} funds have history.')
    if metric not in METRIC_LABELS:
        raise ValueError(f'Unknown trigger: {metric}')
    if cadence not in ('monthly', 'quarterly', 'semiannual', 'annual'):
        raise ValueError('Review cadence must be monthly, quarterly, semiannual or annual.')
    controls = controls or {}
    if set(controls) - PROFILE_FIELDS:
        raise ValueError(f'Unknown turnover controls: {sorted(set(controls)-PROFILE_FIELDS)}')
    score = ScoreConfig()
    if selected.outcome_type.iloc[0] == 'Quarterly':
        score = replace(score, period_center=45, period_width=18.75)
    config = replace(base, client_name=name, portfolio_size=n, universe=tickers, review=cadence,
                     trigger='ecr' if metric == 'ecr' else 'metric_floor',
                     threshold=floor if metric == 'ecr' else 1.5,
                     buffer_metric=metric if metric != 'ecr' else 'buffer_integrity',
                     metric_floor=floor if metric != 'ecr' else .8, **controls)
    return ResearchCase(Experiment(name, score, config), filters)


def grid_cases(grid, catalogue, available, base):
    allowed = {'universes', 'portfolio_sizes', 'review_cadences', 'triggers', 'turnover_profiles', 'include_single_ticker_reference'}
    if set(grid) - allowed:
        raise ValueError(f'Unknown grid keys: {sorted(set(grid)-allowed)}')
    for key in allowed - {'include_single_ticker_reference'}:
        if not grid.get(key):
            raise ValueError(f'Grid requires nonempty {key}.')
    cases = []
    for filters in grid['universes']:
        universe_name = filters.get('name', filters.get('group', 'F series'))
        if grid.get('include_single_ticker_reference', True):
            cases.append(make_case(f'{universe_name} / Super Buffer reference / N=1 / ECR < 1.5 / monthly',
                                   filters, catalogue, available, base, n=1, metric='ecr', floor=1.5))
        for n in grid['portfolio_sizes']:
            for cadence in grid['review_cadences']:
                for metric, floors in grid['triggers'].items():
                    if not floors:
                        raise ValueError(f'Trigger {metric} needs at least one threshold.')
                    for floor in floors:
                        for profile in grid['turnover_profiles']:
                            unknown = set(profile) - PROFILE_FIELDS - {'name', 'turnover_budget_by_size'}
                            if unknown:
                                raise ValueError(f'Unknown profile fields: {sorted(unknown)}')
                            controls = {k:v for k,v in profile.items() if k in PROFILE_FIELDS}
                            if 'turnover_budget_by_size' in profile:
                                if 'max_turnover_per_review' in controls:
                                    raise ValueError('Specify a single turnover budget or budgets by size, not both.')
                                try:
                                    controls['max_turnover_per_review'] = profile['turnover_budget_by_size'][str(n)]
                                except KeyError as exc:
                                    raise ValueError(f'Turnover profile needs a budget for N={n}.') from exc
                            name = f'{universe_name} / N={n} / {METRIC_LABELS.get(metric, metric)} < {floor:g} / {cadence} / {profile["name"]}'
                            cases.append(make_case(name, filters, catalogue, available, base, n, metric, floor, cadence, controls))
    names = [c.experiment.name for c in cases]
    configs = [replace(c.experiment.backtest, client_name='Research') for c in cases]
    if len(set(names)) != len(names) or len(set(configs)) != len(configs):
        raise ValueError('Grid contains duplicate names or equivalent configurations.')
    return cases


def preflight_cases(session, cases):
    """Reject invalid windows and incomparable launches before writing reports."""
    from backtesting.engine import _start_index
    import pandas as pd
    windows = None
    for case in cases:
        e = case.experiment
        data = session.prepare(e.score, e.backtest.universe)
        validate_window(data, e.backtest)
        end = pd.Timestamp(e.backtest.end_date) if e.backtest.end_date else data.dates[-1]
        stop = data.dates.searchsorted(end, side='right')
        current = [(c, data.dates[_start_index(data, c, e.backtest)], data.dates[stop-1]) for c in e.backtest.cohorts]
        if windows is not None and current != windows:
            raise ValueError('Comparisons require identical launch and end dates. Narrow the window or run these universes separately.')
        windows = current


def ensure_archive_root(root):
    """Fail before simulation if the requested destination is not writable."""
    from tempfile import TemporaryFile
    path = archive_root_path(root)
    path.mkdir(parents=True, exist_ok=True)
    with TemporaryFile(dir=path) as probe:
        probe.write(b'ECR archive write check')
        probe.flush()
    return path


def run_and_archive(session, case, archive_root, catalogue, folder_name=None, *, plots=True):
    e = case.experiment
    began = time.perf_counter()
    result = session.run(e.score, e.backtest)
    result['universe'] = catalogue.loc[catalogue.ticker.isin(e.backtest.universe)].copy()
    result['requested_archive_root'] = str(archive_root)
    result['sources'] = {**result['sources'], 'universe_filters': case.filters}
    folder = save_archive(build_downloads(result, plots=plots), archive_root, e.name, result['strategy_id'], folder_name=folder_name)
    print(f'  Saved ({time.perf_counter()-began:.1f}s): {folder}', flush=True)
    return result, folder
