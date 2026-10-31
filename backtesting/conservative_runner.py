"""Shared, deterministic research configurations for the two command-line runners."""
import argparse
from dataclasses import dataclass, fields, replace
from itertools import product
from pathlib import Path
import time

from backtesting.research import Experiment
from config.identity import rule_fingerprint
from config.settings import BacktestConfig, ScoreConfig
from config.spec import RunSpec
from core.universe import BUFFER_BANDS, GROUPS, filter_catalog, load_catalog, validate_universe
from studio.archive import archive_root_path, default_archive_root, save_archive, save_report
from studio.service import build_downloads, get_session, plan_cohort_windows, run_backtest, write_report

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID = ROOT / 'config/conservative-grid.json'
METRIC_LABELS = {'ecr': 'ECR', 'buffer_integrity': 'Buffer', 'dbb': 'DBB', 'cap_integrity': 'Cap'}
PROFILE_FIELDS = {'min_selection_improvement', 'max_replacements_per_review', 'max_turnover_per_review',
                  'weight_band', 'min_holding_days', 'confirmation_reviews', 'entry_metric_margin',
                  'reentry_cooldown_days'}
SELECTION_LABELS = {'highest_ecr': 'ECR', 'highest_dbb': 'DBB', 'highest_buffer_integrity': 'Buffer',
                    'highest_cap_integrity': 'Cap'}
# Optional grid dimensions: (grid key, BacktestConfig field, label for case names).
SETTING_DIMENSIONS = (('selections', 'selection', lambda v: f'select {SELECTION_LABELS[v]}'),
                      ('execution_lags', 'execution_lag', lambda v: 'next close' if v else 'same close'),
                      ('costs_bps', 'cost_bps', lambda v: f'{v:g} bps'),
                      ('comparators', 'comparator', lambda v: f'trigger {v}'))


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
    parser.add_argument('--archive-root', '--output', default=default_archive_root(),
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
    return get_session()


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
              cadence='monthly', controls=None, rule='floor', score_overrides=None, settings=None):
    """One research case. rule='floor' fires below `floor`; rule='drop' fires when `metric` falls by `floor`
    (a fraction) from its value at purchase. settings override selection, lag, cost or comparator."""
    tickers, selected = resolve_universe(filters, catalogue, available)
    if n > len(tickers):
        raise ValueError(f'{name}: {n} positions requested but only {len(tickers)} funds have history.')
    if metric not in METRIC_LABELS:
        raise ValueError(f'Unknown trigger: {metric}')
    if rule not in ('floor', 'drop') or (rule == 'drop' and metric == 'ecr'):
        raise ValueError('Trigger rules are floor (any metric) or drop (a component metric).')
    if cadence not in ('daily', 'monthly', 'quarterly', 'semiannual', 'annual'):
        raise ValueError('Review cadence must be daily, monthly, quarterly, semiannual or annual.')
    controls = controls or {}
    if set(controls) - PROFILE_FIELDS:
        raise ValueError(f'Unknown turnover controls: {sorted(set(controls)-PROFILE_FIELDS)}')
    settings = settings or {}
    if set(settings) - {field for _, field, _ in SETTING_DIMENSIONS}:
        raise ValueError(f'Unknown case settings: {sorted(settings)}')
    score = ScoreConfig()
    if selected.outcome_type.iloc[0] == 'Quarterly':
        score = replace(score, period_center=45, period_width=18.75)
    score = replace(score, **(score_overrides or {}))
    if rule == 'drop':
        trigger = dict(trigger='buffer_drop', buffer_metric=metric, buffer_drop=floor)
    elif metric == 'ecr':
        trigger = dict(trigger='ecr', threshold=floor)
    else:
        trigger = dict(trigger='metric_floor', buffer_metric=metric, metric_floor=floor)
    config = replace(base, client_name=name, portfolio_size=n, universe=tickers, review=cadence,
                     **trigger, **settings, **controls)
    return ResearchCase(Experiment(name, score, config), filters)


def _variants(grid, key, default):
    values = grid.get(key, [default])
    if not isinstance(values, list) or not values:
        raise ValueError(f'Grid {key} must be a nonempty list.')
    return values


def grid_cases(grid, catalogue, available, base):
    """Expand a grid JSON into named cases. Optional dimensions (score_profiles, selections, execution_lags,
    costs_bps, comparators, trigger_drops) default to the base settings and appear in names only when they vary."""
    required = {'universes', 'portfolio_sizes', 'review_cadences', 'triggers', 'turnover_profiles'}
    optional = {'include_single_ticker_reference', 'score_profiles', 'trigger_drops',
                *(key for key, _, _ in SETTING_DIMENSIONS)}
    if set(grid) - required - optional:
        raise ValueError(f'Unknown grid keys: {sorted(set(grid)-required-optional)}')
    for key in required:
        if not grid.get(key):
            raise ValueError(f'Grid requires nonempty {key}.')
    score_profiles = _variants(grid, 'score_profiles', {'name': 'Standard ECR'})
    score_fields = {f.name for f in fields(ScoreConfig)}
    for profile in score_profiles:
        if set(profile) - score_fields - {'name'} or 'name' not in profile:
            raise ValueError(f'Score profiles need a name and ScoreConfig fields only: {profile}')
    dimensions = [(field, label, _variants(grid, key, getattr(base, field))) for key, field, label in SETTING_DIMENSIONS]
    triggers = [('floor', metric, values) for metric, values in grid['triggers'].items()]
    triggers += [('drop', metric, values) for metric, values in grid.get('trigger_drops', {}).items()]
    cases = []

    def add(name, filters, *args, **kwargs):
        for profile in score_profiles:
            for combination in product(*(values for _, _, values in dimensions)):
                settings = {field: value for (field, _, _), value in zip(dimensions, combination)}
                suffix = [label(value) for (field, label, values), value in zip(dimensions, combination) if len(values) > 1]
                if len(score_profiles) > 1:
                    suffix.insert(0, profile['name'])
                overrides = {k: v for k, v in profile.items() if k != 'name'}
                cases.append(make_case(' / '.join([name, *suffix]), filters, catalogue, available, base, *args,
                                       score_overrides=overrides, settings=settings, **kwargs))

    for filters in grid['universes']:
        universe_name = filters.get('name', filters.get('group', 'F series'))
        if grid.get('include_single_ticker_reference', True):
            add(f'{universe_name} / Super Buffer reference / N=1 / ECR < 1.5 / monthly', filters, n=1, metric='ecr', floor=1.5)
        for n in grid['portfolio_sizes']:
            for cadence in grid['review_cadences']:
                for rule, metric, values in triggers:
                    if not values:
                        raise ValueError(f'Trigger {metric} needs at least one threshold.')
                    for value in values:
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
                            rule_text = (f'{METRIC_LABELS[metric]} < {value:g}' if rule == 'floor'
                                         else f'{METRIC_LABELS[metric]} drop {value:.0%}')
                            add(f'{universe_name} / N={n} / {rule_text} / {cadence} / {profile["name"]}', filters,
                                n, metric, value, cadence, controls, rule=rule)
    names = [c.experiment.name for c in cases]
    rules = [rule_fingerprint(c.experiment.score, c.experiment.backtest) for c in cases]
    if len(set(names)) != len(names) or len(set(rules)) != len(rules):
        raise ValueError('Grid contains duplicate names or economically equivalent configurations.')
    return cases


def preflight_cases(session, cases):
    """Reject invalid windows and incomparable launches before writing reports."""
    windows = None
    for case in cases:
        e = case.experiment
        data = session.prepare(e.score, e.backtest.universe)
        current = [(w['cohort'], w['start'], w['end']) for w in plan_cohort_windows(data, e.backtest)]
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


def run_and_archive(session, case, archive_root, catalogue, folder_name=None, *, plots=True, excel='full', package=True):
    """Simulate one case and archive its report.

    package=True keeps a downloadable results.zip beside the extracted report (single runs).
    Suites pass package=False and excel='compact': the report is written straight into its folder.
    """
    e = case.experiment
    began = time.perf_counter()
    result = run_backtest(RunSpec(e.score, e.backtest, name=e.name, filters=case.filters), session)
    result['universe'] = catalogue.loc[catalogue.ticker.isin(e.backtest.universe)].copy()
    result['requested_archive_root'] = str(archive_root)
    result['sources'] = {**result['sources'], 'universe_filters': case.filters}
    if package:
        folder = save_archive(build_downloads(result, plots=plots, excel=excel), archive_root, e.name,
                              result['strategy_id'], folder_name=folder_name)
    else:
        folder = save_report(lambda path: write_report(result, path, plots=plots, excel=excel), archive_root,
                             e.name, result['strategy_id'], folder_name=folder_name)
    print(f'  Saved ({time.perf_counter()-began:.1f}s): {folder}', flush=True)
    return result, folder


# Kept in memory per suite case; detailed ledgers are already on disk.
COMPARISON_KEYS = ('summary', 'daily', 'parameters', 'timings', 'strategy_id', 'sources',
                   'rule_fingerprint', 'run_fingerprint')


def run_suite_case(case, archive_root, folder_name):
    """One suite case in any process: compact report written in place, comparison inputs returned."""
    result, folder = run_and_archive(research_session(), case, archive_root, load_catalog(), folder_name=folder_name,
                                     plots=False, excel='compact', package=False)
    return {key: result[key] for key in COMPARISON_KEYS}, folder

