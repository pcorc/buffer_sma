"""Where every chassis setting lives in the Studio, with lossless state <-> configuration transforms.

A test asserts FIELD_SPEC covers every ScoreConfig and BacktestConfig field, so a new engine
setting cannot silently go missing from the interface.
"""
from dataclasses import asdict, fields
import datetime

from config.settings import BacktestConfig, ScoreConfig

TRIGGER_RULES = {
    'floor': 'Falls below a level',
    'drop': 'Falls a set % from its value at purchase',
    'either': 'ECR below a level, or the component drops',
    'both': 'ECR below a level, and the component drops',
}
RULE_TRIGGERS = {'drop': 'buffer_drop', 'either': 'ecr_or_buffer', 'both': 'ecr_and_buffer'}
HOLD_BENCHMARKS = {'auto': 'Automatic (legacy inference)', 'initial_selection': 'Fund(s) bought at launch',
                   'launch_fund': 'Launch-month F-series fund'}

# field -> (Studio section, session-state keys)
FIELD_SPEC = {
    'client_name': ('Client configuration', ('client',)),
    'universe': ('Fund universe', ('selected_tickers',)),
    'portfolio_size': ('01 Portfolio', ('structure', 'fund_count')),
    'review': ('02 Schedule', ('cadence',)),
    'selection': ('03 Selection', ('selection',)),
    'trigger': ('04 Trigger', ('trigger_rule', 'trigger_metric')),
    'buffer_metric': ('04 Trigger', ('trigger_metric', 'buffer_metric_carry')),
    'threshold': ('04 Trigger', ('ecr_threshold',)),
    'metric_floor': ('04 Trigger', ('component_threshold',)),
    'buffer_drop': ('04 Trigger', ('buffer_drop_pct',)),
    'start_date': ('05 Test window', ('start',)),
    'end_date': ('05 Test window', ('end',)),
    'cohorts': ('05 Test window', ('cohorts',)),
    'execution_lag': ('Trading assumptions', ('lag',)),
    'cost_bps': ('Trading assumptions', ('cost',)),
    'min_remaining_days': ('Trading assumptions', ('remaining_days',)),
    'candidate_audit': ('Trading assumptions', ('audit',)),
    'min_selection_improvement': ('Turnover & weight controls', ('score_gap',)),
    'max_replacements_per_review': ('Turnover & weight controls', ('replacement_limit',)),
    'max_turnover_per_review': ('Turnover & weight controls', ('turnover_budget',)),
    'weight_band': ('Turnover & weight controls', ('use_bands', 'band_pp')),
    'comparator': ('Advanced rule settings', ('comparator',)),
    'initial_selection': ('Advanced rule settings', ('initial_selection',)),
    'hold_benchmark': ('Advanced rule settings', ('hold_benchmark',)),
    'min_holding_days': ('Advanced rule settings', ('min_holding_days',)),
    'confirmation_reviews': ('Advanced rule settings', ('confirmation_reviews',)),
    'entry_metric_margin': ('Advanced rule settings', ('entry_metric_margin',)),
    'reentry_cooldown_days': ('Advanced rule settings', ('reentry_cooldown_days',)),
    **{name: ('ECR formula', ('score_' + name,)) for name in (f.name for f in fields(ScoreConfig))},
}
WINDOW_KEYS = ('start', 'end', 'cohorts')


def _percent(value):
    # Display as percent without exposing binary floating-point residue (0.07 * 100).
    return round(value * 100, 10)


def state_from_config(score, config, *, window=False, universe=False):
    """Widget values for a configuration. The window and universe are opt-in: presets keep the user's."""
    trigger_rule = {'ecr': 'floor', 'metric_floor': 'floor', 'buffer_drop': 'drop',
                    'ecr_or_buffer': 'either', 'ecr_and_buffer': 'both'}[config.trigger]
    state = {
        'client': config.client_name,
        'structure': 'Single ticker' if config.portfolio_size == 1 else 'Fund of funds',
        'fund_count': max(2, config.portfolio_size), 'cadence': config.review, 'selection': config.selection,
        'trigger_rule': trigger_rule, 'trigger_metric': 'ecr' if config.trigger == 'ecr' else config.buffer_metric,
        'buffer_metric_carry': config.buffer_metric, 'ecr_threshold': float(config.threshold),
        'component_threshold': float(config.metric_floor), 'buffer_drop_pct': _percent(config.buffer_drop),
        'lag': config.execution_lag, 'cost': float(config.cost_bps), 'remaining_days': config.min_remaining_days,
        'audit': config.candidate_audit, 'score_gap': float(config.min_selection_improvement),
        'replacement_limit': config.max_replacements_per_review,
        'turnover_budget': _percent(config.max_turnover_per_review),
        'use_bands': config.weight_band is not None,
        'band_pp': _percent(config.weight_band) if config.weight_band is not None else 5.,
        'comparator': config.comparator, 'initial_selection': config.initial_selection,
        'hold_benchmark': config.hold_benchmark or 'auto', 'min_holding_days': config.min_holding_days,
        'confirmation_reviews': config.confirmation_reviews,
        'entry_metric_margin': float(config.entry_metric_margin),
        'reentry_cooldown_days': config.reentry_cooldown_days,
    }
    state.update({'score_' + name: float(value) for name, value in asdict(score).items()})
    if window:
        state.update({'start': datetime.date.fromisoformat(config.start_date),
                      'end': datetime.date.fromisoformat(config.end_date) if config.end_date else None,
                      'cohorts': list(config.cohorts)})
    if universe and config.universe is not None:
        state['selected_tickers'] = list(config.universe)
    return state


def config_from_state(state, universe=None):
    """The configuration the Studio will run; raises ValueError with a user-facing message."""
    start, end = state.get('start'), state.get('end')
    if start is None or end is None:
        raise ValueError('Choose both a start and an end date.')
    rule, metric = state['trigger_rule'], state['trigger_metric']
    if rule == 'floor':
        trigger = 'ecr' if metric == 'ecr' else 'metric_floor'
    elif metric == 'ecr':
        raise ValueError('This trigger rule watches a component. Choose DBB, Buffer Integrity or Cap Integrity.')
    else:
        trigger = RULE_TRIGGERS[rule]
    n = 1 if state['structure'] == 'Single ticker' else int(state['fund_count'])
    universe = state.get('selected_tickers') if universe is None else universe
    score = ScoreConfig(**{f.name: float(state['score_' + f.name]) for f in fields(ScoreConfig)})
    config = BacktestConfig(
        client_name=state['client'], portfolio_size=n, selection=state['selection'],
        universe=tuple(universe) if universe else None, threshold=float(state['ecr_threshold']),
        comparator=state['comparator'], review=state['cadence'], min_holding_days=int(state['min_holding_days']),
        start_date=start.isoformat(), end_date=end.isoformat(), cohorts=tuple(state['cohorts']),
        min_remaining_days=int(state['remaining_days']), execution_lag=int(state['lag']),
        cost_bps=float(state['cost']), candidate_audit=bool(state['audit']),
        initial_selection='highest_score' if n > 1 else state['initial_selection'], trigger=trigger,
        buffer_metric=metric if metric != 'ecr' else state['buffer_metric_carry'],
        buffer_drop=float(state['buffer_drop_pct']) / 100, metric_floor=float(state['component_threshold']),
        confirmation_reviews=int(state['confirmation_reviews']),
        min_selection_improvement=float(state['score_gap']),
        entry_metric_margin=float(state['entry_metric_margin']),
        reentry_cooldown_days=int(state['reentry_cooldown_days']),
        max_replacements_per_review=int(state['replacement_limit']),
        max_turnover_per_review=float(state['turnover_budget']) / 100,
        weight_band=float(state['band_pp']) / 100 if n > 1 and state['use_bands'] else None,
        hold_benchmark=None if state['hold_benchmark'] == 'auto' else state['hold_benchmark'])
    return score, config


def state_keys():
    return sorted({key for _, keys in FIELD_SPEC.values() for key in keys})
