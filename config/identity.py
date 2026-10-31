"""Economic identity of a run, independent of names, audit tables and field spelling.

strategy_id (engine) remains the artifact id; it hashes every field, including the run name.
rule_fingerprint answers "is this the same strategy?"; run_fingerprint adds the data and the
resolved launch windows, answering "are these the same results?".
"""
from dataclasses import asdict
import hashlib
import json

from config.settings import hold_basis

ENGINE_VERSION = 'ecr-ar-v2'
# Labels and reporting switches: they never change NAVs, holdings or trades.
COSMETIC_FIELDS = ('client_name', 'candidate_audit', 'start_date', 'end_date', 'cohorts')
# Parameters a trigger never reads; normalizing them lets equivalent configurations match.
UNUSED_BY_TRIGGER = {
    'ecr': ('buffer_metric', 'buffer_drop', 'metric_floor'),
    'metric_floor': ('threshold', 'buffer_drop'),
    'buffer_drop': ('threshold', 'metric_floor'),
    'ecr_or_buffer': ('metric_floor',),
    'ecr_and_buffer': ('metric_floor',),
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def rule_spec(score, config, tickers=None):
    """The economically relevant settings; tickers resolves a default (None) universe when known."""
    backtest = asdict(config)
    for name in COSMETIC_FIELDS + UNUSED_BY_TRIGGER[config.trigger]:
        backtest.pop(name)
    universe = config.universe if config.universe is not None else tickers
    backtest['universe'] = sorted(universe) if universe is not None else 'default'
    backtest['hold_benchmark'] = hold_basis(config)
    return {'engine': ENGINE_VERSION, 'score': asdict(score), 'backtest': backtest}


def rule_fingerprint(score, config, tickers=None):
    return _digest(rule_spec(score, config, tickers))


def run_fingerprint(rule, data_id, windows):
    """windows: (cohort, start, end) for each launch actually simulated."""
    return _digest({'rule': rule, 'data_id': data_id,
                    'windows': sorted((str(c), str(s), str(e)) for c, s, e in windows)})


def identify(score, config, tickers, data_id, summary):
    rule = rule_fingerprint(score, config, tickers)
    windows = zip(summary['Cohort'], summary['Start'].dt.date, summary['End'].dt.date)
    return {'engine_version': ENGINE_VERSION, 'rule_fingerprint': rule,
            'run_fingerprint': run_fingerprint(rule, data_id, windows)}
