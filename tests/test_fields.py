"""Studio <-> chassis parity: every field is reachable and survives a state round trip unchanged."""
from dataclasses import asdict, fields, replace
import json
import unittest

from config.settings import BacktestConfig, MONTHS, ScoreConfig
from config.spec import RunSpec
from studio.fields import FIELD_SPEC, config_from_state, state_from_config
from studio.service import ROOT, load_preset

UNIVERSE = tuple('F' + month for month in MONTHS)


def synthetic_configs():
    base = BacktestConfig(universe=UNIVERSE, initial_selection='highest_score', start_date='2022-01-03',
                          end_date='2025-06-30', cohorts=('JAN', 'JUL'))
    return [
        replace(base, comparator='<=', threshold=1.25, trigger='ecr', min_holding_days=21, review='daily'),
        replace(base, trigger='buffer_drop', buffer_metric='cap_integrity', buffer_drop=.35, comparator='<'),
        replace(base, trigger='ecr_or_buffer', buffer_metric='dbb', buffer_drop=.07, threshold=1.75),
        replace(base, trigger='ecr_and_buffer', buffer_metric='buffer_integrity', buffer_drop=.2),
        replace(base, initial_selection='cohort', hold_benchmark='launch_fund', review='semiannual', cost_bps=7.5),
        replace(base, portfolio_size=5, trigger='metric_floor', metric_floor=.85, entry_metric_margin=.07,
                confirmation_reviews=2, reentry_cooldown_days=10, min_selection_improvement=.07,
                max_replacements_per_review=2, max_turnover_per_review=.35, weight_band=.07,
                hold_benchmark='initial_selection', execution_lag=0, min_remaining_days=30, candidate_audit=False),
    ]


class FieldSpecTests(unittest.TestCase):
    def test_every_chassis_field_has_a_place_in_the_studio(self):
        expected = {f.name for f in fields(BacktestConfig)} | {f.name for f in fields(ScoreConfig)}
        self.assertEqual(set(FIELD_SPEC), expected)

    def test_presets_and_synthetic_configs_round_trip_exactly(self):
        configs = [load_preset(name) for name in ('super-buffer.json', 'ucit-conservative.json')]
        configs = [(score, replace(config, start_date='2021-01-19', end_date='2026-03-27')) for score, config in configs]
        configs += [(replace(ScoreConfig(), dbb_weight=1.8, sigma_down=.035), c) for c in synthetic_configs()]
        for score, config in configs:
            with self.subTest(config=config):
                state = state_from_config(score, config, window=True, universe=True)
                self.assertEqual(config_from_state(state), (score, config))

    def test_component_rules_need_a_component(self):
        state = state_from_config(ScoreConfig(), synthetic_configs()[1], window=True, universe=True)
        state['trigger_metric'] = 'ecr'
        with self.assertRaisesRegex(ValueError, 'Choose DBB'):
            config_from_state(state)


class RunSpecTests(unittest.TestCase):
    def test_every_saved_shape_loads_to_the_same_run(self):
        score, config = ScoreConfig(), synthetic_configs()[-1]
        plain = {'score': asdict(score), 'backtest': asdict(config)}
        manifest = {**plain, 'strategy_id': 'abc', 'data_id': 'd', 'sources': {'metrics': {'sha256': 'x'}},
                    'git_commit': 'e350c19', 'rule_fingerprint': 'r', 'run_fingerprint': 'f', 'timings': {}}
        plan_entry = {'name': 'C001', 'filters': {'group': 'F series'}, **plain}
        for raw in (plain, manifest, plan_entry, json.dumps(plain)):
            spec = RunSpec.from_json(raw)
            self.assertEqual((spec.score, spec.backtest), (score, config))
        self.assertEqual(RunSpec.from_json(manifest).provenance['data_id'], 'd')
        self.assertEqual(RunSpec.from_json(RunSpec.from_json(plan_entry).to_json()), RunSpec.from_json(plan_entry))

    def test_unknown_keys_are_rejected_not_ignored(self):
        with self.assertRaisesRegex(ValueError, 'Unknown configuration keys'):
            RunSpec.from_json({'backtest': {}, 'surprise': 1})
        with self.assertRaisesRegex(ValueError, 'Unknown BacktestConfig fields'):
            RunSpec.from_json({'backtest': {'threshhold': 1.5}})

    def test_repository_presets_load(self):
        for name in ('super-buffer.json', 'ucit-conservative.json', 'ecr.json'):
            RunSpec.from_json(ROOT / 'config' / name)
