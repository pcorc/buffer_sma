"""Rule and run fingerprints identify economics, not names or field spelling."""
from dataclasses import fields, replace
import unittest

from config.identity import COSMETIC_FIELDS, UNUSED_BY_TRIGGER, rule_fingerprint
from config.settings import BacktestConfig, MONTHS, ScoreConfig

UNIVERSE = tuple('F' + month for month in MONTHS)
# One valid alternative value per economically relevant field.
CHANGES = {
    'portfolio_size': 3, 'selection': 'highest_dbb', 'universe': UNIVERSE[:6], 'threshold': 1.25,
    'comparator': '<=', 'review': 'quarterly', 'min_holding_days': 21, 'min_remaining_days': 30,
    'execution_lag': 0, 'cost_bps': 5., 'initial_selection': 'cohort', 'trigger': 'ecr_or_buffer',
    'buffer_metric': 'dbb', 'buffer_drop': .3, 'metric_floor': .7, 'confirmation_reviews': 2,
    'min_selection_improvement': .1, 'entry_metric_margin': .05, 'reentry_cooldown_days': 10,
    'max_replacements_per_review': 1, 'max_turnover_per_review': .5, 'weight_band': .05,
    'hold_benchmark': 'launch_fund',
}


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.score = ScoreConfig()
        self.config = BacktestConfig(universe=UNIVERSE, initial_selection='highest_score', execution_lag=1,
                                     comparator='<', trigger='metric_floor')

    def test_cosmetic_fields_and_universe_order_do_not_change_identity(self):
        base = rule_fingerprint(self.score, self.config)
        for config in (replace(self.config, client_name='Renamed'), replace(self.config, candidate_audit=False),
                       replace(self.config, start_date='2022-01-03', end_date='2025-01-02', cohorts=('JAN',)),
                       replace(self.config, universe=tuple(reversed(UNIVERSE))),
                       replace(self.config, threshold=2., buffer_drop=.5),  # unread by a floor trigger
                       replace(self.config, hold_benchmark='initial_selection')):  # the inferred basis made explicit
            self.assertEqual(rule_fingerprint(self.score, config), base)

    def test_every_economic_field_changes_identity(self):
        covered = set(CHANGES) | set(COSMETIC_FIELDS)
        self.assertEqual(covered, {f.name for f in fields(BacktestConfig)}, 'Classify every new BacktestConfig field.')
        for name, value in CHANGES.items():
            # Weight bands are only valid for multi-fund portfolios.
            config = replace(self.config, portfolio_size=3) if name == 'weight_band' else self.config
            changed = replace(config, **{name: value})
            if name in UNUSED_BY_TRIGGER[changed.trigger]:
                continue
            with self.subTest(field=name):
                self.assertNotEqual(rule_fingerprint(self.score, changed), rule_fingerprint(self.score, config))
        base = rule_fingerprint(self.score, self.config)
        for field in fields(ScoreConfig):
            with self.subTest(score=field.name):
                score = replace(self.score, **{field.name: getattr(self.score, field.name) * 1.1})
                self.assertNotEqual(rule_fingerprint(score, self.config), base)

    def test_default_universe_matches_its_explicit_tickers(self):
        legacy = BacktestConfig()
        explicit = replace(legacy, universe=UNIVERSE, hold_benchmark='launch_fund')
        self.assertEqual(rule_fingerprint(self.score, legacy, UNIVERSE), rule_fingerprint(self.score, explicit))
        # The same tickers with a different buy-and-hold comparison are different results.
        self.assertNotEqual(rule_fingerprint(self.score, replace(explicit, hold_benchmark='initial_selection')),
                            rule_fingerprint(self.score, explicit))
