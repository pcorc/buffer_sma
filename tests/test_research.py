from dataclasses import replace
import unittest
import numpy as np
from test_ecr import example_data, ROOT
from config.settings import ScoreConfig, BacktestConfig
from backtesting.engine import prepare_data, simulate, trigger_signals
from backtesting.research import ResearchSession, initial_experiments

class ResearchTests(unittest.TestCase):
    def test_trigger_combinations(self):
        for mode, expected in [('buffer_drop', True), ('ecr_or_buffer', True), ('ecr_and_buffer', False), ('ecr', False)]:
            config = BacktestConfig(trigger=mode, buffer_drop=.2)
            fired, ecr, buffer, drop = trigger_signals(2., .8, 1., config)
            self.assertEqual(fired, expected)
            self.assertFalse(ecr); self.assertTrue(buffer)
            self.assertAlmostEqual(drop, .2)
        self.assertFalse(trigger_signals(2., 0., 0., BacktestConfig(trigger='buffer_drop'))[0])

    def test_highest_entry_and_prior_signal(self):
        data = prepare_data(*example_data())
        config = BacktestConfig(cohorts=('JAN',), initial_selection='highest_score', review='daily')
        result = simulate(data, config)
        self.assertEqual(result['diagnostics'].Fund.iloc[0], 'FFEB')
        self.assertEqual(len(result['diagnostics']), 1)
        self.assertAlmostEqual(result['daily']['Strategy NAV'].iloc[-1], 242.)
        with self.assertRaisesRegex(ValueError, 'prior session'):
            simulate(data, replace(config, execution_lag=1))

    def test_buffer_anchor_resets_on_trade_and_minimum_hold_applies(self):
        metrics, bench, cycles = example_data()
        metrics.loc[metrics.ticker=='FJAN','remaining_cap_net'] = .1
        mask = (metrics.ticker=='FJAN') & (metrics.date >= '2021-01-20')
        metrics.loc[mask,'remaining_buffer_net'] = .07
        result = simulate(prepare_data(metrics,bench,cycles), BacktestConfig(cohorts=('JAN',),
                          trigger='buffer_drop', review='daily', min_holding_days=2))
        self.assertEqual(str(result['diagnostics'].Date.iloc[1].date()), '2021-01-21')
        self.assertAlmostEqual(result['daily']['Entry Buffer Score'].iloc[2], 1.)
        self.assertFalse(result['decisions']['Triggered'].iloc[0])

    def test_cache_reuses_scores_for_trigger_changes(self):
        session = ResearchSession(ROOT/'input_data/website_metrics.csv', ROOT/'input_data/benchmark_ts.csv',ROOT/'input_data/roll_dates.csv')
        config = BacktestConfig(cohorts=('FEB',), candidate_audit=False)
        a = session.run(backtest=config)
        b = session.run(backtest=replace(config,trigger='buffer_drop'))
        self.assertEqual(session.score_builds, 1)
        self.assertNotEqual(a['strategy_id'], b['strategy_id'])
        np.testing.assert_allclose(session.prepare().scores, prepare_data(session.metrics,session.benchmarks,session.cycles).scores)
        self.assertEqual(len(initial_experiments()), 4)

class ConservativeTests(unittest.TestCase):
    def test_legacy_allocation_config_is_rejected(self):
        with self.assertRaises(TypeError):
            ScoreConfig(component_weights=(.75,.125,.125))

    def test_conservative_grid_varies_only_buffer_trigger(self):
        experiments = initial_experiments()
        self.assertTrue(all(e.score == ScoreConfig() for e in experiments))
        self.assertEqual(experiments[0].backtest.trigger, 'ecr')
        self.assertEqual([e.backtest.metric_floor for e in experiments[1:]], [.7,.8,.9])
        for e in experiments[1:]:
            self.assertEqual(e.backtest.trigger, 'metric_floor')
            self.assertEqual(e.backtest.buffer_metric, 'buffer_integrity')
            self.assertEqual(e.backtest.comparator, '<')

    def test_metric_floor_is_independent_of_entry_and_ecr(self):
        config=BacktestConfig(trigger='metric_floor',metric_floor=.8,comparator='<')
        self.assertTrue(trigger_signals(2.9,.79,.4,config)[0])
        self.assertFalse(trigger_signals(.2,.8,1.,config)[0])
        self.assertTrue(trigger_signals(2.9,.8,.4,replace(config,comparator='<='))[0])
        for metric in ['dbb','buffer_integrity','cap_integrity']:
            BacktestConfig(trigger='metric_floor',buffer_metric=metric)
        with self.assertRaises(ValueError):BacktestConfig(metric_floor=80)

    def test_cap_floor_selects_highest_ecr(self):
        data=prepare_data(*example_data())
        config=BacktestConfig(cohorts=('JAN',),review='daily',trigger='metric_floor',buffer_metric='cap_integrity',metric_floor=.8)
        result=simulate(data,config)
        self.assertEqual(result['diagnostics'].Fund.tolist(),['FJAN','FFEB'])
        self.assertEqual(result['decisions']['Fresh Reference Score'].iloc[0],1.)
        self.assertEqual(result['decisions']['Metric Fraction of Fresh'].iloc[0],0.)
