"""Excel expectations are cached workbook values, independent of Python scoring."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from backtesting.data_pipeline import load_metrics, load_benchmarks, load_cycles
from backtesting.engine import DIAGNOSTIC_COLUMNS, prepare_data, simulate, threshold_fires
from config.settings import BacktestConfig, ScoreConfig
from core.scoring import COMPONENTS, score_funds
from utils.exporters import export_results

ROOT = Path(__file__).resolve().parents[1]


def example_data():
    dates = pd.to_datetime(['2021-01-19', '2021-01-20', '2021-01-21', '2021-01-22', '2021-01-25'])
    rows = []
    for fund, prices in [('FJAN', [100, 110, 121, 121, 121]), ('FFEB', [100, 200, 220, 242, 242])]:
        for date, nav in zip(dates, prices):
            rows.append({'ticker': fund, 'date': date, 'fund_value': float(nav), 'reference_asset_return': 0.,
                         'remaining_outcome_days': 180., 'remaining_cap_net': .1 if fund == 'FFEB' else 0.,
                         'remaining_buffer_net': .1, 'downside_before_buffer_net': 0.,
                         'fund_cap_net': .1, 'buffer_net': .1})
    metrics = pd.DataFrame(rows)
    benchmarks = pd.DataFrame({'SPY': [100, 101, 102, 103, 104], 'BUFR': [100]*5}, index=dates)
    cycles = pd.to_datetime(['2021-01-15', '2021-01-22'])
    return metrics, benchmarks, cycles


class ScoreTests(unittest.TestCase):
    def test_excel_golden_components(self):
        frame = pd.read_csv(ROOT / 'tests/fixtures/website_metrics_golden.csv')
        before = frame.copy(deep=True)
        actual = score_funds(frame)
        np.testing.assert_allclose(actual[list(COMPONENTS)], frame[list(COMPONENTS)], rtol=0, atol=1e-12)
        self.assertTrue(actual.score_valid.all())
        pd.testing.assert_frame_equal(frame, before)

    def test_asymmetry_and_period_peak(self):
        frame = example_data()[0].iloc[:3].copy()
        frame['remaining_cap_net'] = .1
        frame['reference_asset_return'] = [-.07, .07, .05]
        frame['remaining_outcome_days'] = [180., 105., 255.]
        actual = score_funds(frame)
        self.assertAlmostEqual(actual.reset_proximity.iloc[0], np.exp(-.5))
        self.assertGreater(actual.reset_proximity.iloc[0], actual.reset_proximity.iloc[1])
        self.assertAlmostEqual(actual.reset_proximity.iloc[2], np.exp(-.5))
        np.testing.assert_allclose(actual.period_position, [1, np.exp(-.5), np.exp(-.5)])

    def test_invalid_rows_do_not_become_zero_scores(self):
        frame = example_data()[0].iloc[:4].copy()
        frame.loc[0, 'buffer_net'] = 0
        frame.loc[1, 'fund_cap_net'] = -1
        frame.loc[2, 'reference_asset_return'] = np.nan
        frame.loc[3, 'remaining_outcome_days'] = -1
        actual = score_funds(frame)
        self.assertFalse(actual.score_valid.any())
        self.assertTrue(actual[list(COMPONENTS)].isna().all().all())

    def test_expiry_and_long_period_remain_formula_inputs(self):
        frame = example_data()[0].iloc[:2].copy()
        frame['remaining_outcome_days'] = [0, 371]
        actual = score_funds(frame)
        self.assertTrue(actual.score_valid.all())
        np.testing.assert_allclose(actual.period_position, np.exp(-(np.array([0, 371])-180)**2/(2*75**2)))

    def test_integrity_gates_and_no_undocumented_cap_clamp(self):
        frame = example_data()[0].iloc[:3].copy()
        frame['remaining_buffer_net'] = [.2, -.1, .1]
        frame['remaining_cap_net'] = [.2, .1, -.1]
        actual = score_funds(frame)
        np.testing.assert_allclose(actual.buffer_integrity, [1, 0, 1])
        np.testing.assert_allclose(actual.cap_integrity, [1, 1, -1])
        np.testing.assert_allclose(actual.ecr2, [3, .3, .3])

    def test_parameters_are_effective_and_validated(self):
        frame = example_data()[0]
        self.assertFalse(score_funds(frame).ecr2.equals(score_funds(frame, ScoreConfig(period_bonus=0)).ecr2))
        for kwargs in [{'sigma_up': 0}, {'period_width': -1}, {'dbb_weight': float('nan')}, {'cap_exponent': -1}]:
            with self.assertRaises(ValueError):
                ScoreConfig(**kwargs)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.inputs = example_data()
        self.data = prepare_data(*self.inputs)
        self.config = BacktestConfig(cohorts=('JAN',), review='daily')

    def test_same_close_switch_earns_outgoing_return_on_signal_day(self):
        result = simulate(self.data, self.config)
        np.testing.assert_allclose(result['daily']['Strategy NAV'], [100, 110, 121, 133.1, 133.1])
        self.assertEqual(result['diagnostics']['Event Type'].tolist(), ['initial', 'switch'])
        self.assertEqual(result['diagnostics'].Date.iloc[1], pd.Timestamp('2021-01-20'))
        self.assertEqual(result['summary'].Switches.iloc[0], 1)

    def test_next_close_delay(self):
        metrics, bench, cycles = self.inputs
        # Candidate jumps 30% on execution day; the outgoing holding earns 10%.
        metrics.loc[(metrics.ticker == 'FFEB') & (metrics.date == '2021-01-21'), 'fund_value'] = 260.
        data = prepare_data(metrics, bench, cycles)
        result = simulate(data, replace(self.config, execution_lag=1))
        event = result['diagnostics'].iloc[1]
        self.assertEqual(event['Signal Date'], pd.Timestamp('2021-01-20'))
        self.assertEqual(event['Execution Date'], pd.Timestamp('2021-01-21'))
        self.assertEqual(result['daily']['Fund at Close'].tolist()[:3], ['FJAN', 'FJAN', 'FFEB'])
        self.assertAlmostEqual(result['daily']['Strategy NAV'].iloc[2], 121.)

    def test_pending_trade_at_end_is_not_executed(self):
        result = simulate(self.data, replace(self.config, execution_lag=1, end_date='2021-01-20'))
        self.assertEqual(len(result['diagnostics']), 1)
        self.assertEqual(result['decisions'].Action.iloc[-1], 'unexecuted_end')

    def test_threshold_boundary_and_minimum_hold(self):
        at = float(self.data.scores[0, self.data.tickers.index('FJAN')])
        self.assertTrue(threshold_fires(at, replace(self.config, threshold=at)))
        self.assertFalse(threshold_fires(at, replace(self.config, threshold=at, comparator='<')))
        self.assertFalse(threshold_fires(at + .01, replace(self.config, threshold=at)))
        result = simulate(self.data, replace(self.config, min_holding_days=2))
        self.assertEqual(result['diagnostics'].Date.iloc[1], pd.Timestamp('2021-01-21'))

    def test_monthly_only_reviews_cycle_dates(self):
        result = simulate(self.data, replace(self.config, review='monthly'))
        self.assertEqual(result['decisions'].Date.tolist(), [pd.Timestamp('2021-01-22')])
        self.assertEqual(result['diagnostics'].Date.iloc[1], pd.Timestamp('2021-01-22'))

    def test_ties_keep_current_holding_without_extra_snapshot(self):
        metrics, bench, cycles = self.inputs
        metrics['remaining_cap_net'] = .1
        data = prepare_data(metrics, bench, cycles)
        result = simulate(data, replace(self.config, threshold=3))
        self.assertEqual(len(result['diagnostics']), 1)
        self.assertTrue((result['decisions'].Action == 'no_switch').all())
        self.assertTrue((result['candidates'].loc[result['candidates'].Best, 'Rank'] == 1).all())

    def test_missing_held_price_is_an_error(self):
        metrics, bench, cycles = self.inputs
        metrics = metrics.loc[~((metrics.ticker == 'FFEB') & (metrics.date == pd.Timestamp('2021-01-22')))]
        with self.assertRaisesRegex(ValueError, 'Missing held NAV'):
            simulate(prepare_data(metrics, bench, cycles), self.config)

    def test_missing_interior_benchmarks_are_explicit(self):
        metrics, bench, cycles = self.inputs
        bench = bench.drop(pd.Timestamp('2021-01-20'))
        result = simulate(prepare_data(metrics, bench, cycles), self.config)
        self.assertTrue(pd.isna(result['daily']['BUFR NAV'].iloc[1]))
        self.assertEqual(result['summary']['Missing Benchmark Sessions'].iloc[0], 1)
        self.assertAlmostEqual(result['summary']['BUFR Total Return'].iloc[0], 0)

    def test_future_changes_do_not_change_past_decisions_or_nav(self):
        original = simulate(self.data, self.config)
        metrics, bench, cycles = self.inputs
        metrics.loc[metrics.date >= '2021-01-22', ['fund_value', 'remaining_buffer_net']] *= 1.5
        revised = simulate(prepare_data(metrics, bench, cycles), self.config)
        for key in ('daily', 'diagnostics', 'decisions'):
            columns = [c for c in original[key] if c not in ('Strategy ID', 'Event ID', 'Review ID', 'Data ID')]
            a = original[key].loc[original[key].Date < '2021-01-22', columns].reset_index(drop=True)
            b = revised[key].loc[revised[key].Date < '2021-01-22', columns].reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b)

    def test_cost_is_applied_once_per_switch(self):
        free = simulate(self.data, self.config)
        charged = simulate(self.data, replace(self.config, cost_bps=10))
        self.assertAlmostEqual(charged['daily']['Strategy NAV'].iloc[-1],
                               free['daily']['Strategy NAV'].iloc[-1] * .999)

    def test_run_identity_includes_parameters(self):
        a = simulate(self.data, self.config)
        b = simulate(self.data, replace(self.config, threshold=1.4))
        c = simulate(prepare_data(*self.inputs, ScoreConfig(sigma_up=.1)), self.config)
        self.assertEqual(len({x['strategy_id'] for x in (a, b, c)}), 3)


class WorkbookRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = prepare_data(load_metrics(ROOT / 'input_data/website_metrics.csv'),
                                load_benchmarks(ROOT / 'input_data/benchmark_ts.csv'),
                                load_cycles(ROOT / 'input_data/roll_dates.csv'))
        cls.result = simulate(cls.data)

    def test_all_166_generated_events_match_excel(self):
        expected = pd.read_csv(ROOT / 'tests/fixtures/diagnostics_golden.csv', parse_dates=['Date'])
        actual = self.result['diagnostics'][DIAGNOSTIC_COLUMNS].reset_index(drop=True)
        self.assertEqual(len(actual), 166)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=0)

    def test_all_cohorts_and_candidates_have_provenance(self):
        self.assertEqual(len(self.result['summary']), 12)
        self.assertEqual(self.result['diagnostics']['Event ID'].nunique(), 166)
        candidates = self.result['candidates']
        self.assertTrue(candidates.loc[candidates.Eligible, 'Reset Prox.'].notna().all())
        self.assertFalse(candidates.loc[candidates['Reset Prox.'].isna(), 'Eligible'].any())
        self.assertEqual(self.result['summary']['Missing Benchmark Sessions'].unique().tolist(), [12])

    def test_no_audit_preserves_events_and_nav(self):
        lean = simulate(self.data, BacktestConfig(candidate_audit=False))
        for key in ('daily', 'diagnostics'):
            columns = [c for c in lean[key] if c not in ('Strategy ID', 'Event ID')]
            pd.testing.assert_frame_equal(lean[key][columns], self.result[key][columns])
        self.assertTrue(lean['candidates'].empty)

    def test_excel_csv_export_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            export_results(self.result, output, self.result['parameters'], plots=False)
            actual = pd.read_excel(output / 'ecr2_results.xlsx', sheet_name='ECR 2.0 Diagnostics')
            self.assertEqual(len(actual), 166)
            self.assertTrue((output / 'manifest.json').is_file())
            with self.assertRaises(FileExistsError):
                export_results(self.result, output, {}, plots=False)


class AdapterTests(unittest.TestCase):
    def test_percentage_csv_is_converted_once(self):
        actual = load_metrics(ROOT / 'input_data/data.csv')
        jan = actual.loc[(actual.ticker == 'FJAN') & (actual.date == '2021-01-19')].iloc[0]
        self.assertAlmostEqual(jan.reference_asset_return, .0079)
        self.assertAlmostEqual(jan.remaining_cap_net, .1293)

    def test_duplicate_keys_are_rejected(self):
        frame = example_data()[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.csv'
            pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_metrics(path)


if __name__ == '__main__':
    unittest.main()
