"""At N=1 the single-fund loop and the portfolio engine are two implementations of one policy."""
from dataclasses import replace
from itertools import product
import unittest

import pandas as pd

from backtesting.portfolio import simulate_portfolio
from backtesting.engine import simulate
from config.settings import BacktestConfig, MONTHS, ScoreConfig
from studio.service import load_dataset

UNIVERSE = tuple('F' + month for month in MONTHS)
SUMMARY = ['Start', 'End', 'Observations', 'Total Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe',
           'Annualized Downside Deviation', 'Sortino', 'Max Drawdown', 'Switches', 'One-way Turnover',
           'Trading Cost NAV Units', 'SPY Total Return', 'BUFR Total Return', 'vs BUFR Cumulative Excess']
DAILY = ['Date', 'Return Fund', 'Fund at Close', 'Strategy NAV', 'SPY NAV', 'BUFR NAV', 'Hold NAV',
         'Held ECR Score', 'Held Buffer Score', 'Entry Buffer Score']
DECISIONS = ['Date', 'Held Fund', 'Held Score', 'Holding Days', 'Triggered', 'Action', 'ECR Signal', 'Buffer Signal']


class EngineParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, _ = load_dataset(ScoreConfig(), UNIVERSE)

    def test_single_fund_paths_agree(self):
        base = BacktestConfig(universe=UNIVERSE, initial_selection='highest_score', comparator='<',
                              cohorts=('JAN', 'JUL'), candidate_audit=False)
        for lag, cost, (trigger, floor), review in product((0, 1), (0., 5.), (('ecr', 1.5), ('metric_floor', .8)),
                                                           ('monthly', 'quarterly')):
            config = replace(base, execution_lag=lag, cost_bps=cost, trigger=trigger, review=review,
                             threshold=floor if trigger == 'ecr' else 1.5,
                             metric_floor=floor if trigger != 'ecr' else .8)
            with self.subTest(lag=lag, cost=cost, trigger=trigger, review=review):
                single, portfolio = simulate(self.data, config), simulate_portfolio(self.data, config)
                pd.testing.assert_frame_equal(single['summary'][SUMMARY], portfolio['summary'][SUMMARY],
                                              check_dtype=False, rtol=0, atol=1e-10)
                pd.testing.assert_frame_equal(single['daily'][DAILY], portfolio['daily'][DAILY],
                                              check_dtype=False, rtol=0, atol=1e-10)
                pd.testing.assert_frame_equal(single['decisions'][DECISIONS].reset_index(drop=True),
                                              portfolio['decisions'][DECISIONS].reset_index(drop=True),
                                              check_dtype=False, rtol=0, atol=1e-10)
                self.assertEqual(single['diagnostics'][['Date', 'Fund', 'Event Type']].values.tolist(),
                                 portfolio['diagnostics'][['Date', 'Fund', 'Event Type']].values.tolist())
