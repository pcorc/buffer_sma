from dataclasses import replace
import unittest
import numpy as np
import pandas as pd

from backtesting.engine import prepare_data, simulate
from backtesting.scenarios import conservative_suite, quarterly_score_hypotheses
from config.settings import BacktestConfig
from core.scoring import score_funds
from core.turnover import controlled_plan
from test_portfolio import portfolio_data


def with_prior(inputs):
    metrics,bench,cycles=inputs
    prior=metrics.loc[metrics.date=='2021-01-19'].copy()
    prior['date']=pd.Timestamp('2021-01-15')
    bench=bench.copy();bench.loc[pd.Timestamp('2021-01-15')]=100.
    return pd.concat([prior,metrics],ignore_index=True),bench.sort_index(),cycles


class TurnoverTests(unittest.TestCase):
    def setUp(self):
        self.config=BacktestConfig(portfolio_size=2,initial_selection='highest_score',cohorts=('JAN',),
                                  review='daily',trigger='metric_floor',metric_floor=.75,comparator='<')

    def test_confirmation_and_recovery_reset(self):
        result=simulate(prepare_data(*portfolio_data()),replace(self.config,confirmation_reviews=2))
        self.assertEqual(result['diagnostics'].query('`Event Type` == "switch"').Date.iloc[0],pd.Timestamp('2021-01-21'))
        self.assertIn('awaiting_confirmation',result['decisions']['Control Reason'].tolist())
        metrics,bench,cycles=portfolio_data()
        metrics.loc[(metrics.ticker=='FJAN') & (metrics.date=='2021-01-21'),'remaining_buffer_net']=.09
        result=simulate(prepare_data(metrics,bench,cycles),replace(self.config,confirmation_reviews=2))
        self.assertEqual(result['diagnostics'].query('`Event Type` == "switch"').Date.iloc[0],pd.Timestamp('2021-01-25'))

    def test_budget_prioritizes_worst_trigger_and_limits_whole_sleeves(self):
        result=simulate(prepare_data(*portfolio_data(both=True)),replace(self.config,max_replacements_per_review=1,max_turnover_per_review=.55))
        changes=result['trades'].loc[result['trades'].Date==pd.Timestamp('2021-01-20')]
        self.assertEqual(changes['From Fund'].tolist(),['FFEB'])
        self.assertEqual(changes.Fund.tolist(),['FMAR'])
        self.assertLessEqual(changes['One-way Turnover'].sum(),.55)
        holdings=result['holdings'].groupby('Date')
        self.assertTrue((holdings.Fund.nunique()==2).all())

    def test_too_small_budget_defers_instead_of_partial_exit(self):
        result=simulate(prepare_data(*portfolio_data()),replace(self.config,max_turnover_per_review=.1))
        self.assertEqual(result['summary'].Switches.iloc[0],0)
        self.assertIn('turnover_budget',result['decisions']['Control Reason'].tolist())

    def test_weight_band_is_optional_and_charges_only_transferred_capital(self):
        data=prepare_data(*portfolio_data())
        config=replace(self.config,metric_floor=0.,cost_bps=100.)
        unbanded=simulate(data,config)
        banded=simulate(data,replace(config,weight_band=.01))
        row=banded['daily'].loc[banded['daily'].Date==pd.Timestamp('2021-01-20')].iloc[0]
        self.assertAlmostEqual(row['Strategy NAV'],115-.025)
        weights=banded['holdings'].loc[banded['holdings'].Date==row.Date].Weight
        np.testing.assert_allclose(weights,[.5,.5])
        self.assertEqual(banded['summary'].Switches.iloc[0],0)
        self.assertGreater(banded['summary']['Weight Rebalances'].iloc[0],0)
        self.assertEqual(unbanded['summary']['Weight Rebalances'].iloc[0],0)
        wide=simulate(data,replace(config,weight_band=.1))
        np.testing.assert_allclose(wide['daily']['Strategy NAV'],unbanded['daily']['Strategy NAV'])
        self.assertTrue((wide['weight_reviews'].Action=='within_band').all())

    def test_wide_weight_band_does_not_change_replacement_assignment(self):
        data=prepare_data(*portfolio_data(both=True))
        a=simulate(data,self.config)
        b=simulate(data,replace(self.config,weight_band=1.))
        np.testing.assert_allclose(a['daily']['Strategy NAV'],b['daily']['Strategy NAV'])
        self.assertEqual(a['holdings'].Fund.tolist(),b['holdings'].Fund.tolist())
        self.assertEqual(a['diagnostics'].Fund.tolist(),b['diagnostics'].Fund.tolist())

    def test_next_close_weight_reset_earns_outgoing_returns(self):
        data=prepare_data(*with_prior(portfolio_data()))
        result=simulate(data,replace(self.config,metric_floor=0,weight_band=.01,execution_lag=1,cost_bps=100))
        row=result['daily'].loc[result['daily'].Date==pd.Timestamp('2021-01-21')].iloc[0]
        self.assertAlmostEqual(row['Strategy NAV'],132.5-.0575)
        self.assertAlmostEqual(result['trades']['One-way Turnover'].iloc[0],5.75/132.5)
        self.assertEqual(result['trades']['Signal Date'].iloc[0],pd.Timestamp('2021-01-20'))
        np.testing.assert_allclose(result['holdings'].loc[result['holdings'].Date==row.Date].Weight,[.5,.5])

    def test_execution_rechecks_budget_after_price_drift(self):
        result=simulate(prepare_data(*with_prior(portfolio_data())),replace(self.config,metric_floor=0,
                        weight_band=.01,execution_lag=1,max_turnover_per_review=.03))
        trade=result['trades'].iloc[0]
        self.assertEqual(trade.Action,'deferred_execution_budget')
        self.assertEqual(trade['One-way Turnover'],0.)
        self.assertEqual(result['summary']['Weight Rebalances'].iloc[0],0)

    def test_next_close_replacement_can_be_deferred_by_actual_weight(self):
        metrics,bench,cycles=with_prior(portfolio_data())
        metrics.loc[(metrics.ticker=='FJAN') & (metrics.date=='2021-01-21'),'fund_value']=160.
        result=simulate(prepare_data(metrics,bench,cycles),replace(self.config,execution_lag=1,
                        max_turnover_per_review=.49,end_date='2021-01-21'))
        self.assertEqual(result['summary'].Switches.iloc[0],0)
        self.assertEqual(result['trades'].Action.tolist(),['deferred_execution_budget'])
        self.assertIn('scheduled',result['decisions'].Action.tolist())
        self.assertEqual(result['holdings'].loc[result['holdings'].Date==pd.Timestamp('2021-01-21')].Fund.tolist(),['FJAN','FFEB'])

    def test_repurchase_cooldown_uses_actual_sale_date(self):
        metrics,bench,cycles=portfolio_data()
        metrics=metrics.loc[metrics.ticker.isin(['FJAN','FMAR'])].copy()
        metrics.loc[(metrics.ticker=='FJAN') & (metrics.date>='2021-01-21'),'remaining_buffer_net']=.099
        metrics.loc[(metrics.ticker=='FMAR') & (metrics.date>='2021-01-21'),'remaining_buffer_net']=.04
        data=prepare_data(metrics,bench,cycles)
        free=simulate(data,replace(self.config,portfolio_size=1))
        restricted=simulate(data,replace(self.config,portfolio_size=1,reentry_cooldown_days=5))
        self.assertEqual(free['diagnostics'].Fund.tolist(),['FJAN','FMAR','FJAN'])
        self.assertEqual(restricted['diagnostics'].Fund.tolist(),['FJAN','FMAR'])

    def test_combined_trade_budget_and_accounting(self):
        result=simulate(prepare_data(*portfolio_data(both=True)),replace(self.config,weight_band=.01,max_turnover_per_review=.55,cost_bps=20))
        trades=result['trades']
        self.assertTrue((trades.groupby('Date')['One-way Turnover'].sum()<=.55+1e-12).all())
        self.assertAlmostEqual(trades['Cost NAV Units'].sum(),result['summary']['Trading Cost NAV Units'].iloc[0])
        self.assertAlmostEqual(trades['One-way Turnover'].sum(),result['summary']['One-way Turnover'].iloc[0])
        np.testing.assert_allclose(result['holdings'].groupby('Date').Capital.sum(),result['daily']['Strategy NAV'])

    def test_controls_work_for_single_ticker_too(self):
        result=simulate(prepare_data(*portfolio_data()),replace(self.config,portfolio_size=1,confirmation_reviews=2))
        self.assertEqual(result['diagnostics'].query('`Event Type` == "switch"').Date.iloc[0],pd.Timestamp('2021-01-21'))
        self.assertTrue((result['holdings'].Weight==1).all())

    def test_future_changes_do_not_rewrite_controlled_history(self):
        inputs=portfolio_data(both=True)
        config=replace(self.config,confirmation_reviews=2,weight_band=.02)
        a=simulate(prepare_data(*inputs),config)
        metrics,bench,cycles=inputs
        metrics.loc[metrics.date>='2021-01-22','fund_value']*=3
        b=simulate(prepare_data(metrics,bench,cycles),config)
        for key in ['daily','holdings','trades','weight_reviews']:
            cols=[c for c in a[key] if c!='Strategy ID']
            pd.testing.assert_frame_equal(a[key].loc[a[key].Date<'2021-01-22',cols],b[key].loc[b[key].Date<'2021-01-22',cols])

    def test_entry_margin_cooldown_and_minimum_improvement(self):
        config=replace(self.config,entry_metric_margin=.15,min_selection_improvement=.2,reentry_cooldown_days=5)
        args=dict(held=np.array([0,1]),triggered=[0],ranking=np.array([1.5,2,1.6,2.]),
                  eligible=np.ones(4,dtype=bool),tickers=['A','B','C','D'],metric=np.array([.7,.9,.95,.95]),
                  weights=np.array([.5,.5]),last_exit=np.array([-100,-100,-100,8]),index=10,config=config)
        plan,reasons=controlled_plan(**args)
        self.assertFalse(plan);self.assertEqual(reasons[0],'insufficient_improvement')
        args['index']=13
        plan,_=controlled_plan(**args)
        self.assertEqual(plan,{0:3})
        args['metric'][3]=.89
        plan,_=controlled_plan(**args)
        self.assertFalse(plan)

    def test_config_validation(self):
        for kwargs in [{'confirmation_reviews':0},{'reentry_cooldown_days':-1},{'max_replacements_per_review':True},
                       {'weight_band':-.1},{'weight_band':float('nan')},{'max_turnover_per_review':0},
                       {'max_turnover_per_review':1.1},{'entry_metric_margin':.3}, {'min_selection_improvement':-.1}]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):replace(self.config,**kwargs)
        with self.assertRaises(ValueError):replace(self.config,portfolio_size=1,weight_band=.05)
        with self.assertRaises(ValueError):replace(self.config,trigger='ecr',entry_metric_margin=.1)

    def test_scenarios_are_distinct_and_quarterly_period_scales(self):
        scenarios=conservative_suite()
        self.assertEqual(len(scenarios),35)
        self.assertEqual(len({(e.score,e.backtest) for e in scenarios}),35)
        self.assertTrue(all(e.backtest.selection=='highest_ecr' for e in scenarios))
        self.assertTrue(all(e.backtest.portfolio_size in (3,5) for e in scenarios[1:]))
        frame=portfolio_data()[0].iloc[:3].copy()
        frame['remaining_outcome_days']=[180.,105.,255.]
        annual=score_funds(frame)
        frame['remaining_outcome_days']/=4
        quarter=score_funds(frame,quarterly_score_hypotheses()['quarterly_same_return_tolerance'])
        np.testing.assert_allclose(annual.period_position,quarter.period_position)
