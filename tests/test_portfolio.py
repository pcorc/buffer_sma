from dataclasses import replace
import json
import unittest
import numpy as np
import pandas as pd
from config.settings import BacktestConfig
from backtesting.engine import prepare_data, simulate
from test_ecr import ROOT


def portfolio_data(both=False):
    dates = pd.to_datetime(['2021-01-19','2021-01-20','2021-01-21','2021-01-22','2021-01-25'])
    rows=[]
    prices={'FJAN':[100,110,121,121,121], 'FFEB':[100,120,144,144,144],
            'FMAR':[100,150,180,180,180], 'FAPR':[100,100,100,100,100]}
    buffers={'FJAN':[.9,.4,.4,.4,.4], 'FFEB':[.8,.3 if both else .8,.8,.8,.8],
             'FMAR':[.7,.95,.95,.95,.95], 'FAPR':[.6,.9 if both else .6,.9 if both else .6,.9,.9]}
    for fund in prices:
        for i,date in enumerate(dates):
            rows.append({'ticker':fund,'date':date,'fund_value':float(prices[fund][i]),
                         'reference_asset_return':0.,'remaining_outcome_days':180.,
                         'remaining_cap_net':.1,'remaining_buffer_net':buffers[fund][i]*.1,
                         'downside_before_buffer_net':0.,'fund_cap_net':.1,'buffer_net':.1})
    metrics=pd.DataFrame(rows)
    bench=pd.DataFrame({'SPY':100.,'BUFR':100.},index=dates)
    cycles=pd.to_datetime(['2021-01-15','2021-01-22'])
    return metrics,bench,cycles


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.config=BacktestConfig(portfolio_size=2,initial_selection='highest_score',cohorts=('JAN',),
                                  review='daily',trigger='metric_floor',buffer_metric='buffer_integrity',
                                  metric_floor=.75,comparator='<')

    def test_only_breached_sleeve_changes_and_weights_drift(self):
        r=simulate(prepare_data(*portfolio_data()),self.config)
        initial=r['holdings'].loc[lambda x: x.Date == pd.Timestamp('2021-01-19')]
        self.assertEqual(initial.Fund.tolist(),['FJAN','FFEB'])
        np.testing.assert_allclose(initial.Weight,[.5,.5])
        switched=r['holdings'].loc[lambda x: x.Date == pd.Timestamp('2021-01-20')]
        self.assertEqual(switched.Fund.tolist(),['FMAR','FFEB'])
        np.testing.assert_allclose(switched.Capital,[55,60])
        np.testing.assert_allclose(r['daily']['Strategy NAV'],[100,115,138,138,138])
        self.assertEqual(r['summary'].Switches.iloc[0],1)
        self.assertEqual(r['diagnostics']['Event Type'].tolist(),['initial','initial','switch'])
        sums=r['holdings'].groupby('Date').Capital.sum()
        np.testing.assert_allclose(sums,r['daily']['Strategy NAV'])
        self.assertTrue((r['holdings'].groupby('Date').Fund.nunique()==2).all())

    def test_cost_only_applies_to_replaced_capital(self):
        r=simulate(prepare_data(*portfolio_data()),replace(self.config,cost_bps=100))
        self.assertAlmostEqual(r['daily']['Strategy NAV'].iloc[1],114.45)
        self.assertAlmostEqual(r['daily']['Strategy NAV'].iloc[2],137.34)
        self.assertAlmostEqual(r['summary']['Trading Cost NAV Units'].iloc[0],.55)

    def test_initial_next_close_requires_prior_history(self):
        with self.assertRaisesRegex(ValueError, 'prior session'):
            simulate(prepare_data(*portfolio_data()),replace(self.config,execution_lag=1))

    def test_next_close_with_history(self):
        metrics,bench,cycles=portfolio_data()
        prior=metrics.loc[metrics.date=='2021-01-19'].copy();prior['date']=pd.Timestamp('2021-01-15')
        metrics=pd.concat([prior,metrics],ignore_index=True)
        bench.loc[pd.Timestamp('2021-01-15')]=100.;bench=bench.sort_index()
        r=simulate(prepare_data(metrics,bench,cycles),replace(self.config,execution_lag=1,cost_bps=100))
        self.assertEqual(r['holdings'].loc[lambda x: x.Date == pd.Timestamp('2021-01-20')].Fund.tolist(),['FJAN','FFEB'])
        self.assertAlmostEqual(r['daily']['Strategy NAV'].iloc[2],131.895)
        event=r['diagnostics'].iloc[-1]
        self.assertEqual(event['Signal Date'],pd.Timestamp('2021-01-20'))
        self.assertEqual(event['Execution Date'],pd.Timestamp('2021-01-21'))

    def test_simultaneous_triggers_cannot_duplicate_funds(self):
        r=simulate(prepare_data(*portfolio_data(both=True)),self.config)
        selected=r['holdings'].loc[lambda x: x.Date == pd.Timestamp('2021-01-20')]
        self.assertEqual(set(selected.Fund),{'FMAR','FAPR'})
        self.assertEqual(r['summary'].Switches.iloc[0],2)
        self.assertEqual(len(r['diagnostics']),4)

    def test_no_spurious_trades_when_universe_is_fully_held(self):
        r=simulate(prepare_data(*portfolio_data()),replace(self.config,portfolio_size=4))
        self.assertEqual(r['summary'].Switches.iloc[0],0)
        self.assertEqual(len(r['diagnostics']),4)

    def test_insufficient_universe_fails_instead_of_duplication(self):
        with self.assertRaisesRegex(ValueError,'only 4 eligible'):
            simulate(prepare_data(*portfolio_data()),replace(self.config,portfolio_size=5))
        for value in [0,13,True,2.5]:
            with self.assertRaises(ValueError):replace(self.config,portfolio_size=value)

    def test_minimum_hold_is_per_position(self):
        r=simulate(prepare_data(*portfolio_data()),replace(self.config,min_holding_days=2))
        self.assertEqual(r['diagnostics'].iloc[-1].Date,pd.Timestamp('2021-01-21'))

    def test_future_data_cannot_change_earlier_portfolio(self):
        metrics,bench,cycles=portfolio_data()
        a=simulate(prepare_data(metrics,bench,cycles),self.config)
        metrics.loc[metrics.date >= '2021-01-22','fund_value'] *= 2
        b=simulate(prepare_data(metrics,bench,cycles),self.config)
        for key in ['daily','holdings']:
            cols=[c for c in a[key] if c!='Strategy ID']
            pd.testing.assert_frame_equal(a[key].loc[a[key].Date<'2021-01-22',cols],
                                          b[key].loc[b[key].Date<'2021-01-22',cols])

    def test_client_presets_keep_trigger_and_selection_independent(self):
        super_config=json.loads((ROOT/'config/super-buffer.json').read_text())['backtest']
        ucits=json.loads((ROOT/'config/ucit-conservative.json').read_text())['backtest']
        self.assertEqual((super_config['trigger'],super_config['threshold'],super_config['comparator'],super_config['portfolio_size']),('ecr',1.5,'<',1))
        self.assertEqual((ucits['trigger'],ucits['buffer_metric']),('metric_floor','buffer_integrity'))
        for config in (super_config,ucits):self.assertEqual(config['selection'],'highest_ecr')
