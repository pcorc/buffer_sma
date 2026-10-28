"""Selection/cadence semantics and portable frontend artifacts."""
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import unittest
from zipfile import ZipFile

import numpy as np
import pandas as pd

from backtesting.engine import prepare_data, simulate
from backtesting.schedule import review_dates
from config.settings import BacktestConfig, ScoreConfig
from test_portfolio import portfolio_data


def ranking_data():
    metrics, bench, cycles = portfolio_data()
    # Components deliberately disagree: selecting one is not equivalent to ECR.
    components = {'FJAN': (.95, .4, .9), 'FFEB': (.7, 1., .4),
                  'FMAR': (.5, .8, 1.), 'FAPR': (.1, .1, .1)}
    for fund, (dbb, buffer, cap) in components.items():
        mask = metrics.ticker == fund
        metrics.loc[mask, 'downside_before_buffer_net'] = np.log(dbb)/5
        metrics.loc[mask, 'remaining_buffer_net'] = buffer*.1
        metrics.loc[mask, 'remaining_cap_net'] = cap*.1
    metrics.loc[(metrics.ticker == 'FFEB') & (metrics.date > '2021-01-19'), 'remaining_buffer_net'] = .098
    metrics.loc[(metrics.ticker == 'FMAR') & (metrics.date > '2021-01-19'), 'remaining_cap_net'] = .098
    mask = (metrics.ticker == 'FAPR') & (metrics.date > '2021-01-19')
    metrics.loc[mask, 'downside_before_buffer_net'] = np.log(.99)/5
    metrics.loc[mask, ['remaining_buffer_net', 'remaining_cap_net']] = .099
    return prepare_data(metrics, bench, cycles)


class SelectionTests(unittest.TestCase):
    def test_each_ranking_controls_entry_and_replacement_for_single_and_fof(self):
        data = ranking_data()
        for selection, expected in {'highest_ecr': ['FMAR','FFEB'], 'highest_dbb': ['FJAN','FFEB'],
                                    'highest_buffer_integrity': ['FFEB','FMAR'], 'highest_cap_integrity': ['FMAR','FJAN']}.items():
            for n in (1,2):
                with self.subTest(selection=selection,n=n):
                    config = BacktestConfig(portfolio_size=n, selection=selection, initial_selection='highest_score',
                                            review='daily', cohorts=('JAN',), threshold=3, comparator='<')
                    result = simulate(data,config)
                    initial = result['holdings'].loc[result['holdings'].Date == pd.Timestamp('2021-01-19')]
                    self.assertEqual(initial.Fund.tolist(),expected[:n])
                    switches = result['diagnostics'].query('`Event Type` == "switch"')
                    self.assertIn('FAPR',switches.Fund.tolist())
                    self.assertEqual(set(switches.Selection),{selection})
                    if selection != 'highest_ecr':
                        self.assertAlmostEqual(switches.loc[switches.Fund=='FAPR','Selection Score at Signal'].iloc[0],.99)
                    # Rank-one candidate audit uses the selected component, not ECR.
                    audit = result['candidates'].loc[(result['candidates'].Date == pd.Timestamp('2021-01-20')) & (result['candidates'].Rank == 1)]
                    expected_top = 'FAPR'
                    self.assertEqual(audit.Fund.iloc[0],expected_top)

    def test_component_selection_does_not_change_independent_buffer_trigger(self):
        data = ranking_data()
        config = BacktestConfig(selection='highest_dbb', initial_selection='highest_score', review='daily',
                                cohorts=('JAN',), trigger='metric_floor', buffer_metric='buffer_integrity',
                                metric_floor=.4, comparator='<')
        equal = simulate(data, config)
        self.assertEqual(equal['summary'].Switches.iloc[0],0)
        below = simulate(data,replace(config,metric_floor=.41))
        self.assertEqual(below['diagnostics'].Fund.tolist(),['FJAN','FAPR'])


class CadenceTests(unittest.TestCase):
    def test_calendar_months_and_holiday_adjusted_dates(self):
        dates = pd.bdate_range('2021-01-01','2022-12-30')
        cycles = pd.DatetimeIndex([pd.Timestamp(year,month,1)+pd.offsets.WeekOfMonth(week=2,weekday=4)
                                  for year in (2021,2022) for month in range(1,13)])
        for cadence, months in [('monthly',set(range(1,13))),('quarterly',{3,6,9,12}),('semiannual',{6,12}),('annual',{12})]:
            actual = review_dates(dates,cycles,0,len(dates),cadence)
            self.assertEqual(actual,set(cycles[cycles.month.isin(months)]))
            self.assertEqual(len(actual),len(months)*2)
        self.assertEqual(review_dates(dates,cycles,0,len(dates),'daily'),set(dates))

    def test_no_observation_or_missing_review_month_fails(self):
        dates = pd.bdate_range('2021-01-01','2021-12-31')
        cycles = pd.to_datetime(['2021-03-19','2021-06-18','2021-09-17','2021-12-17'])
        with self.assertRaisesRegex(ValueError,'does not cover'):
            review_dates(dates,cycles[:-1],0,len(dates),'annual')
        missing = dates[dates != pd.Timestamp('2021-06-18')]
        with self.assertRaisesRegex(ValueError,'No observation'):
            review_dates(missing,cycles,0,len(missing),'quarterly')

    def test_both_engines_obey_cadence(self):
        from studio.service import load_dataset
        data, _ = load_dataset(ScoreConfig())
        for n in (1,3):
            for cadence, months in [('quarterly',{3,6,9,12}),('semiannual',{6,12}),('annual',{12})]:
                result = simulate(data,BacktestConfig(portfolio_size=n, initial_selection='highest_score',
                                  cohorts=('JAN',),review=cadence,end_date='2021-12-31'))
                self.assertEqual(set(result['decisions'].Date.dt.month),months)


try:
    import streamlit
    import plotly
    HAS_APP = True
except ImportError:
    HAS_APP = False


@unittest.skipUnless(HAS_APP, 'Install requirements-app.txt to run application tests')
class StudioTests(unittest.TestCase):
    def test_downloads_have_workbook_config_graphs_and_provenance(self):
        from studio.service import build_downloads, config_json
        config = BacktestConfig(cohorts=('JAN',),review='daily',initial_selection='highest_score',
                                selection='highest_dbb',trigger='metric_floor',metric_floor=.41,comparator='<')
        result = simulate(ranking_data(),config)
        downloads = build_downloads(result)
        with ZipFile(BytesIO(downloads['zip'])) as archive:
            names = archive.namelist()
            for name in ['ecr_results.xlsx','config.json','manifest.json','charts/jan-report.png',
                         'charts/jan-performance.html','charts/jan-holdings.html','charts/plotly.min.js']:
                self.assertIn(name,names)
            manifest = json.loads(archive.read('manifest.json'))
            self.assertEqual(manifest['backtest']['selection'],'highest_dbb')
            self.assertEqual(archive.read('ecr_results.xlsx'),downloads['excel'])
        book = pd.ExcelFile(BytesIO(downloads['excel']))
        self.assertTrue({'Summary','Daily NAV','ECR Diagnostics','Review Decisions','Candidate Audit','Portfolio Holdings','Run Manifest'}.issubset(book.sheet_names))
        raw = json.loads(config_json(ScoreConfig(),config))
        raw['backtest']['cohorts'] = tuple(raw['backtest']['cohorts'])
        self.assertEqual(BacktestConfig(**raw['backtest']),config)

    def test_app_presets_invalid_window_and_successful_run(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'), default_timeout=60).run()
        self.assertFalse(app.exception)
        app.selectbox(key='preset').select('UCITS Conservative').run()
        self.assertEqual(app.selectbox(key='structure').value,'Fund of funds')
        self.assertEqual(app.number_input(key='fund_count').value,3)
        self.assertEqual(app.number_input(key='component_threshold').value,.8)
        app.selectbox(key='selection').select('highest_dbb').run()
        app.selectbox(key='cadence').select('quarterly').run()
        app.multiselect(key='cohorts').set_value([]).run()
        self.assertTrue(app.button[0].disabled)
        app.multiselect(key='cohorts').set_value(['JAN']).run()
        app.button[0].click().run(timeout=120)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['view'],'Results')
        result = app.session_state['result']
        self.assertEqual(result['parameters']['backtest']['selection'],'highest_dbb')
        self.assertEqual(result['parameters']['backtest']['portfolio_size'],3)
        self.assertEqual(len(app.metric),4)
        # Configuration survives navigating away and back; result is immutable.
        app.segmented_control(key='view').set_value('Configure').run()
        self.assertEqual(app.selectbox(key='selection').value,'highest_dbb')
        self.assertEqual(app.number_input(key='fund_count').value,3)
        app.selectbox(key='preset').select('Super Buffer').run()
        self.assertEqual(app.selectbox(key='trigger_metric').value,'ecr')
        self.assertEqual(app.selectbox(key='ecr_threshold').value,1.5)
        self.assertEqual(app.session_state['result']['parameters']['backtest']['portfolio_size'],3)

    def test_conservative_preset_and_weight_controls_export(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=60).run()
        app.selectbox(key='preset').select('UCITS Conservative').run()
        app.number_input(key='score_gap').set_value(.1).run()
        app.number_input(key='turnover_budget').set_value(40.).run()
        app.checkbox(key='use_bands').check().run()
        self.assertFalse(app.exception)
        self.assertFalse({'holding_days', 'confirmations', 'entry_margin', 'cooldown'} & {w.key for w in app.number_input})
        self.assertAlmostEqual(app.number_input(key='score_gap').value,.1)
        self.assertAlmostEqual(app.number_input(key='turnover_budget').value,40.)
        self.assertTrue(app.checkbox(key='use_bands').value)
        app.number_input(key='band_pp').set_value(1.).run()
        app.button[0].click().run(timeout=120)
        self.assertFalse(app.exception)
        saved=app.session_state['result']['parameters']['backtest']
        self.assertEqual(saved['weight_band'],.01)
        self.assertEqual(saved['entry_metric_margin'],0.)
        self.assertEqual(saved['min_holding_days'],0)
        self.assertEqual(saved['reentry_cooldown_days'],0)
        self.assertEqual(saved['confirmation_reviews'],1)
        book=pd.ExcelFile(BytesIO(app.session_state['downloads']['excel']))
        self.assertIn('Executed Trades',book.sheet_names)
        self.assertIn('Weight Reviews',book.sheet_names)
        app.segmented_control(key='view').set_value('Configure').run()
        app.selectbox(key='preset').select('Super Buffer').run()
        self.assertFalse(app.checkbox(key='use_bands').value)
        self.assertEqual(app.number_input(key='turnover_budget').value,100.)
