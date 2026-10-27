from dataclasses import replace
from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd

from backtesting.engine import prepare_data, simulate
from backtesting.data_pipeline import load_metric_collection
from config.settings import BacktestConfig, ScoreConfig
from core.universe import load_catalog, filter_catalog, validate_universe
from studio.archive import save_archive, DEFAULT_ARCHIVE_ROOT
from test_portfolio import portfolio_data


def fixture_downloads():
    data=BytesIO()
    with ZipFile(data,'w') as z:
        z.writestr('ecr_results.xlsx',b'test-workbook')
        z.writestr('charts/jan-report.png',b'test-image')
        z.writestr('manifest.json','{}')
    return {'zip':data.getvalue()}


class UniverseTests(unittest.TestCase):
    def test_filters_use_metadata_and_have_one_compact_buffer_category(self):
        cat=load_catalog()
        self.assertEqual(len(filter_catalog(cat)),12)
        self.assertTrue(filter_catalog(cat,group='D series').empty)
        self.assertTrue(filter_catalog(cat,group='Quarterly structures',outcome='Quarterly').empty)
        self.assertEqual(filter_catalog(cat,months=['JAN','FEB'],references=['SPY'],buffer_band='Up to 10%').ticker.tolist(),['FJAN','FFEB'])
        self.assertTrue(filter_catalog(cat,buffer_band='Deferred protection').empty)
        self.assertTrue(filter_catalog(cat,strategies=['Deep Buffer']).empty)

    def test_custom_universe_need_not_contain_launch_month_fund(self):
        metrics,bench,cycles=portfolio_data()
        # A January launch selects the best available fund; it doesn't imply FJAN.
        chosen=('FFEB','FMAR')
        data=prepare_data(metrics,bench,cycles,universe=chosen)
        for n in (1,2):
            for confirmations in (1,2):
                config=BacktestConfig(universe=chosen,portfolio_size=n,cohorts=('JAN',),initial_selection='highest_score',
                                      review='daily',confirmation_reviews=confirmations)
                result=simulate(data,config)
                self.assertTrue(set(result['holdings'].Fund)<=set(chosen))
                self.assertEqual(result['daily']['Hold NAV'].iloc[0],100)
                self.assertAlmostEqual(result['daily']['Hold NAV'].iloc[1],120 if n==1 else 135)
                self.assertEqual(result['summary'].Start.iloc[0],pd.Timestamp('2021-01-19'))

    def test_non_f_tickers_can_be_prepared_and_selected_explicitly(self):
        metrics,bench,cycles=portfolio_data()
        mapping={'FJAN':'DABC','FFEB':'GABC','FMAR':'UCIT','FAPR':'QTR'}
        metrics.ticker=metrics.ticker.map(mapping)
        chosen=('DABC','GABC','UCIT','QTR')
        data=prepare_data(metrics,bench,cycles,universe=chosen)
        result=simulate(data,BacktestConfig(universe=chosen,portfolio_size=3,initial_selection='highest_score',cohorts=('JAN',),review='daily'))
        self.assertEqual(set(result['holdings'].Fund)-set(chosen),set())
        self.assertTrue((result['holdings'].groupby('Date').Fund.nunique()==3).all())

    def test_wrong_prepared_universe_or_missing_history_is_rejected(self):
        data=prepare_data(*portfolio_data())
        with self.assertRaisesRegex(ValueError,'does not match'):
            simulate(data,BacktestConfig(universe=('FJAN',)))
        with self.assertRaisesRegex(ValueError,'No history'):
            prepare_data(*portfolio_data(),universe=('DABC',))
        for value in [(),('FJAN','FJAN'),'FJAN',('fjan',)]:
            with self.assertRaises(ValueError):BacktestConfig(universe=value)
        with self.assertRaisesRegex(ValueError,'not loaded'):
            validate_universe(load_catalog(),['FJAN'],[])

    def test_metadata_drives_future_groups_without_ticker_prefix_guessing(self):
        cat=load_catalog().iloc[:2].copy()
        cat.loc[cat.index[0],['ticker','series','group','outcome_type','outcome_month']]=['X123','D','UCITS','Quarterly','Quarterly']
        self.assertEqual(filter_catalog(cat,group='UCITS',outcome='Quarterly').ticker.tolist(),['X123'])
        self.assertEqual(filter_catalog(cat,group='D series',outcome='Quarterly').ticker.tolist(),['X123'])
        with self.assertRaisesRegex(ValueError,'separate runs'):
            validate_universe(cat,cat.ticker.tolist(),cat.ticker.tolist())
        cat['currency']='GBP'
        with self.assertRaisesRegex(ValueError,'Currency conversion'):
            validate_universe(cat,['X123'],['X123'])

    def test_supplemental_metric_files_are_validated_together(self):
        frame=portfolio_data()[0]
        with tempfile.TemporaryDirectory() as temp:
            a,b=Path(temp)/'a.csv',Path(temp)/'b.csv'
            frame.loc[frame.ticker=='FJAN'].to_csv(a,index=False)
            frame.loc[frame.ticker!='FJAN'].to_csv(b,index=False)
            combined=load_metric_collection([a,b])
            self.assertEqual(len(combined),len(frame))
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                load_metric_collection([a,a])


class ArchiveTests(unittest.TestCase):
    def test_each_run_has_its_own_complete_folder_and_zip(self):
        data=fixture_downloads()
        with tempfile.TemporaryDirectory() as temp:
            first=save_archive(data,temp,'UCITS / research:*','abcdef1234')
            second=save_archive(data,temp,'UCITS / research:*','abcdef1234')
            self.assertNotEqual(first,second)
            for folder in (first,second):
                self.assertEqual((folder/'results.zip').read_bytes(),data['zip'])
                self.assertTrue((folder/'ecr_results.xlsx').exists())
                self.assertTrue((folder/'charts/jan-report.png').exists())
                self.assertTrue((folder/'ARCHIVE_COMPLETE.txt').exists())

    def test_unavailable_archive_never_claims_success(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch('studio.archive.Path.write_bytes',side_effect=PermissionError('read-only')):
                with self.assertRaises(PermissionError):save_archive(fixture_downloads(),temp,'Run','abc')
            self.assertEqual(list(Path(temp).iterdir()),[])
        with self.assertRaises(OSError):save_archive(fixture_downloads(),'relative/folder','Run','abc')
        if os.name!='nt':
            with self.assertRaisesRegex(OSError,'Windows shared drive'):
                save_archive(fixture_downloads(),DEFAULT_ARCHIVE_ROOT,'Run','abc')

    def test_zip_cannot_write_outside_run_folder(self):
        data=BytesIO()
        with ZipFile(data,'w') as z:z.writestr('../outside.txt','no')
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(OSError):save_archive({'zip':data.getvalue()},temp,'Run','abc')
            self.assertEqual(list(Path(temp).iterdir()),[])


try:
    from streamlit.testing.v1 import AppTest
    APP_AVAILABLE=True
except ImportError:
    APP_AVAILABLE=False


@unittest.skipUnless(APP_AVAILABLE,'Install requirements-app.txt')
class StudioUniverseTests(unittest.TestCase):
    def test_pending_groups_profiles_custom_selection_and_archive(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.number_input(key='cost').value,0.)
        self.assertEqual(app.number_input(key='holding_days').value,0)
        self.assertEqual(app.selectbox(key='lag').value,1)
        app.selectbox(key='universe_group').select('Quarterly structures').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.number_input(key='score_period_center').value,45.)
        self.assertEqual(app.number_input(key='score_period_width').value,18.75)
        self.assertEqual(app.number_input(key='score_sigma_down').value,.07)
        self.assertTrue(app.button[0].disabled)
        app.selectbox(key='universe_group').select('D series').run()
        self.assertTrue(app.button[0].disabled)
        app.selectbox(key='universe_group').select('Custom universe').run()
        app.multiselect(key='selected_tickers').set_value(['FFEB','FMAR']).run()
        self.assertFalse(app.button[0].disabled)
        app.selectbox(key='cadence').select('quarterly').run()
        with tempfile.TemporaryDirectory() as temp:
            app.text_input(key='archive_root').set_value(temp).run()
            app.button[0].click().run(timeout=120)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['view'],'Results')
            result=app.session_state['result']
            self.assertEqual(set(result['holdings'].Fund)-{'FFEB','FMAR'},set())
            self.assertEqual(set(result['decisions'].Date.dt.month),{3,6,9,12})
            folder=Path(app.session_state['archive_path'])
            self.assertTrue((folder/'results.zip').exists())
            self.assertTrue((folder/'universe.csv').exists())
            self.assertTrue(list((folder/'charts').glob('*.png')))
            self.assertFalse(app.session_state['archive_error'])
            book=pd.ExcelFile(BytesIO(app.session_state['downloads']['excel']))
            self.assertIn('Fund Universe',book.sheet_names)
            app.segmented_control(key='view').set_value('Configure').run()
            self.assertEqual(set(app.multiselect(key='selected_tickers').value),{'FFEB','FMAR'})
