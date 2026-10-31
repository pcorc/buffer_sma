from argparse import Namespace
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from backtesting.conservative_runner import (DEFAULT_GRID, base_config, grid_cases, make_case,
                                             preflight_cases, research_session, resolve_universe)
from config.settings import ScoreConfig
from core.universe import load_catalog
from visualization.conservative_comparison import export_batch_comparison, top_five_tables


class ConservativeRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = research_session()
        cls.catalogue = load_catalog()
        cls.base = base_config(Namespace(start_date='2021-01-19',end_date='2022-12-30',cohorts=('JAN',),cost_bps=0.))
        cls.grid = json.loads(DEFAULT_GRID.read_text())
        cls.available = cls.session.metrics.ticker.unique()

    def test_grid_is_unique_and_removed_controls_are_neutral(self):
        cases = grid_cases(self.grid,self.catalogue,self.available,self.base)
        self.assertEqual(len(cases),85)
        self.assertEqual(len({c.experiment.name for c in cases}),85)
        self.assertEqual(cases[0].experiment.backtest.portfolio_size,1)
        self.assertEqual(cases[0].experiment.backtest.threshold,1.5)
        for case in cases:
            config = case.experiment.backtest
            self.assertEqual(case.experiment.score,ScoreConfig())
            self.assertEqual(config.selection,'highest_ecr')
            self.assertEqual(config.comparator,'<')
            self.assertEqual(config.execution_lag,1)
            self.assertEqual(config.min_holding_days,0)
            self.assertEqual(config.reentry_cooldown_days,0)
            self.assertEqual(config.confirmation_reviews,1)
            self.assertEqual(config.entry_metric_margin,0)
        # The session is shared process-wide; all 85 cases must reuse a single score panel.
        before = self.session.score_builds
        preflight_cases(self.session,cases)
        self.assertLessEqual(self.session.score_builds-before,1)

    def test_optional_dimensions_cover_every_setting(self):
        grid = {'universes':[{'name':'F annual','group':'F series','outcome':'Annual'}],'portfolio_sizes':[3],
                'review_cadences':['monthly'],'triggers':{'buffer_integrity':[.8]},
                'trigger_drops':{'cap_integrity':[.25]},
                'turnover_profiles':[{'name':'Held','min_holding_days':21,'confirmation_reviews':2,'reentry_cooldown_days':5}],
                'score_profiles':[{'name':'Standard ECR'},{'name':'Defensive','dbb_weight':1.8,'reset_weight':.9}],
                'selections':['highest_ecr','highest_dbb'],'execution_lags':[1],'costs_bps':[0.,5.],'comparators':['<'],
                'include_single_ticker_reference':False}
        cases = grid_cases(grid,self.catalogue,self.available,self.base)
        self.assertEqual(len(cases),2*2*2*2)  # triggers × score profiles × selections × costs
        names = [c.experiment.name for c in cases]
        self.assertIn('F annual / N=3 / Cap drop 25% / monthly / Held / Defensive / select DBB / 5 bps',names)
        case = cases[names.index('F annual / N=3 / Cap drop 25% / monthly / Held / Defensive / select DBB / 5 bps')]
        config = case.experiment.backtest
        self.assertEqual((config.trigger,config.buffer_metric,config.buffer_drop),('buffer_drop','cap_integrity',.25))
        self.assertEqual((config.selection,config.cost_bps,config.min_holding_days,config.confirmation_reviews,
                          config.reentry_cooldown_days),('highest_dbb',5.,21,2,5))
        self.assertEqual((case.experiment.score.dbb_weight,case.experiment.score.reset_weight),(1.8,.9))
        with self.assertRaisesRegex(ValueError,'Score profiles'):
            grid_cases({**grid,'score_profiles':[{'name':'x','dbb_wieght':2}]},self.catalogue,self.available,self.base)

    def test_filters_and_grid_reject_missing_data_typos_and_duplicates(self):
        chosen,_ = resolve_universe({'months':['JAN','FEB','MAR'],'references':['SPY']},self.catalogue,self.available)
        self.assertEqual(chosen,('FFEB','FJAN','FMAR'))
        with self.assertRaises(ValueError):
            resolve_universe({'group':'UCITS'},self.catalogue,self.available)
        with self.assertRaisesRegex(ValueError,'Unknown universe'):
            resolve_universe({'month':['JAN']},self.catalogue,self.available)
        with self.assertRaisesRegex(ValueError,'match the universe'):
            resolve_universe({'months':['JAN'],'tickers':['FFEB']},self.catalogue,self.available)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            grid_cases({**self.grid,'portfolio_sizes':[3,3]},self.catalogue,self.available,self.base)
        with self.assertRaisesRegex(ValueError,'Unknown profile'):
            grid_cases({**self.grid,'turnover_profiles':[{'name':'bad','cooldown':2}]},self.catalogue,self.available,self.base)
        with self.assertRaisesRegex(ValueError,'equivalent'):
            # Same economics under two names: B states the default budget explicitly.
            grid_cases({**self.grid,'turnover_profiles':[{'name':'A'},{'name':'B','max_turnover_per_review':1.}]},
                       self.catalogue,self.available,self.base)
        with self.assertRaisesRegex(ValueError,'only 3 funds'):
            make_case('too many',{'months':['JAN','FEB','MAR']},self.catalogue,self.available,self.base,n=5)
        with self.assertRaisesRegex(ValueError,'budget for N=4'):
            grid_cases({**self.grid,'portfolio_sizes':[4]},self.catalogue,self.available,self.base)

    def test_comparisons_use_exact_windows_and_export_every_iteration(self):
        cases = [make_case(f'Buffer {floor}',{},self.catalogue,self.available,self.base,floor=floor) for floor in (.7,.9)]
        preflight_cases(self.session,cases)
        wrong = replace(cases[1],experiment=replace(cases[1].experiment,backtest=replace(cases[1].experiment.backtest,start_date='2022-01-01')))
        with self.assertRaisesRegex(ValueError,'identical launch'):
            preflight_cases(self.session,[cases[0],wrong])
        results = {c.experiment.name:self.session.run(c.experiment.score,c.experiment.backtest) for c in cases}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            folders = {name:output/'runs'/str(i) for i,name in enumerate(results)}
            table = export_batch_comparison(results,cases,folders,output)
            self.assertEqual(len(table),2)
            self.assertEqual(set(table['Configuration ID']), {'C001', 'C002'})
            book = pd.ExcelFile(output/'comparison.xlsx')
            self.assertTrue({'Comparison','Configurations','Launch Cohorts','NAV Jan'}.issubset(book.sheet_names))
            nav = pd.read_excel(book,sheet_name='NAV Jan',index_col=0)
            for name,result in results.items():
                self.assertAlmostEqual(nav[name].iloc[-1],result['daily']['Strategy NAV'].iloc[-1],places=11)
            book.close()
            self.assertTrue((output/'index.html').exists())
            self.assertEqual(len(list((output/'charts').glob('*.png'))),3)
            self.assertEqual(len(list((output/'charts').glob('*.html'))),3)
            self.assertTrue((output/'charts/top5-sharpe.html').exists())
            self.assertIn('Parameter sets', (output/'charts/top5-drawdown.html').read_text())
            self.assertTrue((output/'top_five.csv').exists())

    def test_top_five_metrics_rank_independently_and_exclude_undefined_values(self):
        import numpy as np
        table = pd.DataFrame({'Mean Total Return':[.1,.2,.3,.4,.5,.6,.7],
                              'Worst Drawdown':[-.01,-.2,-.3,-.4,-.5,-.6,-.7],
                              'Mean Sharpe':[2.,2.,np.nan,np.inf,1.,.5,-.1]},index=list('ABCDEFG'))
        ranked = top_five_tables(table)
        self.assertEqual(ranked['total-return'].index.tolist(),list('GFEDC'))
        self.assertEqual(ranked['drawdown'].index.tolist(),list('ABCDE'))
        self.assertEqual(ranked['sharpe'].index.tolist(),list('ABEFG'))

    def test_no_plots_preserves_excel_and_rankings_without_calling_renderers(self):
        from io import BytesIO
        from zipfile import ZipFile
        from studio.service import build_downloads
        case = make_case('No plots',{},self.catalogue,self.available,self.base)
        result = self.session.run(case.experiment.score,case.experiment.backtest)
        with patch('visualization.performance_plots.plot_results') as overview, \
             patch('visualization.performance_plots.plot_studio_results') as reports, \
             patch('visualization.research_plots.dashboard') as interactive:
            downloads = build_downloads(result,plots=False)
            overview.assert_not_called(); reports.assert_not_called(); interactive.assert_not_called()
        with ZipFile(BytesIO(downloads['zip'])) as archive:
            self.assertIn('ecr_results.xlsx',archive.namelist())
            self.assertFalse(any(n.startswith('charts/') or n.endswith(('.png','.html','.js')) for n in archive.namelist()))
            self.assertFalse(json.loads(archive.read('manifest.json'))['plots_included'])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with patch('visualization.conservative_comparison.render_top_five') as renderer:
                export_batch_comparison({'No plots':result},[case],{'No plots':output/'runs/001'},output,plots='none')
                renderer.assert_not_called()
            self.assertFalse((output/'charts').exists())
            self.assertTrue((output/'comparison.xlsx').exists())
            self.assertTrue((output/'top_five.csv').exists())
            html = (output/'index.html').read_text()
            self.assertIn('All plotting was disabled',html)
            self.assertNotIn('href="charts/',html)

    def test_batch_failure_records_partial_state_without_complete_marker(self):
        from scripts.run_conservative_suite import main
        with tempfile.TemporaryDirectory() as temp:
            with patch('scripts.run_conservative_suite.run_suite_case',side_effect=OSError('disk full')):
                with self.assertRaises(SystemExit) as caught:
                    main(['--archive-root',temp,'--limit','1','--cohorts','JAN','--end-date','2022-12-30'])
            self.assertEqual(caught.exception.code,1)
            output = next(Path(temp).iterdir())
            self.assertEqual(json.loads((output/'batch-status.json').read_text())['state'],'incomplete')
            self.assertFalse((output/'ARCHIVE_COMPLETE.txt').exists())
