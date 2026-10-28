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
from visualization.conservative_comparison import export_batch_comparison


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
        preflight_cases(self.session,cases)
        self.assertEqual(self.session.score_builds,1)

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
            grid_cases({**self.grid,'turnover_profiles':[{'name':'bad','confirmation_reviews':2}]},self.catalogue,self.available,self.base)
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
            book = pd.ExcelFile(output/'comparison.xlsx')
            self.assertTrue({'Comparison','Configurations','Launch Cohorts','NAV Jan'}.issubset(book.sheet_names))
            nav = pd.read_excel(book,sheet_name='NAV Jan',index_col=0)
            for name,result in results.items():
                self.assertAlmostEqual(nav[name].iloc[-1],result['daily']['Strategy NAV'].iloc[-1],places=11)
            book.close()
            self.assertTrue((output/'index.html').exists())
            self.assertTrue((output/'charts/jan-shortlist.png').exists())
            self.assertTrue((output/'charts/jan-comparison.html').exists())
            self.assertTrue((output/'charts/risk-return.png').exists())

    def test_batch_failure_records_partial_state_without_complete_marker(self):
        from scripts.run_conservative_suite import main
        with tempfile.TemporaryDirectory() as temp:
            with patch('scripts.run_conservative_suite.run_and_archive',side_effect=OSError('disk full')):
                with self.assertRaises(SystemExit) as caught:
                    main(['--archive-root',temp,'--limit','1','--cohorts','JAN','--end-date','2022-12-30'])
            self.assertEqual(caught.exception.code,1)
            output = next(Path(temp).iterdir())
            self.assertEqual(json.loads((output/'batch-status.json').read_text())['state'],'incomplete')
            self.assertFalse((output/'ARCHIVE_COMPLETE.txt').exists())
