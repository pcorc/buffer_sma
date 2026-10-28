from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from backtesting.regimes import Regime, evaluate_windows, load_export, nav_metrics, simple_configurations


def config_rows():
    return pd.DataFrame([
        {'Experiment':'simple','N':3,'Review':'monthly','Trigger':'buffer_integrity','Threshold':.8,
         'Universe':'FJAN, FFEB, FMAR','Score improvement':0.,'Max replacements':0,'Turnover budget':1.,'Weight band':np.nan},
        {'Experiment':'limited','N':3,'Review':'monthly','Trigger':'buffer_integrity','Threshold':.8,
         'Universe':'FJAN, FFEB, FMAR','Score improvement':0.,'Max replacements':1,'Turnover budget':.4,'Weight band':.05},
    ])


class RegimeTests(unittest.TestCase):
    def setUp(self):
        self.dates=pd.bdate_range('2021-12-31',periods=6)
        self.nav=pd.Series([100.,120.,90.,110.,110.,130.],index=self.dates)

    def test_period_statistics_reset_only_the_evaluation_peak(self):
        frames={'Jan':self.nav.to_frame('simple'),'Feb':(self.nav*2).to_frame('simple')}
        periods=[Regime('Decline','2022-01-03','2022-01-06'),Regime('Recovery','2022-01-04','2022-01-07')]
        details,summary,coverage=evaluate_windows(frames,periods)
        self.assertEqual(len(details),4)
        self.assertTrue(coverage.Status.eq('Covered').all())
        decline=details.loc[details.Regime.eq('Decline')]
        np.testing.assert_allclose(decline['Max Drawdown'],-.25)
        np.testing.assert_allclose(decline['Total Return'],110/120-1)
        recovery=details.loc[details.Regime.eq('Recovery')]
        np.testing.assert_allclose(recovery['Max Drawdown'],0)
        np.testing.assert_allclose(recovery['Total Return'],130/90-1)
        self.assertAlmostEqual(nav_metrics(self.nav)['Max Drawdown'],-.25)
        self.assertTrue(summary['Launch Tests'].eq(2).all())

    def test_missing_boundary_is_disclosed_not_shifted(self):
        periods=[Regime('Available','2022-01-03','2022-01-06'),Regime('Too early','2021-12-30','2022-01-06')]
        details,_,coverage=evaluate_windows({'Jan':self.nav.to_frame('simple')},periods)
        self.assertEqual(details.Regime.tolist(),['Available'])
        self.assertTrue(coverage.loc[coverage.Regime.eq('Too early'),'Status'].iloc[0].startswith('Unavailable'))
        with self.assertRaisesRegex(ValueError,'No complete'):
            evaluate_windows({'Jan':self.nav.to_frame('simple')},periods[1:])

    def test_simple_filter_uses_parameters_not_scenario_names(self):
        configs=config_rows()
        self.assertEqual(simple_configurations(configs).Experiment.tolist(),['simple'])
        configs.loc[0,'Review']='quarterly'
        with self.assertRaisesRegex(ValueError,'No simple'):simple_configurations(configs)
        self.assertEqual(len(simple_configurations(configs,'all')),1)

    def test_export_audit_checks_all_cases_before_filtering(self):
        configs=config_rows()
        frame=pd.DataFrame({'simple':self.nav,'limited':self.nav*2})
        summary=pd.DataFrame([{'Experiment':name,'Cohort':'Jan','Start':frame.index[0],'End':frame.index[-1],**nav_metrics(frame[name])} for name in frame])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'comparison.xlsx'
            def write():
                with pd.ExcelWriter(path,engine='openpyxl') as w:
                    configs.to_excel(w,sheet_name='Configurations',index=False)
                    summary.to_excel(w,sheet_name='Launch Cohorts',index=False)
                    summary.to_excel(w,sheet_name='Comparison',index=False)
                    frame.to_excel(w,sheet_name='NAV Jan')
            write()
            selected,curves,audit,_=load_export(path)
            self.assertEqual(selected.Experiment.tolist(),['simple'])
            self.assertEqual(curves['Jan'].columns.tolist(),['simple'])
            self.assertEqual(len(audit),2)
            summary.loc[summary.Experiment.eq('limited'),'Max Drawdown']=-.1
            write()
            with self.assertRaisesRegex(ValueError,'does not reconcile'):load_export(path)
            summary.loc[summary.Experiment.eq('limited'),'Max Drawdown']=-.25
            summary.loc[0,'Start']='2022-01-03'
            write()
            with self.assertRaisesRegex(ValueError,'reported dates'):load_export(path)

    def test_invalid_navs_are_never_filled(self):
        for bad in ([100.,np.nan,110.],[100.,0.,110.],[100.,np.inf,110.]):
            with self.assertRaisesRegex(ValueError,'finite positive'):nav_metrics(pd.Series(bad))
