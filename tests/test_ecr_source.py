"""The ecr database loads with the engine's exact frame contracts, identity and provenance."""
from dataclasses import replace
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import _ecr_db  # noqa: F401 (pins ECR_AS_OF before any data loads)
from backtesting import ecr_source
from backtesting.data_pipeline import METRIC_COLUMNS
from backtesting.engine import prepare_data
from config.settings import BacktestConfig, MONTHS
from config.spec import RunSpec
from core.universe import COLUMNS
from studio.service import get_session, input_changes, source_mismatches

F_SERIES = tuple('F' + month for month in MONTHS)


class SourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = ecr_source.load_bundle()

    def test_frames_keep_the_engine_contract(self):
        metrics, benchmarks, cycles, catalog = self.bundle[:4]
        self.assertEqual(list(metrics.columns), list(METRIC_COLUMNS))
        self.assertEqual({k: str(v) for k, v in metrics.dtypes.items()},
                         {'ticker': 'object', 'date': 'datetime64[ns]', 'remaining_outcome_days': 'int64',
                          **{c: 'float64' for c in METRIC_COLUMNS[2:] if c != 'remaining_outcome_days'}})
        self.assertTrue(set(F_SERIES) <= set(metrics.ticker))
        self.assertEqual(list(benchmarks.columns), ['SPY', 'BUFR'])
        self.assertEqual(str(benchmarks.index.dtype), 'datetime64[ns]')
        # BUFR's pre-launch and 2024 gaps stay NaN rows; nothing is dropped or filled.
        self.assertGreater(int(benchmarks.BUFR.isna().sum()), 0)
        self.assertTrue(benchmarks.SPY.notna().all())
        self.assertGreater(len(cycles), 0)
        self.assertEqual(str(cycles.dtype), 'datetime64[ns]')
        self.assertEqual(list(catalog.columns), COLUMNS)
        self.assertEqual(catalog.loc[catalog.series == 'F', 'ticker'].tolist(), list(F_SERIES))

    def test_default_universe_is_the_f_series(self):
        data = [prepare_data(self.bundle.metrics, self.bundle.benchmarks, self.bundle.cycles, universe=u)
                for u in (None, F_SERIES)]
        self.assertEqual(data[0].tickers, list(sorted(F_SERIES)))
        self.assertEqual(data[0].data_id, data[1].data_id)

    def test_precomputed_scores_are_never_loaded(self):
        stored = {'dbb_score', 'buffer_integrity', 'cap_integrity', 'reset_proximity', 'period_position', 'ecr_2_0'}
        self.assertFalse(stored & set(self.bundle.metrics.columns))

    def test_sources_name_each_table_and_its_content(self):
        for name, table in ecr_source.TABLES.items():
            source = self.bundle.sources[name]
            self.assertEqual(source['table'], table)
            self.assertEqual(len(source['sha256']), 64)
            self.assertGreater(source['rows'], 0)
        self.assertLessEqual(self.bundle.sources['metrics']['date_max'], _ecr_db.AS_OF)


class ProvenanceTests(unittest.TestCase):
    def test_changed_and_file_era_sources_are_reported(self):
        current = get_session().sources
        self.assertEqual(source_mismatches(current), [])
        tampered = {**current, 'benchmarks': {**current['benchmarks'], 'sha256': '0' * 64}}
        self.assertEqual(source_mismatches(tampered), ['benchmarks'])
        legacy = {'metrics': {'path': 'input_data/website_metrics.csv', 'sha256': '0' * 64}}
        self.assertEqual(source_mismatches(legacy), ['metrics (recorded from a file)'])

    def test_saved_runs_are_checked_by_their_own_data_id(self):
        current = get_session().prepare(universe=F_SERIES).data_id
        spec = RunSpec(backtest=BacktestConfig(universe=F_SERIES),
                       provenance={'data_id': current, 'sources': {'metrics': {'path': 'x', 'sha256': 'y'}}})
        self.assertEqual(input_changes(spec), [])
        self.assertEqual(input_changes(replace(spec, provenance={'data_id': '0' * 64})), ['data'])


class ConnectionTests(unittest.TestCase):
    def test_missing_or_unreachable_database_is_an_os_error_without_secrets(self):
        with patch.dict(os.environ, {'ECR_DB_URL': ''}), patch.object(ecr_source, 'ENV_FILE', Path('no-such.env')):
            with self.assertRaisesRegex(ecr_source.DataSourceError, 'ECR_DB_URL'):
                ecr_source.load_bundle()
        url = 'mysql+pymysql://reader:s3cret-pw@127.0.0.1:1/ecr?connect_timeout=2'
        with patch.dict(os.environ, {'ECR_DB_URL': url}):
            with self.assertRaises(OSError) as caught:
                ecr_source.signature(max_age=0)
        self.assertNotIn('s3cret-pw', str(caught.exception))

    def test_env_file_saved_by_windows_tools_is_read(self):
        url = 'mysql+pymysql://reader:pw@example:3306/ecr'
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'ECR_DB_URL': ''}):
            env = Path(temp) / '.env'
            with patch.object(ecr_source, 'ENV_FILE', env):
                env.write_text(f'# local\nECR_DB_URL="{url}"\n', encoding='utf-8-sig')  # with a byte-order mark
                self.assertEqual(ecr_source.database_url(), url)
                env.write_text(f'ECR_DB_URL={url}\n', encoding='utf-16')
                with self.assertRaisesRegex(ecr_source.DataSourceError, 'UTF-8'):
                    ecr_source.database_url()


if __name__ == '__main__':
    unittest.main()
