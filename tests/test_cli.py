"""The command line replays a Studio configuration to the same economics."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

import main
from backtesting.engine import simulate
from config.settings import MONTHS
from studio.service import config_json, load_dataset, load_preset

UNIVERSE = tuple('F' + month for month in MONTHS)


class CommandLineTests(unittest.TestCase):
    def test_studio_config_replays_through_main(self):
        score, preset = load_preset('ucit-conservative.json')
        config = replace(preset, universe=UNIVERSE, cohorts=('JAN', 'APR'), candidate_audit=False)
        expected = simulate(load_dataset(score, UNIVERSE)[0], config)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.json'
            path.write_text(config_json(score, config))
            output = Path(temp) / 'run'
            actual = main.main(['--config', str(path), '--output', str(output), '--no-excel', '--no-plots'])
            self.assertEqual(actual['strategy_id'], expected['strategy_id'])
            self.assertEqual(actual['run_fingerprint'], expected['run_fingerprint'])
            for key in ('summary', 'daily', 'decisions', 'diagnostics', 'holdings'):
                pd.testing.assert_frame_equal(actual[key], expected[key])
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['strategy_id'], expected['strategy_id'])
            self.assertEqual(manifest['rule_fingerprint'], expected['rule_fingerprint'])
            self.assertTrue((output / 'summary.csv').exists())
