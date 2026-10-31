"""Economic regression on real data: every listed configuration must reproduce its stored fingerprint."""
import json
import unittest

from golden_cases import FIXTURE, compute


class GoldenTests(unittest.TestCase):
    def test_fingerprints_match_fixture(self):
        expected = json.loads(FIXTURE.read_text())
        actual = compute()
        self.assertEqual(set(actual), set(expected), 'Golden case list changed; regenerate deliberately.')
        for name in expected:
            with self.subTest(case=name):
                # Readable values first, so a failure shows which cohort moved.
                for key in ('switches', 'final_nav', 'max_drawdown'):
                    self.assertEqual(actual[name][key], expected[name][key], key)
                self.assertEqual(actual[name]['frames'], expected[name]['frames'])
