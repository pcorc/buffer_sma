"""Timing budgets for the paths the Studio waits on. Opt-in: machines differ, so set ECR_PERF=1 to run."""
from dataclasses import replace
import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtesting.engine import simulate
from config.spec import RunSpec
from studio.service import ROOT, get_session

# Seconds, best of five; measured budgets with 1.5x headroom for an ordinary workstation.
BUDGETS = {'super-buffer.json': .5, 'ucit-conservative.json': 1.}


@unittest.skipUnless(os.environ.get('ECR_PERF'), 'Set ECR_PERF=1 to check timing budgets')
class PerformanceTests(unittest.TestCase):
    def test_twelve_launch_simulations_fit_their_budgets(self):
        session = get_session()
        for name, budget in BUDGETS.items():
            spec = RunSpec.from_json(ROOT / 'config' / name)
            data = session.prepare(spec.score, spec.backtest.universe)
            times = []
            for _ in range(5):
                began = time.perf_counter()
                simulate(data, replace(spec.backtest, candidate_audit=False))
                times.append(time.perf_counter() - began)
            with self.subTest(preset=name):
                self.assertLess(min(times), budget * 1.5, f'{name}: best {min(times):.2f}s')
