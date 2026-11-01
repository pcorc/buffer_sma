"""Reusable in-memory research sessions. Loading and scoring are not repeated per trigger."""
from collections import OrderedDict
import copy
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time

import pandas as pd

from backtesting.engine import plan_cohort_windows, prepare_data, simulate
from config.settings import BacktestConfig, ScoreConfig


@dataclass(frozen=True)
class Experiment:
    name: str
    score: ScoreConfig
    backtest: BacktestConfig


class ResearchSession:
    """Hold one loaded dataset; retain up to eight score panels for parameter experiments.

    Build it with from_source() (the ecr database) or from in-memory frames. Results are newly
    simulated each time; callers should treat the cached prepared data as read-only.
    """
    def __init__(self, metrics, benchmarks, cycles, sources=None):
        self.sources = copy.deepcopy(sources or {})
        self.metrics = metrics
        self.benchmarks = benchmarks
        self.cycles = cycles
        self._prepared = OrderedDict()
        self._lock = threading.Lock()
        self.score_builds = 0

    @classmethod
    def from_source(cls, as_of=None):
        from backtesting.ecr_source import load_bundle
        bundle = load_bundle(as_of)
        return cls(bundle.metrics, bundle.benchmarks, bundle.cycles, bundle.sources)

    def prepare(self, config=ScoreConfig(), universe=None):
        # prepare_data sorts tickers, so universe order must not create a second panel.
        key = (config, tuple(sorted(universe)) if universe is not None else None)
        # The Studio may serve several browser sessions from one process.
        with self._lock:
            if key not in self._prepared:
                self._prepared[key] = prepare_data(self.metrics, self.benchmarks, self.cycles, config, universe=universe)
                self.score_builds += 1
                if len(self._prepared) > 8:
                    self._prepared.popitem(last=False)
            self._prepared.move_to_end(key)
            return self._prepared[key]

    def run(self, score=ScoreConfig(), backtest=BacktestConfig()):
        t = time.perf_counter()
        data = self.prepare(score, backtest.universe)
        plan_cohort_windows(data, backtest)
        prepared = time.perf_counter()
        result = simulate(data, backtest)
        result['timings'] = {'prepare_seconds': prepared-t, 'simulate_seconds': time.perf_counter()-prepared}
        result['sources'] = copy.deepcopy(self.sources)
        return result

    def compare(self, experiments):
        names = [e.name for e in experiments]
        if len(names) != len(set(names)) or not names:
            raise ValueError('Experiments need nonempty unique names')
        results = {e.name: self.run(e.score, e.backtest) for e in experiments}
        # Comparisons require the exact same cohort windows, not just the same count.
        windows = None
        for result in results.values():
            current = result['summary'][['Cohort', 'Start', 'End']].sort_values('Cohort').reset_index(drop=True)
            if windows is not None and not current.equals(windows):
                raise ValueError('Experiments must share the same cohort start/end dates')
            windows = current
        return results, comparison_table(results)


def comparison_table(results):
    rows = []
    for name, result in results.items():
        s = result['summary']
        rows.append({'Experiment': name, 'Mean Total Return': s['Total Return'].mean(),
                     'Mean vs BUFR': s['vs BUFR Cumulative Excess'].mean(),
                     'Worst Cohort vs BUFR': s['vs BUFR Cumulative Excess'].min(),
                     'Positive Cohorts': int((s['vs BUFR Cumulative Excess'] > 0).sum()),
                     'Cohorts': len(s), 'Mean Sharpe': s.Sharpe.mean(),
                     'Portfolio Size': int(s['Portfolio Size'].iloc[0]),
                     'Mean Downside Deviation': s['Annualized Downside Deviation'].mean(),
                     'Mean Sortino': s.Sortino.mean(),
                     'Mean One-way Turnover': s['One-way Turnover'].mean(),
                     'Mean Annualized Turnover': (s['One-way Turnover'] * 365 / (s.End-s.Start).dt.days.replace(0, float('nan'))).mean(),
                     'Mean Weight Rebalances': s['Weight Rebalances'].mean() if 'Weight Rebalances' in s else 0.,
                     'Worst Drawdown': s['Max Drawdown'].min(), 'Mean Switches': s.Switches.mean(),
                     'Mean Switches per Year': s['Switches per Year'].mean(),
                     'Simulation Seconds': result.get('timings', {}).get('simulate_seconds', float('nan')),
                     'Strategy ID': result['strategy_id']})
    return pd.DataFrame(rows).set_index('Experiment')


def export_comparison(results, output):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    comparison_table(results).to_csv(path / 'comparison.csv')
    cohorts = pd.concat([r['summary'].assign(Experiment=name) for name, r in results.items()], ignore_index=True)
    cohorts.to_csv(path / 'cohort_comparison.csv', index=False)
    manifest = {name: {'parameters': r['parameters'], 'sources': r['sources'], 'timings': r['timings']}
                for name, r in results.items()}
    (path / 'experiments.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    return path


def conservative_experiments(score=ScoreConfig(), backtest=BacktestConfig()):
    """Fixed multiplicative ECR selection; vary only the buffer-integrity exit floor."""
    from dataclasses import replace
    common = replace(backtest, initial_selection='highest_score', candidate_audit=False,
                     buffer_metric='buffer_integrity', comparator='<')
    tests = [Experiment('ECR threshold baseline', score, replace(common, trigger='ecr'))]
    for floor in (.7, .8, .9):
        tests.append(Experiment(f'Buffer integrity below {floor:.0%} fresh', score,
                     replace(common, trigger='metric_floor', metric_floor=floor)))
    return tests


def initial_experiments(score=ScoreConfig(), backtest=BacktestConfig()):
    """The notebook's initial comparison uses the confirmed conservative experiment."""
    return conservative_experiments(score, backtest)


def client_experiments(score=ScoreConfig(), backtest=BacktestConfig(), sizes=(1,3,5), floors=(.7,.8,.9)):
    """Compare UCITS portfolio sizes/floors against a separately saved Super Buffer rule."""
    from dataclasses import replace
    common = replace(backtest, candidate_audit=False, initial_selection='highest_score',
                     selection='highest_ecr', comparator='<')
    tests = [Experiment('Super Buffer / ECR < 1.5 / N=1', score,
                        replace(common, client_name='Super Buffer', portfolio_size=1,
                                trigger='ecr', threshold=1.5))]
    for n in sizes:
        for floor in floors:
            tests.append(Experiment(f'UCITS / N={n} / buffer < {floor:.0%}', score,
                         replace(common, client_name='UCITS Conservative', portfolio_size=n,
                                 trigger='metric_floor', buffer_metric='buffer_integrity', metric_floor=floor)))
    return tests
