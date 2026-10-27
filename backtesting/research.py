"""Reusable in-memory research sessions. Loading and scoring are not repeated per trigger."""
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time

import pandas as pd

from backtesting.data_pipeline import file_hash, load_benchmarks, load_cycles, load_metrics
from backtesting.engine import prepare_data, simulate
from config.settings import BacktestConfig, ScoreConfig


@dataclass(frozen=True)
class Experiment:
    name: str
    score: ScoreConfig
    backtest: BacktestConfig


class ResearchSession:
    """Load a dataset once; retain up to eight score panels for parameter experiments.

    Instantiate a new session after changing source files. Results are newly simulated
    each time; callers should treat the cached prepared data as read-only.
    """
    def __init__(self, metrics, benchmarks, cycles):
        paths = {'metrics': Path(metrics), 'benchmarks': Path(benchmarks), 'cycles': Path(cycles)}
        self.sources = {k: {'path': str(p.resolve()), 'sha256': file_hash(p)} for k, p in paths.items()}
        self.metrics = load_metrics(metrics)
        self.benchmarks = load_benchmarks(benchmarks)
        self.cycles = load_cycles(cycles)
        self._prepared = OrderedDict()
        self.score_builds = 0

    def prepare(self, config=ScoreConfig()):
        if config not in self._prepared:
            self._prepared[config] = prepare_data(self.metrics, self.benchmarks, self.cycles, config)
            self.score_builds += 1
            if len(self._prepared) > 8:
                self._prepared.popitem(last=False)
        self._prepared.move_to_end(config)
        return self._prepared[config]

    def run(self, score=ScoreConfig(), backtest=BacktestConfig()):
        t = time.perf_counter()
        data = self.prepare(score)
        prepared = time.perf_counter()
        result = simulate(data, backtest)
        result['timings'] = {'prepare_seconds': prepared-t, 'simulate_seconds': time.perf_counter()-prepared}
        result['sources'] = self.sources
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
                     'Worst Drawdown': s['Max Drawdown'].min(), 'Mean Switches': s.Switches.mean(),
                     'Mean Switches per Year': s['Switches per Year'].mean(),
                     'Simulation Seconds': result['timings']['simulate_seconds'],
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


def initial_experiments(score=ScoreConfig(), backtest=BacktestConfig()):
    """Controlled variations of the same multiplicative ECR formula."""
    from dataclasses import replace
    common = replace(backtest, candidate_audit=False, trigger='ecr')
    top = replace(common, initial_selection='highest_score')
    return [
        Experiment('Workbook / cohort entry', score, replace(common, initial_selection='cohort')),
        Experiment('Workbook / highest entry', score, top),
        Experiment('DBB emphasis (1.8 / 0.9)', replace(score, dbb_weight=1.8, reset_weight=.9), top),
        Experiment('Stricter buffer gate (1.0)', replace(score, buffer_exponent=1.), top),
        Experiment('Stricter cap gate (0.8)', replace(score, cap_exponent=.8), top),
        Experiment('Buffer decline only', score, replace(top, trigger='buffer_drop')),
        Experiment('ECR OR buffer decline', score, replace(top, trigger='ecr_or_buffer')),
        Experiment('ECR AND buffer decline', score, replace(top, trigger='ecr_and_buffer')),
    ]


def conservative_experiments(score=ScoreConfig(), backtest=BacktestConfig()):
    """Separate ECR parameter allocation from fresh-component exit thresholds."""
    from dataclasses import replace
    common = replace(backtest, initial_selection='highest_score', candidate_audit=False)
    allocations = [('Equal', (1,1,1)), ('DBB 75%', (.75,.125,.125)),
                   ('Buffer 75%', (.125,.75,.125)), ('Cap 75%', (.125,.125,.75))]
    tests = [Experiment(f'Weights / {name}', replace(score, component_weights=w),
                        replace(common, trigger='ecr')) for name,w in allocations]
    # Method 2 deliberately uses the unweighted baseline ECR for replacement selection.
    baseline = replace(score, component_weights=(1/3,1/3,1/3))
    for metric in ('dbb','buffer_integrity','cap_integrity'):
        for floor in (.7,.8,.9):
            tests.append(Experiment(f'{metric} below {floor:.0%} fresh', baseline,
                         replace(common, trigger='metric_floor', buffer_metric=metric,
                                 metric_floor=floor, comparator='<')))
    return tests
