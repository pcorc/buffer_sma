"""Run the workbook ECR 2.0 model. Use --help for reproducible controls."""
import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from backtesting.data_pipeline import file_hash, load_benchmarks, load_cycles, load_metrics
from backtesting.engine import prepare_data, simulate
from config.settings import BacktestConfig, ScoreConfig
from utils.exporters import export_results

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics', type=Path, default=ROOT / 'input_data/website_metrics.csv',
                   help='Website Metrics XLSX, normalized decimal CSV, or original percentage CSV')
    p.add_argument('--benchmarks', type=Path, default=ROOT / 'input_data/benchmark_ts.csv')
    p.add_argument('--cycles', type=Path, default=ROOT / 'input_data/roll_dates.csv')
    p.add_argument('--config', type=Path, help='JSON object with score and backtest settings')
    p.add_argument('--output', type=Path, help='New output directory (must not exist)')
    p.add_argument('--threshold', type=float)
    p.add_argument('--comparator', choices=['<', '<='])
    p.add_argument('--review', choices=['daily', 'monthly'])
    p.add_argument('--min-holding-days', type=int)
    p.add_argument('--min-remaining-days', type=int)
    p.add_argument('--execution-lag', type=int, choices=[0, 1])
    p.add_argument('--cost-bps', type=float)
    p.add_argument('--start-date')
    p.add_argument('--end-date')
    p.add_argument('--cohorts', help='Comma-separated launch months, e.g. JAN,FEB')
    p.add_argument('--initial-selection', choices=['cohort', 'highest_score'])
    p.add_argument('--trigger', choices=['ecr', 'buffer_drop', 'ecr_or_buffer', 'ecr_and_buffer'])
    p.add_argument('--buffer-metric', choices=['buffer_integrity', 'dbb'])
    p.add_argument('--buffer-drop', type=float)
    for field in fields(ScoreConfig):
        p.add_argument('--' + field.name.replace('_', '-'), type=float)
    p.add_argument('--no-audit', action='store_true', help='Omit all-candidate table; keep diagnostics/decisions')
    p.add_argument('--no-excel', action='store_true')
    p.add_argument('--no-plots', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    supplied = json.loads(args.config.read_text()) if args.config else {}
    if set(supplied) - {'score', 'backtest'}:
        raise ValueError('Config only accepts score and backtest objects')
    score_options = supplied.get('score', {}).copy()
    backtest_options = supplied.get('backtest', {}).copy()
    for field in fields(ScoreConfig):
        if getattr(args, field.name) is not None:
            score_options[field.name] = getattr(args, field.name)
    for field in fields(BacktestConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            backtest_options[field.name] = value
    if args.cohorts:
        backtest_options['cohorts'] = tuple(c.strip().upper() for c in args.cohorts.split(','))
    elif 'cohorts' in backtest_options:
        backtest_options['cohorts'] = tuple(backtest_options['cohorts'])
    if args.no_audit:
        backtest_options['candidate_audit'] = False
    score_config, config = ScoreConfig(**score_options), BacktestConfig(**backtest_options)
    started = time.perf_counter()
    data = prepare_data(load_metrics(args.metrics), load_benchmarks(args.benchmarks),
                        load_cycles(args.cycles), score_config)
    prepared = time.perf_counter()
    result = simulate(data, config)
    simulated = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or ROOT / 'output' / f'ecr2-{stamp}-{result["strategy_id"]}'
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    manifest = {'created_utc': stamp, **result['parameters'], 'strategy_id': result['strategy_id'],
                'git_commit': commit, 'git_dirty': dirty,
                'sources': {name: {'path': str(getattr(args, name).resolve()),
                                   'sha256': file_hash(getattr(args, name))}
                            for name in ('metrics', 'benchmarks', 'cycles')},
                'load_score_seconds': prepared - started, 'simulation_seconds': simulated - prepared,
                'execution': ('signal at close; theoretical execution at same close' if config.execution_lag == 0
                              else 'signal at close; execution at next observed trading close'),
                'score_config': asdict(score_config)}
    export_results(result, output, manifest, excel=not args.no_excel, plots=not args.no_plots)
    print(result['summary'].to_string(index=False))
    print(f'Prepared/scored in {prepared-started:.3f}s; simulated in {simulated-prepared:.3f}s')
    print(f'{len(result["diagnostics"])} entry/switch snapshots. Results: {output}')
    return result


if __name__ == '__main__':
    main()
