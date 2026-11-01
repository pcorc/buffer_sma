"""Run the workbook ECR model on the ecr database (ECR_DB_URL). Use --help for reproducible controls."""
import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time

from backtesting.engine import simulate
from config.settings import BacktestConfig, ScoreConfig
from config.spec import RunSpec
from studio.service import load_dataset
from utils.exporters import export_results

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, help='Saved configuration, run manifest.json or suite plan entry')
    p.add_argument('--output', type=Path, help='New output directory (must not exist)')
    p.add_argument('--threshold', type=float)
    p.add_argument('--client-name')
    p.add_argument('--universe', help='Comma-separated explicit eligible tickers, e.g. FJAN,FFEB,FMAR')
    p.add_argument('--portfolio-size', type=int, help='Number of distinct funds, fixed for this backtest')
    p.add_argument('--selection', choices=['highest_ecr', 'highest_dbb', 'highest_buffer_integrity', 'highest_cap_integrity'])
    p.add_argument('--comparator', choices=['<', '<='])
    p.add_argument('--review', choices=['daily', 'monthly', 'quarterly', 'semiannual', 'annual'])
    p.add_argument('--min-holding-days', type=int)
    p.add_argument('--min-remaining-days', type=int)
    p.add_argument('--execution-lag', type=int, choices=[0, 1])
    p.add_argument('--cost-bps', type=float)
    p.add_argument('--start-date')
    p.add_argument('--end-date')
    p.add_argument('--cohorts', help='Comma-separated launch months, e.g. JAN,FEB')
    p.add_argument('--initial-selection', choices=['cohort', 'highest_score'])
    p.add_argument('--trigger', choices=['ecr', 'buffer_drop', 'ecr_or_buffer', 'ecr_and_buffer', 'metric_floor'])
    p.add_argument('--buffer-metric', choices=['buffer_integrity', 'dbb', 'cap_integrity'])
    p.add_argument('--metric-floor', type=float, help='Fraction of ideal fresh component score 1, e.g. 0.8')
    p.add_argument('--buffer-drop', type=float)
    p.add_argument('--hold-benchmark', choices=['initial_selection', 'launch_fund'])
    for name in ('confirmation_reviews', 'reentry_cooldown_days', 'max_replacements_per_review'):
        p.add_argument('--' + name.replace('_', '-'), type=int)
    for name in ('min_selection_improvement', 'entry_metric_margin', 'max_turnover_per_review', 'weight_band'):
        p.add_argument('--' + name.replace('_', '-'), type=float)
    for field in fields(ScoreConfig):
        p.add_argument('--' + field.name.replace('_', '-'), type=float)
    p.add_argument('--no-audit', action='store_true', help='Omit all-candidate table; keep diagnostics/decisions')
    p.add_argument('--no-excel', action='store_true')
    p.add_argument('--no-plots', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    # A saved Studio config, a run manifest or a suite plan entry; flags below override it.
    spec = RunSpec.from_json(args.config) if args.config else RunSpec()
    score_options, backtest_options = asdict(spec.score), asdict(spec.backtest)
    for field in fields(ScoreConfig):
        if getattr(args, field.name) is not None:
            score_options[field.name] = getattr(args, field.name)
    for field in fields(BacktestConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            backtest_options[field.name] = value
    if args.universe:
        backtest_options['universe'] = tuple(t.strip().upper() for t in args.universe.split(','))
    if args.cohorts:
        backtest_options['cohorts'] = tuple(c.strip().upper() for c in args.cohorts.split(','))
    if args.no_audit:
        backtest_options['candidate_audit'] = False
    score_config, config = ScoreConfig(**score_options), BacktestConfig(**backtest_options)
    started = time.perf_counter()
    # The same loaded dataset as the Studio, so a saved config replays to the same economics.
    data, sources = load_dataset(score_config, universe=config.universe)
    prepared = time.perf_counter()
    result = simulate(data, config)
    simulated = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or ROOT / 'output' / f'ecr-{stamp}-{result["strategy_id"]}'
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    manifest = {'created_utc': stamp, **result['parameters'], 'strategy_id': result['strategy_id'],
                **{key: result[key] for key in ('engine_version', 'rule_fingerprint', 'run_fingerprint')},
                'git_commit': commit, 'git_dirty': dirty,
                'sources': sources,
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
