"""UI-independent configuration, data loading, and downloadable run artifacts."""
from dataclasses import asdict
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

from backtesting.data_pipeline import file_hash, load_metrics, load_benchmarks, load_cycles
from backtesting.engine import prepare_data
from config.settings import BacktestConfig, ScoreConfig
from utils.exporters import export_results

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = {name: ROOT / 'input_data' / file for name, file in (
    ('metrics', 'website_metrics.csv'), ('benchmarks', 'benchmark_ts.csv'), ('cycles', 'roll_dates.csv'))}
METRICS = {'ecr': 'ECR', 'dbb': 'Downside Before Buffer',
           'buffer_integrity': 'Buffer Integrity', 'cap_integrity': 'Cap Integrity'}
SELECTIONS = {f'highest_{key}': f'Highest {label}' for key, label in METRICS.items()}
CADENCES = {'monthly': 'Monthly', 'quarterly': 'Quarterly', 'semiannual': 'Semiannual', 'annual': 'Annual'}
SCHEDULE_NOTES = {
    'monthly': 'Review on each monthly roll date.',
    'quarterly': 'Review on the March, June, September and December roll dates.',
    'semiannual': 'Review on the June and December roll dates.',
    'annual': 'Review on the December roll date.',
}


def source_signature():
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in SOURCE_PATHS.values())


def load_dataset(score):
    sources = {name: {'path': str(p), 'sha256': file_hash(p)} for name, p in SOURCE_PATHS.items()}
    data = prepare_data(load_metrics(SOURCE_PATHS['metrics']), load_benchmarks(SOURCE_PATHS['benchmarks']),
                        load_cycles(SOURCE_PATHS['cycles']), score)
    return data, sources


def config_json(score, config):
    return json.dumps({'score': asdict(score), 'backtest': asdict(config)}, indent=2) + '\n'


def load_preset(name):
    raw = json.loads((ROOT / 'config' / name).read_text())
    backtest = raw['backtest']
    if 'cohorts' in backtest:
        backtest['cohorts'] = tuple(backtest['cohorts'])
    return ScoreConfig(**raw.get('score', {})), BacktestConfig(**backtest)


def validate_window(data, config):
    """Validate every launch before an expensive simulation or report build."""
    from backtesting.engine import _start_index
    from backtesting.schedule import review_dates
    import pandas as pd
    end = pd.Timestamp(config.end_date) if config.end_date else data.dates[-1]
    if pd.Timestamp(config.start_date) >= end:
        raise ValueError('Choose an end date after the start date.')
    if end > data.dates[-1]:
        raise ValueError('End date exceeds the available observations.')
    stop = int(data.dates.searchsorted(end, side='right'))
    for cohort in config.cohorts:
        start = _start_index(data, cohort, config)
        if start + 1 >= stop:
            raise ValueError(f'{cohort}: choose a longer window containing its launch month and at least two trading sessions.')
        if start < config.execution_lag:
            raise ValueError(f'{cohort}: no prior session is available for the initial selection; move the start date forward.')
        review_dates(data.dates, data.cycles, start, stop, config.review)


def build_downloads(result):
    """Build artifacts in an isolated temporary directory; return immutable bytes."""
    from visualization.research_plots import dashboard, holdings_figure
    score = ScoreConfig(**result['parameters']['score'])
    config = BacktestConfig(**result['parameters']['backtest'])
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    manifest = {**result['parameters'], 'strategy_id': result['strategy_id'],
                'created_utc': datetime.now(timezone.utc).isoformat(), 'git_commit': commit,
                'git_dirty': dirty, 'sources': result.get('sources', {}),
                'execution': 'Next observed close' if config.execution_lag else 'Theoretical same close',
                'review_calendar': SCHEDULE_NOTES.get(config.review, 'Every observed session'),
                'timings': result.get('timings', {})}
    with TemporaryDirectory(prefix='ecr-studio-') as temp:
        output = Path(temp) / 'report'
        export_results(result, output, manifest)
        (output / 'config.json').write_text(config_json(score, config))
        charts = output / 'charts'
        charts.mkdir()
        from visualization.performance_plots import plot_studio_results
        plot_studio_results(result, charts)
        for cohort in result['summary'].Cohort:
            dashboard(result, cohort).write_html(charts / f'{cohort.lower()}-performance.html', include_plotlyjs='directory')
            holdings_figure(result, cohort).write_html(charts / f'{cohort.lower()}-holdings.html', include_plotlyjs='directory')
        (output / 'README.txt').write_text(
            'ECR Backtest Studio\n\n'
            'ecr_results.xlsx: summary, daily NAV, holdings, decisions, candidates and diagnostics.\n'
            'charts/: PNG reports and interactive HTML charts. Extract the whole ZIP before opening HTML.\n'
            'manifest.json: exact parameters, source hashes, code version and execution assumptions.\n'
            'config.json: replay with python main.py --config /path/to/config.json\n\n'
            'Review cadence checks the trigger; trades are conditional. Only breached sleeves are replaced.\n'
            'Ties prefer incumbents. A breached incumbent may stay if it still ranks highest.\n'
            'A single-fund Hold NAV is the launch-month fund; a FoF Hold NAV is the initial basket.\n'
            'Initial purchases are not charged a switch cost. Later costs apply to replaced capital only.\n')
        archive = BytesIO()
        with ZipFile(archive, 'w', ZIP_DEFLATED) as zipped:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    zipped.write(path, path.relative_to(output))
        return {'excel': (output / 'ecr_results.xlsx').read_bytes(), 'zip': archive.getvalue(),
                'config': config_json(score, config)}
