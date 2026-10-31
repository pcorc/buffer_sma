"""UI-independent configuration, data loading, and downloadable run artifacts."""
from dataclasses import asdict
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import threading
import time
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

from backtesting.data_pipeline import file_hash, load_benchmarks, load_cycles, metric_paths, load_metric_collection
from core.universe import CATALOG_PATH, EXTRA_METRICS, load_catalog
from backtesting.engine import plan_cohort_windows, prepare_data  # noqa: F401 (re-exported)
from config.settings import BacktestConfig, ScoreConfig
from config.spec import RunSpec
from utils.exporters import export_results

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = {name: ROOT / 'input_data' / file for name, file in (
    ('metrics', 'website_metrics.csv'), ('benchmarks', 'benchmark_ts.csv'), ('cycles', 'roll_dates.csv'))}
METRICS = {'ecr': 'ECR', 'dbb': 'Downside Before Buffer',
           'buffer_integrity': 'Buffer Integrity', 'cap_integrity': 'Cap Integrity'}
SELECTIONS = {f'highest_{key}': f'Highest {label}' for key, label in METRICS.items()}
CADENCES = {'monthly': 'Monthly', 'quarterly': 'Quarterly', 'semiannual': 'Semiannual', 'annual': 'Annual',
            'daily': 'Daily (research)'}
SCHEDULE_NOTES = {
    'daily': 'Review at every observed close. A research setting: expect far more turnover than a client schedule.',
    'monthly': 'Review on each monthly roll date.',
    'quarterly': 'Review on the March, June, September and December roll dates.',
    'semiannual': 'Review on the June and December roll dates.',
    'annual': 'Review on the December roll date.',
}


def source_signature():
    paths = [*SOURCE_PATHS.values(), CATALOG_PATH, *sorted(EXTRA_METRICS.glob('*.csv'))]
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)


def load_dataset(score, universe=None):
    paths = metric_paths(SOURCE_PATHS['metrics'], EXTRA_METRICS)
    sources = {name: {'path': str(p), 'sha256': file_hash(p)} for name, p in SOURCE_PATHS.items()}
    sources['catalogue'] = {'path': str(CATALOG_PATH), 'sha256': file_hash(CATALOG_PATH)}
    sources['supplemental_metrics'] = [{'path': str(p), 'sha256': file_hash(p)} for p in paths[1:]]
    data = prepare_data(load_metric_collection(paths), load_benchmarks(SOURCE_PATHS['benchmarks']),
                        load_cycles(SOURCE_PATHS['cycles']), score, universe=universe)
    return data, sources


_SESSIONS = {}
_SESSION_LOCK = threading.Lock()


def get_session():
    """One loaded dataset per process, rebuilt when any source file changes (Studio, scripts, notebook)."""
    from backtesting.research import ResearchSession
    signature = source_signature()
    with _SESSION_LOCK:
        if signature not in _SESSIONS:
            session = ResearchSession(SOURCE_PATHS['metrics'], SOURCE_PATHS['benchmarks'], SOURCE_PATHS['cycles'],
                                      supplemental_metrics=sorted(EXTRA_METRICS.glob('*.csv')))
            session.sources['catalogue'] = {'path': str(CATALOG_PATH), 'sha256': file_hash(CATALOG_PATH)}
            _SESSIONS.clear()
            _SESSIONS[signature] = session
        return _SESSIONS[signature]


def run_backtest(spec, session=None):
    """Validate windows, simulate and attach provenance; no files are written."""
    result = (session or get_session()).run(spec.score, spec.backtest)
    result['spec'] = spec.to_dict()
    return result


def source_mismatches(sources):
    """Names of recorded inputs whose content differs from the current repository files."""
    current = {name: file_hash(path) for name, path in SOURCE_PATHS.items()}
    return sorted(name for name, recorded in (sources or {}).items()
                  if name in current and isinstance(recorded, dict) and recorded.get('sha256') != current[name])


def available_catalogue():
    metrics = load_metric_collection(metric_paths(SOURCE_PATHS['metrics'], EXTRA_METRICS))
    catalogue = load_catalog()
    catalogue['history_available'] = catalogue.ticker.isin(metrics.ticker.unique())
    return catalogue


def config_json(score, config):
    return json.dumps({'score': asdict(score), 'backtest': asdict(config)}, indent=2) + '\n'


def load_preset(name):
    spec = RunSpec.from_json(ROOT / 'config' / name)
    return spec.score, spec.backtest


def describe_rule(config):
    """The trigger exactly as the engine applies it, e.g. 'Buffer Integrity < 0.8'."""
    ecr = f'ECR {config.comparator} {config.threshold:g}'
    component = METRICS[config.buffer_metric]
    # Relative drops always include the boundary (value <= entry × (1 − drop)).
    drop = f'{component} down ≥{config.buffer_drop:.0%} from purchase'
    return {'ecr': ecr, 'metric_floor': f'{component} {config.comparator} {config.metric_floor:g}',
            'buffer_drop': drop, 'ecr_or_buffer': f'{ecr} or {drop}', 'ecr_and_buffer': f'{ecr} and {drop}'}[config.trigger]


_GIT_STATE = {}


def git_state(max_age=60):
    """(commit, dirty) for manifests, read at most once a minute: suites write hundreds of reports."""
    cached = _GIT_STATE.get('value')
    if cached is None or time.monotonic() - _GIT_STATE['at'] > max_age:
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = None, None
        _GIT_STATE.update(value=(commit, dirty), at=time.monotonic())
    return _GIT_STATE['value']


def write_report(result, output, *, plots=True, excel='full'):
    """Write the complete report for one run into a new directory (CSV, Excel, manifest, config, charts)."""
    score = ScoreConfig(**result['parameters']['score'])
    config = BacktestConfig(**result['parameters']['backtest'])
    commit, dirty = git_state()
    manifest = {**result['parameters'], 'strategy_id': result['strategy_id'],
                **{key: result.get(key) for key in ('engine_version', 'rule_fingerprint', 'run_fingerprint')},
                'created_utc': datetime.now(timezone.utc).isoformat(), 'git_commit': commit,
                'git_dirty': dirty, 'sources': result.get('sources', {}),
                'execution': 'Next observed close' if config.execution_lag else 'Theoretical same close',
                'review_calendar': SCHEDULE_NOTES.get(config.review, 'Every observed session'),
                'timings': result.get('timings', {}),
                'requested_archive_root': result.get('requested_archive_root'),
                'plots_included': plots, 'excel': excel}
    output = Path(output)
    export_results(result, output, manifest, excel=excel, plots=plots)
    (output / 'config.json').write_text(config_json(score, config))
    if plots:
        from visualization.research_plots import dashboard, holdings_figure
        from visualization.performance_plots import plot_studio_results
        charts = output / 'charts'
        charts.mkdir()
        plot_studio_results(result, charts)
        for cohort in result['summary'].Cohort:
            dashboard(result, cohort).write_html(charts / f'{cohort.lower()}-performance.html', include_plotlyjs='directory')
            holdings_figure(result, cohort).write_html(charts / f'{cohort.lower()}-holdings.html', include_plotlyjs='directory')
    chart_note = ('charts/: PNG reports and interactive HTML charts. Extract the whole ZIP before opening HTML.\n'
                  if plots else 'Charts are not included in this package. Build the full report for PNG and HTML charts.\n')
    workbook_note = {'full': 'ecr_results.xlsx: summary, daily NAV, holdings, decisions, candidates and diagnostics.\n',
                     'compact': 'ecr_results.xlsx: summary, NAV by launch, diagnostics and trades; ledgers are in the CSV files.\n'
                     }.get(excel, 'No Excel workbook; every table is a CSV file.\n')
    (output / 'README.txt').write_text(
        'ECR Backtest Studio\n\n'
        + workbook_note + chart_note +
        'manifest.json: exact parameters, source hashes, code version and execution assumptions.\n'
        'config.json: replay with python main.py --config /path/to/config.json\n\n'
        'Review cadence checks the trigger; trades are conditional. Only breached sleeves are replaced.\n'
        'Ties prefer incumbents. A breached incumbent may stay if it still ranks highest.\n'
        'With an explicit universe, Hold NAV is the initially selected fund/basket held without trading.\n'
        'Legacy configurations without a universe retain the single-fund launch-month benchmark.\n'
        'Initial purchases are not charged a switch cost. Later costs apply to replaced capital only.\n')
    return output


# Already-compressed formats gain nothing from deflate; storing them is faster.
STORED_SUFFIXES = {'.xlsx', '.png', '.zip', '.parquet'}


def build_downloads(result, *, plots=True, excel='full'):
    """Build artifacts in an isolated temporary directory; return immutable bytes."""
    score = ScoreConfig(**result['parameters']['score'])
    config = BacktestConfig(**result['parameters']['backtest'])
    with TemporaryDirectory(prefix='ecr-studio-') as temp:
        output = write_report(result, Path(temp) / 'report', plots=plots, excel=excel)
        archive = BytesIO()
        with ZipFile(archive, 'w', ZIP_DEFLATED) as zipped:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    zipped.write(path, path.relative_to(output),
                                 compress_type=ZIP_STORED if path.suffix in STORED_SUFFIXES else ZIP_DEFLATED)
        workbook = output / 'ecr_results.xlsx'
        return {'excel': workbook.read_bytes() if workbook.exists() else None, 'zip': archive.getvalue(),
                'config': config_json(score, config)}
