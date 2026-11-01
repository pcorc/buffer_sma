"""The ecr database schema is the only source of input data. Set ECR_DB_URL, e.g.

    mysql+pymysql://reader:password@host:3306/ecr

or put that line, ECR_DB_URL=..., in the repository's gitignored .env file.
ECR_AS_OF (YYYY-MM-DD) optionally caps metrics, benchmarks and roll dates at a date, so a run or the
test suite can replay a frozen window of a growing database. A sqlite:///path/ecr.db URL is accepted
as a local stand-in: the file is attached as schema ecr and the same queries run unchanged.
"""
from collections import namedtuple
import os
from pathlib import Path
import threading
import time

import pandas as pd

from backtesting.data_pipeline import (METRIC_COLUMNS, content_hash, normalize_benchmarks, normalize_cycles,
                                       normalize_metrics)
from config.settings import MONTHS
from core.universe import COLUMNS as CATALOG_COLUMNS

TABLES = {'metrics': 'ecr.website_metrics', 'benchmarks': 'ecr.benchmark_prices',
          'cycles': 'ecr.roll_dates', 'catalogue': 'ecr.fund_catalog'}
# Explicit lists only: the table's precomputed score columns share names with score_funds output.
QUERIES = {
    'metrics': (f'SELECT {", ".join("`date`" if c == "date" else c for c in METRIC_COLUMNS)} '
                'FROM ecr.website_metrics', '`date`'),
    'benchmarks': ('SELECT price_date AS `Date`, spy AS SPY, bufr AS BUFR FROM ecr.benchmark_prices', 'price_date'),
    'cycles': ("SELECT roll_date FROM ecr.roll_dates WHERE cadence = 'monthly'", 'roll_date'),
    'catalogue': (f'SELECT {", ".join("fund_group AS `group`" if c == "group" else c for c in CATALOG_COLUMNS)} '
                  'FROM ecr.fund_catalog', None),
}
TEXT_COLUMNS = [c for c in CATALOG_COLUMNS if c not in ('buffer_start', 'buffer_end')]

Bundle = namedtuple('Bundle', 'metrics benchmarks cycles catalog sources signature')


class DataSourceError(OSError):
    """The ecr database could not be read. An OSError, so existing load-failure handling applies."""


_ENGINES = {}
_LOCK = threading.Lock()
_SIGNATURE = {}
_BUNDLES = {}


ENV_FILE = Path(__file__).resolve().parents[1] / '.env'


def database_url():
    url = os.environ.get('ECR_DB_URL', '').strip()
    if not url and ENV_FILE.is_file():
        try:
            lines = ENV_FILE.read_text(encoding='utf-8-sig').splitlines()  # tolerates a byte-order mark
        except UnicodeDecodeError:
            raise DataSourceError(f'{ENV_FILE} must be saved as UTF-8 (PowerShell: Set-Content -Encoding utf8)') from None
        for line in lines:
            key, _, value = line.partition('=')
            if key.strip() == 'ECR_DB_URL':
                url = value.strip().strip('"\'')
    if not url:
        raise DataSourceError('Set ECR_DB_URL to the ecr database, e.g. mysql+pymysql://user:password@host:3306/ecr')
    return url


def as_of_default():
    return os.environ.get('ECR_AS_OF', '').strip() or None


def _engine():
    url = database_url()
    key = (os.getpid(), url)  # spawned workers build their own pool
    if key not in _ENGINES:
        try:
            from sqlalchemy import create_engine, event
            from sqlalchemy.engine import make_url
        except ImportError as exc:
            raise DataSourceError('Install requirements.txt (SQLAlchemy, PyMySQL) to read the ecr database') from exc
        try:
            parsed = make_url(url)
        except Exception:
            raise DataSourceError('ECR_DB_URL is not a valid database URL') from None
        if parsed.get_backend_name() == 'sqlite':
            path = parsed.database
            engine = create_engine('sqlite://')

            @event.listens_for(engine, 'connect')
            def _attach(connection, _):
                connection.execute('ATTACH DATABASE ? AS ecr', (path,))
        else:
            # One transaction per load: REPEATABLE READ gives every table the same snapshot.
            try:
                engine = create_engine(parsed, pool_pre_ping=True, pool_recycle=3600, isolation_level='REPEATABLE READ')
            except ImportError as exc:  # the URL's driver, e.g. PyMySQL, is not installed
                raise DataSourceError(f'Install requirements.txt to read the ecr database ({exc})') from None
        _ENGINES[key] = engine
    return _ENGINES[key]


def with_connection(work):
    """Run work(connection) in one read transaction; driver errors become DataSourceError without secrets."""
    engine = _engine()
    from sqlalchemy.exc import SQLAlchemyError

    def failure(exc):
        detail = str(getattr(exc, 'orig', None) or exc).splitlines()[0]
        password = engine.url.password
        if password:
            detail = detail.replace(password, '***')
        return DataSourceError(f'Unable to read the ecr database ({engine.url.render_as_string(hide_password=True)}): '
                               f'{detail}')
    try:
        connection = engine.connect()
    except (SQLAlchemyError, RuntimeError) as exc:  # PyMySQL raises RuntimeError when 'cryptography' is missing
        raise failure(exc) from None
    with connection:
        try:
            return work(connection)
        except SQLAlchemyError as exc:
            raise failure(exc) from None


def _query(connection, name, as_of):
    from sqlalchemy import text
    sql, date_column = QUERIES[name]
    params = {}
    if as_of is not None and date_column:
        sql += (' AND ' if ' WHERE ' in sql else ' WHERE ') + f'{date_column} <= :as_of'
        params['as_of'] = pd.Timestamp(as_of).date().isoformat()
    return pd.read_sql(text(sql), connection, params=params, coerce_float=True)


def read_frames(connection, as_of=None):
    """Raw query results for each source, before normalization."""
    return {name: _query(connection, name, as_of) for name in QUERIES}


def normalize_frames(raw):
    """Exact engine contracts: int64 days, datetime64[ns] dates, NaN benchmark gaps, workbook catalogue order."""
    metrics = raw['metrics'].copy()
    metrics['ticker'] = metrics.ticker.astype(object)
    metrics['date'] = pd.to_datetime(metrics['date']).astype('datetime64[ns]')
    if metrics.remaining_outcome_days.notna().all():
        metrics['remaining_outcome_days'] = metrics.remaining_outcome_days.astype('int64')
    benchmarks = raw['benchmarks'].copy()
    benchmarks['Date'] = pd.to_datetime(benchmarks['Date']).astype('datetime64[ns]')
    benchmarks[['SPY', 'BUFR']] = benchmarks[['SPY', 'BUFR']].astype(float)
    catalog = raw['catalogue'].copy()
    catalog[TEXT_COLUMNS] = catalog[TEXT_COLUMNS].fillna('').astype(str)
    order = {month: i for i, month in enumerate([*MONTHS, 'Quarterly'])}
    catalog = (catalog.assign(_month=catalog.outcome_month.map(order))
               .sort_values(['series', '_month', 'ticker']).drop(columns='_month').reset_index(drop=True))
    return (normalize_metrics(metrics), normalize_benchmarks(benchmarks),
            normalize_cycles(pd.to_datetime(raw['cycles'].roll_date).astype('datetime64[ns]')), catalog)


def describe(metrics, benchmarks, cycles, catalog, as_of=None):
    """Provenance per source: the table, a content hash of exactly what was loaded, rows and date range."""
    def entry(name, frame, dates=None):
        record = {'table': TABLES[name], 'sha256': content_hash(frame), 'rows': len(frame)}
        if dates is not None and len(dates):
            record.update(date_min=str(dates.min().date()), date_max=str(dates.max().date()))
        if as_of is not None and dates is not None:
            record['as_of'] = str(pd.Timestamp(as_of).date())
        return record
    return {'metrics': entry('metrics', metrics, metrics.date), 'benchmarks': entry('benchmarks', benchmarks, benchmarks.index),
            'cycles': entry('cycles', cycles, cycles), 'catalogue': entry('catalogue', catalog)}


def signature(max_age=None):
    """A cheap token that changes whenever any source table changes; cached briefly for Studio reruns."""
    max_age = float(os.environ.get('ECR_SIGNATURE_TTL', 30)) if max_age is None else max_age
    url = database_url()
    with _LOCK:
        cached = _SIGNATURE.get(url)
        if cached and time.monotonic() - cached[0] <= max_age:
            return cached[1]

    def work(connection):
        if connection.dialect.name == 'mysql':
            # CHECKSUM also catches in-place UPDATEs that row counts would miss.
            rows = connection.exec_driver_sql('CHECKSUM TABLE ' + ', '.join(TABLES.values())).all()
        else:
            rows = [(t, connection.exec_driver_sql(f'SELECT COUNT(*) FROM {t}').scalar()) for t in TABLES.values()]
        return tuple((str(t), None if c is None else int(c)) for t, c in rows)
    value = with_connection(work)
    with _LOCK:
        _SIGNATURE[url] = (time.monotonic(), value)
    return value


def load_bundle(as_of=None):
    """All four sources from one consistent read, normalized; memoized until the signature changes.

    The frames are shared: treat them as read-only.
    """
    as_of = as_of if as_of is not None else as_of_default()
    current = signature()
    key = (database_url(), current, as_of)
    with _LOCK:
        if key in _BUNDLES:
            return _BUNDLES[key]

    def work(connection):
        metrics, benchmarks, cycles, catalog = normalize_frames(read_frames(connection, as_of))
        return Bundle(metrics, benchmarks, cycles, catalog,
                      describe(metrics, benchmarks, cycles, catalog, as_of), current)
    bundle = with_connection(work)
    with _LOCK:
        _BUNDLES.clear()
        _BUNDLES[key] = bundle
    return bundle


def clear_cache():
    with _LOCK:
        _SIGNATURE.clear()
        _BUNDLES.clear()
