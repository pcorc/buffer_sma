"""One run specification shared by the Studio, CLI, notebook and suite scripts."""
from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path

from config.settings import BacktestConfig, ScoreConfig

# Keys the various saved shapes may carry besides score/backtest; they describe, never configure.
METADATA_KEYS = {'name', 'filters', 'tags', 'parent_run_id', 'baseline_run_id', 'batch_id'}
MANIFEST_KEYS = {'model', 'data_id', 'strategy_id', 'portfolio_policy', 'created_utc', 'git_commit', 'git_dirty',
                 'sources', 'supplemental_metrics', 'execution', 'review_calendar', 'timings',
                 'requested_archive_root', 'plots_included', 'load_score_seconds', 'simulation_seconds',
                 'score_config', 'engine_version', 'rule_fingerprint', 'run_fingerprint', 'regime'}


@dataclass(frozen=True)
class RunSpec:
    score: ScoreConfig = ScoreConfig()
    backtest: BacktestConfig = BacktestConfig()
    name: str | None = None
    filters: dict | None = None
    tags: tuple[str, ...] = ()
    parent_run_id: str | None = None
    baseline_run_id: str | None = None
    batch_id: str | None = None
    # Recorded provenance of a loaded manifest (source hashes, data id); never used to configure a run.
    provenance: dict = field(default_factory=dict, compare=False, hash=False)

    @classmethod
    def from_json(cls, source):
        """Accept a {score, backtest} config, a run manifest or a suite plan entry (dict, JSON text or path)."""
        if isinstance(source, Path) or (isinstance(source, str) and not source.lstrip().startswith('{')):
            source = Path(source).read_text(encoding='utf-8')
        raw = json.loads(source) if isinstance(source, str) else dict(source)
        if not isinstance(raw, dict) or 'backtest' not in raw:
            raise ValueError('A configuration needs a "backtest" object (and optionally "score").')
        unknown = set(raw) - {'score', 'backtest'} - METADATA_KEYS - MANIFEST_KEYS
        if unknown:
            raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        return cls(score=_build(ScoreConfig, raw.get('score') or {}), backtest=_build(BacktestConfig, raw['backtest']),
                   name=raw.get('name'), filters=raw.get('filters'), tags=tuple(raw.get('tags') or ()),
                   parent_run_id=raw.get('parent_run_id'), baseline_run_id=raw.get('baseline_run_id'),
                   batch_id=raw.get('batch_id'),
                   provenance={k: raw[k] for k in ('data_id', 'sources', 'strategy_id', 'rule_fingerprint',
                                                   'run_fingerprint', 'git_commit') if k in raw})

    def to_dict(self):
        """Replayable through main.py --config; metadata is kept, provenance is not."""
        data = {'score': asdict(self.score), 'backtest': asdict(self.backtest)}
        data.update({k: v for k, v in (('name', self.name), ('filters', self.filters), ('tags', list(self.tags)),
                                       ('parent_run_id', self.parent_run_id), ('baseline_run_id', self.baseline_run_id),
                                       ('batch_id', self.batch_id)) if v})
        return data

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2) + '\n'


def _build(kind, values):
    if not isinstance(values, dict):
        raise ValueError(f'{kind.__name__} settings must be an object.')
    names = {f.name for f in fields(kind)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f'Unknown {kind.__name__} fields: {sorted(unknown)}')
    values = dict(values)
    # JSON has no tuples; the dataclasses are frozen and hashable.
    for name in ('universe', 'cohorts'):
        if isinstance(values.get(name), list):
            values[name] = tuple(values[name])
    return kind(**values)
