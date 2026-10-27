"""Explicit, serializable parameters for the Website Metrics AR model."""
from dataclasses import asdict, dataclass
import math

MONTHS = ('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC')


@dataclass(frozen=True)
class ScoreConfig:
    dbb_weight: float = 1.5
    reset_weight: float = 1.2
    buffer_exponent: float = 0.5
    cap_exponent: float = 0.3
    period_bonus: float = 0.3
    dbb_decay: float = 5.0
    sigma_down: float = 0.07
    sigma_up: float = 0.05
    period_center: float = 180.0
    period_width: float = 75.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('sigma_down', 'sigma_up', 'period_width'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')


@dataclass(frozen=True)
class BacktestConfig:
    threshold: float = 1.5
    comparator: str = '<='
    review: str = 'monthly'
    min_holding_days: int = 0
    start_date: str = '2021-01-19'
    end_date: str | None = None
    cohorts: tuple[str, ...] = MONTHS
    min_remaining_days: int = 0
    execution_lag: int = 0
    cost_bps: float = 0.0
    candidate_audit: bool = True
    initial_selection: str = 'cohort'
    trigger: str = 'ecr'
    buffer_metric: str = 'buffer_integrity'
    buffer_drop: float = 0.20

    def __post_init__(self):
        if self.initial_selection not in ('cohort', 'highest_score'):
            raise ValueError('initial_selection must be cohort or highest_score')
        if self.trigger not in ('ecr', 'buffer_drop', 'ecr_or_buffer', 'ecr_and_buffer'):
            raise ValueError('unknown trigger')
        if self.buffer_metric not in ('buffer_integrity', 'dbb'):
            raise ValueError('buffer_metric must be buffer_integrity or dbb')
        if not math.isfinite(self.buffer_drop) or not 0 < self.buffer_drop <= 1:
            raise ValueError('buffer_drop must be in (0, 1]')
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError('threshold must be finite and nonnegative')
        if self.comparator not in ('<', '<='):
            raise ValueError('comparator must be < or <=')
        if self.review not in ('daily', 'monthly'):
            raise ValueError('review must be daily or monthly')
        for name in ('min_holding_days', 'min_remaining_days'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.execution_lag not in (0, 1):
            raise ValueError('execution_lag must be 0 (same close) or 1 (next close)')
        if not math.isfinite(self.cost_bps) or not 0 <= self.cost_bps < 10000:
            raise ValueError('cost_bps must be in [0, 10000)')
        if not self.cohorts or len(set(self.cohorts)) != len(self.cohorts):
            raise ValueError('cohorts must be nonempty and unique')
        if any(c not in MONTHS for c in self.cohorts):
            raise ValueError('unknown cohort')
