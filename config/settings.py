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
    client_name: str = 'Research'
    portfolio_size: int = 1
    selection: str = 'highest_ecr'
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
    metric_floor: float = 0.80
    confirmation_reviews: int = 1
    min_selection_improvement: float = 0.0
    entry_metric_margin: float = 0.0
    reentry_cooldown_days: int = 0
    max_replacements_per_review: int = 0
    max_turnover_per_review: float = 1.0
    weight_band: float | None = None

    def __post_init__(self):
        for name in ('confirmation_reviews', 'reentry_cooldown_days', 'max_replacements_per_review'):
            value = getattr(self, name)
            minimum = 1 if name == 'confirmation_reviews' else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        for name in ('min_selection_improvement', 'entry_metric_margin'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.entry_metric_margin and (self.trigger != 'metric_floor' or self.metric_floor + self.entry_metric_margin > 1):
            raise ValueError('Entry margin requires a metric-floor trigger and floor + margin <= 1')
        if not math.isfinite(self.max_turnover_per_review) or not 0 < self.max_turnover_per_review <= 1:
            raise ValueError('max_turnover_per_review must be in (0, 1]')
        if self.weight_band is not None:
            if not math.isfinite(self.weight_band) or not 0 <= self.weight_band <= 1:
                raise ValueError('weight_band must be None or a fraction in [0, 1]')
            if self.portfolio_size < 2:
                raise ValueError('Weight bands require at least two funds')
        if not isinstance(self.client_name, str) or not self.client_name.strip():
            raise ValueError('client_name must be a nonempty string')
        if isinstance(self.portfolio_size, bool) or not isinstance(self.portfolio_size, int) or not 1 <= self.portfolio_size <= 12:
            raise ValueError('portfolio_size must be an integer from 1 to 12')
        if self.selection not in ('highest_ecr', 'highest_dbb', 'highest_buffer_integrity', 'highest_cap_integrity'):
            raise ValueError('Unknown selection metric')
        if self.portfolio_size > 1 and self.initial_selection != 'highest_score':
            raise ValueError('Multi-fund portfolios require initial_selection=highest_score')
        if self.initial_selection not in ('cohort', 'highest_score'):
            raise ValueError('initial_selection must be cohort or highest_score')
        if self.trigger not in ('ecr', 'buffer_drop', 'ecr_or_buffer', 'ecr_and_buffer', 'metric_floor'):
            raise ValueError('unknown trigger')
        if self.buffer_metric not in ('buffer_integrity', 'dbb', 'cap_integrity'):
            raise ValueError('buffer_metric must be buffer_integrity, dbb or cap_integrity')
        if not math.isfinite(self.metric_floor) or not 0 <= self.metric_floor <= 1:
            raise ValueError('metric_floor must be a fraction in [0, 1] of ideal fresh score 1')
        if not math.isfinite(self.buffer_drop) or not 0 < self.buffer_drop <= 1:
            raise ValueError('buffer_drop must be in (0, 1]')
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError('threshold must be finite and nonnegative')
        if self.comparator not in ('<', '<='):
            raise ValueError('comparator must be < or <=')
        if self.review not in ('daily', 'monthly', 'quarterly', 'semiannual', 'annual'):
            raise ValueError('Unknown review cadence')
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


def turnover_controls_enabled(config):
    return (config.confirmation_reviews > 1 or config.min_selection_improvement > 0
            or config.entry_metric_margin > 0 or config.reentry_cooldown_days > 0
            or config.max_replacements_per_review > 0 or config.max_turnover_per_review < 1
            or config.weight_band is not None)


def replacement_controls_enabled(config):
    # Weight bands and confirmation must not independently change candidate assignment.
    return (config.min_selection_improvement > 0 or config.entry_metric_margin > 0
            or config.reentry_cooldown_days > 0 or config.max_replacements_per_review > 0
            or config.max_turnover_per_review < 1)
