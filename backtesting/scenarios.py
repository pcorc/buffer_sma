"""Named conservative hypotheses, never presented as optimized allocations."""
from dataclasses import replace

from backtesting.research import Experiment
from config.settings import BacktestConfig, ScoreConfig


def conservative_suite(backtest=None):
    """35 scenarios with shared windows/costs; ECR always selects the replacement."""
    base = backtest or BacktestConfig(execution_lag=1, cost_bps=5)
    base = replace(base, initial_selection='highest_score', selection='highest_ecr', comparator='<',
                   candidate_audit=False, review='monthly', min_holding_days=0,
                   confirmation_reviews=1, min_selection_improvement=0, entry_metric_margin=0,
                   reentry_cooldown_days=0, max_replacements_per_review=0,
                   max_turnover_per_review=1, weight_band=None)
    score = ScoreConfig()
    tests = [Experiment('Super Buffer / ECR 1.5', score, replace(base,client_name='Super Buffer',portfolio_size=1,trigger='ecr',threshold=1.5))]
    for n in (3,5):
        common = replace(base,client_name='UCITS Conservative',portfolio_size=n,trigger='metric_floor')
        def add(label, config, scoring=score):
            tests.append(Experiment(f'UCITS / N={n} / {label}',scoring,config))
        add('ECR 1.5 reference',replace(common,trigger='ecr',threshold=1.5))
        for floor in (.7,.8,.9):
            add(f'Buffer {floor:.0%}',replace(common,buffer_metric='buffer_integrity',metric_floor=floor))
        for floor in (.8,.9,.95):
            add(f'DBB {floor:.0%}',replace(common,buffer_metric='dbb',metric_floor=floor))
        buffer = replace(common,buffer_metric='buffer_integrity',metric_floor=.8)
        controls = dict(confirmation_reviews=2, min_selection_improvement=.1, entry_metric_margin=.05,
                        min_holding_days=21, reentry_cooldown_days=21, max_replacements_per_review=1,
                        max_turnover_per_review=.4 if n==3 else .25, weight_band=.05)
        add('Buffer 80% + ECR improvement 0.15',replace(buffer,min_selection_improvement=.15))
        add('Buffer 80% + two-review confirmation',replace(buffer,confirmation_reviews=2))
        add('Buffer 80% + enter at 90%',replace(buffer,entry_metric_margin=.1))
        add('Buffer 80% + hold 42 / cooldown 21',replace(buffer,min_holding_days=42,reentry_cooldown_days=21))
        add('Buffer 80% + replacement budget',replace(buffer,max_replacements_per_review=1,max_turnover_per_review=controls['max_turnover_per_review']))
        add('Buffer 80% + weight band 5pp',replace(buffer,weight_band=.05))
        add('Buffer 80% + combined controls',replace(buffer,**controls))
        add('DBB 90% + combined controls',replace(common,buffer_metric='dbb',metric_floor=.9,**controls))
        add('Buffer 80% + defensive ECR',buffer,replace(score,dbb_weight=1.8,reset_weight=.9,buffer_exponent=.75,cap_exponent=.2))
        add('Buffer 80% + quarterly reviews',replace(buffer,review='quarterly'))
    return tests


def quarterly_score_hypotheses():
    """Parameter proposals only. These do not convert annual funds to quarterly funds.

    A 90-day vs 360-day convention keeps the relative period-position curve equal.
    Half-width sigmas assume square-root-of-time return dispersion. Real calibration
    must use quarterly fund NAV, cap, buffer, returns-since-reset and outcome dates.
    """
    relative_period = replace(ScoreConfig(),period_center=45,period_width=18.75)
    return {'quarterly_same_return_tolerance': relative_period,
            'quarterly_half_return_tolerance': replace(relative_period,sigma_down=.035,sigma_up=.025)}
