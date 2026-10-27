"""Indexed ECR simulation; scores are computed once for all cohorts."""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from config.settings import BacktestConfig, MONTHS, ScoreConfig
from core.scoring import score_funds
from core.selection import rank_candidates

DIAGNOSTIC_MAP = {
    'Ref Asset Return': 'reference_asset_return',
    'Rem. Outcome Days': 'remaining_outcome_days',
    'Rem. Cap (net)': 'remaining_cap_net',
    'Rem. Buffer (net)': 'remaining_buffer_net',
    'Downside Before Buffer (net)': 'downside_before_buffer_net',
    'DBB': 'dbb', 'Buffer Integ.': 'buffer_integrity', 'Cap Integ.': 'cap_integrity',
    'Reset Prox.': 'reset_proximity', 'Period Pos.': 'period_position', 'ECR': 'ecr',
}
DIAGNOSTIC_COLUMNS = ['Cohort', 'Switch #', 'Date', 'Fund', *DIAGNOSTIC_MAP]


@dataclass
class PreparedData:
    rows: pd.DataFrame
    dates: pd.DatetimeIndex
    tickers: list[str]
    nav: np.ndarray
    scores: np.ndarray
    days: np.ndarray
    benchmarks: np.ndarray
    cycles: pd.DatetimeIndex
    score_config: ScoreConfig
    data_id: str
    buffers: np.ndarray
    dbb: np.ndarray
    caps: np.ndarray


def prepare_data(metrics, benchmarks, cycles, score_config=ScoreConfig()):
    # Only the F-series universe is supported; remove unrelated assets once.
    frame = metrics.loc[metrics.ticker.isin(['F' + c for c in MONTHS])].copy()
    if frame.empty:
        raise ValueError('No F-series observations')
    if frame.duplicated(['date', 'ticker']).any():
        raise ValueError('Duplicate ticker/date observations')
    frame = frame.sort_values(['date', 'ticker']).reset_index(drop=True)
    dates = pd.DatetimeIndex(frame.date.unique()).sort_values()
    tickers = sorted(frame.ticker.unique())
    scored = pd.concat([frame, score_funds(frame, score_config)], axis=1)

    def matrix(column):
        return scored.pivot(index='date', columns='ticker', values=column).reindex(
            index=dates, columns=tickers).to_numpy(dtype=float)

    digest = hashlib.sha256()
    for value in (frame, benchmarks):
        digest.update(pd.util.hash_pandas_object(value, index=True).values.tobytes())
    digest.update(pd.DatetimeIndex(cycles).asi8.tobytes())
    return PreparedData(scored.set_index(['date', 'ticker']), dates, tickers,
                        matrix('fund_value'), matrix('ecr'), matrix('remaining_outcome_days'),
                        benchmarks.reindex(dates)[['SPY', 'BUFR']].to_numpy(dtype=float),
                        pd.DatetimeIndex(cycles).sort_values(), score_config, digest.hexdigest(),
                        matrix('buffer_integrity'), matrix('dbb'), matrix('cap_integrity'))


def threshold_fires(score, config):
    if not np.isfinite(score):
        raise ValueError('Held fund has no valid ECR score on review date')
    return score <= config.threshold if config.comparator == '<=' else score < config.threshold


def trigger_signals(score, buffer_value, entry_buffer, config):
    ecr = threshold_fires(score, config)
    if not np.isfinite(buffer_value):
        raise ValueError('Invalid buffer metric on review date')
    # A zero entry score has no meaningful relative deterioration baseline.
    deterioration = 1 - buffer_value / entry_buffer if entry_buffer > 0 else np.nan
    buffer = bool(entry_buffer > 0 and buffer_value <= entry_buffer * (1 - config.buffer_drop))
    if config.trigger == 'metric_floor':
        buffer = bool(buffer_value <= config.metric_floor if config.comparator == '<='
                      else buffer_value < config.metric_floor)
    fired = {'ecr': ecr, 'buffer_drop': buffer, 'ecr_or_buffer': ecr or buffer,
             'ecr_and_buffer': ecr and buffer, 'metric_floor': buffer}[config.trigger]
    return bool(fired), bool(ecr), buffer, deterioration


def _start_index(data, cohort, config):
    earliest = pd.Timestamp(config.start_date)
    month = MONTHS.index(cohort) + 1
    for cycle in data.cycles[data.cycles.month == month]:
        # First observed trading session strictly after the cohort's cycle date.
        index = data.dates.searchsorted(cycle, side='right')
        if index < len(data.dates) and data.dates[index] >= earliest:
            # Do not jump across months of missing observations.
            if (data.dates[index] - cycle).days > 7:
                continue
            return int(index)
    raise ValueError(f'No initial cycle/trading date for {cohort}')


def simulate(data: PreparedData, config=BacktestConfig()):
    if config.portfolio_size > 1:
        from backtesting.portfolio import simulate_portfolio
        return simulate_portfolio(data, config)
    model_name = 'ecr-ar-v1'
    parameters = {'model': model_name, 'score': asdict(data.score_config),
                  'backtest': asdict(config), 'data_id': data.data_id}
    strategy_id = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:16]
    end = pd.Timestamp(config.end_date) if config.end_date else data.dates[-1]
    if pd.Timestamp(config.start_date) > end:
        raise ValueError('start_date is after end_date')
    stop = int(data.dates.searchsorted(end, side='right'))
    reviews = set(data.cycles)
    # Shared across cohorts: eligibility and buffer metrics do not depend on holdings.
    eligibility = (np.isfinite(data.scores) & np.isfinite(data.nav) & (data.nav > 0)
                   & (data.days >= config.min_remaining_days))
    buffer_panel = {'buffer_integrity': data.buffers, 'dbb': data.dbb,
                    'cap_integrity': data.caps}[config.buffer_metric]
    diagnostics, decisions, candidates, daily, summaries = [], [], [], [], []

    for cohort in config.cohorts:
        initial_fund = 'F' + cohort
        if initial_fund not in data.tickers:
            raise ValueError(f'Missing cohort fund {initial_fund}')
        held = initial = data.tickers.index(initial_fund)
        start = _start_index(data, cohort, config)
        if start >= stop:
            raise ValueError(f'No observations in requested period for {cohort}')
        initial_signal = start
        if config.initial_selection == 'highest_score':
            # Keep cohort execution dates comparable; next-close entry uses the previous session's scores.
            initial_signal = start - config.execution_lag
            if initial_signal < 0:
                raise ValueError('No prior session for initial selection signal')
            available = np.flatnonzero(eligibility[initial_signal])
            if not len(available):
                raise ValueError('No eligible initial candidates')
            held = int(rank_candidates(data.scores[initial_signal], eligibility[initial_signal], data.tickers)[0])
        if config.review == 'monthly':
            expected = pd.period_range(data.dates[start], data.dates[stop - 1], freq='M')
            if not set(expected).issubset(set(data.cycles.to_period('M'))):
                raise ValueError('Monthly cycle calendar does not cover the requested period')
            for cycle in data.cycles:
                if data.dates[start] <= cycle <= data.dates[stop - 1] and cycle not in data.dates:
                    raise ValueError(f'No observation on cycle date {cycle.date()}; check holiday calendar')
        bench = data.benchmarks[start:stop]
        if not np.isfinite(bench[[0, -1]]).all() or (bench <= 0).any():
            raise ValueError(f'Missing benchmark endpoint or invalid NAV for {cohort}')
        # Preserve interior gaps as NaN; never invent prices or omit strategy sessions.
        benchmark_gaps = int((~np.isfinite(bench).all(axis=1)).sum())
        hold_prices = data.nav[start:stop, initial]
        if not np.isfinite(hold_prices).all() or (hold_prices <= 0).any():
            raise ValueError(f'Missing or invalid buy-and-hold NAV for {cohort}')
        if not np.isfinite(data.scores[start, held]) or data.days[start, held] < config.min_remaining_days:
            raise ValueError(f'Invalid initial score for {initial_fund}')
        nav, last_trade, switches, event_number = 100.0, start, 0, 0
        trading_cost = 0.0
        entry_buffer = buffer_panel[start, held]
        pending = None
        cohort_daily = []

        def snapshot(i, fund, signal_i, event_type, outgoing=None, signal_score=None):
            nonlocal event_number
            event_number += 1
            date, ticker = data.dates[i], data.tickers[fund]
            row = data.rows.loc[(date, ticker)]
            if not row.score_valid:
                raise ValueError(f'Invalid snapshot for {ticker} on {date.date()}')
            event = {'Cohort': cohort.title(), 'Switch #': event_number, 'Date': date, 'Fund': ticker}
            event.update({label: row[column] for label, column in DIAGNOSTIC_MAP.items()})
            event.update({'Strategy ID': strategy_id, 'Event ID': f'{strategy_id}/{cohort}/{event_number}',
                          'Event Type': event_type, 'Signal Date': data.dates[signal_i],
                          'Execution Date': date, 'From Fund': outgoing,
                          'Selection Score at Signal': signal_score if signal_score is not None else row.ecr,
                          'Threshold': config.threshold, 'Comparator': config.comparator,
                          'Review': config.review, 'Data ID': data.data_id,
                          'Model': model_name, 'Trigger': config.trigger,
                          'Buffer Metric': config.buffer_metric, 'Buffer Drop Threshold': config.buffer_drop,
                          'Metric Floor': config.metric_floor, 'Fresh Reference Score': 1.0})
            diagnostics.append(event)

        snapshot(start, held, initial_signal, 'initial', signal_score=data.scores[initial_signal, held])
        for i in range(start, stop):
            date = data.dates[i]
            return_fund = data.tickers[held]
            if not np.isfinite(data.nav[i, held]) or data.nav[i, held] <= 0:
                raise ValueError(f'Missing held NAV for {return_fund} on {date.date()}')
            if i > start:
                nav *= data.nav[i, held] / data.nav[i - 1, held]
            # Pending selections earn the outgoing fund's return until next-close execution.
            if pending is not None:
                new, signal_i, signal_score = pending
                if not np.isfinite(data.nav[i, new]) or data.nav[i, new] <= 0:
                    raise ValueError('Missing execution NAV for pending selection')
                snapshot(i, new, signal_i, 'switch', data.tickers[held], signal_score)
                held, last_trade, switches = new, i, switches + 1
                entry_buffer = buffer_panel[i, held]
                trading_cost += nav * config.cost_bps / 10000
                nav *= 1 - config.cost_bps / 10000
                pending = None
            if i > start and (config.review == 'daily' or date in reviews):
                held_score = data.scores[i, held]
                age = i - last_trade
                eligible = eligibility[i]
                signal, ecr_signal, buffer_signal, deterioration = trigger_signals(
                    held_score, buffer_panel[i, held], entry_buffer, config)
                fired = signal and age >= config.min_holding_days
                available = np.flatnonzero(eligible)
                if not len(available):
                    raise ValueError(f'No eligible funds on {date.date()}')
                best = int(rank_candidates(data.scores[i], eligible, data.tickers, [held])[0])
                action = 'hold'
                if fired:
                    action = 'no_switch' if best == held else 'switch'
                    if best != held:
                        if config.execution_lag:
                            if i + 1 < stop:
                                pending = (best, i, data.scores[i, best])
                                action = 'scheduled'
                            else:
                                action = 'unexecuted_end'
                        else:
                            snapshot(i, best, i, 'switch', data.tickers[held], data.scores[i, best])
                            trading_cost += nav * config.cost_bps / 10000
                            nav *= 1 - config.cost_bps / 10000
                            switches += 1
                review_id = f'{strategy_id}/{cohort}/review/{date.date()}'
                decisions.append({'Strategy ID': strategy_id, 'Review ID': review_id,
                                  'Cohort': cohort.title(), 'Date': date, 'Held Fund': data.tickers[held],
                                  'Held Score': held_score, 'Best Fund': data.tickers[best],
                                  'Best Score': data.scores[i, best], 'Holding Days': age,
                                  'Triggered': fired, 'Action': action,
                                  'Threshold': config.threshold, 'Comparator': config.comparator,
                                  'ECR Signal': ecr_signal, 'Buffer Signal': buffer_signal,
                                  'Buffer Metric': config.buffer_metric,
                                  'Entry Buffer Score': entry_buffer, 'Held Buffer Score': buffer_panel[i, held],
                                  'Buffer Deterioration': deterioration, 'Buffer Drop Threshold': config.buffer_drop,
                                  'Metric Floor': config.metric_floor, 'Fresh Reference Score': 1.0,
                                  'Metric Fraction of Fresh': buffer_panel[i, held],
                                  'Trigger': config.trigger})
                if config.candidate_audit:
                    order = sorted(available, key=lambda j: (-data.scores[i, j], j != held, data.tickers[j]))
                    ranks = {j: n + 1 for n, j in enumerate(order)}
                    for j, ticker in enumerate(data.tickers):
                        candidates.append({'Strategy ID': strategy_id, 'Review ID': review_id,
                                           'Cohort': cohort.title(), 'Date': date, 'Fund': ticker,
                                           'ECR': data.scores[i, j], 'Eligible': bool(eligible[j]),
                                           'Rank': ranks.get(j), 'Best': j == best, 'Action': action})
                if action == 'switch':
                    held, last_trade = best, i
                    entry_buffer = buffer_panel[i, held]
            record = {'Strategy ID': strategy_id, 'Cohort': cohort.title(), 'Date': date,
                      'Return Fund': return_fund, 'Fund at Close': data.tickers[held],
                      'Held ECR Score': data.scores[i, held], 'Held Buffer Score': buffer_panel[i, held],
                      'Entry Buffer Score': entry_buffer,
                      'Strategy NAV': nav,
                      'SPY NAV': 100 * data.benchmarks[i, 0] / data.benchmarks[start, 0],
                      'BUFR NAV': 100 * data.benchmarks[i, 1] / data.benchmarks[start, 1],
                      'Hold NAV': 100 * data.nav[i, initial] / data.nav[start, initial]}
            cohort_daily.append(record)
        daily.extend(cohort_daily)
        curve = pd.DataFrame(cohort_daily)
        returns = curve['Strategy NAV'].pct_change().dropna()
        total = nav / 100 - 1
        elapsed = (data.dates[stop - 1] - data.dates[start]).days
        annual = (nav / 100)**(365 / elapsed) - 1 if elapsed else np.nan
        volatility = returns.std() * np.sqrt(252)
        downside = np.sqrt(np.mean(np.minimum(returns, 0)**2)) * np.sqrt(252)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
        summaries.append({'Strategy ID': strategy_id, 'Cohort': cohort.title(),
                          'Start': data.dates[start], 'End': data.dates[stop - 1], 'Observations': len(curve),
                          'Total Return': total, 'Annualized Return': annual,
                          'Annualized Volatility': volatility, 'Sharpe': sharpe,
                          'Annualized Downside Deviation': downside,
                          'Sortino': returns.mean() * 252 / downside if downside > 0 else np.nan,
                          'Client': config.client_name, 'Portfolio Size': config.portfolio_size,
                          'Max Drawdown': (curve['Strategy NAV'] / curve['Strategy NAV'].cummax() - 1).min(),
                          'Switches': switches, 'Switches per Year': switches * 365 / elapsed if elapsed else np.nan,
                          'One-way Turnover': float(switches), 'Trading Cost NAV Units': trading_cost,
                          'Missing Benchmark Sessions': benchmark_gaps,
                          'SPY Total Return': curve['SPY NAV'].iloc[-1] / 100 - 1,
                          'BUFR Total Return': curve['BUFR NAV'].iloc[-1] / 100 - 1,
                          'vs BUFR Cumulative Excess': total - (curve['BUFR NAV'].iloc[-1] / 100 - 1)})
    candidate_frame = pd.DataFrame(candidates)
    if not candidate_frame.empty:
        raw = data.rows.reset_index().rename(columns={'date': 'Date', 'ticker': 'Fund',
                                                      **{v: k for k, v in DIAGNOSTIC_MAP.items()}})
        columns = ['Date', 'Fund', *[c for c in DIAGNOSTIC_MAP if c != 'ECR']]
        candidate_frame = candidate_frame.merge(raw[columns], on=['Date', 'Fund'], how='left',
                                                validate='many_to_one')
    daily_frame = pd.DataFrame(daily)
    holdings = daily_frame[['Strategy ID', 'Cohort', 'Date', 'Fund at Close', 'Strategy NAV',
                            'Held ECR Score', 'Held Buffer Score', 'Entry Buffer Score']].rename(
        columns={'Fund at Close': 'Fund', 'Strategy NAV': 'Capital'})
    holdings['Position'] = 1
    holdings['Weight'] = 1.0
    return {'parameters': parameters, 'strategy_id': strategy_id,
            'summary': pd.DataFrame(summaries), 'daily': daily_frame, 'holdings': holdings,
            'diagnostics': pd.DataFrame(diagnostics), 'decisions': pd.DataFrame(decisions),
            'candidates': candidate_frame}
