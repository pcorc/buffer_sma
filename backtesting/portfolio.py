"""Multiple funded sleeves using the same ECR ranking and per-holding trigger rules."""
from dataclasses import asdict
import hashlib
import json

import numpy as np
import pandas as pd

from backtesting.engine import DIAGNOSTIC_MAP, _start_index, trigger_signals
from backtesting.schedule import review_dates
from config.settings import hold_basis
from core.turnover import controlled_plan, weight_band_breached
from core.selection import rank_candidates, selection_panel


def simulate_portfolio(data, config):
    n = config.portfolio_size
    parameters = {'model': 'ecr-ar-v1', 'score': asdict(data.score_config),
                  'backtest': asdict(config), 'data_id': data.data_id,
                  'portfolio_policy': ('equal-initial-sleeves; replace-breached-only; ' +
                                       ('review-date-weight-bands' if config.weight_band is not None else 'weights-drift'))}
    strategy_id = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:16]
    end = pd.Timestamp(config.end_date) if config.end_date else data.dates[-1]
    if pd.Timestamp(config.start_date) > end:
        raise ValueError('start_date is after end_date')
    stop = int(data.dates.searchsorted(end, side='right'))
    ranking = selection_panel(data, config.selection)
    eligibility = (np.isfinite(data.scores) & np.isfinite(ranking) & np.isfinite(data.nav) & (data.nav > 0)
                   & (data.days >= config.min_remaining_days))
    metric = {'buffer_integrity': data.buffers, 'dbb': data.dbb, 'cap_integrity': data.caps}[config.buffer_metric]
    diagnostics, decisions, candidates, daily, holdings, summaries = [], [], [], [], [], []
    trades, weight_reviews = [], []

    # Boxed once: indexing a DatetimeIndex per session is a measurable share of the loop.
    dates = list(data.dates)
    for cohort in config.cohorts:
        start = _start_index(data, cohort, config)
        if start >= stop:
            raise ValueError(f'No observations in requested period for {cohort}')
        signal_i = start - config.execution_lag if config.initial_selection == 'highest_score' else start
        if signal_i < 0:
            raise ValueError('No prior session for initial selection signal')
        ranked = rank_candidates(ranking[signal_i], eligibility[signal_i], data.tickers)
        if len(ranked) < n:
            raise ValueError(f'{cohort}: requested {n} funds, only {len(ranked)} eligible on {dates[signal_i].date()}')
        held = np.array(ranked[:n], dtype=int)
        if n == 1 and config.initial_selection == 'cohort':
            held[0] = data.tickers.index('F' + cohort)
        initial = held.copy()
        if hold_basis(config) == 'launch_fund':
            if 'F' + cohort not in data.tickers:
                raise ValueError(f'Missing cohort fund F{cohort}')
            initial[0] = data.tickers.index('F' + cohort)
        reviews = review_dates(data.dates, data.cycles, start, stop, config.review)
        bench = data.benchmarks[start:stop]
        if not np.isfinite(bench[[0, -1]]).all() or (bench <= 0).any():
            raise ValueError(f'Missing benchmark endpoint or invalid NAV for {cohort}')
        benchmark_gaps = int((~np.isfinite(bench).all(axis=1)).sum())
        initial_prices = data.nav[start:stop, initial]
        if not np.isfinite(initial_prices).all() or (initial_prices <= 0).any():
            raise ValueError(f'Missing or invalid buy-and-hold NAV for {cohort}')
        hold_curve = 100 * np.ascontiguousarray(initial_prices / data.nav[start, initial]).mean(axis=1)
        if not np.isfinite(data.scores[start, held]).all():
            raise ValueError('Invalid initial portfolio score')
        capital = np.full(n, 100. / n)
        last_trade = np.full(n, start, dtype=int)
        entry_metric = metric[start, held].copy()
        event_number = switches = 0
        cumulative_cost = 0.
        cumulative_turnover = 0.
        pending = None
        confirmation = np.zeros(n, dtype=int)
        last_exit = np.full(len(data.tickers), -100000, dtype=int)
        weight_rebalances = 0
        cohort_daily = []

        def snapshot(i, slot, fund, signal_index, event_type, outgoing=None):
            nonlocal event_number
            event_number += 1
            row = data.rows.loc[(dates[i], data.tickers[fund])]
            if not row.score_valid:
                raise ValueError(f'Invalid execution snapshot for {data.tickers[fund]}')
            event = {'Cohort': cohort.title(), 'Switch #': event_number, 'Date': dates[i],
                     'Fund': data.tickers[fund]}
            event.update({label: row[column] for label, column in DIAGNOSTIC_MAP.items()})
            event.update({'Strategy ID': strategy_id, 'Event ID': f'{strategy_id}/{cohort}/{event_number}',
                          'Position': slot+1, 'Portfolio Size': n, 'Event Type': event_type,
                          'Signal Date': dates[signal_index], 'Execution Date': dates[i],
                          'From Fund': outgoing, 'Selection Score at Signal': ranking[signal_index, fund], 'Selection': config.selection,
                          'Threshold': config.threshold, 'Comparator': config.comparator,
                          'Review': config.review, 'Data ID': data.data_id, 'Model': 'ecr-ar-v1',
                          'Trigger': config.trigger, 'Buffer Metric': config.buffer_metric,
                          'Metric Floor': config.metric_floor, 'Fresh Reference Score': 1.0,
                          'Capital Before Cost': capital[slot]})
            diagnostics.append(event)

        def record_outcome(row, outcome, i):
            if row is not None:
                row.update({'Execution Outcome': outcome, 'Execution Date': dates[i]})

        def execute(plan, signal_index, i, rebalance_weights=False, rows=None, weight_row=None):
            nonlocal switches, cumulative_cost, cumulative_turnover, weight_rebalances
            rows = rows or {}
            total_before = float(capital.sum())
            spent = 0.0
            # Worst-first priority matters only when the budget can defer trades; otherwise keep stable slot order.
            for slot, new in (plan.items() if config.max_turnover_per_review < 1 else sorted(plan.items())):
                old = held[slot]
                if new == old:
                    continue
                fraction = capital[slot] / total_before
                # Next-close prices can change sleeve weights; recheck the actual budget.
                if spent + fraction > config.max_turnover_per_review + 1e-12:
                    record_outcome(rows.get(slot), 'deferred_execution_budget', i)
                    trades.append({'Cohort': cohort.title(), 'Date': dates[i],
                                   'Signal Date': dates[signal_index], 'Position': slot+1,
                                   'Type': 'replacement', 'Action': 'deferred_execution_budget',
                                   'From Fund': data.tickers[old], 'Fund': data.tickers[new],
                                   'One-way Turnover': 0., 'Cost NAV Units': 0.})
                    continue
                if not np.isfinite(data.nav[i, new]) or data.nav[i, new] <= 0:
                    raise ValueError('Missing execution NAV for pending selection')
                if new in held:
                    raise ValueError('Replacement plan would duplicate holdings')
                snapshot(i, slot, new, signal_index, 'switch', data.tickers[old])
                cost = capital[slot] * config.cost_bps / 10000
                trades.append({'Cohort': cohort.title(), 'Date': dates[i],
                               'Signal Date': dates[signal_index], 'Position': slot+1,
                               'Type': 'replacement', 'Action': 'executed',
                               'From Fund': data.tickers[old], 'Fund': data.tickers[new],
                               'One-way Turnover': fraction, 'Cost NAV Units': cost})
                spent += fraction
                cumulative_turnover += fraction
                capital[slot] -= cost
                cumulative_cost += cost
                held[slot] = new
                last_exit[old] = i
                last_trade[slot] = i
                confirmation[slot] = 0
                entry_metric[slot] = metric[i, new]
                switches += 1
                record_outcome(rows.get(slot), 'executed', i)
            if rebalance_weights:
                # Band breach is decided at signal time; the next-close target is equal weights.
                gross = float(capital.sum())
                turnover_nav = float(np.abs(capital - gross/n).sum() / 2)
                fraction = turnover_nav / total_before
                action = 'deferred_execution_budget'
                cost = 0.
                if spent + fraction <= config.max_turnover_per_review + 1e-12:
                    action = 'executed' if turnover_nav > 1e-12 else 'no_change'
                    cost = turnover_nav * config.cost_bps / 10000
                    capital[:] = (gross - cost) / n
                    cumulative_cost += cost
                    cumulative_turnover += fraction
                    weight_rebalances += int(action == 'executed')
                record_outcome(weight_row, action, i)
                trades.append({'Cohort': cohort.title(), 'Date': dates[i],
                               'Signal Date': dates[signal_index], 'Type': 'weight_rebalance',
                               'Action': action, 'One-way Turnover': fraction if action == 'executed' else 0.,
                               'Cost NAV Units': cost})
            if len(set(held)) != n:
                raise ValueError('Replacement plan would duplicate holdings')

        for slot, fund in enumerate(held):
            snapshot(start, slot, fund, signal_i, 'initial')
        for i in range(start, stop):
            date = dates[i]
            return_funds = ','.join(data.tickers[j] for j in held)
            if not np.isfinite(data.nav[i, held]).all() or (data.nav[i, held] <= 0).any():
                raise ValueError(f'Missing held NAV on {date.date()}')
            if i > start:
                capital *= data.nav[i, held] / data.nav[i-1, held]
            if pending is not None:
                plan, signal_index, rebalance_weights, rows, weight_row = pending
                execute(plan, signal_index, i, rebalance_weights, rows, weight_row)
                pending = None
            if i > start and date in reviews:
                signals = [trigger_signals(data.scores[i, j], metric[i, j], entry_metric[slot], config)
                           for slot, j in enumerate(held)]
                confirmation[:] = [confirmation[slot]+1 if sig[0] else 0 for slot,sig in enumerate(signals)]
                triggered = [slot for slot, sig in enumerate(signals)
                             if sig[0] and i-last_trade[slot] >= config.min_holding_days
                             and confirmation[slot] >= config.confirmation_reviews]
                reasons = {slot: ('not_triggered' if not sig[0] else
                                  'minimum_hold' if i-last_trade[slot] < config.min_holding_days else
                                  'awaiting_confirmation' if confirmation[slot] < config.confirmation_reviews else
                                  'no_better_candidate') for slot,sig in enumerate(signals)}
                # Controls only restrict the standard plan, so one route serves every configuration.
                plan, control_reasons = controlled_plan(held, triggered, ranking[i], eligibility[i],
                                                       data.tickers, data.scores[i] if config.trigger == 'ecr' else metric[i], capital/capital.sum(),
                                                       last_exit, i, config)
                reasons.update(control_reasons)
                top = rank_candidates(ranking[i], eligibility[i], data.tickers, held)
                top = top[0] if top else None
                rebalance_weights = weight_band_breached(capital, config.weight_band)
                weight_row = None
                if config.weight_band is not None:
                    weight_action = (('unexecuted_end' if config.execution_lag and i+1 >= stop else
                                      'scheduled' if config.execution_lag else 'execution_check') if rebalance_weights else 'within_band')
                    weight_reviews.append({'Cohort': cohort.title(), 'Date': date,
                                           'Max Weight Deviation': float(np.abs(capital/capital.sum()-1/n).max()),
                                           'Weight Band': config.weight_band, 'Breached': rebalance_weights,
                                           'Action': weight_action,
                                           'Execution Outcome': {'within_band': 'none', 'unexecuted_end': 'unexecuted_end'}.get(weight_action, 'pending'),
                                           'Execution Date': pd.NaT})
                    if weight_action in ('scheduled', 'execution_check'):
                        weight_row = weight_reviews[-1]
                rows = {}
                review_id = f'{strategy_id}/{cohort}/review/{date.date()}'
                for slot, fund in enumerate(held):
                    new = plan.get(slot, fund)
                    action = 'hold'
                    if slot in triggered:
                        action = 'no_switch' if new == fund else ('switch' if not config.execution_lag else
                                                                 ('scheduled' if i+1 < stop else 'unexecuted_end'))
                    _, ecr_signal, buffer_signal, decline = signals[slot]
                    decisions.append({'Strategy ID': strategy_id, 'Review ID': review_id, 'Position': slot+1,
                                      'Cohort': cohort.title(), 'Date': date, 'Held Fund': data.tickers[fund],
                                      'Held Score': data.scores[i, fund], 'Best Fund': data.tickers[new],
                                      'Best Score': ranking[i, new],
                                      'Top Ranked Fund': data.tickers[top] if top is not None else None,
                                      'Top Ranked Score': ranking[i, top] if top is not None else np.nan,
                                      'Planned Fund': data.tickers[new], 'Planned Score': ranking[i, new],
                                      'Selection': config.selection, 'Holding Days': i-last_trade[slot],
                                      'Triggered': slot in triggered, 'Raw Trigger': signals[slot][0],
                                      'Confirmation Reviews': int(confirmation[slot]), 'Control Reason': reasons[slot],
                                      'Action': action, 'Trigger': config.trigger,
                                      'ECR Signal': ecr_signal, 'Buffer Signal': buffer_signal,
                                      'Buffer Metric': config.buffer_metric, 'Held Buffer Score': metric[i, fund],
                                      'Entry Buffer Score': entry_metric[slot], 'Buffer Deterioration': decline,
                                      'Buffer Drop Threshold': config.buffer_drop, 'Metric Floor': config.metric_floor,
                                      'Fresh Reference Score': 1., 'Metric Fraction of Fresh': metric[i, fund],
                                      'Threshold': config.threshold, 'Comparator': config.comparator,
                                      'Execution Outcome': {'hold': 'none', 'no_switch': 'none',
                                                            'unexecuted_end': 'unexecuted_end'}.get(action, 'pending'),
                                      'Execution Date': pd.NaT})
                    if action in ('switch', 'scheduled'):
                        rows[slot] = decisions[-1]
                if config.candidate_audit:
                    ranked = rank_candidates(ranking[i], eligibility[i], data.tickers, held)
                    ranks = {j: rank+1 for rank,j in enumerate(ranked)}
                    retained = {j for slot,j in enumerate(held) if slot not in triggered}
                    for j,ticker in enumerate(data.tickers):
                        candidates.append({'Strategy ID': strategy_id, 'Review ID': review_id,
                                           'Cohort': cohort.title(), 'Date': date, 'Fund': ticker,
                                           'ECR': data.scores[i,j], 'Selection': config.selection, 'Selection Score': ranking[i,j], 'Eligible': bool(eligibility[i,j]),
                                           'Rank': ranks.get(j), 'Retained Position': j in retained,
                                           'Selected': j in plan.values(), 'Available for Replacement': j not in retained,
                                           'Passes Entry Margin': not config.entry_metric_margin or metric[i,j] >= config.metric_floor+config.entry_metric_margin,
                                           'Outside Cooldown': i-last_exit[j] >= config.reentry_cooldown_days,
                                           'Current Holding': j in held})
                if config.execution_lag:
                    if i+1 < stop and (rebalance_weights or any(new != held[slot] for slot,new in plan.items())):
                        pending = (plan, i, rebalance_weights, rows, weight_row)
                else:
                    execute(plan, i, i, rebalance_weights, rows, weight_row)
            total = float(capital.sum())
            weights = capital / total
            record = {'Strategy ID': strategy_id, 'Cohort': cohort.title(), 'Date': date,
                      'Return Fund': return_funds, 'Fund at Close': ','.join(data.tickers[j] for j in held),
                      'Held ECR Score': float(weights @ data.scores[i, held]),
                      'Held Buffer Score': float(weights @ metric[i, held]),
                      'Entry Buffer Score': float(weights @ entry_metric),
                      'Strategy NAV': total, 'Position Count': n, 'Cumulative Trading Cost': cumulative_cost,
                      'SPY NAV': 100 * data.benchmarks[i,0] / data.benchmarks[start,0],
                      'BUFR NAV': 100 * data.benchmarks[i,1] / data.benchmarks[start,1],
                      'Hold NAV': hold_curve[i - start]}
            cohort_daily.append(record)
            for slot,fund in enumerate(held):
                holdings.append({'Strategy ID': strategy_id, 'Cohort': cohort.title(), 'Date': date,
                                 'Position': slot+1, 'Fund': data.tickers[fund], 'Capital': capital[slot],
                                 'Weight': weights[slot], 'Held ECR Score': data.scores[i,fund],
                                 'Held Buffer Score': metric[i,fund], 'Entry Buffer Score': entry_metric[slot]})
        daily.extend(cohort_daily)
        curve = pd.DataFrame(cohort_daily)
        returns = curve['Strategy NAV'].pct_change().dropna()
        elapsed = (dates[stop-1] - dates[start]).days
        total_return = capital.sum()/100-1
        downside = np.sqrt(np.mean(np.minimum(returns,0)**2))*np.sqrt(252)
        summaries.append({'Strategy ID': strategy_id, 'Cohort': cohort.title(), 'Client': config.client_name,
                          'Portfolio Size': n, 'Start': dates[start], 'End': dates[stop-1],
                          'Observations': len(curve), 'Total Return': total_return,
                          'Annualized Return': (capital.sum()/100)**(365/elapsed)-1 if elapsed else np.nan,
                          'Annualized Volatility': returns.std()*np.sqrt(252),
                          'Sharpe': returns.mean()/returns.std()*np.sqrt(252) if returns.std()>0 else np.nan,
                          'Annualized Downside Deviation': downside,
                          'Sortino': returns.mean()*252/downside if downside>0 else np.nan,
                          'Max Drawdown': (curve['Strategy NAV']/curve['Strategy NAV'].cummax()-1).min(),
                          'Switches': switches, 'Switches per Year': switches*365/elapsed if elapsed else np.nan,
                          'One-way Turnover': cumulative_turnover, 'Trading Cost NAV Units': cumulative_cost,
                          'Weight Rebalances': weight_rebalances,
                          'Missing Benchmark Sessions': benchmark_gaps,
                          'SPY Total Return': curve['SPY NAV'].iloc[-1]/100-1,
                          'BUFR Total Return': curve['BUFR NAV'].iloc[-1]/100-1,
                          'vs BUFR Cumulative Excess': total_return - (curve['BUFR NAV'].iloc[-1]/100-1)})
    candidate_frame = pd.DataFrame(candidates)
    if not candidate_frame.empty:
        raw = data.rows.reset_index().rename(columns={'date':'Date','ticker':'Fund',
                                                     **{v:k for k,v in DIAGNOSTIC_MAP.items()}})
        columns = ['Date','Fund',*[c for c in DIAGNOSTIC_MAP if c != 'ECR']]
        candidate_frame = candidate_frame.merge(raw[columns],on=['Date','Fund'],how='left',validate='many_to_one')
    return {'parameters': parameters, 'strategy_id': strategy_id, 'summary': pd.DataFrame(summaries),
            'daily': pd.DataFrame(daily), 'holdings': pd.DataFrame(holdings),
            'diagnostics': pd.DataFrame(diagnostics), 'decisions': pd.DataFrame(decisions),
            'candidates': candidate_frame, 'trades': pd.DataFrame(trades),
            'weight_reviews': pd.DataFrame(weight_reviews)}
