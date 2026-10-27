"""Deterministic replacement controls; all inputs belong to the signal session."""
import numpy as np

from core.selection import rank_candidates


def controlled_plan(held, triggered, ranking, eligible, tickers, metric, weights,
                    last_exit, index, config):
    """Replace worst triggered components first, never rotate already-held funds.

    A budget cannot partly sell a sleeve: it defers the entire replacement.
    This preserves N and makes a low turnover budget's consequences explicit.
    """
    used = set(held)
    plan, reasons = {}, {}
    spent = 0.0
    count = 0
    for slot in sorted(triggered, key=lambda slot: (metric[held[slot]], tickers[held[slot]])):
        old = held[slot]
        reason = 'no_better_candidate'
        free = eligible.copy()
        free[list(used)] = False
        if config.entry_metric_margin > 0:
            free &= metric >= config.metric_floor + config.entry_metric_margin
        if config.reentry_cooldown_days > 0:
            free &= index - last_exit >= config.reentry_cooldown_days
        candidates = rank_candidates(ranking, free, tickers)
        if candidates:
            new = candidates[0]
            gain = ranking[new] - ranking[old]
            if gain <= 0 or gain + 1e-12 < config.min_selection_improvement:
                reason = 'insufficient_improvement'
            elif config.max_replacements_per_review and count >= config.max_replacements_per_review:
                reason = 'replacement_limit'
            elif spent + weights[slot] > config.max_turnover_per_review + 1e-12:
                reason = 'turnover_budget'
            else:
                plan[slot] = new
                used.add(new)
                count += 1
                spent += weights[slot]
                reason = 'replacement_allowed'
        reasons[slot] = reason
    return plan, reasons


def weight_band_breached(capital, band):
    if band is None:
        return False
    weights = capital / np.sum(capital)
    return bool(np.any(np.abs(weights - 1 / len(capital)) > band + 1e-12))
