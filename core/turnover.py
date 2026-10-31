"""Deterministic replacement controls; all inputs belong to the signal session."""
import numpy as np

from core.selection import replacement_plan


def controlled_plan(held, triggered, ranking, eligible, tickers, metric, weights,
                    last_exit, index, config):
    """The standard replacement plan, restricted by controls; non-binding controls change nothing.

    Entry margin and cooldown restrict which outside funds may be bought. Improvement,
    count and budget gates are checked worst triggered metric first; a slot that fails
    keeps its fund, and the standard plan is rebuilt for the remaining slots so the
    trades that survive still buy the best available funds. A budget cannot partly sell
    a sleeve: it defers the entire replacement, preserving N.
    """
    free = eligible.copy()
    if config.entry_metric_margin > 0:
        free &= metric >= config.metric_floor + config.entry_metric_margin
    if config.reentry_cooldown_days > 0:
        free &= index - last_exit >= config.reentry_cooldown_days
    for slot in triggered:
        # Staying put is neither a purchase nor a re-entry.
        free[held[slot]] = eligible[held[slot]]
    order = sorted(triggered, key=lambda slot: (metric[held[slot]], tickers[held[slot]]))
    active, reasons = list(order), {}
    while True:
        # Each pass removes at least one slot or finishes, so this ends within N passes.
        proposal = replacement_plan(held, active, ranking, free, tickers)
        plan, spent, rejected = {}, 0.0, False
        for slot in active:
            old, new = held[slot], proposal[slot]
            if new == old:
                reasons[slot] = 'no_better_candidate'
                continue
            if ranking[new] - ranking[old] + 1e-12 < config.min_selection_improvement:
                reasons[slot] = 'insufficient_improvement'
            elif config.max_replacements_per_review and len(plan) >= config.max_replacements_per_review:
                reasons[slot] = 'replacement_limit'
            elif spent + weights[slot] > config.max_turnover_per_review + 1e-12:
                reasons[slot] = 'turnover_budget'
            else:
                plan[slot] = new
                spent += weights[slot]
                reasons[slot] = 'replacement_allowed'
                continue
            active.remove(slot)
            rejected = True
            break
        if not rejected:
            return plan, reasons


def weight_band_breached(capital, band):
    if band is None:
        return False
    weights = capital / np.sum(capital)
    return bool(np.any(np.abs(weights - 1 / len(capital)) > band + 1e-12))
