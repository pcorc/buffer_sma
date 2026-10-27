"""Candidate ranking is independent of the rule that triggered a review."""
import numpy as np


def rank_candidates(scores, eligible, tickers, preferred=()):
    preferred = set(preferred)
    return sorted(np.flatnonzero(eligible),
                  key=lambda j: (-scores[j], j not in preferred, tickers[j]))


def replacement_plan(held, triggered_positions, scores, eligible, tickers):
    """Jointly rank free candidates; preserve untouched positions and unique tickers.

    Triggered incumbents may remain if still among the top available ECR funds.
    Exact ties retain incumbents. Insufficient alternatives produce an explicit no-op.
    """
    triggered = sorted(triggered_positions)
    retained = {fund for slot, fund in enumerate(held) if slot not in triggered}
    free = eligible.copy()
    for fund in retained:
        free[fund] = False
    ranked = rank_candidates(scores, free, tickers, (held[slot] for slot in triggered))
    selected = ranked[:len(triggered)]
    # Keep selected incumbents in their existing sleeves, avoiding pointless swaps.
    plan = {slot: held[slot] for slot in triggered if held[slot] in selected}
    replacements = [fund for fund in selected if fund not in plan.values()]
    for slot in triggered:
        if slot not in plan:
            plan[slot] = replacements.pop(0) if replacements else held[slot]
    return plan
