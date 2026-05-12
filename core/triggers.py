"""
Trigger functions for determining WHEN to switch funds.
"""

import pandas as pd
import numpy as np
from core.selections import compute_ecr_scores

def trigger_rebalance_time_period(current_date, roll_dates_list):
    """
    Time-based trigger: fires on specific roll dates.

    Parameters:
      current_date: Current date being evaluated
      roll_dates_list: List of dates when rebalancing should occur

    Returns:
      Boolean: True if current_date is in roll_dates_list
    """
    return current_date in roll_dates_list


def trigger_remaining_cap_threshold(fund_data_row, threshold):
    """
    Threshold trigger: fires when remaining cap % falls below threshold.

    Parameters:
      fund_data_row: Series with fund data for current date
      threshold: Threshold as decimal (e.g., 0.25 for 25%)

    Returns:
      Boolean: True if Cap_Remaining_Pct < threshold
    """
    if pd.isna(fund_data_row['Cap_Remaining_Pct']):
        return False
    return fund_data_row['Cap_Remaining_Pct'] < threshold


def trigger_downside_before_buffer_threshold(fund_data_row, threshold):
    """
    Threshold trigger: fires when downside before buffer falls below threshold.
    Useful for detecting when fund enters buffer zone.

    Parameters:
      fund_data_row: Series with fund data for current date
      threshold: Threshold as decimal (e.g., 0.0 for in-buffer)

    Returns:
      Boolean: True if Downside Before Buffer <= threshold
    """
    downside_col = 'Downside Before Buffer'
    if downside_col not in fund_data_row.index:
        return False

    downside_value = abs(fund_data_row[downside_col] / 100)
    if pd.isna(downside_value):
        return False

    return threshold <= downside_value


def trigger_ref_asset_return_threshold(fund_data_row, threshold):
    """
    Threshold trigger: fires when reference asset return crosses threshold.

    Parameters:
      fund_data_row: Series with fund data for current date
      threshold: Threshold as decimal (e.g., -0.05 for -5%)

    Returns:
      Boolean: True if Reference Asset Return crosses threshold
    """
    ref_return_col = 'Reference Asset Return'
    if ref_return_col not in fund_data_row.index:
        return False

    ref_return = fund_data_row[ref_return_col] / 100
    if pd.isna(ref_return):
        return False

    if threshold < 0:
        return ref_return <= threshold
    else:
        return ref_return >= threshold


def trigger_remaining_buffer_threshold(fund_data_row, threshold):
    """
    Threshold trigger: fires when buffer cushion is depleted beyond threshold.

    Fires when: Buffer_Depletion_Ratio > threshold

    Buffer_Depletion_Ratio = (Original_Buffer - Remaining_Buffer) / Original_Buffer

    Example (F-series, Original Buffer = 10%):
    - threshold = 0.15 → fires when 15% of buffer used (Remaining = 8.5%)
    - threshold = 0.50 → fires when 50% of buffer used (Remaining = 5.0%)
    - threshold = 0.85 → fires when 85% of buffer used (Remaining = 1.5%)

    Interpretation:
    - threshold = 0.15 (15%): Early warning, conservative trigger
      Market decline is small, but we rotate proactively
    - threshold = 0.50 (50%): Moderate depletion
      Market decline is significant, buffer halfway consumed
    - threshold = 0.85 (85%): Late trigger, deep into buffer zone
      Market decline is severe, buffer almost exhausted

    This is BEARISH because we're detecting market stress (buffer consumption).
    Higher threshold = more patient, waits for severe drawdown before rotating.

    Parameters:
      fund_data_row: Series with fund data for current date
      threshold: Depletion ratio as decimal (e.g., 0.50 for 50% depletion)

    Returns:
      Boolean: True if buffer depletion exceeds threshold
    """
    buffer_col = 'Remaining Buffer'
    original_buffer_col = 'Original_Buffer'

    if buffer_col not in fund_data_row.index or original_buffer_col not in fund_data_row.index:
        return False

    remaining_buffer = fund_data_row[buffer_col]
    original_buffer = fund_data_row[original_buffer_col]

    # Handle missing values
    if pd.isna(remaining_buffer) or pd.isna(original_buffer):
        return False

    # Convert Remaining Buffer from percentage to decimal if needed
    # Original_Buffer is already in decimal from preprocessor
    remaining_buffer_decimal = remaining_buffer / 100

    # Calculate buffer depletion ratio
    # When market rallies: remaining > original → depletion is negative (no trigger)
    # When market falls: remaining < original → depletion is positive (potential trigger)
    buffer_depletion = (original_buffer - remaining_buffer_decimal) / original_buffer

    # Fire when depletion exceeds threshold
    # Example: threshold=0.50 means fire when 50% of buffer cushion is used up
    return buffer_depletion > threshold


def cap_or_buffer_utilization_threshold(df_universe, current_holdings, current_date, threshold=0.90):
    """
    Trigger rotation when EITHER cap OR buffer utilization exceeds threshold.

    This is a compound OR condition:
    - Triggers if cap is 90%+ utilized (only 10% upside remaining)
    - OR if buffer is 90%+ utilized (only 10% protection remaining)

    Cap Utilization = (Original Cap - Remaining Cap) / Original Cap
    Buffer Utilization = (Original Buffer - Remaining Buffer Net) / Original Buffer

    Parameters:
        df_universe: DataFrame with all available funds
        current_holdings: DataFrame with current position
        current_date: Current date
        threshold: Utilization threshold (default 0.90 = 90%)

    Returns:
        bool: True if rotation should occur, False otherwise
    """
    if current_holdings.empty:
        return False

    current_fund = current_holdings.iloc[0]

    # =========================================================================
    # Check Cap Utilization
    # =========================================================================

    if 'Cap_Utilization' in current_fund:
        cap_util = current_fund['Cap_Utilization']
    elif 'Remaining Cap' in current_fund and 'Original_Cap' in current_fund:
        remaining_cap = current_fund['Remaining Cap']
        original_cap = current_fund['Original_Cap']

        if original_cap > 0:
            cap_util = 1.0 - (remaining_cap / 100.0 / original_cap)
        else:
            cap_util = 0.0
    else:
        cap_util = 0.0

    # =========================================================================
    # Check Buffer Utilization
    # =========================================================================

    if 'Remaining Buffer' in current_fund and 'Original_Buffer' in current_fund:
        remaining_buffer = current_fund['Remaining Buffer']
        original_buffer = current_fund['Original_Buffer']

        if original_buffer > 0:
            # Buffer utilization = portion of original buffer consumed
            buffer_util = max(0.0, (original_buffer - remaining_buffer / 100.0) / original_buffer)
        else:
            buffer_util = 0.0
    else:
        buffer_util = 0.0

    # =========================================================================
    # Compound OR Logic
    # =========================================================================

    should_trigger = (cap_util >= threshold) or (buffer_util >= threshold)

    # Debug logging (optional - comment out in production)
    if should_trigger:
        trigger_reason = []
        if cap_util >= threshold:
            trigger_reason.append(f"Cap {cap_util * 100:.1f}% utilized")
        if buffer_util >= threshold:
            trigger_reason.append(f"Buffer {buffer_util * 100:.1f}% utilized")
        print(f"  → Trigger: {' OR '.join(trigger_reason)}")

    return should_trigger

def trigger_cap_utilization_threshold(fund_data_row, threshold):
    """
    Combined trigger: fires when EITHER cap utilization OR buffer utilization
    exceeds threshold.

    This is the 'Existing 90%' baseline strategy:
    - Upside event:   Cap_Utilization >= threshold  (market rallied, cap nearly consumed)
    - Downside event: Buffer_Utilization >= threshold (market fell, buffer nearly consumed)

    Fires on whichever happens first — captures both directional scenarios.

    Parameters:
      fund_data_row: Series with fund data for current date
      threshold: Threshold as decimal (e.g., 0.90 for 90%)

    Returns:
      Boolean: True if Cap_Utilization >= threshold OR Buffer_Utilization >= threshold
    """
    cap_util    = fund_data_row.get('Cap_Utilization', np.nan)
    buffer_util = fund_data_row.get('Buffer_Utilization', np.nan)

    cap_triggered    = pd.notna(cap_util)    and cap_util    >= threshold
    buffer_triggered = pd.notna(buffer_util) and buffer_util >= threshold

    return cap_triggered or buffer_triggered


def trigger_buffer_utilization_threshold(fund_data_row, threshold):
    """
    Threshold trigger: fires when buffer utilization exceeds threshold.

    Buffer_Utilization = (Original_Buffer_Net - Remaining_Buffer_Net) / Original_Buffer_Net

    A rising buffer utilization means the market is falling into the buffer zone.
    Fire when buffer has been consumed beyond the threshold.

    Example: threshold=0.90 fires when 90% of the buffer has been used up.

    Parameters:
      fund_data_row: Series with fund data for current date
      threshold: Threshold as decimal (e.g., 0.90 for 90%)

    Returns:
      Boolean: True if Buffer_Utilization >= threshold
    """
    if 'Buffer_Utilization' not in fund_data_row.index:
        return False
    if pd.isna(fund_data_row['Buffer_Utilization']):
        return False
    return fund_data_row['Buffer_Utilization'] >= threshold


def trigger_ecr_percentile(
    current_fund,
    df_universe,
    series,
    w_dbb, w_buffer, w_cap,
    spread_history,
    percentile_threshold,
    min_holding_days,
    days_since_rebalance,
    rolling_window=252,
    min_absolute_spread=0.05  # ← ADD: require at least 5% spread

):
    """
    Trigger that fires when the ECR spread between the current fund and the
    best available fund exceeds a rolling percentile threshold.

    Spread = (best_ecr - current_ecr) / best_ecr

    Logic:
    1. Enforce minimum holding period — no trigger before min_holding_days
    2. Score all funds using shared compute_ecr_scores helper
    3. Compute today's spread between current fund and best fund
    4. Append spread to rolling history
    5. Fire when today's spread >= percentile_threshold of rolling history

    Parameters:
        current_fund: Ticker of currently held fund (e.g. 'FMAR')
        df_universe: DataFrame of all available funds on current date
        series: Fund series letter (e.g. 'F')
        w_dbb: DBB weight for ECR scoring
        w_buffer: Buffer Integrity weight for ECR scoring
        w_cap: Cap Integrity weight for ECR scoring
        spread_history: List of daily spreads — mutated in place each call
        percentile_threshold: Float 0-1, e.g. 0.75 = fire at 75th percentile
        min_holding_days: Minimum trading days before trigger can fire
        days_since_rebalance: Trading days elapsed since last rebalance
        rolling_window: Lookback days for percentile calculation (default 252)

    Returns:
        tuple: (triggered: bool, spread: float)
            triggered — whether to fire the rebalance
            spread    — today's spread value (caller appends to history)
    """
    from core.selections import compute_ecr_scores

    # Step 1: Enforce minimum holding period
    if days_since_rebalance < min_holding_days:
        # Still compute and return spread for history tracking, but don't fire
        score_dict = compute_ecr_scores(df_universe, series, w_dbb, w_buffer, w_cap)
        current_score = score_dict.get(current_fund, np.nan)
        if not score_dict or pd.isna(current_score):
            return False, np.nan
        best_score = max(score_dict.values())
        spread = (best_score - current_score) / best_score if best_score > 0 else 0.0
        return False, spread

    # Step 2: Score all funds
    score_dict = compute_ecr_scores(df_universe, series, w_dbb, w_buffer, w_cap)

    if not score_dict:
        return False, np.nan

    current_score = score_dict.get(current_fund, np.nan)

    if pd.isna(current_score):
        return False, np.nan

    # Step 3: Compute today's spread
    best_score = max(score_dict.values())
    best_fund = max(score_dict, key=score_dict.get)

    # No point switching to the same fund
    if best_fund == current_fund:
        spread = 0.0
        return False, spread

    spread = (best_score - current_score) / best_score if best_score > 0 else 0.0

    # Step 4: Need enough history to compute a meaningful percentile
    # Use expanding window until rolling_window days are available
    window = spread_history[-rolling_window:] if len(spread_history) >= rolling_window else spread_history

    if len(window) < 20:
        # Not enough history yet — don't fire, just track
        return False, spread

    # Step 5: Absolute spread gate — must exceed minimum before percentile check
    if spread < min_absolute_spread:
        return False, spread

    # Step 6: Compute percentile rank of today's spread in rolling window
    percentile_rank = np.mean(np.array(window) <= spread)

    # Fire only if BOTH conditions met: absolute size AND percentile rank
    triggered = percentile_rank >= percentile_threshold

    return triggered, spread

def trigger_ecr_score_threshold(
    current_fund,
    df_universe,
    series,
    w_dbb, w_buffer, w_cap,
    score_threshold,
    min_holding_days,
    days_since_rebalance
):
    """
    Trigger that fires when the current fund's ECR score falls below
    an absolute threshold.

    No history needed — purely compares current score against a fixed floor.
    Fires when: current_ecr < score_threshold AND days_since_rebalance >= min_holding_days

    Parameters:
        current_fund: Ticker of currently held fund
        df_universe: DataFrame of all available funds on current date
        series: Fund series letter
        w_dbb: DBB weight
        w_buffer: Buffer Integrity weight
        w_cap: Cap Integrity weight
        score_threshold: Float — fire when current score drops below this
        min_holding_days: Minimum days before trigger can fire
        days_since_rebalance: Trading days since last rebalance

    Returns:
        bool: True if trigger fires
    """

    if days_since_rebalance < min_holding_days:
        return False

    score_dict = compute_ecr_scores(df_universe, series, w_dbb, w_buffer, w_cap)
    current_score = score_dict.get(current_fund, np.nan)

    if pd.isna(current_score):
        return False

    return current_score < score_threshold


def trigger_ecr_percentile_rank(
    current_fund,
    df_universe,
    series,
    w_dbb, w_buffer, w_cap,
    percentile_threshold,
    min_holding_days,
    days_since_rebalance
):
    """
    Trigger that fires when the current fund's ECR score falls below
    a given percentile rank within the available universe on that date.

    Example: percentile_threshold=50 fires when the current fund scores
    below the median of all available funds — i.e. it is in the bottom half.

    Parameters:
        percentile_threshold: int — 0 to 100 (e.g. 50 = median, 25 = bottom quartile)
        min_holding_days: minimum trading days before trigger can fire
        days_since_rebalance: trading days since last rebalance

    Returns:
        bool: True if trigger fires
    """
    from core.selections import compute_ecr_scores

    if days_since_rebalance < min_holding_days:
        return False

    score_dict = compute_ecr_scores(df_universe, series, w_dbb, w_buffer, w_cap)

    if not score_dict:
        return False

    current_score = score_dict.get(current_fund, np.nan)

    if pd.isna(current_score):
        return False

    all_scores   = np.array(list(score_dict.values()))
    pct_cutoff   = np.percentile(all_scores, percentile_threshold)

    return current_score < pct_cutoff


def trigger_ecr_rank_threshold(
        current_fund,
        df_universe,
        series,
        w_dbb, w_buffer, w_cap,
        rank_threshold,
        min_holding_days,
        days_since_rebalance
):
    """
    Trigger that fires when the current fund's ordinal rank in the universe
    exceeds a threshold.

    Ranks are 1-indexed with 1 = best score (highest ECR). Fires when
    current_rank > rank_threshold.

    Example: rank_threshold = 1.5 → fires when current_rank ≥ 2
             (i.e. anytime the current fund is not the top-ranked fund)

    Note: Uses v1 ECR composite via compute_ecr_scores (same as other
    ecr_* triggers). Selection function — which may use par-proximity —
    is called separately by the engine.

    Parameters:
        rank_threshold: float — fires when current_rank > threshold (1 = best)
        min_holding_days: minimum trading days before trigger can fire
        days_since_rebalance: trading days since last rebalance

    Returns:
        bool: True if trigger fires
    """
    if days_since_rebalance < min_holding_days:
        return False

    score_dict = compute_ecr_scores(df_universe, series, w_dbb, w_buffer, w_cap)

    if not score_dict:
        return False

    current_score = score_dict.get(current_fund, np.nan)
    if pd.isna(current_score):
        return False

    # Rank high-to-low (best score = rank 1)
    sorted_funds = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    current_rank = next(
        (i + 1 for i, (f, _) in enumerate(sorted_funds) if f == current_fund),
        None
    )

    if current_rank is None:
        return False

    return current_rank > rank_threshold




# Trigger registry for dynamic lookup
TRIGGER_REGISTRY = {
    'rebalance_time_period': trigger_rebalance_time_period,
    'remaining_cap_threshold': trigger_remaining_cap_threshold,
    'cap_utilization_threshold': trigger_cap_utilization_threshold,
    'downside_before_buffer_threshold': trigger_downside_before_buffer_threshold,
    'ref_asset_return_threshold': trigger_ref_asset_return_threshold,
    'remaining_buffer_threshold': trigger_remaining_buffer_threshold,
    'cap_or_buffer_utilization_threshold': cap_or_buffer_utilization_threshold,
    'ecr_percentile_trigger': trigger_ecr_percentile,
    'ecr_score_threshold': trigger_ecr_score_threshold,
    'buffer_utilization_threshold': trigger_buffer_utilization_threshold,
    'ecr_percentile_rank': trigger_ecr_percentile_rank,
    'ecr_rank_threshold': trigger_ecr_rank_threshold,
}


def get_trigger_function(trigger_type):
    """
    Get trigger function by name.

    Parameters:
      trigger_type: String name of trigger

    Returns:
      Function reference
    """
    if trigger_type not in TRIGGER_REGISTRY:
        raise ValueError(f"Unknown trigger type: {trigger_type}")

    return TRIGGER_REGISTRY[trigger_type]