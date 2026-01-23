"""
Trigger functions for determining WHEN to switch funds.
"""

import pandas as pd


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


def trigger_cap_utilization_threshold(fund_data_row, threshold):
    """
    Threshold trigger: fires when cap utilization exceeds threshold.

    Parameters:
      fund_data_row: Series with fund data for current date
      threshold: Threshold as decimal (e.g., 0.75 for 75%)

    Returns:
      Boolean: True if Cap_Utilization >= threshold
    """
    if pd.isna(fund_data_row['Cap_Utilization']):
        return False
    return fund_data_row['Cap_Utilization'] >= threshold


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
    downside_col = 'Downside Before Buffer (%)'
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
    ref_return_col = 'Reference Asset Return (%)'
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

# Trigger registry for dynamic lookup
TRIGGER_REGISTRY = {
    'rebalance_time_period': trigger_rebalance_time_period,
    'remaining_cap_threshold': trigger_remaining_cap_threshold,
    'cap_utilization_threshold': trigger_cap_utilization_threshold,
    'downside_before_buffer_threshold': trigger_downside_before_buffer_threshold,
    'ref_asset_return_threshold': trigger_ref_asset_return_threshold,
    'remaining_buffer_threshold': trigger_remaining_buffer_threshold,

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