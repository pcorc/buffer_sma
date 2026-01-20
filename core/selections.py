"""
Selection functions for determining WHAT fund to switch to.
"""

import pandas as pd
import numpy as np


# Add these after your existing functions, before SELECTION_REGISTRY

def select_remaining_cap(df_universe, current_date, series='F'):
    """
    Legacy alias for select_remaining_cap_highest.
    Selects fund with highest remaining cap (most upside potential).
    """
    return select_remaining_cap_highest(df_universe, current_date, series)


def select_cap_utilization(df_universe, current_date, series='F'):
    """
    Legacy alias for select_cap_utilization_lowest.
    Selects fund with lowest cap utilization (most cap remaining).
    """
    return select_cap_utilization_lowest(df_universe, current_date, series)

def select_most_recent_launch(df_universe, current_date, series='F'):
    """
    Select the fund with the most recent roll date.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker
    """
    if df_universe.empty:
        return None

    valid_funds = df_universe[df_universe['Roll_Date'] <= current_date].copy()

    if valid_funds.empty:
        return df_universe.loc[df_universe['Remaining Outcome Days'].idxmax(), 'Fund']

    most_recent_idx = valid_funds['Roll_Date'].idxmax()
    return valid_funds.loc[most_recent_idx, 'Fund']


def select_remaining_cap_highest(df_universe, current_date, series='F'):
    """
    Select the fund with the HIGHEST remaining cap (bullish).

    Seeks maximum upside potential by choosing funds with the most cap remaining.
    If multiple funds have the same cap, selects the most recent launch.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker with highest remaining cap
    """
    if df_universe.empty:
        return None

    # Find maximum remaining cap
    max_cap = df_universe['Remaining Cap'].max()

    # Filter to funds with max cap
    max_cap_funds = df_universe[df_universe['Remaining Cap'] == max_cap].copy()

    # If tie, use most recent launch as tiebreaker
    if len(max_cap_funds) > 1:
        return select_most_recent_launch(max_cap_funds, current_date, series)

    return max_cap_funds.iloc[0]['Fund']


def select_remaining_cap_lowest(df_universe, current_date, series='F'):
    """
    Select the fund with the LOWEST remaining cap (bearish/conservative).

    Chooses funds with caps nearly exhausted - more conservative positioning.
    If multiple funds have the same cap, selects the most recent launch.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker with lowest remaining cap
    """
    if df_universe.empty:
        return None

    # Find minimum remaining cap
    min_cap = df_universe['Remaining Cap'].min()

    # Filter to funds with min cap
    min_cap_funds = df_universe[df_universe['Remaining Cap'] == min_cap].copy()

    # If tie, use most recent launch as tiebreaker
    if len(min_cap_funds) > 1:
        return select_most_recent_launch(min_cap_funds, current_date, series)

    return min_cap_funds.iloc[0]['Fund']


def select_downside_buffer_highest(df_universe, current_date, series='F'):
    """
    Select the fund with the HIGHEST downside before buffer (bullish).

    Chooses funds with the most cushion before hitting buffer zone.
    Higher downside % means more room to fall = more aggressive positioning.
    If multiple funds have the same downside, selects the most recent launch.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker with highest downside before buffer
    """
    if df_universe.empty:
        return None

    downside_col = 'Downside Before Buffer (%)'

    if downside_col not in df_universe.columns:
        return df_universe.iloc[0]['Fund']

    # Filter out any null values
    valid_funds = df_universe[df_universe[downside_col].notna()].copy()

    if valid_funds.empty:
        return df_universe.iloc[0]['Fund']

    # Find maximum downside
    max_downside = valid_funds[downside_col].max()

    # Filter to funds with max downside
    max_downside_funds = valid_funds[valid_funds[downside_col] == max_downside].copy()

    # If tie, use most recent launch as tiebreaker
    if len(max_downside_funds) > 1:
        return select_most_recent_launch(max_downside_funds, current_date, series)

    return max_downside_funds.iloc[0]['Fund']


def select_downside_buffer_lowest(df_universe, current_date, series='F'):
    """
    Select the fund with the LOWEST downside before buffer (bearish/defensive).

    Chooses funds closest to or in the buffer zone - maximum downside protection.
    Lower downside % means closer to buffer = more defensive positioning.
    If multiple funds have the same downside, selects the most recent launch.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker with lowest downside before buffer
    """
    if df_universe.empty:
        return None

    downside_col = 'Downside Before Buffer (%)'

    if downside_col not in df_universe.columns:
        return df_universe.iloc[0]['Fund']

    # Filter out any null values
    valid_funds = df_universe[df_universe[downside_col].notna()].copy()

    if valid_funds.empty:
        return df_universe.iloc[0]['Fund']

    # Find minimum downside
    min_downside = valid_funds[downside_col].min()

    # Filter to funds with min downside
    min_downside_funds = valid_funds[valid_funds[downside_col] == min_downside].copy()

    # If tie, use most recent launch as tiebreaker
    if len(min_downside_funds) > 1:
        return select_most_recent_launch(min_downside_funds, current_date, series)

    return min_downside_funds.iloc[0]['Fund']


def select_cap_utilization_lowest(df_universe, current_date, series='F'):
    """
    Select the fund with the LOWEST cap utilization (bullish).

    Low cap utilization = most cap remaining = maximum upside potential.
    Example: 25% utilization means 75% of cap is still available.
    If multiple funds have the same utilization, selects the most recent launch.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker with lowest cap utilization
    """
    if df_universe.empty:
        return None

    if 'Cap_Utilization' not in df_universe.columns:
        return df_universe.iloc[0]['Fund']

    # Filter out any null values
    valid_funds = df_universe[df_universe['Cap_Utilization'].notna()].copy()

    if valid_funds.empty:
        return df_universe.iloc[0]['Fund']

    # Find minimum utilization
    min_util = valid_funds['Cap_Utilization'].min()

    # Filter to funds with min utilization
    min_util_funds = valid_funds[valid_funds['Cap_Utilization'] == min_util].copy()

    # If tie, use most recent launch as tiebreaker
    if len(min_util_funds) > 1:
        return select_most_recent_launch(min_util_funds, current_date, series)

    return min_util_funds.iloc[0]['Fund']


def select_cap_utilization_highest(df_universe, current_date, series='F'):
    """
    Select the fund with the HIGHEST cap utilization (bearish/conservative).

    High cap utilization = little cap remaining = conservative positioning.
    Example: 85% utilization means only 15% of cap is available.
    If multiple funds have the same utilization, selects the most recent launch.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker with highest cap utilization
    """
    if df_universe.empty:
        return None

    if 'Cap_Utilization' not in df_universe.columns:
        return df_universe.iloc[0]['Fund']

    # Filter out any null values
    valid_funds = df_universe[df_universe['Cap_Utilization'].notna()].copy()

    if valid_funds.empty:
        return df_universe.iloc[0]['Fund']

    # Find maximum utilization
    max_util = valid_funds['Cap_Utilization'].max()

    # Filter to funds with max utilization
    max_util_funds = valid_funds[valid_funds['Cap_Utilization'] == max_util].copy()

    # If tie, use most recent launch as tiebreaker
    if len(max_util_funds) > 1:
        return select_most_recent_launch(max_util_funds, current_date, series)

    return max_util_funds.iloc[0]['Fund']


def select_highest_outcome_and_cap(df_universe, current_date, series='F'):
    """
    Select fund with highest combined Remaining Outcome Days + Remaining Cap.

    If multiple funds have the same score, selects the most recent launch.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker
    """
    if df_universe.empty:
        return None

    df_universe = df_universe.copy()
    df_universe['Combined_Score'] = (
            df_universe['Remaining Outcome Days'] / 365 * 100 +
            df_universe['Remaining Cap']
    )

    # Find maximum score
    max_score = df_universe['Combined_Score'].max()

    # Filter to funds with max score
    max_score_funds = df_universe[df_universe['Combined_Score'] == max_score].copy()

    # If tie, use most recent launch as tiebreaker
    if len(max_score_funds) > 1:
        return select_most_recent_launch(max_score_funds, current_date, series)

    return max_score_funds.iloc[0]['Fund']


def select_cost_analysis(df_universe, current_date, series='F'):
    """
    Select fund with lowest cost per day of protection.

    If multiple funds have the same cost, selects the most recent launch.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker
    """
    if df_universe.empty:
        return None

    df_universe = df_universe.copy()
    downside_col = 'Downside Before Buffer (%)'
    df_universe['Cost_Per_Day'] = np.inf

    valid_mask = (
            (df_universe['Remaining Outcome Days'] > 0) &
            (df_universe[downside_col].notna()) &
            (df_universe[downside_col] != 0)
    )

    df_universe.loc[valid_mask, 'Cost_Per_Day'] = (
            df_universe.loc[valid_mask, 'Fund Value (USD)'] /
            (abs(df_universe.loc[valid_mask, downside_col] / 100) *
             (df_universe.loc[valid_mask, 'Remaining Outcome Days'] / 365))
    )

    # Find minimum cost
    min_cost = df_universe['Cost_Per_Day'].min()

    # Filter to funds with min cost
    min_cost_funds = df_universe[df_universe['Cost_Per_Day'] == min_cost].copy()

    # If tie, use most recent launch as tiebreaker
    if len(min_cost_funds) > 1:
        return select_most_recent_launch(min_cost_funds, current_date, series)

    return min_cost_funds.iloc[0]['Fund']


# Selection registry for dynamic lookup
SELECTION_REGISTRY = {
    'select_most_recent_launch': select_most_recent_launch,
    'select_remaining_cap': select_remaining_cap,  # Add this
    'select_cap_utilization': select_cap_utilization,  # Add this
    'select_remaining_cap_highest': select_remaining_cap_highest,
    'select_remaining_cap_lowest': select_remaining_cap_lowest,
    'select_cap_utilization_lowest': select_cap_utilization_lowest,
    'select_cap_utilization_highest': select_cap_utilization_highest,
    'select_downside_buffer_highest': select_downside_buffer_highest,
    'select_downside_buffer_lowest': select_downside_buffer_lowest,
    'select_highest_outcome_and_cap': select_highest_outcome_and_cap,
    'select_cost_analysis': select_cost_analysis
}


def get_selection_function(selection_name):
    """
    Get selection function by name.

    Parameters:
      selection_name: String name of selection function

    Returns:
      Function reference
    """
    if selection_name not in SELECTION_REGISTRY:
        raise ValueError(f"Unknown selection function: {selection_name}")

    return SELECTION_REGISTRY[selection_name]