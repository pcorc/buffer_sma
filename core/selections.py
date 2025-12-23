"""
Selection functions for determining WHAT fund to switch to.
"""

import pandas as pd
import numpy as np


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


def select_remaining_cap(df_universe, current_date, series='F'):
    """
    Select the fund with the highest remaining cap.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker
    """
    if df_universe.empty:
        return None

    max_cap_idx = df_universe['Remaining Cap'].idxmax()
    return df_universe.loc[max_cap_idx, 'Fund']


def select_cap_utilization(df_universe, current_date, series='F'):
    """
    Select the fund with the lowest cap utilization.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker
    """
    if df_universe.empty:
        return None

    min_util_idx = df_universe['Cap_Utilization'].idxmin()
    return df_universe.loc[min_util_idx, 'Fund']


def select_highest_outcome_and_cap(df_universe, current_date, series='F'):
    """
    Select fund with highest combined Remaining Outcome Days + Remaining Cap.

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

    max_score_idx = df_universe['Combined_Score'].idxmax()
    return df_universe.loc[max_score_idx, 'Fund']


def select_cost_analysis(df_universe, current_date, series='F'):
    """
    Select fund with lowest cost per day of protection.

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

    min_cost_idx = df_universe['Cost_Per_Day'].idxmin()
    return df_universe.loc[min_cost_idx, 'Fund']


# Selection registry for dynamic lookup
SELECTION_REGISTRY = {
    'select_most_recent_launch': select_most_recent_launch,
    'select_remaining_cap': select_remaining_cap,
    'select_cap_utilization': select_cap_utilization,
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