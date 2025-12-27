"""
Date utility functions for backtesting framework.
"""

import pandas as pd
from pandas.tseries.offsets import BDay


def get_trading_date(roll_date: pd.Timestamp, available_dates: pd.Series) -> pd.Timestamp:
    """
    Get T+1 business day after roll_date, ensuring data exists.

    Roll dates are typically third Fridays. T+1 means the next business day,
    which is usually Monday (3 calendar days later).

    Parameters:
        roll_date: The roll date (third Friday)
        available_dates: Series of available dates to check against

    Returns:
        pd.Timestamp of T+1 business day with data availability
    """
    # Calculate T+1
    trading_date = roll_date + BDay(1)

    # Verify data exists on trading date
    available_dates_set = set(pd.to_datetime(available_dates))

    # If T+1 not available, find next available date (up to 5 business days)
    max_attempts = 5
    attempts = 0

    while trading_date not in available_dates_set and attempts < max_attempts:
        trading_date += BDay(1)
        attempts += 1

    if trading_date not in available_dates_set:
        raise ValueError(
            f"No trading date found within {max_attempts} business days after roll_date {roll_date.strftime('%Y-%m-%d')}"
        )

    return trading_date


def get_rebalance_trading_dates(
        rebalance_frequency: str,
        roll_dates_dict: dict,
        df_dates: pd.Series,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Get trading dates (T+1) for rebalancing based on frequency.

    Parameters:
        rebalance_frequency: 'M', 'Q', 'S', or 'A'
        roll_dates_dict: Dictionary from load_roll_dates()
        df_dates: Series of available dates to check data availability
        start_date: Start of backtest period
        end_date: End of backtest period

    Returns:
        List of tuples: [(roll_date, trading_date), ...]
        roll_date = third Friday when we capture metrics
        trading_date = T+1 business day when we execute trades
    """
    # Get roll dates for this frequency
    if rebalance_frequency not in roll_dates_dict:
        raise ValueError(f"Unknown rebalance frequency: {rebalance_frequency}")

    roll_dates = roll_dates_dict[rebalance_frequency]

    # Filter to backtest period
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    roll_dates = [d for d in roll_dates if start_date <= d <= end_date]

    # Calculate trading dates
    trading_date_pairs = []
    for roll_date in roll_dates:
        try:
            trading_date = get_trading_date(roll_date, df_dates)
            trading_date_pairs.append((roll_date, trading_date))
        except ValueError as e:
            print(f"  ⚠️  Skipping roll date {roll_date.strftime('%Y-%m-%d')}: {e}")
            continue

    return trading_date_pairs