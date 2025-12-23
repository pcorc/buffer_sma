"""
Data loading utilities.
Handles loading and initial cleaning of raw data files.
"""

import pandas as pd
import numpy as np


def load_fund_data(file_path):
    """
    Load raw fund data from CSV.

    Parameters:
      file_path: Path to data.csv

    Returns:
      DataFrame with fund data
    """
    print(f"Loading fund data from {file_path}...")

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Fund', 'Date']).reset_index(drop=True)

    # Rename column if needed
    if 'Remaining Cap (%)' in df.columns:
        df.rename(columns={'Remaining Cap (%)': 'Remaining Cap'}, inplace=True)

    # Compute daily returns if not present
    if 'daily_return' not in df.columns:
        df['daily_return'] = df.groupby('Fund')['Fund Value (USD)'].pct_change().fillna(0)

    print(f"  Loaded {len(df)} rows for {df['Fund'].nunique()} funds")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    return df


def load_benchmark_data(file_path):
    """
    Load benchmark time series (SPY, BUFR).

    Parameters:
      file_path: Path to benchmark_ts.csv

    Returns:
      DataFrame with Date, SPY, BUFR columns
    """
    print(f"Loading benchmark data from {file_path}...")

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Calculate daily returns
    df['spy_return'] = df['SPY'].pct_change().fillna(0)
    df['bufr_return'] = df['BUFR'].pct_change().fillna(0)

    print(f"  Loaded {len(df)} rows")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    return df


def load_roll_dates(file_path):
    """
    Load roll dates for different rebalancing frequencies.

    Parameters:
      file_path: Path to roll_dates.csv

    Returns:
      Dict with keys: 'monthly', 'quarterly', 'semi_annual', 'annual'
      Values are lists of datetime objects
    """
    print(f"Loading roll dates from {file_path}...")

    df = pd.read_csv(file_path)

    roll_dates_dict = {}
    for col in ['monthly', 'quarterly', 'semi_annual', 'annual']:
        if col in df.columns:
            dates = pd.to_datetime(df[col].dropna())
            roll_dates_dict[col] = dates.tolist()
            print(f"  Loaded {len(dates)} {col} dates")

    return roll_dates_dict


def validate_data_alignment(df_fund, df_benchmark):
    """
    Validate that fund and benchmark data have overlapping date ranges.

    Parameters:
      df_fund: Fund DataFrame
      df_benchmark: Benchmark DataFrame

    Returns:
      Tuple of (common_start_date, common_end_date)
    """
    fund_start = df_fund['Date'].min()
    fund_end = df_fund['Date'].max()
    bench_start = df_benchmark['Date'].min()
    bench_end = df_benchmark['Date'].max()

    common_start = max(fund_start, bench_start)
    common_end = min(fund_end, bench_end)

    if common_start > common_end:
        raise ValueError("No overlapping dates between fund and benchmark data!")

    print(f"\nData alignment:")
    print(f"  Common date range: {common_start.date()} to {common_end.date()}")
    print(f"  Total days: {(common_end - common_start).days}")

    return common_start, common_end