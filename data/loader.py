"""
Data loading utilities.
Handles loading and initial cleaning of raw data files.
"""

import pandas as pd
import numpy as np


def load_fund_data(file_path: str, series: str = 'F') -> pd.DataFrame:
    """
    Load and preprocess fund data.

    Parameters:
      file_path: Path to the CSV file
      series: Fund series to filter (default 'F')

    Returns:
      Preprocessed DataFrame
    """
    # Load with low_memory=False to handle mixed types
    df = pd.read_csv(file_path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])

    df = df.rename(columns={
        'Remaining Cap (%)': 'Remaining Cap',
        'Remaining Cap Net (%)': 'Remaining Cap Net',
        'Remaining Buffer (%)': 'Remaining Buffer',
        'Remaining Buffer Net (%)': 'Remaining Buffer Net',
        # Add any other columns you want to rename
    })

    # Convert numeric columns that might have been read as strings
    numeric_cols = [
        'Fund Value (USD)', 'Fund Return (%)',
        'Reference Asset Value (USD)', 'Reference Asset Return (%)',
        'Remaining Outcome Days', 'Remaining Cap',  # Note: Updated name
        'Remaining Cap Net', 'Reference Asset Return to Realize Cap (%)',
        'Remaining Buffer', 'Remaining Buffer Net',  # Note: Updated names
        'Downside Before Buffer (%)', 'Downside Before Buffer Net (%)',
        'Reference Asset to Buffer End (%)', 'Unrealized Option Payoff (%)',
        'Unrealized Option Payoff Net (%)'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter to only specified series
    if series:
        original_count = df['Fund'].nunique()
        df = df[df['Fund'].str.startswith(series)].copy()
        filtered_count = df['Fund'].nunique()
        print(f"  Filtered: {original_count} total funds → {filtered_count} {series}-series funds")

    df = df.sort_values(['Fund', 'Date'])

    # Fix FutureWarning by adding fill_method=None
    df['daily_return'] = df.groupby('Fund')['Fund Value (USD)'].pct_change(fill_method=None).fillna(0)

    return df


def load_benchmark_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess benchmark time series data.

    Parameters:
      file_path: Path to the CSV file

    Returns:
      Preprocessed DataFrame
    """
    df = pd.read_csv(file_path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Calculate daily returns for each benchmark with fill_method=None
    for col in ['SPY', 'BUFR']:
        if col in df.columns:
            df[f'{col.lower()}_return'] = df[col].pct_change(fill_method=None).fillna(0)
            df[f'{col}_daily_return'] = df[col].pct_change(fill_method=None).fillna(0)

    return df


def load_roll_dates(file_path: str) -> dict[str, list]:
    """
    Load roll dates for different rebalancing frequencies.

    Parameters:
      file_path: Path to roll_dates.csv

    Returns:
      Dict with keys: 'M', 'Q', 'S', 'A', plus legacy keys
      Values are sorted lists of datetime objects
    """
    df = pd.read_csv(file_path, low_memory=False)
    roll_dates_dict = {}

    # Map column names to frequency codes
    column_mapping = {
        'monthly': 'M',
        'quarterly': 'Q',
        'semi_annual': 'S',
        'annual': 'A'
    }

    for col in df.columns:
        if col in column_mapping:
            dates = pd.to_datetime(df[col].dropna()).sort_values()
            freq_code = column_mapping[col]
            roll_dates_dict[freq_code] = dates.tolist()
            # Also keep legacy key for backwards compatibility
            roll_dates_dict[col] = dates.tolist()

    return roll_dates_dict


def validate_data_alignment(df_fund: pd.DataFrame, df_benchmark: pd.DataFrame) -> tuple[bool, list[str], pd.Timestamp, pd.Timestamp]:
    """
    Validate that fund and benchmark data have overlapping date ranges.

    Parameters:
      df_fund: Fund DataFrame
      df_benchmark: Benchmark DataFrame

    Returns:
      Tuple of (is_valid, error_messages, common_start_date, common_end_date)
    """
    errors = []

    fund_start = df_fund['Date'].min()
    fund_end = df_fund['Date'].max()
    bench_start = df_benchmark['Date'].min()
    bench_end = df_benchmark['Date'].max()

    common_start = max(fund_start, bench_start)
    common_end = min(fund_end, bench_end)

    if common_start > common_end:
        errors.append(
            f"No overlapping dates! Fund: {fund_start.date()} to {fund_end.date()}, "
            f"Benchmark: {bench_start.date()} to {bench_end.date()}"
        )
        return False, errors, None, None

    return True, errors, common_start, common_end