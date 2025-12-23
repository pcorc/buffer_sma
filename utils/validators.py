"""
Data validation utilities.
"""

import pandas as pd
import numpy as np


def validate_fund_data(df):
    """
    Validate fund data structure and content.

    Parameters:
      df: Fund DataFrame

    Returns:
      Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check required columns
    required_cols = [
        'Date', 'Fund', 'Fund Value (USD)', 'Remaining Outcome Days',
        'Remaining Cap', 'Reference Asset Value (USD)', 'Reference Asset Return (%)',
        'Downside Before Buffer (%)'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Check for null values in critical columns
    if 'Date' in df.columns:
        if df['Date'].isna().any():
            errors.append("Found null values in Date column")

    if 'Fund' in df.columns:
        if df['Fund'].isna().any():
            errors.append("Found null values in Fund column")

    # Check date format
    if 'Date' in df.columns:
        try:
            pd.to_datetime(df['Date'])
        except:
            errors.append("Date column cannot be converted to datetime")

    # Check for duplicate rows
    if not df.empty:
        duplicates = df.duplicated(subset=['Date', 'Fund'], keep=False)
        if duplicates.any():
            num_dupes = duplicates.sum()
            errors.append(f"Found {num_dupes} duplicate Date-Fund combinations")

    # Check fund naming convention
    if 'Fund' in df.columns:
        valid_series = ['F', 'G', 'D']
        invalid_funds = df[~df['Fund'].str[0].isin(valid_series)]['Fund'].unique()
        if len(invalid_funds) > 0:
            errors.append(f"Found funds with invalid series: {invalid_funds[:5]}")

    is_valid = len(errors) == 0

    return is_valid, errors


def validate_benchmark_data(df):
    """
    Validate benchmark data structure and content.

    Parameters:
      df: Benchmark DataFrame

    Returns:
      Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check required columns
    required_cols = ['Date', 'SPY', 'BUFR']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Check for null values
    for col in ['Date', 'SPY', 'BUFR']:
        if col in df.columns:
            if df[col].isna().any():
                errors.append(f"Found null values in {col} column")

    # Check date format
    if 'Date' in df.columns:
        try:
            pd.to_datetime(df['Date'])
        except:
            errors.append("Date column cannot be converted to datetime")

    is_valid = len(errors) == 0

    return is_valid, errors


def validate_roll_dates(roll_dates_dict):
    """
    Validate roll dates dictionary.

    Parameters:
      roll_dates_dict: Dict with frequency keys and date list values

    Returns:
      Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check required frequencies
    required_freqs = ['monthly', 'quarterly', 'semi_annual', 'annual']
    missing_freqs = [freq for freq in required_freqs if freq not in roll_dates_dict]
    if missing_freqs:
        errors.append(f"Missing required frequencies: {missing_freqs}")

    # Check that each frequency has dates
    for freq, dates in roll_dates_dict.items():
        if not dates or len(dates) == 0:
            errors.append(f"No dates found for frequency: {freq}")
        else:
            # Check that dates are sorted
            sorted_dates = sorted(dates)
            if dates != sorted_dates:
                errors.append(f"Dates not sorted for frequency: {freq}")

    is_valid = len(errors) == 0

    return is_valid, errors


def validate_data_alignment(df_fund, df_benchmark):
    """
    Validate that fund and benchmark data have overlapping date ranges.

    Parameters:
      df_fund: Fund DataFrame
      df_benchmark: Benchmark DataFrame

    Returns:
      Tuple of (is_valid, error_messages, common_start, common_end)
    """
    errors = []

    df_fund['Date'] = pd.to_datetime(df_fund['Date'])
    df_benchmark['Date'] = pd.to_datetime(df_benchmark['Date'])

    fund_start = df_fund['Date'].min()
    fund_end = df_fund['Date'].max()
    bench_start = df_benchmark['Date'].min()
    bench_end = df_benchmark['Date'].max()

    common_start = max(fund_start, bench_start)
    common_end = min(fund_end, bench_end)

    if common_start > common_end:
        errors.append("No overlapping dates between fund and benchmark data!")
        is_valid = False
        return is_valid, errors, None, None

    # Check for sufficient overlap
    overlap_days = (common_end - common_start).days
    if overlap_days < 365:
        errors.append(f"Warning: Only {overlap_days} days of overlap (less than 1 year)")

    is_valid = len(errors) == 0

    return is_valid, errors, common_start, common_end


def print_validation_results(validation_name, is_valid, errors):
    """
    Print validation results in formatted way.

    Parameters:
      validation_name: Name of validation being performed
      is_valid: Boolean indicating if validation passed
      errors: List of error messages
    """
    print(f"\n{'=' * 80}")
    print(f"VALIDATION: {validation_name}")
    print(f"{'=' * 80}")

    if is_valid:
        print("✅ PASSED - No issues found")
    else:
        print("❌ FAILED - Issues found:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")

    print(f"{'=' * 80}\n")