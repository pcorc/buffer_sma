"""
Data validation utilities.
"""

import pandas as pd
import numpy as np


def validate_fund_data(df: pd.DataFrame, series: str = 'F') -> tuple[bool, list[str], pd.DataFrame]:
    """
    Validate fund data structure and content, automatically cleaning duplicates.

    Parameters:
      df: Fund DataFrame
      series: Expected fund series prefix (default 'F')

    Returns:
      Tuple of (is_valid, error_messages, cleaned_df)
    """
    errors = []
    warnings = []
    cleaned_df = df.copy()

    # Check required columns
    required_cols = [
        'Date', 'Fund', 'Fund Value (USD)', 'Remaining Outcome Days',
        'Remaining Cap', 'Reference Asset Value (USD)', 'Reference Asset Return (%)',
        'Downside Before Buffer (%)'
    ]

    missing_cols = [col for col in required_cols if col not in cleaned_df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Check for null values in critical columns
    if 'Date' in cleaned_df.columns:
        if cleaned_df['Date'].isna().any():
            errors.append("Found null values in Date column")

    if 'Fund' in cleaned_df.columns:
        if cleaned_df['Fund'].isna().any():
            errors.append("Found null values in Fund column")

    # Check date format
    if 'Date' in cleaned_df.columns:
        try:
            cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])
        except Exception as e:
            errors.append(f"Date column cannot be converted to datetime: {str(e)}")

    # Check for and automatically clean duplicate rows
    if not cleaned_df.empty:
        duplicates = cleaned_df.duplicated(subset=['Date', 'Fund'], keep=False)
        if duplicates.any():
            # Find duplicate groups
            dupe_groups = cleaned_df[duplicates].groupby(['Date', 'Fund']).size().reset_index(name='Count')
            num_dupe_groups = len(dupe_groups)

            print(f"\n🧹 Cleaning {num_dupe_groups} duplicate Date-Fund combinations:")

            for _, row in dupe_groups.iterrows():
                date = row['Date']
                fund = row['Fund']
                count = row['Count']

                # Get the duplicate rows for this Date-Fund combo
                mask = (cleaned_df['Date'] == date) & (cleaned_df['Fund'] == fund)
                dupe_rows = cleaned_df[mask]

                # Show what we found
                print(f"  • {pd.to_datetime(date).strftime('%Y-%m-%d')} | {fund} | {count} occurrences")

                # Determine which row to keep
                if 'Fund Return (%)' in dupe_rows.columns:
                    non_zero = dupe_rows[dupe_rows['Fund Return (%)'] != 0]
                    if len(non_zero) > 0:
                        keep_idx = non_zero.index[0]
                    else:
                        keep_idx = dupe_rows.index[-1]
                else:
                    keep_idx = dupe_rows.index[-1]

                # Drop all except the one we want to keep
                drop_indices = dupe_rows.index[dupe_rows.index != keep_idx]
                cleaned_df = cleaned_df.drop(drop_indices)

            warnings.append(f"Automatically cleaned {num_dupe_groups} duplicate Date-Fund combinations")

    # Check fund naming convention (but only warn, don't error)
    if 'Fund' in cleaned_df.columns:
        invalid_funds = cleaned_df[~cleaned_df['Fund'].str.startswith(series)]['Fund'].unique()
        if len(invalid_funds) > 0:
            print(f"\n⚠️  Found {len(invalid_funds)} funds not in {series}-series:")
            for fund in invalid_funds:
                count = len(cleaned_df[cleaned_df['Fund'] == fund])
                print(f"  • {fund} ({count} observations)")
            warnings.append(f"Found funds with invalid series: {invalid_funds.tolist()}")
            print()

    is_valid = len(errors) == 0

    # Print summary
    if errors:
        print("❌ VALIDATION FAILED - Critical errors found:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
    elif warnings:
        print("⚠️  VALIDATION WARNING - Issues found but cleaned:")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    else:
        print("✅ VALIDATION PASSED - No issues found")

    return is_valid, errors + warnings, cleaned_df

def validate_benchmark_data(df: pd.DataFrame) -> tuple[bool, list[str], pd.DataFrame]:
    """
    Validate and clean benchmark data.

    Parameters:
      df: Benchmark DataFrame

    Returns:
      Tuple of (is_valid, error_messages, cleaned_df)
    """
    errors = []
    cleaned_df = df.copy()

    # Check required columns
    required_cols = ['Date', 'SPY', 'BUFR']
    missing_cols = [col for col in required_cols if col not in cleaned_df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors, cleaned_df

    # Convert date
    try:
        cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])
    except Exception as e:
        errors.append(f"Date conversion failed: {str(e)}")
        return False, errors, cleaned_df

    # Drop leading NaN rows for benchmark columns
    for col in ['SPY', 'BUFR']:
        if col in cleaned_df.columns:
            # Find first non-NaN index
            first_valid = cleaned_df[col].first_valid_index()
            if first_valid is not None:
                cleaned_df = cleaned_df.loc[first_valid:].reset_index(drop=True)

    # Check if we have any data left after cleaning
    if cleaned_df.empty:
        errors.append("No valid benchmark data after removing leading NaN values")
        return False, errors, cleaned_df

    is_valid = len(errors) == 0
    return is_valid, errors, cleaned_df


def validate_roll_dates(roll_dates_dict):
    """
    Validate roll dates dictionary.

    Parameters:
      roll_dates_dict: Dict with frequency keys and date list values

    Returns:
      Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check for either new format ('M', 'Q', 'S', 'A') or legacy format
    new_format_keys = ['M', 'Q', 'S', 'A']
    legacy_format_keys = ['monthly', 'quarterly', 'semi_annual', 'annual']

    has_new_format = all(key in roll_dates_dict for key in new_format_keys)
    has_legacy_format = all(key in roll_dates_dict for key in legacy_format_keys)

    if not (has_new_format or has_legacy_format):
        missing_new = [key for key in new_format_keys if key not in roll_dates_dict]
        missing_legacy = [key for key in legacy_format_keys if key not in roll_dates_dict]
        errors.append(
            f"Missing required frequencies. Need either {new_format_keys} or {legacy_format_keys}. "
            f"Missing new format: {missing_new}, Missing legacy format: {missing_legacy}"
        )

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
