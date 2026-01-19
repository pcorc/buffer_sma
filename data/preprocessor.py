"""
Data preprocessing and derived metrics calculation.
"""

import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from config.settings import BUFFER_LEVELS, MONTH_MAP


def get_fund_month(fund_ticker: str) -> str:
    """
    Extract the month abbreviation from fund ticker.

    Examples:
        'FAUG' -> 'AUG'
        'FNOV' -> 'NOV'
    """
    if len(fund_ticker) < 4:
        raise ValueError(f"Invalid fund ticker format: {fund_ticker}")
    return fund_ticker[1:4].upper()


def get_anniversary_roll_dates(fund_ticker: str, monthly_roll_dates: list) -> list:
    """
    Get all anniversary roll dates for a specific fund.

    Parameters:
        fund_ticker: e.g., 'FAUG'
        monthly_roll_dates: List of all monthly roll dates (pd.Timestamp)

    Returns:
        Sorted list of pd.Timestamp objects for this fund's anniversary months
    """
    fund_month = get_fund_month(fund_ticker)
    month_num = MONTH_MAP.get(fund_month)

    if month_num is None:
        raise ValueError(f"Cannot parse month from fund ticker: {fund_ticker}")

    # Filter to only dates in the fund's anniversary month
    anniversary_dates = [
        date for date in monthly_roll_dates
        if date.month == month_num
    ]

    return sorted(anniversary_dates)


def preprocess_fund_data(df_raw: pd.DataFrame, roll_dates_dict: dict) -> pd.DataFrame:
    """
    Enrich raw fund data with derived metrics for each outcome period.

    Uses monthly roll dates to identify each fund's anniversary dates and capture
    initial metrics (Original_Cap, etc.) at the start of each outcome period.

    Key improvements:
    - Uses actual roll dates from roll_dates.csv to identify period starts
    - Matches each fund to its anniversary month roll dates
    - Handles initial fund launches (Nov 2019 for FAUG/FNOV, May 2020+)
    - Validates data availability on roll dates
    - Checks for unexpected variation in Original_Cap within periods

    Parameters:
      df_raw: Raw fund DataFrame
      roll_dates_dict: Dictionary from load_roll_dates() with 'M' key for monthly dates

    Returns:
      DataFrame with additional derived columns:
        - Roll_Date: The anniversary roll date that started this outcome period
        - Outcome_Period_ID: Unique identifier for each fund's outcome period
        - Original_Cap: Cap percentage at period start (captured on roll date)
        - Starting_Fund_Value: Fund NAV at period start
        - Fund_Cap_Value: Fund NAV if cap is reached
        - Starting_Ref_Asset_Value: Reference asset value at period start
        - Ref_Asset_Cap_Value: Reference asset value at cap
        - Buffer_Level: Buffer percentage for this fund series
        - Cap_Utilization: Daily calculation of cap used (0 to 1)
        - Cap_Remaining_Pct: Daily calculation of remaining cap (0 to 1)
    """
    print("\n" + "="*80)
    print("PREPROCESSING FUND DATA")
    print("="*80)

    df = df_raw.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Fund', 'Date']).reset_index(drop=True)

    # Get monthly roll dates - use 'M' key from updated loader
    if 'M' in roll_dates_dict:
        monthly_roll_dates = roll_dates_dict['M']
    else:
        # Fallback to legacy key
        monthly_roll_dates = roll_dates_dict.get('monthly', [])

    if not monthly_roll_dates:
        raise ValueError("No monthly roll dates found in roll_dates_dict")

    # print(f"Using {len(monthly_roll_dates)} monthly roll dates from {monthly_roll_dates[0].strftime('%Y-%m-%d')} to {monthly_roll_dates[-1].strftime('%Y-%m-%d')}")

    # Initialize new columns
    new_columns = [
        'Roll_Date', 'Outcome_Period_ID',
        'Original_Cap', 'Original_Buffer',
        'Starting_Fund_Value', 'Fund_Cap_Value',
        'Starting_Ref_Asset_Value', 'Ref_Asset_Cap_Value',
        'Buffer_Level', 'Total_Outcome_Days', 'Starting_Downside_Before_Buffer',
        'Cap_Utilization', 'Cap_Remaining_Pct'
    ]

    for col in new_columns:
        if col == 'Roll_Date':
            df[col] = pd.NaT
        elif col == 'Outcome_Period_ID':
            df[col] = None
        else:
            df[col] = np.nan

    # Process each fund separately
    funds_processed = 0
    funds_with_warnings = []

    for fund in df['Fund'].unique():
        fund_mask = df['Fund'] == fund
        fund_df = df[fund_mask].copy()
        fund_indices = df[fund_mask].index

        # Get buffer level
        series_letter = fund[0]
        buffer_level = BUFFER_LEVELS.get(series_letter, 0.10)

        # Get fund's first date (launch date)
        launch_date = fund_df['Date'].min()

        # Get anniversary roll dates for this fund
        try:
            anniversary_dates = get_anniversary_roll_dates(fund, monthly_roll_dates)
        except ValueError as e:
            print(f"  ⚠️  Skipping fund {fund}: {e}")
            continue

        # Filter anniversary dates to those >= launch_date
        anniversary_dates = [d for d in anniversary_dates if d >= launch_date]

        # Add launch date as first "roll date" if it's not already an anniversary
        if not anniversary_dates or launch_date < anniversary_dates[0]:
            roll_dates_for_fund = [launch_date] + anniversary_dates
            used_launch_as_first = True
        else:
            roll_dates_for_fund = anniversary_dates
            used_launch_as_first = False

        # Process each outcome period
        for period_idx, roll_date in enumerate(roll_dates_for_fund):
            period_id = period_idx + 1

            # Find the row closest to roll_date (should be exact match or very close)
            date_diffs = abs(fund_df['Date'] - roll_date)
            closest_idx = date_diffs.idxmin()
            min_diff_days = date_diffs.loc[closest_idx].days

            if min_diff_days > 5:
                warning_msg = f"{fund} Period {period_id}: No data on roll date {roll_date.strftime('%Y-%m-%d')}, closest is {fund_df.loc[closest_idx, 'Date'].strftime('%Y-%m-%d')} ({min_diff_days} days away)"
                if fund not in funds_with_warnings:
                    print(f"  ⚠️  {warning_msg}")
                    funds_with_warnings.append(fund)

            start_row = fund_df.loc[closest_idx]
            actual_roll_date = start_row['Date']

            # Capture initial metrics from roll date - THESE ARE THE REFERENCE VALUES
            # These stay constant throughout the outcome period for accurate return calculations
            # Cap metrics
            if actual_roll_date.strftime('%Y-%m-%d') == '2024-09-20':
                x=1

            original_cap = start_row['Remaining Cap'] / 100
            original_buffer = start_row.get('Remaining Buffer (%)', buffer_level * 100) / 100

            # NAV metrics at roll date
            starting_fund_value = start_row['Fund Value (USD)']  # NAV per share at roll
            fund_cap_value = starting_fund_value * (1 + original_cap)

            # Reference Index metrics at roll date
            starting_ref_asset_value = start_row['Reference Asset Value (USD)']
            ref_asset_return_to_cap = start_row['Reference Asset Return to Realize Cap (%)'] / 100
            ref_asset_cap_value = starting_ref_asset_value * (1 + ref_asset_return_to_cap)

            # Outcome period metadata
            total_outcome_days = start_row.get('Remaining Outcome Days', 365)

            # Downside before buffer at roll date
            starting_downside_before_buffer = start_row.get('Downside Before Buffer (%)', 0) / 100

            # Determine end of this period
            if period_idx < len(roll_dates_for_fund) - 1:
                next_roll_date = roll_dates_for_fund[period_idx + 1]
                period_mask = (fund_df['Date'] >= actual_roll_date) & (fund_df['Date'] < next_roll_date)
            else:
                # Last period - goes to end of data
                period_mask = fund_df['Date'] >= actual_roll_date

            period_indices = fund_df[period_mask].index

            # Validation: Check if Original_Cap varies significantly within period
            if len(period_indices) > 0:
                period_remaining_caps = fund_df.loc[period_indices, 'Remaining Cap'] / 100
                max_variation = abs(period_remaining_caps.max() - original_cap)

                if max_variation > 0.02:  # More than 2% variation from original
                    if fund not in funds_with_warnings:
                        print(f"  ⚠️  {fund} Period {period_id}: Original_Cap variation detected: {max_variation:.2%}")
                        funds_with_warnings.append(fund)

                # Assign metrics to entire period
                df.loc[period_indices, 'Roll_Date'] = actual_roll_date
                df.loc[period_indices, 'Outcome_Period_ID'] = f"{fund}_P{period_id}"
                df.loc[period_indices, 'Original_Cap'] = original_cap
                df.loc[period_indices, 'Original_Buffer'] = original_buffer
                df.loc[period_indices, 'Starting_Fund_Value'] = starting_fund_value
                df.loc[period_indices, 'Fund_Cap_Value'] = fund_cap_value
                df.loc[period_indices, 'Starting_Ref_Asset_Value'] = starting_ref_asset_value
                df.loc[period_indices, 'Ref_Asset_Cap_Value'] = ref_asset_cap_value
                df.loc[period_indices, 'Buffer_Level'] = buffer_level
                df.loc[period_indices, 'Total_Outcome_Days'] = total_outcome_days
                df.loc[period_indices, 'Starting_Downside_Before_Buffer'] = starting_downside_before_buffer

        funds_processed += 1

    # Calculate daily derived metrics
    df['Current_Remaining_Cap'] = df['Remaining Cap'] / 100
    df['Cap_Utilization'] = (df['Original_Cap'] - df['Current_Remaining_Cap']) / df['Original_Cap']
    df['Cap_Utilization'] = df['Cap_Utilization'].fillna(0).clip(lower=0, upper=1)
    df['Cap_Remaining_Pct'] = df['Current_Remaining_Cap'] / df['Original_Cap']
    df['Cap_Remaining_Pct'] = df['Cap_Remaining_Pct'].fillna(1).clip(lower=0, upper=1)

    print(f"\n✅ Preprocessing complete- processed {funds_processed} funds")
    print(f"  Created {df['Outcome_Period_ID'].nunique()} outcome periods")
    if funds_with_warnings:
        print(f"  ⚠️  {len(funds_with_warnings)} funds had warnings")
    print("="*80 + "\n")

    return df