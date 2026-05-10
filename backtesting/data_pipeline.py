"""
Unified Data Loading and Preprocessing Pipeline
================================================

Handles all data loading, cleaning, and enrichment in one place.
ALL percentage columns are converted to decimals immediately on load.

This eliminates confusion about which columns are percentages vs decimals.
After this pipeline, EVERYTHING is in decimal format (0.15 not 15%).

Usage:
    from data.data_pipeline import load_and_preprocess_all_data
    
    df_enriched, df_benchmarks, roll_dates_dict = load_and_preprocess_all_data(
        fund_file='data/fund_data.csv',
        benchmark_file='data/benchmarks.csv',
        roll_dates_file='data/roll_dates.csv',
        series='F'
    )
"""

import pandas as pd
import numpy as np
from config.settings import BUFFER_LEVELS, MONTH_MAP

# Global configuration
MIN_ANALYSIS_DATE = pd.Timestamp('2020-07-01')


# ============================================================================
# PUBLIC API
# ============================================================================

def load_and_preprocess_all_data(fund_file, benchmark_file, roll_dates_file, series='F'):
    """
    Complete data pipeline: Load → Clean → Convert → Enrich
    
    This is the ONLY function you need to call to get all data ready for backtesting.
    
    Parameters:
        fund_file: Path to fund CSV
        benchmark_file: Path to benchmark CSV
        roll_dates_file: Path to roll dates CSV
        series: Fund series to filter (default 'F')
    
    Returns:
        df_enriched: Fund data with ALL metrics in DECIMAL format
        df_benchmarks: Benchmark data with daily returns
        roll_dates_dict: Roll dates by frequency
    """

    # Step 1: Load raw CSVs

    df_raw = _load_fund_csv(fund_file, series)
    df_benchmarks = _load_benchmark_csv(benchmark_file)
    roll_dates_dict = _load_roll_dates(roll_dates_file)
    
    # Step 2: Convert ALL percentages to decimals
    df_clean = _convert_percentages_to_decimals(df_raw)
    
    # Step 3: Enrich with roll date metrics
    df_enriched = _enrich_with_roll_dates(df_clean, roll_dates_dict)
    
    # Step 4: Validate
    _validate_decimal_format(df_enriched)
    
    print("\n" + "=" * 80)
    print("✅ DATA PIPELINE COMPLETE")
    print("=" * 80 + "\n")
    
    return df_enriched, df_benchmarks, roll_dates_dict


# ============================================================================
# STEP 1: LOAD RAW DATA
# ============================================================================

def _load_fund_csv(file_path, series):
    """Load raw fund CSV and apply basic filtering."""
    df = pd.read_csv(file_path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter to analysis period
    original_date_range = (df['Date'].min(), df['Date'].max())
    df = df[df['Date'] >= MIN_ANALYSIS_DATE].copy()
    
    print(f"  Date filter: {original_date_range[0].date()} to {original_date_range[1].date()} "
          f"→ {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Standardize column names (remove % suffix for now, easier to work with)
    df = df.rename(columns={
        'Remaining Cap (%)': 'Remaining Cap',
        'Remaining Cap Net (%)': 'Remaining Cap Net',
        'Remaining Buffer (%)': 'Remaining Buffer',
        'Remaining Buffer Net (%)': 'Remaining Buffer Net',
        'Downside Before Buffer (%)': 'Downside Before Buffer',
        'Downside Before Buffer Net (%)': 'Downside Before Buffer Net',
        'Reference Asset to Buffer End (%)': 'Reference Asset to Buffer End',
        'Unrealized Option Payoff (%)': 'Unrealized Option Payoff',
        'Unrealized Option Payoff Net (%)': 'Unrealized Option Payoff Net',
        'Fund Return (%)': 'Fund Return',
        'Reference Asset Return (%)': 'Reference Asset Return',
        'Reference Asset Return to Realize Cap (%)': 'Reference Asset Return to Realize Cap',
        'Original Cap Net (%)': 'Original Cap Net',
        'Original Buffer Net (%)': 'Original Buffer Net'
    })
    
    # Convert to numeric (handle any string values)
    numeric_cols = [
        'Fund Value (USD)', 'Fund Return',
        'Reference Asset Value (USD)', 'Reference Asset Return',
        'Remaining Outcome Days', 'Remaining Cap', 'Remaining Cap Net',
        'Reference Asset Return to Realize Cap',
        'Remaining Buffer', 'Remaining Buffer Net',
        'Downside Before Buffer', 'Downside Before Buffer Net',
        'Reference Asset to Buffer End',
        'Unrealized Option Payoff', 'Unrealized Option Payoff Net',
        'Original Cap Net', 'Original Buffer Net'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter to series
    if series:
        original_count = df['Fund'].nunique()
        df = df[df['Fund'].str.startswith(series)].copy()
        filtered_count = df['Fund'].nunique()
        print(f"  Series filter: {original_count} total funds → {filtered_count} {series}-series funds")
    
    df = df.sort_values(['Fund', 'Date']).reset_index(drop=True)
    
    # Calculate daily returns
    df['daily_return'] = df.groupby('Fund')['Fund Value (USD)'].pct_change(fill_method=None).fillna(0)

    return df


def _load_benchmark_csv(file_path):
    """Load benchmark CSV."""
    df = pd.read_csv(file_path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter to analysis period
    original_date_range = (df['Date'].min(), df['Date'].max())
    df = df[df['Date'] >= MIN_ANALYSIS_DATE].copy()
    
    print(f"  Benchmark filter: {original_date_range[0].date()} to {original_date_range[1].date()} "
          f"→ {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate daily returns
    for col in ['SPY', 'BUFR']:
        if col in df.columns:
            df[f'{col}_daily_return'] = df[col].pct_change(fill_method=None).fillna(0)
    
    print(f"  ✅ Loaded {len(df)} benchmark rows")
    
    return df


def _load_roll_dates(file_path):
    """Load roll dates CSV."""
    df = pd.read_csv(file_path, low_memory=False)
    roll_dates_dict = {}
    
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
            roll_dates_dict[col] = dates.tolist()  # Legacy compatibility
    
    print(f"  ✅ Loaded roll dates: {', '.join([f'{k}={len(v)}' for k, v in roll_dates_dict.items() if len(k) == 1])}")
    
    return roll_dates_dict


# ============================================================================
# STEP 2: CONVERT PERCENTAGES TO DECIMALS
# ============================================================================

def _convert_percentages_to_decimals(df):
    """
    Convert ALL percentage columns to decimal format.
    
    After this function:
    - 15.0% becomes 0.15
    - -9.63% becomes -0.0963
    - 100% becomes 1.0
    
    This is done ONCE at the very beginning so the rest of the codebase
    never has to think about percentage vs decimal format.
    """
    percentage_cols = [
        'Remaining Cap',
        'Remaining Cap Net',
        'Remaining Buffer',
        'Remaining Buffer Net',
        'Downside Before Buffer',
        'Downside Before Buffer Net',
        'Reference Asset to Buffer End',
        'Unrealized Option Payoff',
        'Unrealized Option Payoff Net',
        'Fund Return',
        'Reference Asset Return',
        'Reference Asset Return to Realize Cap',
        'Original Cap Net',
        'Original Buffer Net'
    ]
    
    converted = []
    for col in percentage_cols:
        if col in df.columns:
            df[col] = df[col] / 100
            converted.append(col)
    
    print(f"  ✅ Converted {len(converted)} percentage columns to decimal format")
    
    return df


# ============================================================================
# STEP 3: ENRICH WITH ROLL DATE METRICS
# ============================================================================

def _enrich_with_roll_dates(df, roll_dates_dict):
    """
    Enrich fund data with roll date metrics.
    
    NOTE: Input df already has all percentages converted to decimals.
    So NO /100 conversions happen here!
    """
    print(f"  Global alignment: All funds start at or after {MIN_ANALYSIS_DATE.strftime('%Y-%m-%d')}")
    
    # Get monthly roll dates
    monthly_roll_dates = roll_dates_dict.get('M', roll_dates_dict.get('monthly', []))
    
    if not monthly_roll_dates:
        raise ValueError("No monthly roll dates found")

    new_columns = ['Cap_Utilization']
    funds_processed = 0
    skipped_funds = []
    
    # Process each fund
    for fund in df['Fund'].unique():
        fund_mask = df['Fund'] == fund
        fund_df = df[fund_mask].copy()
        
        series_letter = fund[0]
        buffer_level = BUFFER_LEVELS.get(series_letter, 0.10)
        
        fund_end_date = fund_df['Date'].max()
        
        # Get anniversary roll dates
        try:
            anniversary_dates = _get_anniversary_roll_dates(fund, monthly_roll_dates)
        except ValueError:
            skipped_funds.append(fund)
            continue
        
        # Filter to analysis period
        anniversary_dates = [d for d in anniversary_dates if d >= MIN_ANALYSIS_DATE]
        
        if not anniversary_dates:
            skipped_funds.append(fund)
            continue
        
        # Process each outcome period
        for period_idx, roll_date in enumerate(anniversary_dates):
            if roll_date > fund_end_date:
                continue
            
            exact_match = fund_df[fund_df['Date'] == roll_date]
            
            if exact_match.empty:
                continue
            
            start_row = exact_match.iloc[0]
            
            # CRITICAL: NO /100 HERE - data is already decimal!
            original_cap = start_row['Remaining Cap']
            original_buffer = start_row.get('Remaining Buffer', buffer_level)
            starting_downside_before_buffer = start_row.get('Downside Before Buffer', 0)
            
            # NAV metrics
            starting_fund_value = start_row['Fund Value (USD)']
            fund_cap_value = starting_fund_value * (1 + original_cap)
            
            starting_ref_asset_value = start_row['Reference Asset Value (USD)']
            ref_asset_return_to_cap = start_row['Reference Asset Return to Realize Cap']
            ref_asset_cap_value = starting_ref_asset_value * (1 + ref_asset_return_to_cap)
            
            total_outcome_days = start_row.get('Remaining Outcome Days', 365)
            
            # Determine period end
            if period_idx < len(anniversary_dates) - 1:
                next_roll_date = anniversary_dates[period_idx + 1]
                period_mask = (fund_df['Date'] >= roll_date) & (fund_df['Date'] < next_roll_date)
            else:
                period_mask = fund_df['Date'] >= roll_date
            
            period_indices = fund_df[period_mask].index
            
            # Assign to period
            if len(period_indices) > 0:
                df.loc[period_indices, 'Roll_Date'] = roll_date
                # df.loc[period_indices, 'Original_Cap'] = original_cap
                # df.loc[period_indices, 'Outcome_Period_ID'] = f"{fund}_P{period_idx + 1}"
                # df.loc[period_indices, 'Original_Buffer'] = original_buffer
                # df.loc[period_indices, 'Starting_Fund_Value'] = starting_fund_value
                # df.loc[period_indices, 'Fund_Cap_Value'] = fund_cap_value
                # df.loc[period_indices, 'Starting_Ref_Asset_Value'] = starting_ref_asset_value
                # df.loc[period_indices, 'Ref_Asset_Cap_Value'] = ref_asset_cap_value
                # df.loc[period_indices, 'Buffer_Level'] = buffer_level
                # df.loc[period_indices, 'Total_Outcome_Days'] = total_outcome_days
                # df.loc[period_indices, 'Starting_Downside_Before_Buffer'] = starting_downside_before_buffer
        
        funds_processed += 1
    
    # # Calculate daily derived metrics (NO /100 - data already decimal!)
    df['Original_Cap'] = df['Original Cap Net']
    df['Current_Remaining_Cap'] = df['Remaining Cap']
    df['Cap_Utilization'] = (df['Original_Cap'] - df['Current_Remaining_Cap']) / df['Original_Cap']
    df['Cap_Utilization'] = df['Cap_Utilization'].fillna(0).clip(lower=0, upper=1)
    df['Cap_Remaining_Pct'] = df['Current_Remaining_Cap'] / df['Original_Cap']
    df['Cap_Remaining_Pct'] = df['Cap_Remaining_Pct'].fillna(1).clip(lower=0, upper=1)

    # Buffer utilization — same pattern using net columns
    # Original_Buffer_Net captured at roll date, Remaining Buffer Net updates daily
    df['Buffer_Utilization'] = (
            (df['Original Buffer Net'] - df['Remaining Buffer Net']) / df['Original Buffer Net']
    )
    df['Buffer_Utilization'] = df['Buffer_Utilization'].fillna(0).clip(lower=0, upper=1)
    df['Buffer_Remaining_Pct'] = df['Remaining Buffer Net'] / df['Original Buffer Net']
    df['Buffer_Remaining_Pct'] = df['Buffer_Remaining_Pct'].fillna(1).clip(lower=0, upper=1)

    if skipped_funds:
        print(f"  ⚠️  Skipped {len(skipped_funds)} funds (no valid roll dates)")

    return df


def _get_anniversary_roll_dates(fund_ticker, monthly_roll_dates):
    """Get anniversary roll dates for a specific fund."""
    if len(fund_ticker) < 4:
        raise ValueError(f"Invalid fund ticker: {fund_ticker}")
    
    fund_month = fund_ticker[1:4].upper()
    month_num = MONTH_MAP.get(fund_month)
    
    if month_num is None:
        raise ValueError(f"Cannot parse month from: {fund_ticker}")
    
    anniversary_dates = [d for d in monthly_roll_dates if d.month == month_num]
    
    return sorted(anniversary_dates)


# ============================================================================
# STEP 4: VALIDATION
# ============================================================================

def _validate_decimal_format(df):
    """
    Validate that all percentage columns are now in decimal format.
    
    Checks:
    - Remaining Cap should be ~0.10-0.25 (not 10-25)
    - DBB should be between -1.0 and 0.0 (not -100 to 0)
    - Buffer should be ~0.10 (not 10)
    """
    issues = []
    
    # Check Remaining Cap
    cap_sample = df['Remaining Cap'].dropna()
    if len(cap_sample) > 0:
        if cap_sample.max() > 1.0:
            issues.append(f"❌ Remaining Cap has values > 1.0 (max={cap_sample.max():.2f}) - still in percentage format!")
    
    # Check DBB
    dbb_col = 'Downside Before Buffer'
    if dbb_col in df.columns:
        dbb_sample = df[dbb_col].dropna()
        if len(dbb_sample) > 0:
            if dbb_sample.min() < -1.0:
                issues.append(f"❌ DBB has values < -1.0 (min={dbb_sample.min():.2f}) - still in percentage format!")
    
    # Check Original_Cap
    orig_cap_sample = df['Original Cap Net'].dropna()
    if len(orig_cap_sample) > 0:
        if orig_cap_sample.max() > 1.0:
            issues.append(f"❌ Original Cap Net has values > 1.0 (max={orig_cap_sample.max():.2f}) - still in percentage format!")
    
    if issues:
        print("\n  ⚠️  VALIDATION ISSUES:")
        for issue in issues:
            print(f"    {issue}")
        raise ValueError("Data validation failed - percentages not properly converted!")
    
    print(f"  ✅ Validation passed - all columns in decimal format")
