"""
Data preprocessing and derived metrics calculation.
"""

import pandas as pd
import numpy as np
from config.settings import BUFFER_LEVELS


def preprocess_fund_data(df_raw):
    """
    Enrich raw fund data with derived metrics for each outcome period.

    See original implementation docstring for full details.

    Parameters:
      df_raw: Raw fund DataFrame

    Returns:
      DataFrame with additional derived columns
    """
    print("\n" + "="*80)
    print("PREPROCESSING FUND DATA")
    print("="*80)

    df = df_raw.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Fund', 'Date']).reset_index(drop=True)

    # Initialize new columns
    new_columns = [
        'Roll_Date', 'Outcome_Period_ID', 'Original_Cap', 'Starting_Fund_Value',
        'Fund_Cap_Value', 'Starting_Ref_Asset_Value', 'Ref_Asset_Cap_Value',
        'Buffer_Level', 'Cap_Utilization', 'Cap_Remaining_Pct'
    ]

    for col in new_columns:
        if col in ['Roll_Date']:
            df[col] = pd.NaT
        elif col in ['Outcome_Period_ID']:
            df[col] = None
        else:
            df[col] = np.nan

    # Process each fund separately
    for fund in df['Fund'].unique():
        fund_mask = df['Fund'] == fund
        fund_df = df[fund_mask].copy()

        # Get the actual indices from the original dataframe
        fund_indices = df[fund_mask].index

        # Determine buffer level from series
        series_letter = fund[0]
        buffer_level = BUFFER_LEVELS.get(series_letter, 0.10)

        # Identify outcome period boundaries
        fund_df['Days_Shift'] = fund_df['Remaining Outcome Days'].shift(1)
        period_starts = fund_df[fund_df['Days_Shift'] > fund_df['Remaining Outcome Days']].index

        if len(period_starts) == 0:
            period_starts = [fund_df.index[0]]

        period_id = 0
        for i, start_idx in enumerate(period_starts):
            period_id += 1

            # Determine end of this period
            if i < len(period_starts) - 1:
                end_idx = period_starts[i + 1] - 1
            else:
                end_idx = fund_df.index[-1]

            # Create mask using original df indices
            period_indices = fund_df.loc[start_idx:end_idx].index

            # Get metrics from period start
            start_row = fund_df.loc[start_idx]
            roll_date = start_row['Date']
            original_cap = start_row['Remaining Cap'] / 100
            starting_fund_value = start_row['Fund Value (USD)']
            fund_cap_value = starting_fund_value * (1 + original_cap)
            starting_ref_asset_value = start_row['Reference Asset Value (USD)']

            ref_asset_return_to_cap = start_row['Reference Asset Return to Realize Cap (%)'] / 100
            ref_asset_cap_value = starting_ref_asset_value * (1 + ref_asset_return_to_cap)

            # Fill metrics for this period using the actual indices
            df.loc[period_indices, 'Roll_Date'] = roll_date
            df.loc[period_indices, 'Outcome_Period_ID'] = f"{fund}_{period_id}"
            df.loc[period_indices, 'Original_Cap'] = original_cap
            df.loc[period_indices, 'Starting_Fund_Value'] = starting_fund_value
            df.loc[period_indices, 'Fund_Cap_Value'] = fund_cap_value
            df.loc[period_indices, 'Starting_Ref_Asset_Value'] = starting_ref_asset_value
            df.loc[period_indices, 'Ref_Asset_Cap_Value'] = ref_asset_cap_value
            df.loc[period_indices, 'Buffer_Level'] = buffer_level

    # Calculate daily metrics
    df['Current_Remaining_Cap'] = df['Remaining Cap'] / 100
    df['Cap_Utilization'] = (df['Original_Cap'] - df['Current_Remaining_Cap']) / df['Original_Cap']
    df['Cap_Utilization'] = df['Cap_Utilization'].fillna(0).clip(lower=0, upper=1)
    df['Cap_Remaining_Pct'] = df['Current_Remaining_Cap'] / df['Original_Cap']
    df['Cap_Remaining_Pct'] = df['Cap_Remaining_Pct'].fillna(1).clip(lower=0, upper=1)

    print("\n" + "="*80)
    print("✅ PREPROCESSING FUND DATA --> COMPLETE")
    print("="*80)

    return df