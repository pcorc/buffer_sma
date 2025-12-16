"""
Dynamic Backtest Framework with Fund of Funds (FoF) and Traditional Rebalancing
Author: [Your Name]
Date: [Today's Date]

This script loads historical fund and reference index data, computes necessary returns,
simulates trading strategies (currently in FoF mode, with flexibility for Single Fund),
and produces summary reports and visualizations. The design is modular and parameter‐driven,
making it extensible to support a Fund‐of‐Funds strategy in future.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings
import inspect

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)



# =============================================================================
# 1. Data Loading and Preprocessing Functions
# =============================================================================

def load_and_maintain_data(data_file, spy_file, start_date, ref_asset):
    """
    Loads global fund data and reference index data from the provided file paths,
    then processes them by adjusting the start date (to include T-1 business day),
    cleaning duplicates, and computing daily returns.

    Parameters:
      data_file: Path to the CSV file containing fund data.
      spy_file: Path to the CSV file containing reference index data.
      start_date: The intended start date (string or datetime).
      ref_asset: The column name for the reference index (e.g., 'SPY').

    Returns:
      df: Processed fund DataFrame.
      df_ref_index: Processed reference index DataFrame with the column renamed to 'Ref_Index'.
      end_date: The maximum date present in the fund data.
    """
    # Load fund data.
    df_funds = pd.read_csv(data_file)
    df_funds['Date'] = pd.to_datetime(df_funds['Date'])
    df_funds = df_funds.sort_values(by='Date')
    # Compute daily_return for funds if not already present.
    if 'daily_return' not in df_funds.columns:
        df_funds['daily_return'] = df_funds.groupby('Fund')['Fund Value (USD)'].pct_change().fillna(0)

    # Load reference index data.
    df_ref_index = pd.read_csv(spy_file)
    df_ref_index['Date'] = pd.to_datetime(df_ref_index['Date'])
    df_ref_index = df_ref_index.sort_values(by='Date')

    # Convert start_date to datetime and determine T-1 business day.
    start_date = pd.to_datetime(start_date)
    t_minus_1_date = start_date - pd.offsets.BDay(1)

    # Filter fund data to include dates from T-1 onward.
    df = df_funds[df_funds['Date'] >= t_minus_1_date].copy()

    # Clean duplicates and compute daily returns.
    duplicates = df.duplicated(subset=['Date', 'Fund', 'Fund Value (USD)'], keep=False)
    df = df[~duplicates | (df['Fund Return (%)'] != 0)].drop_duplicates()
    df.rename(columns={'Remaining Cap (%)': 'Remaining Cap'}, inplace=True)
    df['daily_return'] = df.groupby('Fund')['Fund Value (USD)'].pct_change().fillna(0)

    # Filter fund data to only include dates >= the intended start_date.
    df = df[df['Date'] >= start_date]
    end_date = pd.to_datetime(df['Date'].max())

    # Process reference index data: filter to the period [T-1, end_date].
    df_ref_index = df_ref_index[(df_ref_index['Date'] >= t_minus_1_date) & (df_ref_index['Date'] <= end_date)].copy()
    if ref_asset not in df_ref_index.columns:
        raise KeyError(f"'{ref_asset}' not found in the reference index DataFrame columns: {df_ref_index.columns.tolist()}")
    df_ref_index = df_ref_index[['Date', ref_asset]].rename(columns={ref_asset: 'Ref_Index'})
    df_ref_index['daily_return'] = df_ref_index['Ref_Index'].pct_change().fillna(0)

    return df, df_ref_index, end_date


def initial_maintenance_fund_from_dfs(
        df_funds, df_ref_index, fund, provided_start, ref_asset, benchmark_ticker, roll_dates,
        rebalance_type, number_funds
):
    """
    Processes the data for a specific fund and aligns the reference index, FoF benchmark, and single ticker buy & hold time series.
    The function adjusts start dates differently based on rebalance_type.

    Parameters:
      df_funds: DataFrame containing fund data.
      df_ref_index: DataFrame containing reference index data.
      fund: A string representing the specific fund to process (e.g., 'FDEC').
      provided_start: A string or datetime for the provided start date; if None, the fund's inception is used.
      ref_asset: The column name in df_ref_index for the reference index (e.g., 'SPY').
      benchmark_ticker: The benchmark ticker column name in df_ref_index (e.g., 'BUFR').
      roll_dates: List of roll dates (Timestamps) for adjusting fund start dates.
      rebalance_type: 'Single' or 'FoF'. For 'FoF', the effective start will be adjusted based on fund universe availability.
      number_funds: Integer representing the minimum number of funds available (for FoF strategies).

    Returns:
      df_funds_filtered_index: DataFrame for strategy vs. Reference Index.
      df_funds_filtered_fof: DataFrame for strategy vs. FoF benchmark.
      df_funds_filtered_single: DataFrame for strategy vs. single ticker buy & hold.
      effective_start_index: Effective start date for strategy vs. Index.
      effective_start_fof: Effective start date for strategy vs. FoF.
      effective_start_single: Effective start date for strategy vs. single ticker.
      fund_end: The maximum date available for this fund.
    """

    # --------------------------
    # 1️⃣ Process Fund Data
    # --------------------------
    df_fund = df_funds[df_funds['Fund'] == fund].copy()
    if df_fund.empty:
        raise ValueError(f"No data found for fund {fund}")

    df_fund['Date'] = pd.to_datetime(df_fund['Date'])
    df_fund = df_fund.sort_values(by='Date')

    fund_min = df_fund['Date'].min()
    effective_start_single = pd.to_datetime(provided_start) if provided_start else fund_min
    effective_start_single = max(effective_start_single, fund_min)

    df_fund_filtered = df_fund[df_fund['Date'] >= effective_start_single].copy()
    fund_end = pd.to_datetime(df_fund_filtered['Date'].max())

    # --------------------------
    # 2️⃣ Process Reference Index Data
    # --------------------------
    df_ref_index['Date'] = pd.to_datetime(df_ref_index['Date'])
    df_ref_index = df_ref_index.sort_values(by='Date')

    # Ensure the reference asset exists.
    if ref_asset not in df_ref_index.columns:
        raise KeyError(f"'{ref_asset}' not found in reference index data. Columns found: {df_ref_index.columns.tolist()}")

    # The effective start for the index is the later of the fund's effective start and the index's first valid date.
    index_start_date = df_ref_index[df_ref_index[ref_asset].notna()]['Date'].min()
    effective_start_index = max(effective_start_single, index_start_date)

    df_ref_index_fund = df_ref_index[df_ref_index['Date'].between(effective_start_index, fund_end)].copy()
    df_ref_index_fund = df_ref_index_fund[['Date', ref_asset]].rename(columns={ref_asset: 'Ref_Index'})
    df_ref_index_fund['daily_return_index'] = df_ref_index_fund['Ref_Index'].pct_change().fillna(0)

    # --------------------------
    # 3️⃣ Process FoF Benchmark Data
    # --------------------------
    if rebalance_type == 'FoF':
        # Use benchmark_ticker to extract the FoF benchmark.
        if benchmark_ticker in df_ref_index.columns:
            df_benchmark_fof = df_ref_index[['Date', benchmark_ticker]].copy().dropna(subset=[benchmark_ticker])
            benchmark_start_date = df_benchmark_fof['Date'].min()
            # Adjust effective_start for FoF based on availability.
            # For example, if the fund launched before the benchmark was available and we require at least number_funds,
            # we adjust effective_start_fof to the first roll date in the fund’s launch month.
            launch_key = fund[1:4]  # Assuming fund ticker format like 'FJAN' where 'JAN' is the launch key.
            if launch_key in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL'] and effective_start_single < benchmark_start_date:
                # Find the first roll date (from the provided roll_dates) that falls in the same month as launch_key.
                adjusted_candidates = [rd for rd in roll_dates if rd.year == effective_start_single.year and rd.month == pd.to_datetime(launch_key, format='%b').month]
                effective_start_fof = adjusted_candidates[0] if adjusted_candidates else effective_start_single
            else:
                effective_start_fof = effective_start_single
            df_benchmark_fof = df_benchmark_fof[df_benchmark_fof['Date'].between(effective_start_fof, fund_end)].copy()
            df_benchmark_fof = df_benchmark_fof.rename(columns={benchmark_ticker: 'Bmk_FoF'})
            df_benchmark_fof['daily_return_fof'] = df_benchmark_fof['Bmk_FoF'].pct_change().fillna(0)
        else:
            df_benchmark_fof = pd.DataFrame(columns=['Date', 'Bmk_FoF', 'daily_return_fof'])
            effective_start_fof = effective_start_single
    else:
        # For Single strategy, FoF benchmark is not applicable.
        df_benchmark_fof = pd.DataFrame(columns=['Date', 'Bmk_FoF', 'daily_return_fof'])
        effective_start_fof = effective_start_single

    # --------------------------
    # 4️⃣ Process Single Ticker Buy & Hold Data
    # --------------------------
    df_fund_filtered['daily_return_strategy'] = df_fund_filtered.groupby('Fund')['Fund Value (USD)'].pct_change().fillna(0)
    df_funds_filtered_single = df_fund_filtered[['Date', 'Fund', 'Fund Value (USD)', 'daily_return_strategy']].copy()
    df_funds_filtered_single.rename(columns={'Fund Value (USD)': 'NAV_Single'}, inplace=True)
    df_funds_filtered_single['daily_return_single'] = df_funds_filtered_single['daily_return_strategy']
    # For single ticker, always use the fund's inception date.
    effective_start_single = fund_min

    # --------------------------
    # 5️⃣ Merge and Return Final DataFrames
    # --------------------------
    # Strategy vs. Index: Merge fund data with reference index data.
    df_funds_filtered_index = df_fund_filtered[['Date', 'Fund', 'Fund Value (USD)', 'daily_return_strategy']].copy()
    df_funds_filtered_index.rename(columns={'Fund Value (USD)': 'NAV'}, inplace=True)
    df_funds_filtered_index = df_funds_filtered_index.merge(df_ref_index_fund, on='Date', how='inner')

    # Strategy vs. FoF: Merge fund data with FoF benchmark data.
    df_funds_filtered_fof = df_fund_filtered[['Date', 'Fund', 'Fund Value (USD)', 'daily_return_strategy']].copy()
    df_funds_filtered_fof.rename(columns={'Fund Value (USD)': 'NAV'}, inplace=True)
    df_funds_filtered_fof = df_funds_filtered_fof.merge(df_benchmark_fof, on='Date', how='inner')
    # Merge additional reference index info for downside rule analysis.
    df_funds_filtered_fof = df_funds_filtered_fof.merge(df_ref_index_fund[['Date', 'Ref_Index', 'daily_return_index']],
                                                        on='Date', how='left')
    # Strategy vs. Single Ticker: Merge single ticker buy & hold data with reference index data.
    df_funds_filtered_single = df_funds_filtered_single.merge(df_ref_index_fund[['Date', 'Ref_Index', 'daily_return_index']],
                                                              on='Date', how='left')

    return (df_funds_filtered_index,
            df_funds_filtered_fof,
            df_funds_filtered_single,
            effective_start_index,  # Strategy vs. Index start date.
            effective_start_fof,  # FoF start date.
            effective_start_single,  # Single ticker start date.
            fund_end)


# =============================================================================
# 2. Roll Date and Forward Filling Functions
# =============================================================================
def get_roll_dates(start_date, end_date, frequency, launch_month=None, month_mapping=None, file_path=None, universe_df=None):
    """
    Returns roll dates based on either a provided CSV file or by calculating them from a given DataFrame.

    Parameters:
      start_date: Start date (string or datetime).
      end_date: End date (string or datetime).
      frequency: 'monthly', 'quarterly', 'semi-annual', or 'annual'.
      launch_month: Optional launch month (e.g., 'FEB').
      month_mapping: Dictionary mapping month abbreviations (or keys) to full names.
      file_path: Optional file path to a CSV containing roll dates. The CSV must have a header row with columns
                 for each frequency.
      universe_df: Optional DataFrame containing fund data from which roll dates can be calculated.

    Returns:
      roll_dates: A sorted list of pd.Timestamp objects representing the roll dates.
      adjusted_start_date: The (possibly adjusted) start_date used for roll date calculation.

    Raises:
      ValueError: If neither file_path nor universe_df is provided or if required dates cannot be found.
    """
    # If a file_path is provided, load roll dates from CSV.
    if file_path is not None:
        df_roll = pd.read_csv(file_path)
        roll_dates_dict = {}
        for col in df_roll.columns:
            col_dates = df_roll[col].dropna()
            if not col_dates.empty:
                dates = pd.to_datetime(col_dates)
                roll_dates_dict[col.lower()] = sorted(dates.tolist())
        if frequency.lower() in roll_dates_dict:
            return roll_dates_dict[frequency.lower()], pd.to_datetime(start_date)
        else:
            raise ValueError(f"Frequency '{frequency}' not found in the roll dates file.")

    # Otherwise, if a universe DataFrame is provided, calculate roll dates from it.
    elif universe_df is not None:
        universe_df['Date'] = pd.to_datetime(universe_df['Date'])
        unique_dates = universe_df['Date'].drop_duplicates().sort_values()

        # Adjust start_date based on the launch month, if provided.
        adjusted_start_date = pd.to_datetime(start_date)
        if launch_month and month_mapping:
            start_year = adjusted_start_date.year
            launch_dates = unique_dates[
                (unique_dates.dt.year == start_year) &
                (unique_dates.dt.month == list(month_mapping.keys()).index(launch_month) + 1)
                ]
            if not launch_dates.empty:
                adjusted_start_date = launch_dates.iloc[-1]
            else:
                raise ValueError(f"No available dates found for the launch month: {launch_month}")

        filtered_dates = unique_dates[(unique_dates >= adjusted_start_date) &
                                      (unique_dates <= pd.to_datetime(end_date))]

        if frequency == 'monthly':
            roll_dates = filtered_dates.groupby(filtered_dates.dt.to_period('M')).max().tolist()
        elif frequency == 'quarterly':
            roll_dates = []
            current_date = adjusted_start_date
            while current_date <= pd.to_datetime(end_date):
                last_business_day = filtered_dates[
                    (filtered_dates.dt.year == current_date.year) &
                    (filtered_dates.dt.month == current_date.month)
                    ].max()
                if pd.notna(last_business_day):
                    roll_dates.append(last_business_day)
                current_date += pd.DateOffset(months=3)
        elif frequency == 'semi-annual':
            roll_dates = []
            current_date = adjusted_start_date
            while current_date <= pd.to_datetime(end_date):
                last_business_day = filtered_dates[
                    (filtered_dates.dt.year == current_date.year) &
                    (filtered_dates.dt.month == current_date.month)
                    ].max()
                if pd.notna(last_business_day):
                    roll_dates.append(last_business_day)
                current_date += pd.DateOffset(months=6)
        elif frequency == 'annual':
            roll_dates = filtered_dates.groupby(filtered_dates.dt.to_period('A')).max().tolist()
        else:
            raise ValueError("Invalid frequency. Use 'monthly', 'quarterly', 'semi-annual', or 'annual'.")

        return roll_dates, adjusted_start_date

    else:
        raise ValueError("Either file_path or universe_df must be provided.")


def forward_fill_all(df, roll_dates, end_date):
    """
    Applies forward filling on the provided DataFrame.

    This function converts the 'date' and 'roll_date' columns to datetime (if not already),
    groups the DataFrame by ['date', 'roll_date', 'fund'], and for each group, it forward fills
    the data from the day after the group's minimum date (assumed start) to the specified end_date,
    using business days frequency.

    Parameters:
      df: DataFrame containing at least the columns 'date', 'roll_date', and 'fund'.
      roll_dates: Not used internally in this implementation but kept for compatibility.
      end_date: The end date (as a string or datetime) up to which to forward fill.

    Returns:
      A new DataFrame with forward-filled data for each group.
    """
    # Ensure 'date' and 'roll_date' are datetime.
    df['date'] = pd.to_datetime(df['date'])
    df['roll_date'] = pd.to_datetime(df['roll_date'])

    def forward_fill_group(group):
        # Determine the starting date for the group (minimum 'date' in the group)
        group_start = group['date'].min()
        # Create a business-day date range from the day after group_start to end_date.
        date_range = pd.date_range(start=group_start + pd.DateOffset(days=1), end=pd.to_datetime(end_date), freq='B')
        # Use the last available row in the group to fill future dates.
        last_row = group.iloc[-1]
        # Replicate the row values for each date in the date range.
        values = np.tile(last_row.values, (len(date_range), 1))
        # Create a DataFrame with these replicated rows.
        filled_df = pd.DataFrame(values, index=date_range, columns=group.columns)
        filled_df.reset_index(inplace=True)
        filled_df.rename(columns={'index': 'filled_date'}, inplace=True)
        return filled_df

    # Group by 'date', 'roll_date', and 'fund' and apply forward filling.
    processed_df = df.groupby(['date', 'roll_date', 'fund']).apply(forward_fill_group)
    # Remove the multi-index created by groupby and reset index.
    processed_df = processed_df.reset_index(drop=True)
    return processed_df


# =============================================================================
# 3. Trigger and Selection Algorithms
# =============================================================================

def downside_before_buffer_trigger(universe_df, current_date, current_fund, threshold):
    current_fund_data = universe_df[(universe_df['Date'] == current_date) & (universe_df['Fund'] == current_fund)]
    if not current_fund_data.empty:
        downside_before_buffer = current_fund_data.iloc[0]['Downside Before Buffer']
        return downside_before_buffer < threshold
    return False


def ref_asset_return_trigger(universe_df, current_date, current_fund, threshold):
    current_fund_data = universe_df[(universe_df['Date'] == current_date) & (universe_df['Fund'] == current_fund)]
    if not current_fund_data.empty:
        ref_asset_return = current_fund_data.iloc[0]['Reference Asset Return']
        return ref_asset_return > threshold
    return False


def upside_downside_rules(current_value, baseline_value, current_date, index_df, period_start_date, period_end_date, trade_history):
    """
    Checks upside and downside sell rules.
    Upside: if new fund's normalized return (current_value/100 - 1) >= 1.5%.
    Downside: if reference index return over the period <= -5%.
    """
    fund_return = (current_value / 100) - 1  # since new fund simulation resets at 100
    sell_triggered = False
    if index_df is not None:
        period_index = index_df[(index_df['Date'] > period_start_date) & (index_df['Date'] <= period_end_date)]
        if not period_index.empty:
            first_index = period_index.iloc[0]['Ref_Index']
            last_index = period_index.iloc[-1]['Ref_Index']
            index_return = (last_index / first_index) - 1
            if index_return <= -0.05:
                trade_history.append({
                    'Date': current_date,
                    'sell_rule': 'downside',
                    'description': f'Index down {index_return * 100:.2f}% triggered downside sell rule.',
                    'current_value': current_value
                })
                sell_triggered = True
    if fund_return >= 0.015:
        trade_history.append({
            'Date': current_date,
            'sell_rule': 'upside',
            'description': f'New fund up {fund_return * 100:.2f}% triggered upside sell rule.',
            'current_value': current_value
        })
        sell_triggered = True
    return sell_triggered


def remaining_cap_selection(universe_df, current_date):
    eligible_funds = universe_df[universe_df['Date'] == current_date]
    if eligible_funds.empty:
        return None
    return eligible_funds.loc[eligible_funds['Remaining Cap'].idxmax()]['Fund']


def highest_outcome_and_cap_selection(universe_df, current_date):
    eligible_funds = universe_df[universe_df['Date'] == current_date]
    if eligible_funds.empty:
        return None
    return eligible_funds.loc[(eligible_funds['Remaining Outcome Days'] + eligible_funds['Remaining Cap']).idxmax()]['Fund']


def cost_analysis_selection(universe_df, current_date, threshold):
    for index, row in universe_df.iterrows():
        remaining_days = row['Remaining Outcome Days']
        if remaining_days > 0:
            reference_asset_return = row['Reference Asset Return']
            downside = threshold - reference_asset_return
            if reference_asset_return < threshold:
                universe_df.at[index, 'cost_analysis'] = abs((row['Fund Value (USD)'] / downside) * (remaining_days / 365))
            else:
                universe_df.at[index, 'cost_analysis'] = float('inf')
    eligible_funds = universe_df[universe_df['Date'] == current_date]
    if eligible_funds.empty:
        return None
    return eligible_funds.loc[eligible_funds['cost_analysis'].idxmin()]['Fund']


def existing_month_selection(current_date, series, month_mapping):
    """
    Selects the fund corresponding to the current month.
    For example, if series='F' and current_date is in March, returns 'FMAR'.
    """
    month_abbr = current_date.strftime("%b").upper()
    return series + month_abbr


def call_selection_algo(selection_algo, current_date, series, month_mapping):
    """
    Calls the given selection algorithm with appropriate arguments.
    """
    params = inspect.signature(selection_algo).parameters
    if len(params) == 2:
        return selection_algo(current_date, series)
    elif len(params) == 3:
        return selection_algo(current_date, series, month_mapping)
    else:
        raise ValueError("Selection algorithm must accept either 2 or 3 arguments.")


#### DOWNSIDE BEFORE BUFFER

def downside_before_buffer_trigger(df, current_date, current_fund):
    current_fund_data = df[(df['Date'] == current_date) & (df['Fund'] == current_fund)]
    if not current_fund_data.empty:
        downside_before_buffer = current_fund_data.iloc[0]['Downside Before Buffer']
        # Check if the downside is less than -5%
        if downside_before_buffer < -0.05:
            x=1
        return downside_before_buffer < -0.05
    return False


def downside_before_buffer_trading(df, initial_fund, current_date, threshold):
    # Here, we would call all the selection algorithms if the trigger condition is met
    results = {}
    if downside_before_buffer_trigger(df, current_date, initial_fund):
        # results['cost_analysis_selection'] = calculate_cost_analysis_and_selection(df, current_date, threshold)
        results['highest_outcome_and_cap_selection'] = highest_outcome_and_cap_selection(df, current_date)
        results['remaining_cap_selection'] = remaining_cap_selection(df, current_date)
        results['highest_outcome_and_realize_cap_return_selection'] = highest_outcome_and_realize_cap_return_selection(df, current_date)
    return results



# =============================================================================
# 4. Benchmark Generation
# =============================================================================

def generate_benchmarks(df, rebalance_type='FoF', end_date=None, start_date=None, roll_dates=None, rebalance_frequency=None, series=None):
    """
    Generates benchmark performance for the provided fund data.

    Parameters:
      df: Fund DataFrame.
      rebalance_type: 'FoF' or 'Traditional'
      end_date, start_date: Date range.
      roll_dates: List of roll dates.
      rebalance_frequency: Frequency string.
      series: Fund series identifier.

    Returns:
      benchmarks (empty dict) and a DataFrame of daily benchmark performance.
    """
    benchmarks = {}
    daily_benchmark_performance_data = []
    df = df.sort_values(by='Date')
    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if df.empty:
        raise ValueError("No data available after the specified start date.")
    if series:
        df = df[df['Fund'].str.startswith(series)]
    roll_dates = sorted(roll_dates)
    df = df.sort_values(by='Date')
    benchmark_nav = 100.0
    initialized = False
    benchmark_definitions = []

    for i in range(len(roll_dates) - 1):
        start_date_roll = roll_dates[i]
        next_rebalance_date = roll_dates[i + 1]
        rebalance_period_data = df[(df['Date'] >= start_date_roll) & (df['Date'] < next_rebalance_date)]
        if rebalance_period_data.empty:
            continue
        num_funds = rebalance_period_data['Fund'].nunique()
        rebalance_period_data.loc[:, 'weight'] = 1 / num_funds
        benchmark_definitions.append({
            'Rebalance_Date': start_date_roll,
            'Rebalance_Frequency': rebalance_frequency,
            'Funds': list(rebalance_period_data['Fund'].unique()),
            'Weights': list(rebalance_period_data.groupby('Fund')['weight'].first())
        })
        fund_navs = {fund: 100 for fund in rebalance_period_data['Fund'].unique()}
        dates = rebalance_period_data['Date'].unique()
        for idx, current_date in enumerate(dates):
            daily_data = rebalance_period_data[rebalance_period_data['Date'] == current_date]
            daily_return = 0
            for fund in daily_data['Fund']:
                fund_data = daily_data[daily_data['Fund'] == fund]
                if not fund_data.empty:
                    fund_return = fund_data['daily_return'].iloc[0]
                    fund_navs[fund] *= (1 + fund_return)
            total_nav = sum(fund_navs.values())
            fund_weights = {fund: nav / total_nav for fund, nav in fund_navs.items()}
            for fund, weight in fund_weights.items():
                fund_data = daily_data[daily_data['Fund'] == fund]
                if not fund_data.empty:
                    daily_return += weight * fund_data['daily_return'].iloc[0]
            if idx == 0 and not initialized:
                benchmark_nav = 100
                initialized = True
            else:
                benchmark_nav *= (1 + daily_return)
            daily_benchmark_performance_data.append({
                'Date': current_date,
                'Benchmark_NAV': benchmark_nav,
                'Rebalance_Frequency': rebalance_frequency
            })
    daily_benchmark_performance = pd.DataFrame(daily_benchmark_performance_data)
    benchmark_definitions_df = pd.DataFrame(benchmark_definitions)
    benchmark_definitions_df.to_excel("benchmark_definitions.xlsx", index=False)

    return benchmarks, daily_benchmark_performance


# =============================================================================
# 5. Backtesting Simulation Functions
# =============================================================================

def general_rebalancing(threshold, rebalance_frequency, selection_algorithms, trigger_algorithms,
                        rebalance_type, apply_stcg, data, roll_dates, start_date, end_date, series):
    """
    Runs rebalancing simulation over the provided data.

    For FoF mode, it selects funds (equal weighting by default) and simulates portfolio drift between roll dates.

    Returns cumulative performance, trade history, and a daily NAV series DataFrame.
    """
    universe_df = data[data['Fund'].str.startswith(series)].copy().sort_values(by='Date')
    cumulative_performance = {'FoF': {'dates': [], 'standardized_nav': [100]}, 'Traditional': {'dates': [], 'standardized_nav': [100]}}
    trade_history = []
    daily_nav_series = pd.Series([100], index=[start_date])
    total_portfolio_value = 100

    for i in range(len(roll_dates) - 1):
        rebalance_date = roll_dates[i]
        next_rebalance_date = roll_dates[i + 1]
        current_day_data = universe_df[universe_df['Date'] == rebalance_date]
        if not current_day_data.empty:
            if rebalance_type == "FoF":
                current_day_data = current_day_data.sort_values(by='Remaining Cap', ascending=False)
                num_exclude = int(len(current_day_data) * threshold)
                selected_funds = current_day_data if num_exclude == 0 else current_day_data.iloc[:-num_exclude]
                portfolio_values = {}
                num_selected_funds = len(selected_funds)
                equal_weight = 1 / num_selected_funds
                for fund in selected_funds['Fund']:
                    portfolio_values[fund] = total_portfolio_value * equal_weight
                period_data = universe_df[(universe_df['Date'] > rebalance_date) & (universe_df['Date'] <= next_rebalance_date) &
                                          (universe_df['Fund'].isin(selected_funds['Fund']))]
                for date in period_data['Date'].unique():
                    daily_data = period_data[period_data['Date'] == date]
                    for _, row in daily_data.iterrows():
                        fund = row['Fund']
                        if fund not in portfolio_values:
                            portfolio_values[fund] = total_portfolio_value * equal_weight
                        else:
                            portfolio_values[fund] *= (1 + row['daily_return'])
                    total_portfolio_value = sum(portfolio_values.values())
                    daily_nav_series.loc[date] = total_portfolio_value
                updated_nav = daily_nav_series.loc[next_rebalance_date] if next_rebalance_date in daily_nav_series.index else daily_nav_series.iloc[-1]
                cumulative_performance['FoF']['dates'].append(rebalance_date)
                cumulative_performance['FoF']['standardized_nav'].append(updated_nav)
                trade_history.append({
                    'Date': rebalance_date,
                    'previous_funds': list(current_day_data['Fund']),
                    'selected_funds': list(selected_funds['Fund']),
                    'weights': [equal_weight] * num_selected_funds,
                    'NAV': updated_nav,
                    'rebalance_type': 'FoF',
                    'Threshold': threshold,
                    'Rebalance_Frequency': rebalance_frequency
                })
    cumulative_daily_performance_df = daily_nav_series.reset_index().rename(columns={'index': 'Date', 0: 'NAV'})
    trade_history_df = pd.DataFrame(trade_history)
    return cumulative_performance, trade_history_df, cumulative_daily_performance_df


def run_backtest_updated(df_global, df_fund, df_ref_index, thresholds, trigger_algorithms, selection_algorithms,
                         rebalance_type='Single', apply_stcg=False, start_date=None, end_date=None,
                         month_mapping=None, series=None, launch_months=None, roll_dates=None, number_funds=None):
    """
    Runs the backtest simulation for a single fund strategy for one specific launch month using the provided or calculated roll_dates.

    For a Single strategy:
      - The simulation tracks overall NAV continuously; upon a trigger event, it switches funds.
      - When switching, the new fund’s performance segment resets to 100, and the overall NAV is computed as:
            overall NAV = carry * (new fund segment NAV / 100)

    For a FoF strategy:
      - The simulation considers the entire universe (i.e. all funds with a ticker starting with series).
      - At each rebalance date, funds are ranked based on a selection metric.
      - If number_funds is provided, the top n funds are selected and equally weighted.
      - The portfolio then drifts until the next rebalance.

    Parameters:
      df_global: Global fund DataFrame.
      df_fund: Fund-specific DataFrame for the current fund.
      df_ref_index: Fund-specific reference index DataFrame.
      thresholds: List of threshold parameters (used if number_funds is None).
      trigger_algorithms: List of trigger algorithm functions.
      selection_algorithms: List of selection algorithm functions.
      rebalance_type: 'Single' or 'FoF'.
      apply_stcg: Boolean flag.
      start_date: Effective start date for the current fund.
      end_date: End date for the current fund.
      month_mapping: Dictionary mapping month abbreviations to full names.
      series: Fund series identifier (e.g., 'F').
      launch_months: A one-element dictionary for the current launch month.
      roll_dates: List of roll dates (filtered so they are >= start_date).
      number_funds: (Optional) If provided (integer), use this to select the top n funds in FoF mode.

    Returns:
      trade_history_all: List of trade events.
      cumulative_daily_performance_df: DataFrame with overall NAV over time.
      used_frequency: String indicating the frequency used.
    """
    # Ensure launch_months is a dictionary.
    if not isinstance(launch_months, dict):
        raise TypeError("launch_months must be a dictionary, got " + str(type(launch_months)))

    trade_history_all = []
    cumulative_daily = {}

    # Calculate roll dates if not provided.
    if roll_dates is None:
        launch_key = list(launch_months.keys())[0]
        roll_dates, adjusted_start_date = get_roll_dates(start_date, end_date, 'monthly',
                                                         launch_month=launch_key, month_mapping=month_mapping,
                                                         universe_df=df_fund)
        used_frequency = "monthly"
    else:
        used_frequency = "provided"

    current_launch_key = list(launch_months.keys())[0]
    current_fund = series + current_launch_key
    print(f"Running backtest for fund: {current_fund} using launch month {current_launch_key}")

    df_fund['Date'] = pd.to_datetime(df_fund['Date'])
    universe_df = df_fund.copy().sort_values(by='Date')

    # --- Branch based on rebalance_type ---
    if rebalance_type == 'Single':
        # Single Fund Simulation: Start at a normalized base of 100.
        carry = 100.0
        segment_value = 100.0
        overall_value = carry * (segment_value / 100)
        cumulative_daily[pd.to_datetime(start_date)] = overall_value

        selection_algo = selection_algorithms[0]
        trigger_algo = trigger_algorithms[0]
        # If thresholds is not provided, we assume number_funds is used; otherwise, threshold is used.
        threshold_val = thresholds[0] if thresholds is not None and thresholds[0] is not None else None
        if (trigger_algo.__name__ == "upside_downside_rules" and
                selection_algo.__name__ == "existing_month_selection"):
            threshold_used = "N/A"
        else:
            threshold_used = threshold_val

        i = 0
        while i < len(roll_dates) - 1:
            period_start = roll_dates[i]
            period_end = roll_dates[i + 1]
            period_dates = pd.date_range(start=period_start, end=period_end, freq='B')
            for date in period_dates:
                day_data = universe_df[universe_df['Date'] == date]
                if not day_data.empty:
                    daily_ret = day_data.iloc[0]['daily_return']
                    segment_value *= (1 + daily_ret)
                overall_value = carry * (segment_value / 100)
                cumulative_daily[date] = overall_value

            sell_trigger = trigger_algo(segment_value, 100, period_end, df_ref_index, period_start, period_end, trade_history_all)
            if sell_trigger:
                new_fund = call_selection_algo(selection_algo, period_end, series, month_mapping)
                if new_fund != current_fund:
                    trade_history_all.append({
                        'Date': period_end,
                        'action': 'switch_fund',
                        'from_fund': current_fund,
                        'to_fund': new_fund,
                        'value': overall_value
                    })
                    current_fund = new_fund
                    carry = overall_value
                    segment_value = 100.0
                    new_universe = df_global[df_global['Fund'] == current_fund].copy().sort_values(by='Date')
                    new_universe['Date'] = pd.to_datetime(new_universe['Date'])
                    new_universe = new_universe[new_universe['Date'] >= pd.to_datetime(period_end)]
                    if new_universe.empty:
                        available_dates = df_global[df_global['Fund'] == current_fund]['Date']
                        if not available_dates.empty:
                            period_end = available_dates.min()
                            new_universe = df_global[(df_global['Fund'] == current_fund) & (df_global['Date'] >= period_end)].copy().sort_values(by='Date')
                            if 'daily_return' not in new_universe.columns:
                                new_universe['daily_return'] = new_universe.groupby('Fund')['Fund Value (USD)'].pct_change().fillna(0)
                        else:
                            print(f"No data available for fund {current_fund} after period_end {period_end}.")
                    universe_df = new_universe
                    # Reset baseline for new fund segment.
                    baseline_value = 100.0
                    # Update roll_dates to include dates >= period_end (inclusive).
                    roll_dates = [rd for rd in roll_dates if rd >= period_end]
                    i = 0
                    continue
            else:
                baseline_value = segment_value
            i += 1

        cumulative_daily_performance_df = pd.DataFrame(list(cumulative_daily.items()), columns=['Date', 'NAV'])
        cumulative_daily_performance_df = cumulative_daily_performance_df.sort_values(by='Date')
        return trade_history_all, cumulative_daily_performance_df, used_frequency

    elif rebalance_type == 'FoF':
        # Fund-of-Funds Simulation: Work with the entire universe (all funds starting with the series).
        universe_df = df_global[df_global['Fund'].str.startswith(series)].copy().sort_values(by='Date')
        total_portfolio_value = 100.0
        daily_nav_series = pd.Series([100], index=[pd.to_datetime(start_date)])
        trade_history_all = []
        # Loop over roll dates for portfolio rebalancing.
        for i in range(len(roll_dates) - 1):
            rebalance_date = roll_dates[i]
            next_rebalance_date = roll_dates[i + 1]
            current_day_data = universe_df[universe_df['Date'] == rebalance_date]
            if number_funds is not None:
                # Rank funds based on a metric (e.g., Remaining Cap) and select the top number_funds.
                current_day_data = current_day_data.sort_values(by='Remaining Cap', ascending=False)
                selected_funds = current_day_data.head(number_funds)
            else:
                selected_funds = current_day_data[current_day_data['Remaining Cap'] >= thresholds[0]]
            num_selected = len(selected_funds)
            if num_selected == 0:
                print(f"No eligible funds on {rebalance_date}")
                continue
            equal_weight = 1 / num_selected
            portfolio_values = {fund: total_portfolio_value * equal_weight for fund in selected_funds['Fund']}
            period_data = universe_df[(universe_df['Date'] > rebalance_date) &
                                      (universe_df['Date'] <= next_rebalance_date) &
                                      (universe_df['Fund'].isin(selected_funds['Fund']))]
            for date in period_data['Date'].unique():
                daily_data = period_data[period_data['Date'] == date]
                for fund in daily_data['Fund']:
                    fund_data = daily_data[daily_data['Fund'] == fund]
                    if not fund_data.empty:
                        daily_ret = fund_data.iloc[0]['daily_return']
                        portfolio_values[fund] *= (1 + daily_ret)
                total_portfolio_value = sum(portfolio_values.values())
                daily_nav_series.loc[date] = total_portfolio_value
            trade_history_all.append({
                'Date': rebalance_date,
                'previous_funds': list(current_day_data['Fund']),
                'selected_funds': list(selected_funds['Fund']),
                'weights': [equal_weight] * num_selected,
                'NAV': daily_nav_series.loc[next_rebalance_date],
                'rebalance_type': 'FoF',
                'Number_of_Funds': number_funds
            })
        cumulative_daily_performance_df = daily_nav_series.reset_index().rename(columns={'index': 'Date', 0: 'NAV'})
        return trade_history_all, cumulative_daily_performance_df, used_frequency

    else:
        raise ValueError("Invalid rebalance_type. Use 'Single' or 'FoF'.")


def run_all_backtests(df, df_ref_index, thresholds, trigger_algorithms, selection_algorithms,
                      rebalance_type, apply_stcg, start_date, end_date, month_mapping, launch_months,
                      roll_dates_file, series, rebalance_frequencies, number_funds):
    """
    Runs backtests for every combination of launch month and rebalancing frequency using preprocessed data.

    For each launch month:
      - The specific fund ticker is determined by concatenating the series with the launch key.
      - The processed DataFrame 'df' (from load_and_maintain_data) is filtered for that fund,
        which determines the effective start date and fund end.
      - The consolidated get_roll_dates function is used to obtain roll dates for that fund’s period.
      - run_backtest_updated is called to simulate the strategy, passing either a threshold or a number of funds.
        (If both thresholds and number_funds are provided, number_funds takes precedence.)

    Returns:
      perf_dict: Dictionary of strategy performance DataFrames keyed by combination string.
      trade_history_dict: Dictionary of trade history lists keyed by combination string.
      used_frequency_dict: Dictionary mapping each combination key to its used frequency.
      cumulative_daily_performance_dict: Dictionary keyed by combination containing daily NAV DataFrames.
      daily_benchmark_performance_dict: Dictionary keyed by combination containing benchmark NAV DataFrames.
      ref_asset_index_performance_dict: Dictionary keyed by launch month for reference index performance.
    """
    # Ensure launch_months is a dictionary.
    if not isinstance(launch_months, dict):
        launch_months = {month: month_mapping.get(month, month) for month in launch_months}

    perf_dict = {}
    trade_history_dict = {}
    used_frequency_dict = {}
    combined_daily_performance = pd.DataFrame()
    daily_benchmark_performance = pd.DataFrame()
    ref_asset_index_performance_dict = {}

    # Loop over each rebalance frequency and each launch month.
    for freq in rebalance_frequencies:
        for launch_key, launch_full in launch_months.items():
            fund_name = series + launch_key  # e.g., 'FDEC'
            # Filter the processed fund DataFrame for the specific fund.
            df_fund_filtered = df[df['Fund'] == fund_name].copy()
            if df_fund_filtered.empty:
                print(f"No data found for fund {fund_name}")
                continue

            effective_start = df_fund_filtered['Date'].min()
            fund_end = pd.to_datetime(df_fund_filtered['Date'].max())
            # If an overall end_date is provided, choose the earlier of that and the fund's own end.
            if end_date is not None:
                provided_end = pd.to_datetime(end_date)
                fund_end = provided_end if provided_end < fund_end else fund_end

            # Obtain roll dates using the consolidated get_roll_dates function.
            try:
                filtered_roll_dates, adjusted_start = get_roll_dates(
                    start_date=effective_start,
                    end_date=fund_end,
                    frequency=freq,
                    launch_month=launch_key,
                    month_mapping=month_mapping,
                    file_path=roll_dates_file
                )
            except Exception as ex:
                print(f"Error calculating roll dates for fund {fund_name}: {ex}")
                continue

            if not filtered_roll_dates:
                print(f"No roll dates available for fund {fund_name} after effective start {effective_start}. Skipping.")
                continue

            current_launch = {launch_key: launch_full}
            # Determine selection criteria:
            # If thresholds is None (or its first element is None), we use number_funds.
            if thresholds is None or thresholds[0] is None:
                threshold_param = None
                key = f"NumberFunds_{number_funds}_Frequency_{freq}_LaunchMonth_{launch_key}"
            else:
                # If both are provided, number_funds takes precedence.
                threshold_param = None if number_funds is not None else thresholds[0]
                key = f"NumberFunds_{number_funds}_Frequency_{freq}_LaunchMonth_{launch_key}" if number_funds is not None else \
                    f"Threshold_{thresholds[0]}_Frequency_{freq}_LaunchMonth_{launch_key}"

            # Run the backtest simulation for this fund.
            trade_history_all, cum_perf_all, used_freq = run_backtest_updated(
                df_global=df,
                df_fund=df_fund_filtered,
                df_ref_index=df_ref_index[df_ref_index['Date'] >= effective_start].copy(),
                thresholds=[threshold_param] if threshold_param is not None else [None],
                trigger_algorithms=trigger_algorithms,
                selection_algorithms=selection_algorithms,
                rebalance_type=rebalance_type,
                apply_stcg=apply_stcg,
                start_date=effective_start,
                end_date=fund_end,
                month_mapping=month_mapping,
                series=series,
                launch_months=current_launch,
                roll_dates=filtered_roll_dates,
                number_funds=number_funds
            )
            perf_dict[key] = cum_perf_all
            trade_history_dict[key] = trade_history_all
            used_frequency_dict[key] = freq

            # Process the reference index for this fund period.
            df_ref_index_filtered = df_ref_index[df_ref_index['Date'] >= effective_start].copy()
            df_ref_index_filtered['Ref_Index_NAV'] = 100
            for i in range(1, len(df_ref_index_filtered)):
                df_ref_index_filtered.iloc[i, df_ref_index_filtered.columns.get_loc('Ref_Index_NAV')] = \
                    df_ref_index_filtered.iloc[i - 1, df_ref_index_filtered.columns.get_loc('Ref_Index_NAV')] * (
                            1 + df_ref_index_filtered.iloc[i, df_ref_index_filtered.columns.get_loc('daily_return')]
                    )
            df_ref_index_filtered = df_ref_index_filtered[['Date', 'Ref_Index_NAV']]
            ref_asset_key = f"Reference_Asset_{launch_key}"
            ref_asset_index_performance_dict[ref_asset_key] = df_ref_index_filtered

            combined_daily_performance = pd.concat([combined_daily_performance, cum_perf_all], ignore_index=True)
            # (Optionally build daily_benchmark_performance_dict here.)

    # Build dictionaries keyed by combination from combined_daily_performance.
    cumulative_daily_performance_dict = {}
    daily_benchmark_performance_dict = {}

    if number_funds is not None:
        # When using number_funds, ignore thresholds.
        for frequency in rebalance_frequencies:
            for launch_key in launch_months.keys():
                key = f"NumberFunds_{number_funds}_Frequency_{frequency}_LaunchMonth_{launch_key}"
                cumulative_daily_performance_dict[key] = combined_daily_performance[
                    (combined_daily_performance['Rebalance_Frequency'] == frequency) &
                    (combined_daily_performance['Launch_Month'] == launch_key)
                    ].drop(columns=['Threshold', 'Rebalance_Frequency', 'Launch_Month'], errors='ignore')
                daily_benchmark_performance_dict[key] = pd.DataFrame()  # Placeholder
    else:
        for threshold in thresholds:
            for frequency in rebalance_frequencies:
                for launch_key in launch_months.keys():
                    key = f"Threshold_{threshold}_Frequency_{frequency}_LaunchMonth_{launch_key}"
                    cumulative_daily_performance_dict[key] = combined_daily_performance[
                        (combined_daily_performance['Threshold'] == threshold) &
                        (combined_daily_performance['Rebalance_Frequency'] == frequency) &
                        (combined_daily_performance['Launch_Month'] == launch_key)
                        ].drop(columns=['Threshold', 'Rebalance_Frequency', 'Launch_Month'], errors='ignore')
                    daily_benchmark_performance_dict[key] = pd.DataFrame()  # Placeholder

    return (perf_dict, trade_history_dict, used_frequency_dict,
            cumulative_daily_performance_dict, daily_benchmark_performance_dict,
            ref_asset_index_performance_dict)


# =============================================================================
# 6. Reporting and Export Functions
# =============================================================================

def create_backtest_dataframe_all(perf_dict, trade_history_dict, df_ref_index,
                                  selection_algorithms, trigger_algorithms, thresholds,
                                  launch_months, series, rebalance_type, used_frequency_dict,
                                  ref_asset):
    """
    Aggregates performance and trade history data into a summary DataFrame.

    Returns a DataFrame containing overall cumulative return, annualized return, volatility,
    and Sharpe ratio for the strategy and benchmark, along with number of trades.
    """
    if 'Ref_Index' not in df_ref_index.columns:
        if ref_asset in df_ref_index.columns:
            df_ref_index = df_ref_index.rename(columns={ref_asset: 'Ref_Index'})
        else:
            raise KeyError(f"Neither 'Ref_Index' nor '{ref_asset}' found in df_ref_index.")

    records = []
    for key, cum_perf_df in perf_dict.items():
        trade_history_all = trade_history_dict[key]
        used_frequency = used_frequency_dict[key]
        parts = key.split("_")
        launch_key = parts[1]
        launch_full = launch_months[launch_key]
        selection_algo = selection_algorithms[0]
        trigger_algo = trigger_algorithms[0]
        threshold_val = thresholds[0]
        if (trigger_algo.__name__ == "upside_downside_rules" and selection_algo.__name__ == "existing_month_selection"):
            threshold_used = "N/A"
        else:
            threshold_used = threshold_val

        cum_perf_df['Date'] = pd.to_datetime(cum_perf_df['Date'])
        cum_perf_df = cum_perf_df.sort_values(by='Date')
        strategy_start_date = cum_perf_df['Date'].iloc[0]
        strategy_end_date = cum_perf_df['Date'].iloc[-1]
        strat_initial = cum_perf_df['NAV'].iloc[0]
        strat_final = cum_perf_df['NAV'].iloc[-1]
        strat_cum_return = (strat_final / strat_initial * 100) - 100
        days = (strategy_end_date - strategy_start_date).days
        if days > 0:
            strat_ann_return = (strat_final / strat_initial) ** (365 / days) - 1
        else:
            strat_ann_return = np.nan
        strat_daily_ret = cum_perf_df['NAV'].pct_change().dropna()
        strat_vol = strat_daily_ret.std() * np.sqrt(252)
        strat_sharpe = strat_ann_return / strat_vol if strat_vol > 0 else np.nan

        df_ref_index['Date'] = pd.to_datetime(df_ref_index['Date'])
        benchmark_df = df_ref_index[(df_ref_index['Date'] >= strategy_start_date) &
                                    (df_ref_index['Date'] <= strategy_end_date)].sort_values(by='Date').copy()
        if benchmark_df.empty:
            bench_ann_return = np.nan
            bench_vol = np.nan
            bench_sharpe = np.nan
        else:
            if 'daily_return' not in benchmark_df.columns:
                benchmark_df['daily_return'] = benchmark_df['Ref_Index'].pct_change().fillna(0)
            benchmark_df['Ref_Index_NAV'] = 100.0
            for i in range(1, len(benchmark_df)):
                prev_nav = benchmark_df.iloc[i - 1]['Ref_Index_NAV']
                daily_ret = benchmark_df.iloc[i]['daily_return']
                benchmark_df.iloc[i, benchmark_df.columns.get_loc('Ref_Index_NAV')] = prev_nav * (1 + daily_ret)
            bench_initial = benchmark_df['Ref_Index_NAV'].iloc[0]
            bench_final = benchmark_df['Ref_Index_NAV'].iloc[-1]
            if days > 0:
                bench_ann_return = (bench_final / bench_initial) ** (365 / days) - 1
            else:
                bench_ann_return = np.nan
            bench_daily_ret = benchmark_df['Ref_Index_NAV'].pct_change().dropna()
            bench_vol = bench_daily_ret.std() * np.sqrt(252)
            bench_sharpe = bench_ann_return / bench_vol if bench_vol > 0 else np.nan

        trades_df = pd.DataFrame(trade_history_all)
        if not trades_df.empty and 'action' in trades_df.columns:
            num_trades = trades_df[trades_df['action'] == 'switch_fund'].shape[0]
        else:
            num_trades = 0

        if threshold_used == "N/A":
            strategy_name = f"{trigger_algo.__name__}_{used_frequency}_{series}_{launch_key}"
        else:
            strategy_name = f"{trigger_algo.__name__}_{threshold_used}_{used_frequency}_{series}_{launch_key}"

        record = {
            'Strategy Name': strategy_name,
            'Selection Algorithm': selection_algo.__name__,
            'Threshold': threshold_used,
            'Rebalance Frequency': used_frequency,
            'Series': series,
            'Launch Month': launch_full,
            'Trigger Algorithm': trigger_algo.__name__,
            'Strategy Cumulative Return (%)': strat_cum_return,
            'Strategy Annualized Return (%)': strat_ann_return * 100,
            'Strategy Sharpe Ratio': strat_sharpe,
            'Benchmark Cumulative Return (%)': (bench_final / bench_initial * 100) - 100 if not np.isnan(bench_initial) else np.nan,
            'Benchmark Annualized Return (%)': bench_ann_return * 100,
            'Benchmark Sharpe Ratio': bench_sharpe,
            'Difference (%)': strat_cum_return - ((bench_final / bench_initial * 100) - 100),
            'Number of Trades': num_trades,
            'Strategy Volatility (%)': strat_vol * 100,
            'Benchmark Volatility (%)': bench_vol * 100,
            # 'Strategy Max Drawdown (%)': strat_max_dd,  # (calculation exists but commented out)
            # 'Benchmark Max Drawdown (%)': bench_max_dd,  # (calculation exists but commented out)
            'Rebalance Type': rebalance_type
        }
        records.append(record)
    result_df = pd.DataFrame(records)
    result_df = result_df.dropna(subset=['Strategy Cumulative Return (%)']).drop_duplicates().sort_values(by=['Threshold', 'Difference'], ascending=[True, False])
    result_df['Difference'] = result_df['Difference'].map('{:.3f}'.format)
    return result_df


def convert_trade_history_dict_to_df(trade_history_dict):
    """
    Converts a dictionary of trade histories (keyed by combination string) into a single DataFrame by merging trade events that occur on the same Date and Combination.
    Also saves the resulting DataFrame to Excel in the 'Excel_Data' folder as 'trade_history_export.xlsx'.
    """
    records = []
    for combination, trade_list in trade_history_dict.items():
        for trade in trade_list:
            trade.setdefault("Combination", combination)
        temp_df = pd.DataFrame(trade_list)
        grouped = temp_df.groupby(['Date', 'Combination'], as_index=False)
        for _, group in grouped:
            merged = {}
            for col in group.columns:
                non_null_values = group[col].dropna().tolist()
                merged[col] = non_null_values[0] if non_null_values else np.nan
            records.append(merged)
    result_df = pd.DataFrame(records)
    output_dir = os.path.join(os.path.dirname(os.getcwd()), "Excel_Data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "trade_history_export.xlsx")
    result_df.to_excel(output_file, index=False)
    return result_df


def export_backtest_results(perf_dict, trade_history_dict, df_ref_index,
                            selection_algorithms, trigger_algorithms, thresholds,
                            launch_months, series, rebalance_type, used_frequency_dict,
                            ref_asset, output_path):
    """
    Creates summary and trade history DataFrames and exports them to an Excel file with separate sheets.
    """
    backtest_df = create_backtest_dataframe_all(
        perf_dict, trade_history_dict, df_ref_index,
        selection_algorithms, trigger_algorithms, thresholds,
        launch_months, series, rebalance_type, used_frequency_dict,
        ref_asset
    )
    trade_history_df = convert_trade_history_dict_to_df(trade_history_dict)
    with pd.ExcelWriter(output_path) as writer:
        backtest_df.to_excel(writer, sheet_name='Backtest Results', index=False)
        trade_history_df.to_excel(writer, sheet_name='Trade History', index=False)
    print(f"Exported backtest results and trade history to {output_path}")


# =============================================================================
# 7. Visualization Functions
# =============================================================================

def plot_backtest_results(total_cumulative_performance, cumulative_daily_performance_dict, daily_benchmark_performance_dict, ref_asset_index_performance_dict, thresholds,
                          rebalance_frequencies, launch_months, ref_index):
    for threshold in thresholds:
        threshold_str = f"{int(threshold * 100)}%"
        for frequency in rebalance_frequencies:
            for month_abbr, month_name in launch_months.items():
                key = f"Threshold_{threshold}_Frequency_{frequency}_LaunchMonth_{month_abbr}"
                strat_perf = cumulative_daily_performance_dict.get(key)
                bench_perf = daily_benchmark_performance_dict.get(key)
                ref_asset_perf = ref_asset_index_performance_dict.get(f"Reference_Asset_{month_abbr}")
                if strat_perf is None or bench_perf is None or ref_asset_perf is None:
                    print(f"Warning: No data for threshold {threshold_str}, frequency {frequency}, launch month {month_name}")
                    continue
                strat_perf = strat_perf.sort_values(by='Date')
                bench_perf = bench_perf.sort_values(by='Date')
                ref_asset_perf = ref_asset_perf.sort_values(by='Date')
                common_dates = strat_perf['Date'].isin(bench_perf['Date']) & strat_perf['Date'].isin(ref_asset_perf['Date'])
                aligned_strat = strat_perf[common_dates]
                aligned_bench = bench_perf[bench_perf['Date'].isin(aligned_strat['Date'])]
                aligned_ref = ref_asset_perf[ref_asset_perf['Date'].isin(aligned_strat['Date'])]
                if aligned_strat.empty or aligned_bench.empty or aligned_ref.empty:
                    print(f"Warning: No common dates for threshold {threshold_str}, frequency {frequency}, launch month {month_name}")
                    continue
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(aligned_strat['Date'], aligned_strat['NAV'],
                        label=f"Strategy_FoF (Threshold {threshold_str}, {frequency}, {month_name})", linestyle='-', linewidth=1.5, color='blue')
                ax.plot(aligned_bench['Date'], aligned_bench['Benchmark_NAV'],
                        label=f"Benchmark (Threshold {threshold_str}, {frequency}, {month_name})", linestyle='-', linewidth=2, color='red')
                ax.plot(aligned_ref['Date'], aligned_ref['Ref_Index_NAV'],
                        label=f"{ref_index} (Reference Asset)", linestyle='--', linewidth=2, color='green')
                ax.set_title(f"Benchmark, Strategy, and Reference Asset Comparison - Threshold: {threshold_str}, Frequency: {frequency}, Launch Month: {month_name}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Standardized NAV")
                ax.legend(loc='upper left')
                text_box_y_offset = 0.85
                for label, data in [("Strategy_FoF", aligned_strat), ("Benchmark", aligned_bench), (ref_index, aligned_ref)]:
                    if label == "Strategy_FoF" and not aligned_strat.empty:
                        final_nav = aligned_strat['NAV'].iloc[-1]
                        initial_nav = aligned_strat['NAV'].iloc[0]
                    elif label == "Benchmark" and not aligned_bench.empty:
                        final_nav = aligned_bench['Benchmark_NAV'].iloc[-1]
                        initial_nav = aligned_bench['Benchmark_NAV'].iloc[0]
                    elif label == ref_index and not aligned_ref.empty:
                        final_nav = aligned_ref['Ref_Index_NAV'].iloc[-1]
                        initial_nav = aligned_ref['Ref_Index_NAV'].iloc[0]
                    else:
                        continue
                    pct_return = (final_nav / initial_nav - 1) * 100
                    textstr = f"{label}: {pct_return:.2f}% return"
                    ax.text(0.02, text_box_y_offset, textstr, transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
                    text_box_y_offset -= 0.05
                plt.show()


def compare_strategy_vs_hold(perf_dict, df_funds, launch_months, series, window=252):
    """
    For each fund in the launch_months dictionary, calculates the overall and rolling (12-month) metrics
    for both the strategy (from perf_dict) and a simple buy-and-hold benchmark of the fund.

    Returns:
      overall_metrics: DataFrame with overall annualized return, volatility, and Sharpe ratios for strategy vs. holding.
      rolling_metrics: DataFrame with rolling metrics for each fund.
    """
    overall_records = []
    rolling_records = []
    for key, strat_df in perf_dict.items():
        parts = key.split("_")
        if len(parts) < 4:
            continue

        launch_key = parts[1]
        freq = parts[3]

        if launch_key not in launch_months:
            continue

        launch_full = launch_months[launch_key]
        fund_ticker = series + launch_key
        strat_df['Date'] = pd.to_datetime(strat_df['Date'])
        strat_df = strat_df.sort_values(by='Date')
        effective_start = strat_df['Date'].iloc[0]
        fund_end = strat_df['Date'].iloc[-1]
        strat_initial = strat_df['NAV'].iloc[0]
        strat_final = strat_df['NAV'].iloc[-1]
        total_days = (fund_end - effective_start).days
        if total_days > 0:
            strat_ann_return = (strat_final / strat_initial) ** (365 / total_days) - 1
        else:
            strat_ann_return = np.nan
        strat_daily_ret = strat_df['NAV'].pct_change().dropna()
        strat_vol = strat_daily_ret.std() * np.sqrt(252)
        strat_sharpe = strat_ann_return / strat_vol if strat_vol > 0 else np.nan

        fund_data = df_funds[(df_funds['Fund'] == fund_ticker) &
                             (pd.to_datetime(df_funds['Date']) >= effective_start) &
                             (pd.to_datetime(df_funds['Date']) <= fund_end)].copy().sort_values(by='Date')
        if fund_data.empty:
            continue
        if 'daily_return' not in fund_data.columns:
            fund_data['daily_return'] = fund_data.groupby('Fund')['Fund Value (USD)'].pct_change().fillna(0)
        fund_data['Holding_NAV'] = 100 * (1 + fund_data['daily_return']).cumprod()
        hold_initial = fund_data['Holding_NAV'].iloc[0]
        hold_final = fund_data['Holding_NAV'].iloc[-1]
        if total_days > 0:
            hold_ann_return = (hold_final / hold_initial) ** (365 / total_days) - 1
        else:
            hold_ann_return = np.nan
        hold_daily_ret = fund_data['Holding_NAV'].pct_change().dropna()
        hold_vol = hold_daily_ret.std() * np.sqrt(252)
        hold_sharpe = hold_ann_return / hold_vol if hold_vol > 0 else np.nan

        overall_records.append({
            'Fund': fund_ticker,
            'Launch Month': launch_full,
            'Frequency': freq,
            'Strategy Annualized Return (%)': strat_ann_return * 100,
            'Strategy Volatility (%)': strat_vol * 100,
            'Strategy Sharpe Ratio': strat_sharpe,
            'Holding Annualized Return (%)': hold_ann_return * 100,
            'Holding Volatility (%)': hold_vol * 100,
            'Holding Sharpe Ratio': hold_sharpe,
            'Strategy Cumulative Return (%)': (strat_final / strat_initial * 100) - 100,
            'Holding Cumulative Return (%)': (hold_final / hold_initial * 100) - 100
        })

        strat_roll = strat_df.set_index('Date')['NAV']
        hold_roll = fund_data.set_index('Date')['Holding_NAV']

        def calc_rolling_metrics(nav_series, window):
            roll_return = nav_series.rolling(window=window).apply(lambda x: (x[-1] / x[0]) ** (365 / len(x)) - 1, raw=True)
            roll_vol = nav_series.pct_change().rolling(window=window).std() * np.sqrt(252)
            roll_sharpe = roll_return / roll_vol
            return roll_return, roll_vol, roll_sharpe

        strat_roll_return, strat_roll_vol, strat_roll_sharpe = calc_rolling_metrics(strat_roll, window)
        hold_roll_return, hold_roll_vol, hold_roll_sharpe = calc_rolling_metrics(hold_roll, window)
        roll_df = pd.DataFrame({
            'Date': strat_roll_return.index,
            'Strategy Annualized Return (%)': strat_roll_return * 100,
            'Strategy Volatility (%)': strat_roll_vol * 100,
            'Strategy Sharpe Ratio': strat_roll_sharpe,
            'Holding Annualized Return (%)': hold_roll_return * 100,
            'Holding Volatility (%)': hold_roll_vol * 100,
            'Holding Sharpe Ratio': hold_roll_sharpe
        }).dropna().reset_index(drop=True)
        roll_df['Fund'] = fund_ticker
        roll_df['Launch Month'] = launch_full
        roll_df['Frequency'] = freq
        rolling_records.append(roll_df)

    overall_metrics = pd.DataFrame(overall_records)
    rolling_metrics = pd.concat(rolling_records, ignore_index=True) if rolling_records else pd.DataFrame()
    return overall_metrics, rolling_metrics


# =============================================================================
# 7. Export Function
# =============================================================================

def export_backtest_results(perf_dict, trade_history_dict, df_ref_index,
                            selection_algorithms, trigger_algorithms, thresholds,
                            launch_months, series, rebalance_type, used_frequency_dict,
                            ref_asset, output_path):
    """
    Exports backtest summary and trade history DataFrames to an Excel file with separate sheets.
    """
    backtest_df = create_backtest_dataframe_all(
        perf_dict, trade_history_dict, df_ref_index,
        selection_algorithms, trigger_algorithms, thresholds,
        launch_months, series, rebalance_type, used_frequency_dict,
        ref_asset
    )
    trade_history_df = convert_trade_history_dict_to_df(trade_history_dict)
    with pd.ExcelWriter(output_path) as writer:
        backtest_df.to_excel(writer, sheet_name='Backtest Results', index=False)
        trade_history_df.to_excel(writer, sheet_name='Trade History', index=False)
    print(f"Exported backtest results and trade history to {output_path}")


# =============================================================================
# 8. Analysis Function
# =============================================================================


def analyze_results(backtest_df):
    """
    Groups backtest summary results by Strategy_Group (derived from Strategy Name)
    and calculates average cumulative returns and number of trades.
    """
    backtest_df['Strategy_Group'] = backtest_df['Strategy Name'].str[:-4]
    backtest_df['Strategy Cumulative Return (%)'] = pd.to_numeric(backtest_df['Strategy Cumulative Return (%)'], errors='coerce')
    backtest_df['Benchmark Cumulative Return (%)'] = pd.to_numeric(backtest_df['Benchmark Cumulative Return (%)'], errors='coerce')
    backtest_df['Difference'] = pd.to_numeric(backtest_df['Difference'], errors='coerce')
    backtest_df['Number of Trades'] = pd.to_numeric(backtest_df['Number of Trades'], errors='coerce')
    grouped_df = backtest_df.groupby('Strategy_Group').agg(
        Avg_Strategy_Cumulative_Return=('Strategy Cumulative Return (%)', 'mean'),
        Avg_Benchmark_Cumulative_Return=('Benchmark Cumulative Return (%)', 'mean'),
        Avg_Difference=('Difference', 'mean'),
        Avg_Number_of_Trades=('Number of Trades', 'mean')
    ).reset_index()
    grouped_df['Avg_Difference'] = grouped_df['Avg_Difference'].map('{:.3f}'.format)
    return grouped_df


# =============================================================================
# 9. Main Execution Block
# =============================================================================

if __name__ == '__main__':
    # File paths and parameters (update these as needed)
    start_date = '2020-1-31'
    ref_asset = 'SPY'
    data_file = 'C:/Users/PatrickCorcoran/OneDrive - Vest Financial/Documents/Funds/FT/SMA Models/Rachet Algo/MachineLearning_April2025/Excel_Data/data.csv'
    ref_file = 'C:/Users/PatrickCorcoran/OneDrive - Vest Financial/Documents/Funds/FT/SMA Models/Rachet Algo/MachineLearning_April2025/Excel_Data/benchmark_ts.csv'
    roll_dates_file = 'C:/Users/PatrickCorcoran/OneDrive - Vest Financial/Documents/Funds/FT/SMA Models/Rachet Algo/MachineLearning_April2025/Excel_Dataroll_dates.csv'

    df, df_ref_index, end_date = load_and_maintain_data(data_file, ref_file, start_date, ref_asset)

    thresholds = None
    number_funds = 5
    trigger_algorithms = ['rebalance_time_period']  # For now, using a string placeholder.
    selection_algorithms = [remaining_cap_selection]
    rebalance_type = 'FoF'
    apply_stcg = False
    series = 'F'
    rebalance_frequencies = ['quarterly',]
    month_mapping = {'JAN': 'January', 'FEB': 'February', 'MAR': 'March', 'APR': 'April',
                     'MAY': 'May', 'JUN': 'June', 'JUL': 'July', 'AUG': 'August',
                     'SEP': 'September', 'OCT': 'October', 'NOV': 'November', 'DEC': 'December'}
    launch_months = {'JAN': 'January', 'FEB': 'February', 'MAR': 'March', 'APR': 'April',
                     'MAY': 'May', 'JUN': 'June', 'JUL': 'July', 'AUG': 'August',
                     'SEP': 'September', 'OCT': 'October', 'NOV': 'November', 'DEC': 'December'}
    launch_months = {'MAR': 'March',}

    # Call the backtest runner.
    (perf_dict, trade_history_dict, used_frequency_dict,
     cumulative_daily_performance_dict, daily_benchmark_performance_dict,
     ref_asset_index_performance_dict) = run_all_backtests(
         df, df_ref_index, thresholds, trigger_algorithms, selection_algorithms,
         rebalance_type, apply_stcg, start_date, end_date, month_mapping, launch_months,
         roll_dates_file, series, ref_asset, rebalance_frequencies, number_funds
    )
    x = 1

    plot_backtest_results(
        total_cumulative_performance=total_cumulative_performance,
        cumulative_daily_performance_dict=cumulative_daily_performance_dict,
        daily_benchmark_performance_dict=daily_benchmark_performance_dict,
        ref_asset_index_performance_dict=ref_asset_index_performance_dict,
        thresholds=thresholds,
        rebalance_frequencies=rebalance_frequencies,
        launch_months=launch_months,
        ref_index=ref_index
    )

    backtest_df = create_backtest_dataframe(
        trade_history_df=combined_trade_history,
        cumulative_daily_performance_dict=cumulative_daily_performance_dict,
        daily_benchmark_performance_dict=daily_benchmark_performance_dict,
        selection_algorithms=selection_algorithms,
        thresholds=thresholds,
        trigger_algorithms=trigger_algorithms,
        rebalance_frequencies=rebalance_frequencies,
        launch_months=launch_months,
        series=series
    )

    analyzed_results_df = analyze_results(backtest_df)
    print(analyzed_results_df)

    # Compare strategy vs. hold benchmark for each fund and export (FoF extension preparation)
    # WE NEED TO ADD LOGIC IF FOF THEN BUFR BUFZ BUFD ETC
    # OR IF SINGLE FUND THEN
    overall_metrics, rolling_metrics = compare_strategy_vs_hold(
        perf_dict=total_cumulative_performance,
        df_funds=df_funds,
        launch_months=launch_months,
        series=series,
        window=252
    )

    output_file = 'C:/Users/PatrickCorcoran/Documents/Funds/FT/SMA Models/Rachet Algo/MachineLearning_April2025/Excel_Data/backtest_export.xlsx'
    export_backtest_results(
        perf_dict=total_cumulative_performance,
        trade_history_dict=combined_trade_history,
        df_ref_index=df_ref_index,
        selection_algorithms=selection_algorithms,
        trigger_algorithms=trigger_algorithms,
        thresholds=thresholds,
        launch_months=launch_months,
        series=series,
        rebalance_type=rebalance_type,
        used_frequency_dict=total_cumulative_performance,  # Using total_cumulative_performance keys as frequencies
        ref_asset=ref_index,
        output_path=output_file
    )