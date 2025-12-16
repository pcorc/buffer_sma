# Dynamic Backtest Framework with Fund of Funds (FoF) and Traditional Rebalancing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


def initial_maintenance(start_date, ref_asset):
    # Load data
    df = pd.read_csv('/Rachet Algo/UpsideSellRule_February2025/data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    # Adjust start_date to include T-1 business day for daily_return calculation
    start_date = pd.to_datetime(start_date)
    t_minus_1_date = start_date - pd.offsets.BDay(1)
    df = df[df['Date'] >= t_minus_1_date]

    duplicates = df.duplicated(subset=['Date', 'Fund', 'Fund Value (USD)'], keep=False)
    df = df[~duplicates | (df['Fund Return (%)'] != 0)]
    df = df.drop_duplicates()
    df.rename(columns={'Remaining Cap (%)': 'Remaining Cap'}, inplace=True)

    # Calculate daily_return
    df['daily_return'] = df.groupby('Fund')['Fund Value (USD)'].pct_change().fillna(0)

    # Filter out T-1 business day to return the original start date range
    df = df[df['Date'] >= start_date]
    end_date = pd.to_datetime(df['Date'].max())

    # Load reference index data
    df_ref_index = pd.read_csv('/Rachet Algo/UpsideSellRule_February2025/spy_ts.csv')
    df_ref_index['Date'] = pd.to_datetime(df_ref_index['Date'])
    df_ref_index = df_ref_index.sort_values(by='Date')
    # Filter reference index data based on start and end dates
    df_ref_index = df_ref_index[(df_ref_index['Date'] >= t_minus_1_date) & (df_ref_index['Date'] <= end_date)]
    # Filter just to date and ref asset
    df_ref_index = df_ref_index[['Date', ref_asset]].rename(columns={ref_asset: 'Ref_Index'})
    # Calculate daily_return for the reference asset
    df_ref_index['daily_return'] = df_ref_index['Ref_Index'].pct_change().fillna(0)

    return df, df_ref_index, end_date


def forward_fill_and_process(df, roll_dates, end_date):
    """
    This function includes the logic from the 'process_data' and 'forward_fill_data' functions.
    It applies the forward filling process and any other processing required on the DataFrame.
    """
    # Convert 'date' and 'roll_date' to datetime if not already
    df['date'] = pd.to_datetime(df['date'])
    df['roll_date'] = pd.to_datetime(df['roll_date'])

    # Apply the forward filling logic to each group in the DataFrame
    df_grouped = df.groupby(['date', 'roll_date', 'fund'])
    processed_df = df_grouped.apply(lambda group: forward_fill_data(group, roll_dates, end_date))

    # Any additional processing can be done here

    return processed_df


def forward_fill_data(group, start_date, end_date):
    # Forward-fill the data for each fund from start_date to end_date
    # Create a date range from the day after start_date to end_date
    date_range = pd.date_range(start=start_date + pd.DateOffset(days=1), end=end_date, freq='B')

    # Replicate the last available data for each date in date_range
    last_data = group.iloc[-1]  # Get the last row of data in the group
    values = np.tile(last_data.values, (len(date_range), 1))  # Replicate the last row for each date in date_range

    # Create a new DataFrame with forward-filled data
    new_data = pd.DataFrame(values, index=date_range, columns=group.columns)
    new_data.reset_index(inplace=True)  # Reset index to move date from index to column
    new_data.rename(columns={'index': 'daily_date'}, inplace=True)  # Rename 'index' column to 'daily_date'

    return new_data


def calculate_roll_dates(universe_df, start_date, end_date, frequency, launch_month=None, month_mapping=None):
    # Ensure 'Date' column is in datetime format and drop duplicates
    universe_df.loc[:, 'Date'] = pd.to_datetime(universe_df['Date'])

    unique_dates = universe_df['Date'].drop_duplicates().sort_values()

    # Adjust the start_date to match the specified launch month/year
    if launch_month:
        start_year = pd.to_datetime(start_date).year
        # Set the start date to the first available business date for the launch month
        launch_dates = unique_dates[
            (unique_dates.dt.year == start_year) &
            (unique_dates.dt.month == list(month_mapping.keys()).index(launch_month) + 1)
            ]
        if not launch_dates.empty:
            start_date = launch_dates.iloc[-1]
        else:
            raise ValueError(f"No available dates found for the launch month: {launch_month}")

    # Filter the unique dates within the start_date and end_date range
    filtered_dates = unique_dates[
        (unique_dates >= pd.to_datetime(start_date)) &
        (unique_dates <= pd.to_datetime(end_date))
        ]

    # Initialize roll dates list
    roll_dates = []

    # Calculate the roll dates based on the frequency, adjusting cadence from start_date
    if frequency == 'monthly':
        roll_dates = filtered_dates.groupby(filtered_dates.dt.to_period('M')).max().tolist()
    elif frequency == 'quarterly':
        current_date = start_date
        while current_date <= end_date:
            last_business_day = filtered_dates[
                (filtered_dates.dt.year == current_date.year) &
                (filtered_dates.dt.month == current_date.month)
                ].max()
            if pd.notna(last_business_day):
                roll_dates.append(last_business_day)
            current_date += pd.DateOffset(months=3)
    elif frequency == 'semi-annual':
        current_date = start_date
        while current_date <= end_date:
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

    return roll_dates, start_date


# Trigger algorithms
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


# Selection algorithms
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


def generate_benchmarks(df, rebalance_type='FoF', end_date=None, start_date=None, roll_dates=None, rebalance_frequency=None, series=None):
    benchmarks = {}
    daily_benchmark_performance = pd.DataFrame()
    benchmark_definitions = []  # To capture benchmark definitions at each rebalance date

    df = df.sort_values(by='Date')
    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]

    if df.empty:
        raise ValueError("No data available after the specified start date.")

    # Filter funds by series
    if series:
        df = df[df['Fund'].str.startswith(series)]

    roll_dates = sorted(roll_dates)

    # Sort the group by date
    df = df.sort_values(by='Date')

    # Initialize benchmark NAV at 100 and calculate cumulative NAV
    daily_benchmark_performance_data = []
    benchmark_nav = 100
    initialized = False

    for i in range(len(roll_dates) - 1):
        start_date = roll_dates[i]
        next_rebalance_date = roll_dates[i + 1]

        rebalance_period_data = df[(df['Date'] >= start_date) & (df['Date'] < next_rebalance_date)]
        if rebalance_period_data.empty:
            continue

        # Rebalance to equal weights at the start of the period
        num_funds = rebalance_period_data['Fund'].nunique()
        rebalance_period_data.loc[:, 'weight'] = 1 / num_funds

        # Store the benchmark definition for the current rebalance date
        benchmark_definitions.append({
            'Rebalance_Date': start_date,
            'Rebalance_Frequency': rebalance_frequency,
            'Funds': list(rebalance_period_data['Fund'].unique()),
            'Weights': list(rebalance_period_data.groupby('Fund')['weight'].first())
        })

        # Track individual NAVs for each fund and initialize them at equal levels
        fund_navs = {fund: 100 for fund in rebalance_period_data['Fund'].unique()}

        # Calculate daily returns and NAVs over the rebalance period with weight drift
        dates = rebalance_period_data['Date'].unique()
        for idx, current_date in enumerate(dates):
            daily_data = rebalance_period_data[rebalance_period_data['Date'] == current_date]
            daily_return = 0

            # Update each fund's NAV based on its daily return and accumulate the benchmark NAV
            for fund in daily_data['Fund']:
                fund_data = daily_data[daily_data['Fund'] == fund]

                if not fund_data.empty:  # Check if the DataFrame is not empty
                    fund_return = fund_data['daily_return'].iloc[0]
                    fund_navs[fund] *= (1 + fund_return)

            # Calculate total NAV and drifted weights
            total_nav = sum(fund_navs.values())
            fund_weights = {fund: nav / total_nav for fund, nav in fund_navs.items()}

            # Calculate benchmark daily return based on drifted weights
            for fund, weight in fund_weights.items():
                fund_data = daily_data[daily_data['Fund'] == fund]

                if not fund_data.empty:  # Ensure we have data to proceed
                    daily_return += weight * fund_data['daily_return'].iloc[0]

            # Update benchmark NAV
            if idx == 0 and not initialized:
                benchmark_nav = 100
                initialized = True
            else:
                benchmark_nav *= (1 + daily_return)

            daily_benchmark_performance_data.append({'Date': current_date, 'Benchmark_NAV': benchmark_nav, 'Rebalance_Frequency': rebalance_frequency})

    # Append daily benchmark performance to DataFrame
    daily_benchmark_performance = pd.concat([daily_benchmark_performance, pd.DataFrame(daily_benchmark_performance_data)], ignore_index=True)

    # Export benchmark definitions to an Excel file
    benchmark_definitions_df = pd.DataFrame(benchmark_definitions)
    benchmark_definitions_df.to_excel("benchmark_definitions.xlsx", index=False)

    return benchmarks, daily_benchmark_performance


def general_rebalancing(threshold, rebalance_frequency, selection_algorithms, trigger_algorithms, rebalance_type, apply_stcg, data, roll_dates, start_date, end_date, series):
    # Filter funds based on series
    universe_df = data[data['Fund'].str.startswith(series)].copy()

    # Sort the data and calculate daily returns
    universe_df = universe_df.sort_values(by='Date')

    cumulative_performance = {
        'FoF': {'dates': [], 'standardized_nav': [100]},
        'Traditional': {'dates': [], 'standardized_nav': [100]}
    }

    trade_history = []
    cumulative_daily_performance = []
    daily_nav_series = pd.Series([100], index=[start_date])  # Initialize with NAV of 100 at start date

    # Initialize total portfolio value at the start
    total_portfolio_value = 100

    for i in range(len(roll_dates) - 1):
        rebalance_date = roll_dates[i]
        next_rebalance_date = roll_dates[i + 1]

        # Get current day data to rebalance the portfolio
        current_day_data = universe_df[universe_df['Date'] == rebalance_date]

        if not current_day_data.empty:
            # Fund of Funds (FoF) Rebalance Logic
            if rebalance_type == "FoF":
                # Sort by Remaining Cap and exclude bottom percentage of funds based on the threshold
                current_day_data = current_day_data.sort_values(by='Remaining Cap', ascending=False)
                num_exclude = int(len(current_day_data) * threshold)
                selected_funds = current_day_data if num_exclude == 0 else current_day_data.iloc[:-num_exclude]

                # Clear previous portfolio values (wipe clean the old definition)
                portfolio_values = {}

                # Assign new equal weights for selected funds at rebalance
                num_selected_funds = len(selected_funds)
                equal_weight = 1 / num_selected_funds

                # Initialize new fund values for the rebalance period
                for fund in selected_funds['Fund']:
                    portfolio_values[fund] = total_portfolio_value * equal_weight

                # Iterate through each day between roll periods and allow weights to drift
                period_data = universe_df[(universe_df['Date'] > rebalance_date) &
                                          (universe_df['Date'] <= next_rebalance_date) &
                                          (universe_df['Fund'].isin(selected_funds['Fund']))]

                for date in period_data['Date'].unique():
                    daily_data = period_data[period_data['Date'] == date]

                    # Update portfolio value of each fund based on its daily return
                    for _, row in daily_data.iterrows():
                        fund = row['Fund']
                        if fund not in portfolio_values:
                            # Initialize the fund value with equal weight of the total portfolio value
                            portfolio_values[fund] = total_portfolio_value * equal_weight
                        else:
                            # Adjust the fund value based on its daily return
                            portfolio_values[fund] *= (1 + row['daily_return'])

                    # Calculate total portfolio value after drifting weights
                    total_portfolio_value = sum(portfolio_values.values())

                    # Store updated NAV in daily_nav_series
                    daily_nav_series.loc[date] = total_portfolio_value

                # Update cumulative performance for the rebalance date
                updated_nav = daily_nav_series.loc[next_rebalance_date] if next_rebalance_date in daily_nav_series.index else daily_nav_series.iloc[-1]
                cumulative_performance['FoF']['dates'].append(rebalance_date)
                cumulative_performance['FoF']['standardized_nav'].append(updated_nav)

                # Track trade history for FoF
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

    # Convert the daily NAV series to a DataFrame for consistent return
    cumulative_daily_performance_df = daily_nav_series.reset_index().rename(columns={'index': 'Date', 0: 'NAV'})
    trade_history_df = pd.DataFrame(trade_history)

    return cumulative_performance, trade_history_df, cumulative_daily_performance_df


def run_backtest(df, df_ref_index, thresholds, trigger_algorithms, selection_algorithms, rebalance_type='FoF', apply_stcg=False, rebalance_frequencies=None,
                 start_date=None, end_date=None, month_mapping=None, series=None, launch_months=None):
    total_cumulative_performance = {}
    combined_trade_history = pd.DataFrame()
    combined_daily_performance = pd.DataFrame()
    daily_benchmark_performance = pd.DataFrame()
    ref_asset_index_performance_dict = {}

    # Sort the data by date to ensure we start from the earliest date
    df = df.sort_values(by='Date')

    # Loop over each month specified in the launch_months
    for month_abbr, month_name in launch_months.items():
        print(f"Starting backtest for start month: {month_name}")

        # Filter data to start from the specified month and year
        start_year = pd.to_datetime(start_date).year
        df_filtered = df[
            (df['Date'].dt.year > start_year) |
            ((df['Date'].dt.year == start_year) & (df['Date'].dt.month >= list(month_mapping.keys()).index(month_abbr) + 1))
            ]

        for frequency in rebalance_frequencies:
            # Generate roll dates for each frequency
            if 'rebalance_time_period' in trigger_algorithms:
                roll_dates, adjusted_start_date = calculate_roll_dates(
                    df_filtered, start_date=start_date, end_date=end_date, frequency=frequency,
                    launch_month=month_abbr, month_mapping=month_mapping
                )

                # Use adjusted_start_date for the following processes
                start_date = adjusted_start_date

            # Calculate standardized cumulative performance starting at 100 for Ref_Index_NAV
            df_ref_index_filtered = df_ref_index[df_ref_index['Date'] >= start_date].copy()
            df_ref_index_filtered['Ref_Index_NAV'] = 100
            
            for i in range(1, len(df_ref_index_filtered)):
                df_ref_index_filtered.iloc[i, df_ref_index_filtered.columns.get_loc('Ref_Index_NAV')] = \
                    df_ref_index_filtered.iloc[i - 1, df_ref_index_filtered.columns.get_loc('Ref_Index_NAV')] * (
                            1 + df_ref_index_filtered.iloc[i, df_ref_index_filtered.columns.get_loc('daily_return')])
            
            # Drop unnecessary columns and keep only Date and Ref_Index_NAV
            df_ref_index_filtered = df_ref_index_filtered[['Date', 'Ref_Index_NAV']]
            
            # Store the reference asset index performance in the dictionary using the month_abbr
            ref_asset_key = f"Reference_Asset_{month_abbr}"
            ref_asset_index_performance_dict[ref_asset_key] = df_ref_index_filtered


            # Generate the benchmarks and get daily benchmark performance
            benchmarks, daily_benchmark_performance_for_frequency = generate_benchmarks(
                df=df_filtered,
                series=series,  # Pass the series parameter
                rebalance_type=rebalance_type,
                end_date=end_date,
                start_date=start_date,
                roll_dates=roll_dates,
                rebalance_frequency=frequency
            )

            # Add the rebalance frequency and launch month to the daily benchmark performance
            daily_benchmark_performance_for_frequency['Rebalance_Frequency'] = frequency
            daily_benchmark_performance_for_frequency['Launch_Month'] = month_abbr
            daily_benchmark_performance = pd.concat([daily_benchmark_performance, daily_benchmark_performance_for_frequency], ignore_index=True)

            for threshold in thresholds:
                print(f"Starting backtest for frequency {frequency} with threshold {threshold}")

                # Execute general rebalancing
                dbb_performance, dbb_trade_history, dbb_daily_performance = general_rebalancing(
                    threshold=threshold,
                    rebalance_frequency=frequency,
                    selection_algorithms=selection_algorithms,
                    trigger_algorithms=trigger_algorithms,
                    rebalance_type=rebalance_type,
                    apply_stcg=apply_stcg,
                    data=df_filtered,
                    roll_dates=roll_dates,
                    start_date=start_date,
                    end_date=end_date,
                    series=series  # Pass the series parameter
                )

                # Add launch month to trade history
                dbb_trade_history['Launch_Month'] = month_name
                dbb_trade_history['Series'] = series

                # Store results for cumulative performance
                dbb_daily_performance['Threshold'] = threshold
                dbb_daily_performance['Rebalance_Frequency'] = frequency
                dbb_daily_performance['Launch_Month'] = month_abbr  # Add launch month to the results
                total_cumulative_performance[f"{threshold}_{frequency}_{month_abbr}"] = dbb_performance
                combined_trade_history = pd.concat([combined_trade_history, dbb_trade_history], ignore_index=True)
                combined_daily_performance = pd.concat([combined_daily_performance, dbb_daily_performance], ignore_index=True)

    # Create dictionaries for performance based on threshold, frequency, and launch month combinations
    cumulative_daily_performance_dict = {}
    daily_benchmark_performance_dict = {}

    for threshold in thresholds:
        for frequency in rebalance_frequencies:
            for month_abbr in launch_months.keys():
                key = f"Threshold_{threshold}_Frequency_{frequency}_LaunchMonth_{month_abbr}"
                cumulative_daily_performance_dict[key] = combined_daily_performance[
                    (combined_daily_performance['Threshold'] == threshold) &
                    (combined_daily_performance['Rebalance_Frequency'] == frequency) &
                    (combined_daily_performance['Launch_Month'] == month_abbr)
                    ].drop(columns=['Threshold', 'Rebalance_Frequency', 'Launch_Month'])

                daily_benchmark_performance_dict[key] = daily_benchmark_performance[
                    (daily_benchmark_performance['Rebalance_Frequency'] == frequency) &
                    (daily_benchmark_performance['Launch_Month'] == month_abbr)
                    ].drop(columns=['Rebalance_Frequency', 'Launch_Month'])

    return total_cumulative_performance, combined_trade_history, cumulative_daily_performance_dict, daily_benchmark_performance_dict, ref_asset_index_performance_dict




def plot_backtest_results(total_cumulative_performance, cumulative_daily_performance_dict, daily_benchmark_performance_dict, ref_asset_index_performance_dict, thresholds,
                          rebalance_frequencies, launch_months, ref_index):
    for threshold in thresholds:
        threshold_str = f"{int(threshold * 100)}%"
        for frequency in rebalance_frequencies:
            for month_abbr, month_name in launch_months.items():
                key = f"Threshold_{threshold}_Frequency_{frequency}_LaunchMonth_{month_abbr}"
                cumulative_daily_performance = cumulative_daily_performance_dict.get(key)
                daily_benchmark_performance = daily_benchmark_performance_dict.get(key)
                
                # Use month_abbr to access the correct reference asset performance
                ref_asset_key = f"Reference_Asset_{month_abbr}"
                ref_asset_performance = ref_asset_index_performance_dict.get(ref_asset_key)

                if cumulative_daily_performance is None or daily_benchmark_performance is None or ref_asset_performance is None:
                    print(f"Warning: No data available for threshold {threshold_str}, frequency {frequency}, and launch month {month_name}")
                    continue

                # Sort all datasets by date to ensure consistency
                cumulative_daily_performance_sorted = cumulative_daily_performance.sort_values(by='Date')
                daily_benchmark_performance_sorted = daily_benchmark_performance.sort_values(by='Date')
                ref_asset_performance_sorted = ref_asset_performance.sort_values(by='Date')

                # Align the dates of all datasets to ensure consistent comparison
                common_dates = cumulative_daily_performance_sorted['Date'].isin(daily_benchmark_performance_sorted['Date']) & \
                               cumulative_daily_performance_sorted['Date'].isin(ref_asset_performance_sorted['Date'])
                aligned_strategy_performance = cumulative_daily_performance_sorted[common_dates]
                aligned_benchmark_performance = daily_benchmark_performance_sorted[
                    daily_benchmark_performance_sorted['Date'].isin(aligned_strategy_performance['Date'])]
                aligned_ref_asset_performance = ref_asset_performance_sorted[
                    ref_asset_performance_sorted['Date'].isin(aligned_strategy_performance['Date'])]

                # Ensure there is data to plot after aligning dates
                if aligned_strategy_performance.empty or aligned_benchmark_performance.empty or aligned_ref_asset_performance.empty:
                    print(f"Warning: No common dates available for threshold {threshold_str}, frequency {frequency}, and launch month {month_name}")
                    continue

                # Plot daily FoF performance if available
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(aligned_strategy_performance['Date'], aligned_strategy_performance['NAV'],
                        label=f"Strategy_FoF (Threshold {threshold_str}, {frequency}, Launch Month {month_name})", linestyle='-', linewidth=1.5, color='blue')

                # Plot Benchmark daily performance if available
                ax.plot(aligned_benchmark_performance['Date'], aligned_benchmark_performance['Benchmark_NAV'],
                        label=f"Benchmark (Threshold {threshold_str}, {frequency}, Launch Month {month_name})", linestyle='-', linewidth=2, color='red')

                # Plot Reference Asset performance if available
                ax.plot(aligned_ref_asset_performance['Date'], aligned_ref_asset_performance['Ref_Index_NAV'],
                        label=f"{ref_index} (Reference Asset)", linestyle='--', linewidth=2, color='green')

                # Set plot title and labels
                ax.set_title(f"Benchmark, Strategy, and Reference Asset Comparison - Threshold: {threshold_str}, Frequency: {frequency}, Launch Month: {month_name}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Standardized NAV")

                # Add legend
                ax.legend()

                # Show the % return of the strategy, benchmark, and reference asset in a text box on the graph
                text_box_y_offset = 0.85  # Initial vertical offset for text boxes
                for label, data in [
                    ("Strategy_FoF", aligned_strategy_performance),
                    ("Benchmark", aligned_benchmark_performance),
                    (ref_index, aligned_ref_asset_performance)
                ]:
                    if label == "Strategy_FoF" and not aligned_strategy_performance.empty:
                        final_nav = aligned_strategy_performance['NAV'].iloc[-1]
                        initial_nav = aligned_strategy_performance['NAV'].iloc[0]
                    elif label == "Benchmark" and not aligned_benchmark_performance.empty:
                        final_nav = aligned_benchmark_performance['Benchmark_NAV'].iloc[-1]
                        initial_nav = aligned_benchmark_performance['Benchmark_NAV'].iloc[0]
                    elif label == ref_index and not aligned_ref_asset_performance.empty:
                        final_nav = aligned_ref_asset_performance['Ref_Index_NAV'].iloc[-1]
                        initial_nav = aligned_ref_asset_performance['Ref_Index_NAV'].iloc[0]
                    else:
                        continue

                    pct_return = (final_nav / initial_nav - 1) * 100
                    textstr = f"{label}: {pct_return:.2f}% return"
                    ax.text(0.02, text_box_y_offset, textstr, transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
                    text_box_y_offset -= 0.05  # Decrement the offset for next text box

                # Print first date and first Y-axis value for each data series
                # print(f"First Date for Strategy_FoF (Threshold {threshold_str}, {frequency}, Launch Month {month_name}): {aligned_strategy_performance['Date'].iloc[2]}")
                # print(f"First Y-axis value for Strategy_FoF (Threshold {threshold_str}, {frequency}, Launch Month {month_name}): {aligned_strategy_performance['NAV'].iloc[2]}")
                
                # print(f"First Date for Benchmark (Threshold {threshold_str}, {frequency}, Launch Month {month_name}): {aligned_benchmark_performance['Date'].iloc[2]}")
                # print(f"First Y-axis value for Benchmark (Threshold {threshold_str}, {frequency}, Launch Month {month_name}): {aligned_benchmark_performance['Benchmark_NAV'].iloc[2]}")
                
                # print(f"First Date for Reference Asset ({ref_index}, Launch Month {month_name}): {aligned_ref_asset_performance['Date'].iloc[2]}")
                # print(f"First Y-axis value for Reference Asset ({ref_index}, Launch Month {month_name}): {aligned_ref_asset_performance['Ref_Index_NAV'].iloc[2]}")

                plt.show()



def create_backtest_dataframe(trade_history_df, cumulative_daily_performance_dict, daily_benchmark_performance_dict, selection_algorithms, thresholds, trigger_algorithms,
                              rebalance_frequencies, launch_months, series):
    records = []

    # Iterate over all combinations of thresholds, trigger_algorithms, frequencies, launch_months, and selection_algorithms
    for threshold in thresholds:
        for trigger_algorithm in trigger_algorithms:
            for frequency in rebalance_frequencies:
                for month_abbr, month_name in launch_months.items():
                    # Determine strategy name based on trigger_algorithm type, series, and launch month
                    if isinstance(trigger_algorithm, str):
                        strategy_name = f'{trigger_algorithm}_{threshold}_{frequency}_{series}_{month_abbr}'
                    else:
                        strategy_name = f'{trigger_algorithm.__name__}_{threshold}_{frequency}_{series}_{month_abbr}'

                    for selection_algorithm in selection_algorithms:
                        # Construct dictionary key for current threshold, frequency, and launch month
                        key = f"Threshold_{threshold}_Frequency_{frequency}_LaunchMonth_{month_abbr}"

                        # Initialize cumulative return variables
                        strategy_cumulative_return = None
                        benchmark_cumulative_return = None

                        # Retrieve cumulative performance for the strategy
                        cumulative_performance = cumulative_daily_performance_dict.get(key)
                        if cumulative_performance is not None and not cumulative_performance.empty:
                            final_nav = cumulative_performance['NAV'].iloc[-1]
                            initial_nav = cumulative_performance['NAV'].iloc[0]
                            strategy_cumulative_return = (final_nav / initial_nav - 1) * 100  # In percentage

                        # Retrieve cumulative performance for the benchmark
                        benchmark_performance = daily_benchmark_performance_dict.get(key)
                        if benchmark_performance is not None and not benchmark_performance.empty:
                            final_nav = benchmark_performance['Benchmark_NAV'].iloc[-1]
                            initial_nav = benchmark_performance['Benchmark_NAV'].iloc[0]
                            benchmark_cumulative_return = (final_nav / initial_nav - 1) * 100  # In percentage

                        # Number of trades for this combination
                        num_trades = trade_history_df[
                            (trade_history_df['Threshold'] == threshold) &
                            (trade_history_df['Rebalance_Frequency'] == frequency) &
                            (trade_history_df['Launch_Month'] == month_name) &
                            (trade_history_df['Series'] == series) &
                            (trade_history_df['rebalance_type'] == 'FoF')
                            ].shape[0]

                        # Create a record for the backtest DataFrame
                        record = {
                            'Strategy Name': strategy_name,
                            'Selection Algorithm': selection_algorithm.__name__,
                            'Threshold': threshold,
                            'Rebalance Frequency': frequency,
                            'Series': series,
                            'Launch Month': month_name,
                            'Trigger Algorithm': trigger_algorithm if isinstance(trigger_algorithm, str) else trigger_algorithm.__name__,
                            'Strategy Cumulative Return (%)': strategy_cumulative_return,
                            'Benchmark Cumulative Return (%)': benchmark_cumulative_return,
                            'Difference': strategy_cumulative_return - benchmark_cumulative_return if strategy_cumulative_return is not None and benchmark_cumulative_return is not None else None,
                            'Number of Trades': num_trades
                        }
                        records.append(record)

    # Create the backtest DataFrame
    backtest_df = pd.DataFrame(records)

    # Filter out rows with zero cumulative return and remove duplicates
    backtest_df = (
        backtest_df.dropna(subset=['Strategy Cumulative Return (%)'])
        .drop_duplicates()
        .sort_values(by=['Threshold', 'Difference'], ascending=[True, False])
    )

    # Format the 'Difference' column to 3 decimal places
    backtest_df['Difference'] = backtest_df['Difference'].map('{:.3f}'.format)

    return backtest_df


##############################################################################################################################

# Generate benchmarks
start_date = '2020-1-31'  # Example start date
ref_index = 'SPY'
df, df_ref_index, end_date = initial_maintenance(start_date, ref_index)

# Define parameters for backtesting

# Define parameters for backtesting
thresholds = [0.25,]  #  0.35, 0.60
trigger_algorithms = ['rebalance_time_period']  # Rebalance based on time period
selection_algorithms = [remaining_cap_selection]  # Example selection algorithm
rebalance_type = 'FoF'
apply_stcg = True
selected_months = None  # Not used for FoF
rebalance_frequencies = ['monthly',]  #   'quarterly', 'semi-annual'
series = 'F'

month_mapping = {
    'JAN': 'January', 'FEB': 'February', 'MAR': 'March', 'APR': 'April',
    'MAY': 'May', 'JUN': 'June', 'JUL': 'July', 'AUG': 'August',
    'SEP': 'September', 'OCT': 'October', 'NOV': 'November', 'DEC': 'December'
}

launch_months = {
    'JAN': 'January', 'FEB': 'February', 'MAR': 'March', 'APR': 'April', 'MAY': 'May',
    'JUN': 'June', 'JUL': 'July', 'AUG': 'August',
    'SEP': 'September', 'OCT': 'October', 'NOV': 'November', 'DEC': 'December'
}

launch_months = {'DEC': 'December'}

total_cumulative_performance, combined_trade_history, cumulative_daily_performance_dict, daily_benchmark_performance_dict, ref_asset_index_performance_dict = run_backtest(
    df=df,
    df_ref_index=df_ref_index,
    thresholds=thresholds,
    trigger_algorithms=trigger_algorithms,
    selection_algorithms=selection_algorithms,
    rebalance_type='FoF',  # Fund of Funds approach
    apply_stcg=False,  # Do not apply short-term capital gains
    rebalance_frequencies=rebalance_frequencies,
    start_date=start_date,
    end_date=end_date,
    month_mapping=month_mapping,
    series=series,
    launch_months=launch_months
)
x=1


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


# %%


def analyze_results(backtest_df):
    # Extract selection/threshold/rebalance/series pairing from Strategy Name by removing the last 4 characters (i.e., "_JAN", "_FEB", etc.)
    backtest_df['Strategy_Group'] = backtest_df['Strategy Name'].str[:-4]

    # Convert columns to numeric to ensure proper aggregation
    backtest_df['Strategy Cumulative Return (%)'] = pd.to_numeric(backtest_df['Strategy Cumulative Return (%)'], errors='coerce')
    backtest_df['Benchmark Cumulative Return (%)'] = pd.to_numeric(backtest_df['Benchmark Cumulative Return (%)'], errors='coerce')
    backtest_df['Difference'] = pd.to_numeric(backtest_df['Difference'], errors='coerce')
    backtest_df['Number of Trades'] = pd.to_numeric(backtest_df['Number of Trades'], errors='coerce')

    # Group by the Strategy_Group column and calculate the average of strategy and benchmark cumulative returns
    grouped_df = backtest_df.groupby('Strategy_Group').agg(
        Avg_Strategy_Cumulative_Return=('Strategy Cumulative Return (%)', 'mean'),
        Avg_Benchmark_Cumulative_Return=('Benchmark Cumulative Return (%)', 'mean'),
        Avg_Difference=('Difference', 'mean'),
        Avg_Number_of_Trades=('Number of Trades', 'mean')
    ).reset_index()

    # Format the 'Difference' column to 3 decimal places
    grouped_df['Avg_Difference'] = grouped_df['Avg_Difference'].map('{:.3f}'.format)

    return grouped_df


# Example usage
analyzed_results_df = analyze_results(backtest_df)
print(analyzed_results_df)
