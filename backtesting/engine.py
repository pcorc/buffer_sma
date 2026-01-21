"""
Core backtesting engine for single ticker strategy.
"""

import pandas as pd
import numpy as np
from core.triggers import get_trigger_function
from core.selections import get_selection_function


def run_single_ticker_backtest(df_enriched, df_benchmarks, launch_month,
                               trigger_config, selection_func, roll_dates_dict, series='F'):
    """
    Core backtest engine for single ticker strategy.

    Simulates a strategy that:
    1. Starts with a specific launch month fund (e.g., FMAR)
    2. Evaluates trigger conditions daily or on roll dates
    3. When triggered, uses selection algorithm to pick new fund
    4. Tracks performance vs 3 benchmarks (SPY, BUFR, buy-and-hold)

    Parameters:
      df_enriched: Preprocessed fund data with derived metrics
      df_benchmarks: DataFrame with Date, SPY, BUFR columns (with daily returns)
      launch_month: Launch month abbreviation (e.g., 'MAR')
      trigger_config: Dict with 'type' and 'params' keys
      selection_func: Function reference for fund selection
      roll_dates_dict: Dict of roll dates lists by frequency
      series: Fund series letter (default 'F')

    Returns:
      Dict with comprehensive backtest results including:
        - Performance metrics (returns, Sharpe, volatility, max DD)
        - Benchmark comparisons (vs SPY, BUFR, buy-and-hold)
        - Daily NAV series
        - Trade history
    """
    print(f"\n{'=' * 80}")
    print(f"Running backtest: {launch_month} | {trigger_config['type']} | {selection_func.__name__}")
    print(f"{'=' * 80}")

    # Initialize
    current_fund = series + launch_month
    fund_data = df_enriched[df_enriched['Fund'] == current_fund].copy()

    if fund_data.empty:
        print(f"ERROR: No data for fund {current_fund}")
        return None

    BUFR_INCEPTION = pd.Timestamp('2020-07-01')

    # Get the first roll date for this fund (already filtered to >= July 2020 in preprocessing)
    fund_roll_dates = fund_data['Roll_Date'].dropna().unique()

    if len(fund_roll_dates) == 0:
        print(f"ERROR: No valid roll dates for fund {current_fund}")
        return None

    # Start at the first roll date (this is already >= July 2020 from preprocessing)
    first_roll_date = pd.Timestamp(sorted(fund_roll_dates)[0])

    # Ensure we also have benchmark data from this date
    start_date = max(first_roll_date, BUFR_INCEPTION)

    # Filter fund data to start from this aligned date
    fund_data = fund_data[fund_data['Date'] >= start_date].copy()

    if fund_data.empty:
        print(f"ERROR: No fund data after alignment date {start_date.date()}")
        return None

    end_date = fund_data['Date'].max()

    # Prepare benchmark data
    df_bench = df_benchmarks.copy()
    df_bench['Date'] = pd.to_datetime(df_bench['Date'])
    df_bench = df_bench[(df_bench['Date'] >= start_date) & (df_bench['Date'] <= end_date)].copy()
    df_bench = df_bench.sort_values('Date').reset_index(drop=True)

    # Initialize tracking - all start at 100
    strategy_nav = 100.0
    spy_nav = 100.0
    bufr_nav = 100.0
    hold_nav = 100.0

    # Track if this is the first day (to handle initialization)
    first_day = True

    daily_performance = []
    trade_history = []
    num_trades = 0

    # Get business days for iteration
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # Prepare trigger-specific data
    trigger_type = trigger_config['type']
    trigger_params = trigger_config['params']
    trigger_func = get_trigger_function(trigger_type)

    if trigger_type == 'rebalance_time_period':
        from utils.date_utils import get_rebalance_trading_dates

        frequency_map = {
            'monthly': 'M',
            'quarterly': 'Q',
            'semi_annual': 'S',
            'annual': 'A'
        }

        frequency = trigger_params['frequency']
        frequency_code = frequency_map.get(frequency, frequency)

        # Get trading dates (T+1) for rebalancing
        trading_date_pairs = get_rebalance_trading_dates(
            rebalance_frequency=frequency_code,
            roll_dates_dict=roll_dates_dict,
            df_dates=df_enriched['Date'].unique(),
            start_date=start_date,
            end_date=end_date
        )

        # Use trading dates (not roll dates) for actual execution
        trading_dates_list = [trading_date for roll_date, trading_date in trading_date_pairs]

    # Main backtest loop
    for current_date in all_dates:
        # Get current fund data
        current_fund_data = df_enriched[
            (df_enriched['Fund'] == current_fund) &
            (df_enriched['Date'] == current_date)
            ]

        if current_fund_data.empty:
            # No data for current fund on this date, skip
            continue

        current_fund_row = current_fund_data.iloc[0]
        daily_return = current_fund_row['daily_return']

        # On first day, just initialize NAVs at 100 without applying returns
        if first_day:
            # All NAVs stay at 100 on first day
            first_day = False
        else:
            # Update strategy NAV with daily return
            strategy_nav *= (1 + daily_return)

            # Update benchmark NAVs
            bench_data = df_bench[df_bench['Date'] == current_date]
            if not bench_data.empty:
                spy_nav *= (1 + bench_data.iloc[0]['SPY_daily_return'])
                bufr_nav *= (1 + bench_data.iloc[0]['BUFR_daily_return'])

            # Update buy-and-hold NAV (original launch month fund)
            hold_fund = series + launch_month
            hold_data = df_enriched[
                (df_enriched['Fund'] == hold_fund) &
                (df_enriched['Date'] == current_date)
                ]
            if not hold_data.empty:
                hold_nav *= (1 + hold_data.iloc[0]['daily_return'])

        # Store daily performance with comprehensive roll date metrics

        # Current values
        current_fund_nav = current_fund_row.get('Fund Value (USD)', None)
        current_ref_index = current_fund_row.get('Reference Asset Value (USD)', None)
        current_remaining_cap_pct = current_fund_row.get('Remaining Cap', None)
        current_downside_before_buffer = current_fund_row.get('Downside Before Buffer (%)', None)

        # Roll date reference values (constant throughout period)
        roll_date = current_fund_row.get('Roll_Date', None)
        starting_fund_nav = current_fund_row.get('Starting_Fund_Value', None)
        starting_ref_index = current_fund_row.get('Starting_Ref_Asset_Value', None)
        original_cap = current_fund_row.get('Original_Cap', None)
        original_buffer = current_fund_row.get('Original_Buffer', None)

        if current_date.strftime('%Y-%m-%d') == '2024-09-20':
            x=1


        # Calculate returns from roll date
        fund_return_from_roll = None
        ref_index_return_from_roll = None

        if starting_fund_nav and current_fund_nav:
            fund_return_from_roll = (current_fund_nav - starting_fund_nav) / starting_fund_nav

        if starting_ref_index and current_ref_index:
            ref_index_return_from_roll = (current_ref_index - starting_ref_index) / starting_ref_index

        daily_performance.append({
            # Date and identification
            'Date': current_date,
            'Current_Fund': current_fund,
            'Outcome_Period_ID': current_fund_row.get('Outcome_Period_ID', None),
            'Roll_Date': roll_date,

            # Strategy and benchmark NAVs (cumulative performance)
            'Strategy_NAV': strategy_nav,
            'SPY_NAV': spy_nav,
            'BUFR_NAV': bufr_nav,
            'Hold_NAV': hold_nav,

            # Roll date reference values (constant within period)
            'Starting_Fund_NAV': starting_fund_nav,
            'Starting_Ref_Index': starting_ref_index,
            'Original_Cap': original_cap,
            'Original_Buffer': original_buffer,
            'Total_Outcome_Days': current_fund_row.get('Total_Outcome_Days', None),

            # Current day values
            'Current_Fund_NAV': current_fund_nav,
            'Current_Ref_Index': current_ref_index,
            'Current_Remaining_Cap_Pct': current_remaining_cap_pct,
            'Remaining_Outcome_Days': current_fund_row.get('Remaining Outcome Days', None),

            # Calculated returns from roll date
            'Fund_Return_From_Roll': fund_return_from_roll,
            'Ref_Index_Return_From_Roll': ref_index_return_from_roll,

            # Cap metrics
            'Cap_Utilization': current_fund_row.get('Cap_Utilization', None),
            'Cap_Remaining_Pct': current_fund_row.get('Cap_Remaining_Pct', None),

            # Buffer metrics
            'Downside_Before_Buffer_Pct': current_downside_before_buffer,
            'Starting_Downside_Before_Buffer': current_fund_row.get('Starting_Downside_Before_Buffer', None)
        })

        # Evaluate trigger
        triggered = False
        trigger_reason = None

        if trigger_type == 'rebalance_time_period':
            triggered = trigger_func(current_date, trading_dates_list)
            if triggered:
                trigger_reason = f"{frequency}_rebalance"
        else:
            # Threshold-based trigger
            threshold = trigger_params['threshold']
            triggered = trigger_func(current_fund_row, threshold)
            if triggered:
                trigger_reason = f"{trigger_type}={threshold}"

        # If triggered, select new fund
        if triggered:
            # Get universe of available funds on current date
            df_universe = df_enriched[
                (df_enriched['Date'] == current_date) &
                (df_enriched['Fund'].str.startswith(series))
                ].copy()

            if not df_universe.empty:
                new_fund = selection_func(df_universe, current_date, series)

                if new_fund and new_fund != current_fund:
                    # Log trade
                    trade_history.append({
                        'Date': current_date,
                        'From_Fund': current_fund,
                        'To_Fund': new_fund,
                        'Trigger_Reason': trigger_reason,
                        'NAV_at_Switch': strategy_nav
                    })

                    current_fund = new_fund
                    num_trades += 1
                    #print(f"  Trade {num_trades}: {current_date.date()} | {new_fund} | {trigger_reason}")

    # Calculate performance metrics
    df_perf = pd.DataFrame(daily_performance)

    if len(df_perf) < 2:
        print("ERROR: Insufficient data for performance calculation")
        return None

    # Strategy metrics
    strat_total_return = (strategy_nav / 100) - 1
    strat_days = (end_date - start_date).days
    strat_ann_return = ((strategy_nav / 100) ** (365 / strat_days)) - 1 if strat_days > 0 else 0
    strat_daily_returns = df_perf['Strategy_NAV'].pct_change().dropna()
    strat_vol = strat_daily_returns.std() * np.sqrt(252)
    strat_sharpe = strat_ann_return / strat_vol if strat_vol > 0 else 0

    # Calculate max drawdown
    cummax = df_perf['Strategy_NAV'].cummax()
    drawdown = (df_perf['Strategy_NAV'] - cummax) / cummax
    strat_max_dd = drawdown.min()

    # Benchmark metrics
    spy_total_return = (spy_nav / 100) - 1
    spy_ann_return = ((spy_nav / 100) ** (365 / strat_days)) - 1 if strat_days > 0 else 0

    bufr_total_return = (bufr_nav / 100) - 1
    bufr_ann_return = ((bufr_nav / 100) ** (365 / strat_days)) - 1 if strat_days > 0 else 0

    hold_total_return = (hold_nav / 100) - 1
    hold_ann_return = ((hold_nav / 100) ** (365 / strat_days)) - 1 if strat_days > 0 else 0

    # Excess returns
    vs_spy_excess = strat_total_return - spy_total_return
    vs_bufr_excess = strat_total_return - bufr_total_return
    vs_hold_excess = strat_total_return - hold_total_return

    # print(f"\nResults Summary:")
    # print(f"  Strategy Return: {strat_total_return * 100:+.2f}%")
    # print(f"  vs SPY: {vs_spy_excess * 100:+.2f}%")
    # print(f"  vs BUFR: {vs_bufr_excess * 100:+.2f}%")
    # print(f"  vs Hold: {vs_hold_excess * 100:+.2f}%")
    # print(f"  Sharpe Ratio: {strat_sharpe:.2f}")
    # print(f"  Max Drawdown: {strat_max_dd * 100:.2f}%")
    # print(f"  Number of Trades: {num_trades}")

    # Return comprehensive results
    return {
        'launch_month': launch_month,
        'trigger_type': trigger_type,
        'trigger_params': trigger_params,
        'selection_algo': selection_func.__name__,
        'strategy_intent': trigger_config.get('strategy_intent', 'neutral'),  # ADD THIS LINE
        'start_date': start_date,
        'end_date': end_date,
        'num_trades': num_trades,

        'strategy_total_return': strat_total_return,
        'strategy_ann_return': strat_ann_return,
        'strategy_sharpe': strat_sharpe,
        'strategy_volatility': strat_vol,
        'strategy_max_dd': strat_max_dd,

        'spy_total_return': spy_total_return,
        'spy_ann_return': spy_ann_return,

        'bufr_total_return': bufr_total_return,
        'bufr_ann_return': bufr_ann_return,

        'hold_total_return': hold_total_return,
        'hold_ann_return': hold_ann_return,

        'vs_spy_excess': vs_spy_excess,
        'vs_bufr_excess': vs_bufr_excess,
        'vs_hold_excess': vs_hold_excess,

        'daily_performance': df_perf,
        'trade_history': pd.DataFrame(trade_history) if trade_history else pd.DataFrame()
    }