"""
Core backtesting engine for single ticker strategy.
"""

import pandas as pd
import numpy as np
from core.triggers import get_trigger_function
from core.selections import get_selection_function, RebalanceScoringTracker
from config import settings
from utils.date_utils import get_rebalance_trading_dates
from core.selections import compute_ecr_scores

def run_single_ticker_backtest(df_enriched, df_benchmarks, launch_month,
                               trigger_config, selection_func, roll_dates_dict, series='F',
                               df_regimes=None):
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
      df_regimes: Optional DataFrame with S&P 500 regime data (for regime-adaptive strategies)
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
    # print(f"\n{'=' * 80}")
    # print(f"Running backtest: {launch_month} | {trigger_config['type']} | {selection_func.__name__}")
    # print(f"{'=' * 80}")
    print(f"  → Starting: {launch_month} | {trigger_config['type']} | {selection_func.__name__}")

    tracker = RebalanceScoringTracker()

    # Initialize
    current_fund = series + launch_month
    fund_data = df_enriched[df_enriched['Fund'] == current_fund].copy()

    if fund_data.empty:
        print(f"ERROR: No data for fund {current_fund}")
        return None

    BUFR_INCEPTION = pd.Timestamp('2020-07-01')

    # Get roll dates for this fund, filtered to >= BUFR inception
    fund_roll_dates = sorted([
        pd.Timestamp(rd) for rd in fund_data['Roll_Date'].dropna().unique()
        if pd.Timestamp(rd) >= BUFR_INCEPTION
    ])

    if len(fund_roll_dates) == 0:
        print(f"ERROR: No valid roll dates for {current_fund} after {BUFR_INCEPTION.date()}")
        return None

    # Start date is the first available roll date after BUFR inception
    start_date = fund_roll_dates[0]

    # Filter fund data to start from this aligned date
    fund_data = fund_data[fund_data['Date'] >= start_date].copy()

    if fund_data.empty:
        # print(f"ERROR: No fund data after alignment date {start_date.date()}")
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

    # Initialize before main loop
    daily_performance = []
    trade_history = []
    num_trades = 0
    spread_history = []
    days_since_rebalance = 0
    df_universe = None
    df_universe_ecr = None

    # Get business days for iteration
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # Prepare trigger-specific data
    trigger_type = trigger_config['type']
    trigger_params = trigger_config['params']
    trigger_func = get_trigger_function(trigger_type)

    if trigger_type == 'rebalance_time_period':


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

    elif trigger_type == 'ecr_percentile_trigger':
        # Pre-extract params once outside the loop
        ecr_w_dbb    = trigger_params.get('w_dbb', 1)
        ecr_w_buffer = trigger_params.get('w_buffer', 1)
        ecr_w_cap    = trigger_params.get('w_cap', 1)
        ecr_pct_threshold  = trigger_params['percentile_threshold']
        ecr_min_holding    = trigger_params.get('min_holding_days', 21)
        ecr_rolling_window = trigger_params.get('rolling_window', 252)
        ecr_min_abs_spread = trigger_params.get('min_absolute_spread', 0.05)  # ← ADD


        # Parse weight_code string into individual weights if provided
        weight_code = trigger_params.get('weight_code', '111')
        if len(weight_code) == 3 and weight_code.isdigit():
            ecr_w_dbb    = int(weight_code[0])
            ecr_w_buffer = int(weight_code[1])
            ecr_w_cap    = int(weight_code[2])

    elif trigger_type == 'ecr_score_threshold':
        ecr_w_dbb = int(trigger_params.get('weight_code', '111')[0])
        ecr_w_buffer = int(trigger_params.get('weight_code', '111')[1])
        ecr_w_cap = int(trigger_params.get('weight_code', '111')[2])
        ecr_score_thresh = trigger_params['score_threshold']
        ecr_min_holding = trigger_params.get('min_holding_days', 63)

    elif trigger_type == 'ecr_percentile_rank':
        ecr_w_dbb = int(trigger_params.get('weight_code', '111')[0])
        ecr_w_buffer = int(trigger_params.get('weight_code', '111')[1])
        ecr_w_cap = int(trigger_params.get('weight_code', '111')[2])
        ecr_pct_threshold = trigger_params['percentile_threshold']
        ecr_min_holding = trigger_params.get('min_holding_days', 22)

    elif trigger_type == 'ecr_rank_threshold':
        ecr_w_dbb = int(trigger_params.get('weight_code', '111')[0])
        ecr_w_buffer = int(trigger_params.get('weight_code', '111')[1])
        ecr_w_cap = int(trigger_params.get('weight_code', '111')[2])
        ecr_rank_threshold = trigger_params['rank_threshold']
        ecr_min_holding = trigger_params.get('min_holding_days', 22)

    # Main backtest loop
    for current_date in all_dates:
        # Get current fund data
        current_fund_data = df_enriched[
            (df_enriched['Fund'] == current_fund) &
            (df_enriched['Date'] == current_date)
            ]

        if current_fund_data.empty:
            continue

        # Progress indicator for ecr_percentile_trigger (slow daily scoring)
        # if trigger_type in ('ecr_score_threshold', 'ecr_percentile_trigger', 'ecr_percentile_rank') and current_date.day == 1:
        #     print(f"    {launch_month} | {current_date.strftime('%Y-%m')} | "
        #           f"trades={num_trades} | days_held={days_since_rebalance}")

        current_fund_row = current_fund_data.iloc[0]
        daily_return = current_fund_row['daily_return']

        # On first day, just initialize NAVs at 100 without applying returns
        if first_day:
            first_day = False

        else:
            strategy_nav *= (1 + daily_return)

            # Update benchmark NAVs
            bench_data = df_bench[df_bench['Date'] == current_date]
            if not bench_data.empty:
                spy_nav *= (1 + bench_data.iloc[0]['SPY_daily_return'])
                bufr_nav *= (1 + bench_data.iloc[0]['BUFR_daily_return'])

            # Update buy-and-hold NAV
            hold_fund = series + launch_month
            hold_data = df_enriched[
                (df_enriched['Fund'] == hold_fund) &
                (df_enriched['Date'] == current_date)
                ]
            if not hold_data.empty:
                hold_nav *= (1 + hold_data.iloc[0]['daily_return'])

            # Track days since last rebalance for ecr_percentile_trigger
            days_since_rebalance += 1

        # Store daily performance with comprehensive roll date metrics

        # Current values
        current_fund_nav = current_fund_row.get('Fund Value (USD)', None)
        current_ref_index = current_fund_row.get('Reference Asset Value (USD)', None)
        current_remaining_cap_pct = current_fund_row.get('Remaining Cap', None)
        current_downside_before_buffer = current_fund_row.get('Downside Before Buffer', None)

        # Roll date reference values (constant throughout period)
        roll_date = current_fund_row.get('Roll_Date', None)
        starting_fund_nav = current_fund_row.get('Starting_Fund_Value', None)
        starting_ref_index = current_fund_row.get('Starting_Ref_Asset_Value', None)
        original_cap = current_fund_row.get('Original_Cap', None)
        original_buffer = current_fund_row.get('Original_Buffer', None)

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

        elif trigger_type == 'ecr_percentile_trigger':
            # Build universe here — needed before trigger evaluation
            df_universe_ecr = df_enriched[
                (df_enriched['Date'] == current_date) &
                (df_enriched['Fund'].str.startswith(series))
                ].copy()

            if not df_universe_ecr.empty:
                triggered, spread = trigger_func(
                    current_fund=current_fund,
                    df_universe=df_universe_ecr,
                    series=series,
                    w_dbb=ecr_w_dbb,
                    w_buffer=ecr_w_buffer,
                    w_cap=ecr_w_cap,
                    spread_history=spread_history,
                    percentile_threshold=ecr_pct_threshold,
                    min_holding_days=ecr_min_holding,
                    days_since_rebalance=days_since_rebalance,
                    rolling_window=ecr_rolling_window,
                    min_absolute_spread=ecr_min_abs_spread  # ← ADD

                )
                # Always append spread to history regardless of trigger
                if not np.isnan(spread):
                    spread_history.append(spread)

                if triggered:
                    trigger_reason = f"ecr_spread_pct={ecr_pct_threshold}"

        elif trigger_type == 'ecr_score_threshold':
            df_universe_ecr = df_enriched[
                (df_enriched['Date'] == current_date) &
                (df_enriched['Fund'].str.startswith(series))
                ].copy()

            if not df_universe_ecr.empty:
                triggered = trigger_func(
                    current_fund=current_fund,
                    df_universe=df_universe_ecr,
                    series=series,
                    w_dbb=ecr_w_dbb,
                    w_buffer=ecr_w_buffer,
                    w_cap=ecr_w_cap,
                    score_threshold=ecr_score_thresh,
                    min_holding_days=ecr_min_holding,
                    days_since_rebalance=days_since_rebalance
                )
                if triggered:
                    trigger_reason = f"ecr_score<{ecr_score_thresh}"

        elif trigger_type == 'ecr_percentile_rank':
            df_universe_ecr = df_enriched[
                (df_enriched['Date'] == current_date) &
                (df_enriched['Fund'].str.startswith(series))
                ].copy()

            if not df_universe_ecr.empty:
                triggered = trigger_func(
                    current_fund=current_fund,
                    df_universe=df_universe_ecr,
                    series=series,
                    w_dbb=ecr_w_dbb,
                    w_buffer=ecr_w_buffer,
                    w_cap=ecr_w_cap,
                    percentile_threshold=ecr_pct_threshold,
                    min_holding_days=ecr_min_holding,
                    days_since_rebalance=days_since_rebalance
                )
                if triggered:
                    trigger_reason = f"ecr_pct_rank<{ecr_pct_threshold}th"

        elif trigger_type == 'ecr_rank_threshold':
            df_universe_ecr = df_enriched[
                (df_enriched['Date'] == current_date) &
                (df_enriched['Fund'].str.startswith(series))
                ].copy()

            if not df_universe_ecr.empty:
                triggered = trigger_func(
                    current_fund=current_fund,
                    df_universe=df_universe_ecr,
                    series=series,
                    w_dbb=ecr_w_dbb,
                    w_buffer=ecr_w_buffer,
                    w_cap=ecr_w_cap,
                    rank_threshold=ecr_rank_threshold,
                    min_holding_days=ecr_min_holding,
                    days_since_rebalance=days_since_rebalance
                )
                if triggered:
                    trigger_reason = f"ecr_rank>{ecr_rank_threshold}"

        else:
            # All other threshold-based triggers
            threshold = trigger_params['threshold']
            triggered = trigger_func(current_fund_row, threshold)
            if triggered:
                trigger_reason = f"{trigger_type}={threshold}"

        if triggered:
            if trigger_type in ('ecr_percentile_trigger',
                                'ecr_score_threshold',
                                'ecr_relative_spread',
                                'ecr_percentile_rank',
                                'ecr_rank_threshold'):
                df_universe = df_universe_ecr

            else:
                df_universe = df_enriched[
                    (df_enriched['Date'] == current_date) &
                    (df_enriched['Fund'].str.startswith(series))
                    ].copy()

            if not df_universe.empty:
                # Record outgoing fund score before selection
                if trigger_type in ('ecr_score_threshold', 'ecr_relative_spread', 'ecr_percentile_rank', 'ecr_rank_threshold'):
                    all_scores = compute_ecr_scores(df_universe, series, ecr_w_dbb, ecr_w_buffer, ecr_w_cap)
                    outgoing_score = all_scores.get(current_fund, np.nan)
                    best_score = max(all_scores.values()) if all_scores else np.nan
                else:
                    all_scores = {}
                    outgoing_score = np.nan
                    best_score = np.nan

                new_fund = selection_func(
                    df_universe,
                    current_date,
                    series,
                    df_regimes=df_regimes,
                    tracker=tracker,
                    month=launch_month
                )

                if new_fund and new_fund != current_fund:
                    trade_history.append({
                        'Date': current_date,
                        'From_Fund': current_fund,
                        'To_Fund': new_fund,
                        'Trigger_Reason': trigger_reason,
                        'NAV_at_Switch': strategy_nav,
                        'Outgoing_ECR_Score': outgoing_score,
                        'Incoming_ECR_Score': all_scores.get(new_fund, np.nan),
                        'Score_Spread': (best_score - outgoing_score) / best_score
                        if pd.notna(outgoing_score) and best_score > 0 else np.nan,
                    })
                    current_fund = new_fund
                    num_trades += 1
                    days_since_rebalance = 0

                    if trigger_type in ('ecr_percentile_trigger', 'ecr_score_threshold', 'ecr_percentile_rank', 'ecr_rank_threshold'):
                        scores = compute_ecr_scores(df_universe, series, ecr_w_dbb, ecr_w_buffer, ecr_w_cap)
                        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        print(f"    *** TRADE {num_trades} on {current_date.strftime('%Y-%m-%d')}: "
                              f"→ {new_fund} | scores: "
                              f"{' | '.join([f'{f}={s:.3f}' for f, s in ranked])}")

            else:
                print(f"     ❌ No universe data available")

    # Calculate performance metrics
    df_perf = pd.DataFrame(daily_performance)

    if len(df_perf) < 2:
        # print("ERROR: Insufficient data for performance calculation")
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
        'trade_history': pd.DataFrame(trade_history) if trade_history else pd.DataFrame(),

        'scoring_tracker': tracker,
        'spread_history': spread_history,
    }
