"""
Forward regime performance analysis.

Analyzes which trigger/selection combinations perform best when entering
different future market regimes (bull/bear/neutral).

This enables answering questions like:
- "If I'm in January and a bull market is coming, which strategy should I use?"
- "Which defensive combinations work best when entering a bear market?"
"""

import pandas as pd
import numpy as np


def analyze_by_future_regime(results_list, df_forward_regimes, entry_frequency='quarterly'):
    """
    Calculate strategy performance based on FUTURE regime at MULTIPLE entry points.

    For each strategy configuration:
    1. Test entering at multiple dates (quarterly roll dates by default)
    2. For each entry, look up what future regime follows (3M and 6M ahead)
    3. Calculate forward returns from that entry point
    4. Create one row per entry point per strategy

    Parameters:
      results_list: List of result dicts from run_single_ticker_backtest
      df_forward_regimes: DataFrame from classify_forward_regimes()
      entry_frequency: 'monthly', 'quarterly', 'semi_annual', or 'all_dates'

    Returns:
      DataFrame with one row per entry point per strategy:
        - Multiple rows per strategy (one for each entry date)
        - Each row shows performance from that specific entry
    """

    if not results_list:
        print("ERROR: No results to analyze!")
        return pd.DataFrame()

    print("\n" + "=" * 80)
    print("ANALYZING PERFORMANCE BY FUTURE REGIME (MULTIPLE ENTRY POINTS)")

    future_regime_records = []
    total_entries = 0
    skipped_entries = 0

    for result in results_list:
        # Get the daily performance data for this strategy
        daily_perf = result['daily_performance'].copy()
        daily_perf = daily_perf.sort_values('Date').reset_index(drop=True)

        if daily_perf.empty:
            continue

        # Determine entry points based on frequency
        entry_points = _get_entry_points(daily_perf, entry_frequency)

        # For each potential entry point, calculate forward returns
        for entry_date in entry_points:
            total_entries += 1

            # Look up future regime at this entry date
            future_regime_data = df_forward_regimes[
                df_forward_regimes['Date'] == entry_date
                ]

            if future_regime_data.empty:
                skipped_entries += 1
                continue

            future_regime_row = future_regime_data.iloc[0]
            future_regime_3m = future_regime_row['Future_Regime_3M']
            future_regime_6m = future_regime_row['Future_Regime_6M']

            # Skip if both horizons are unknown (end of dataset)
            if future_regime_3m == 'unknown' and future_regime_6m == 'unknown':
                skipped_entries += 1
                continue

            # Get the entry point index in daily_perf
            entry_idx = daily_perf[daily_perf['Date'] == entry_date].index

            if len(entry_idx) == 0:
                skipped_entries += 1
                continue

            entry_idx = entry_idx[0]

            # Get NAVs at entry
            entry_nav = daily_perf.loc[entry_idx, 'Strategy_NAV']
            entry_spy_nav = daily_perf.loc[entry_idx, 'SPY_NAV']
            entry_bufr_nav = daily_perf.loc[entry_idx, 'BUFR_NAV']

            # Calculate 3-month forward returns (63 trading days)
            future_3m_idx = entry_idx + 63
            if future_3m_idx < len(daily_perf):
                forward_3m_strat_nav = daily_perf.loc[future_3m_idx, 'Strategy_NAV']
                forward_3m_spy_nav = daily_perf.loc[future_3m_idx, 'SPY_NAV']
                forward_3m_bufr_nav = daily_perf.loc[future_3m_idx, 'BUFR_NAV']

                forward_3m_return = (forward_3m_strat_nav / entry_nav) - 1
                spy_forward_3m_return = (forward_3m_spy_nav / entry_spy_nav) - 1
                bufr_forward_3m_return = (forward_3m_bufr_nav / entry_bufr_nav) - 1
            else:
                forward_3m_return = np.nan
                spy_forward_3m_return = np.nan
                bufr_forward_3m_return = np.nan

            # Calculate 6-month forward returns (126 trading days)
            future_6m_idx = entry_idx + 126
            if future_6m_idx < len(daily_perf):
                forward_6m_strat_nav = daily_perf.loc[future_6m_idx, 'Strategy_NAV']
                forward_6m_spy_nav = daily_perf.loc[future_6m_idx, 'SPY_NAV']
                forward_6m_bufr_nav = daily_perf.loc[future_6m_idx, 'BUFR_NAV']

                forward_6m_return = (forward_6m_strat_nav / entry_nav) - 1
                spy_forward_6m_return = (forward_6m_spy_nav / entry_spy_nav) - 1
                bufr_forward_6m_return = (forward_6m_bufr_nav / entry_bufr_nav) - 1
            else:
                forward_6m_return = np.nan
                spy_forward_6m_return = np.nan
                bufr_forward_6m_return = np.nan

            # Create record for this entry point
            record = {
                'launch_month': result['launch_month'],
                'trigger_type': result['trigger_type'],
                'trigger_params': str(result['trigger_params']),
                'selection_algo': result['selection_algo'],
                'strategy_intent': result.get('strategy_intent', 'neutral'),
                'entry_date': entry_date,

                # Future regime classifications
                'future_regime_3m': future_regime_3m,
                'future_regime_6m': future_regime_6m,

                # Strategy forward returns
                'forward_3m_return': forward_3m_return,
                'forward_6m_return': forward_6m_return,

                # Benchmark forward returns
                'spy_forward_3m_return': spy_forward_3m_return,
                'spy_forward_6m_return': spy_forward_6m_return,
                'bufr_forward_3m_return': bufr_forward_3m_return,
                'bufr_forward_6m_return': bufr_forward_6m_return,

                # Excess returns
                'excess_vs_spy_3m': forward_3m_return - spy_forward_3m_return if pd.notna(forward_3m_return) else np.nan,
                'excess_vs_spy_6m': forward_6m_return - spy_forward_6m_return if pd.notna(forward_6m_return) else np.nan,
                'excess_vs_bufr_3m': forward_3m_return - bufr_forward_3m_return if pd.notna(forward_3m_return) else np.nan,
                'excess_vs_bufr_6m': forward_6m_return - bufr_forward_6m_return if pd.notna(forward_6m_return) else np.nan,
            }

            future_regime_records.append(record)

    future_regime_df = pd.DataFrame(future_regime_records)

    print("=" * 80 + "\n")

    return future_regime_df


def _get_entry_points(daily_perf, frequency='quarterly'):
    """
    Helper function to determine entry points based on frequency.

    Parameters:
      daily_perf: DataFrame with daily performance data
      frequency: 'monthly', 'quarterly', 'semi_annual', 'all_dates'

    Returns:
      List of pd.Timestamp entry dates
    """
    all_dates = daily_perf['Date'].tolist()

    if frequency == 'all_dates':
        # Use every date (will be VERY slow)
        return all_dates

    elif frequency == 'monthly':
        # Use first day of each month
        monthly_dates = daily_perf.groupby(daily_perf['Date'].dt.to_period('M')).first()['Date']
        return monthly_dates.tolist()

    elif frequency == 'quarterly':
        # Use first day of each quarter (Mar, Jun, Sep, Dec)
        quarterly_dates = daily_perf.groupby(daily_perf['Date'].dt.to_period('Q')).first()['Date']
        return quarterly_dates.tolist()

    elif frequency == 'semi_annual':
        # Use first day of each half-year
        semi_dates = daily_perf[daily_perf['Date'].dt.month.isin([3, 9])]
        semi_dates = semi_dates.groupby(semi_dates['Date'].dt.year)['Date'].first()
        return semi_dates.tolist()

    else:
        # Default to quarterly
        quarterly_dates = daily_perf.groupby(daily_perf['Date'].dt.to_period('Q')).first()['Date']
        return quarterly_dates.tolist()


def rank_strategies_by_future_regime(future_regime_df, horizon='6M', metric='forward_return'):
    """
    Rank strategies by performance within each future regime.

    Parameters:
      future_regime_df: DataFrame from analyze_by_future_regime()
      horizon: '3M' or '6M'
      metric: 'forward_return' or 'excess_vs_spy' or 'excess_vs_bufr'

    Returns:
      DataFrame with top strategies for each future regime
    """
    if future_regime_df.empty:
        return pd.DataFrame()

    # Select appropriate columns based on horizon
    if horizon == '3M':
        regime_col = 'future_regime_3m'
        if metric == 'forward_return':
            perf_col = 'forward_3m_return'
        elif metric == 'excess_vs_spy':
            perf_col = 'excess_vs_spy_3m'
        elif metric == 'excess_vs_bufr':
            perf_col = 'excess_vs_bufr_3m'
    else:  # 6M
        regime_col = 'future_regime_6m'
        if metric == 'forward_return':
            perf_col = 'forward_6m_return'
        elif metric == 'excess_vs_spy':
            perf_col = 'excess_vs_spy_6m'
        elif metric == 'excess_vs_bufr':
            perf_col = 'excess_vs_bufr_6m'

    # Filter out unknown regimes and NaN performance
    valid_data = future_regime_df[
        (future_regime_df[regime_col] != 'unknown') &
        (future_regime_df[perf_col].notna())
    ].copy()

    if valid_data.empty:
        return pd.DataFrame()

    # Create strategy identifier
    valid_data['strategy_id'] = (
        valid_data['launch_month'] + '_' +
        valid_data['trigger_type'] + '_' +
        valid_data['selection_algo']
    )

    # Aggregate by strategy and future regime
    aggregated = valid_data.groupby(['strategy_id', 'launch_month', 'trigger_type',
                                     'selection_algo', 'strategy_intent', regime_col]).agg({
        perf_col: ['mean', 'median', 'std', 'count']
    }).reset_index()

    # Flatten column names - KEEP THE ORIGINAL REGIME COLUMN NAME
    aggregated.columns = [
        'strategy_id', 'launch_month', 'trigger_type', 'selection_algo',
        'strategy_intent', regime_col,  # ← FIXED - preserve '3m' or '6m' suffix
        f'{metric}_mean', f'{metric}_median', f'{metric}_std', 'num_observations'
    ]

    # Rank within each regime
    ranked_results = []

    for regime in ['bull', 'bear', 'neutral']:
        regime_data = aggregated[aggregated[regime_col] == regime].copy()  # ← FIXED - use regime_col

        if regime_data.empty:
            continue

        # Rank by mean performance (descending)
        regime_data = regime_data.sort_values(f'{metric}_mean', ascending=False).reset_index(drop=True)
        regime_data['rank'] = range(1, len(regime_data) + 1)

        ranked_results.append(regime_data)

    if ranked_results:
        return pd.concat(ranked_results, ignore_index=True)
    else:
        return pd.DataFrame()


def summarize_optimal_strategies(future_regime_df, horizon='6M', top_n=5):
    """
    Identify top N strategies for each future regime.

    Parameters:
      future_regime_df: DataFrame from analyze_by_future_regime()
      horizon: '3M' or '6M'
      top_n: Number of top strategies to show per regime

    Returns:
      Dict with keys: 'bull', 'bear', 'neutral'
      Values: DataFrame with top strategies for that regime
    """

    # Rank by excess vs BUFR (most relevant metric)
    ranked = rank_strategies_by_future_regime(
        future_regime_df,
        horizon=horizon,
        metric='excess_vs_bufr'
    )

    if ranked.empty:
        return {}

    optimal_strategies = {}

    # Build the correct column name for this horizon
    regime_col = f'future_regime_{horizon.lower()}'  # ← ADD THIS LINE

    for regime in ['bull', 'bear', 'neutral']:
        # Use the correct column name
        regime_data = ranked[ranked[regime_col] == regime].head(top_n)  # ← FIX THIS LINE

        if not regime_data.empty:
            optimal_strategies[regime] = regime_data[[
                'rank', 'launch_month', 'trigger_type', 'selection_algo',
                'strategy_intent', 'excess_vs_bufr_mean', 'excess_vs_bufr_median',
                'num_observations'
            ]]

    return optimal_strategies


def compare_strategy_intent_vs_future_regime(future_regime_df, horizon='6M'):
    """
    Analyze if strategies perform as intended in future regimes.

    For example:
    - Do "bullish" strategies outperform in future bull markets?
    - Do "bearish" strategies protect better in future bear markets?

    Parameters:
      future_regime_df: DataFrame from analyze_by_future_regime()
      horizon: '3M' or '6M'

    Returns:
      DataFrame showing performance by (strategy_intent × future_regime)
    """
    if future_regime_df.empty:
        return pd.DataFrame()

    # Select columns based on horizon
    if horizon == '3M':
        regime_col = 'future_regime_3m'
        return_col = 'forward_3m_return'
        excess_spy_col = 'excess_vs_spy_3m'
        excess_bufr_col = 'excess_vs_bufr_3m'
    else:
        regime_col = 'future_regime_6m'
        return_col = 'forward_6m_return'
        excess_spy_col = 'excess_vs_spy_6m'
        excess_bufr_col = 'excess_vs_bufr_6m'

    # Filter valid data
    valid_data = future_regime_df[
        (future_regime_df[regime_col] != 'unknown') &
        (future_regime_df[return_col].notna())
        ].copy()

    if valid_data.empty:
        return pd.DataFrame()

    # Aggregate by intent × regime
    intent_regime_summary = valid_data.groupby(['strategy_intent', regime_col]).agg({
        return_col: ['mean', 'median', 'count'],
        excess_spy_col: ['mean', 'median'],
        excess_bufr_col: ['mean', 'median']
    }).round(4)

    # Flatten columns
    intent_regime_summary.columns = [
        'strategy_return_mean', 'strategy_return_median', 'num_strategies',
        'excess_vs_spy_mean', 'excess_vs_spy_median',
        'excess_vs_bufr_mean', 'excess_vs_bufr_median'
    ]

    intent_regime_summary = intent_regime_summary.reset_index()
    intent_regime_summary = intent_regime_summary.rename(columns={regime_col: 'future_regime'})

    return intent_regime_summary


def identify_robust_strategies(future_regime_df, horizon='6M', min_observations=3):
    """
    Find strategies that perform well across ALL future regimes.

    Robust strategies should:
    - Have positive excess returns in bull markets
    - Have positive or minimal negative excess in bear markets
    - Be consistent across neutral markets

    Parameters:
      future_regime_df: DataFrame from analyze_by_future_regime()
      horizon: '3M' or '6M'
      min_observations: Minimum number of observations required per regime

    Returns:
      DataFrame with strategies ranked by consistency across regimes
    """
    if future_regime_df.empty:
        return pd.DataFrame()

    # Select columns based on horizon
    if horizon == '3M':
        regime_col = 'future_regime_3m'
        excess_col = 'excess_vs_bufr_3m'
    else:
        regime_col = 'future_regime_6m'
        excess_col = 'excess_vs_bufr_6m'

    # Filter valid data
    valid_data = future_regime_df[
        (future_regime_df[regime_col] != 'unknown') &
        (future_regime_df[excess_col].notna())
        ].copy()

    if valid_data.empty:
        return pd.DataFrame()

    # Create strategy identifier
    valid_data['strategy_id'] = (
            valid_data['launch_month'] + '_' +
            valid_data['trigger_type'] + '_' +
            valid_data['selection_algo']
    )

    # For each strategy, calculate performance in each regime
    strategy_regime_perf = valid_data.groupby(['strategy_id', regime_col]).agg({
        excess_col: ['mean', 'count']
    }).reset_index()

    strategy_regime_perf.columns = ['strategy_id', 'future_regime', 'excess_mean', 'count']

    # Filter strategies that have observations in all 3 regimes
    strategy_counts = strategy_regime_perf.groupby('strategy_id')['future_regime'].nunique()
    strategies_in_all_regimes = strategy_counts[strategy_counts == 3].index

    filtered_perf = strategy_regime_perf[
        strategy_regime_perf['strategy_id'].isin(strategies_in_all_regimes) &
        (strategy_regime_perf['count'] >= min_observations)
        ]

    if filtered_perf.empty:
        return pd.DataFrame()

    # Pivot to get one row per strategy with columns for each regime
    pivoted = filtered_perf.pivot(
        index='strategy_id',
        columns='future_regime',
        values='excess_mean'
    ).reset_index()

    # Calculate robustness metrics
    if 'bull' in pivoted.columns and 'bear' in pivoted.columns and 'neutral' in pivoted.columns:
        pivoted['min_excess'] = pivoted[['bull', 'bear', 'neutral']].min(axis=1)
        pivoted['avg_excess'] = pivoted[['bull', 'bear', 'neutral']].mean(axis=1)
        pivoted['consistency_score'] = pivoted['min_excess'] * 0.6 + pivoted['avg_excess'] * 0.4

        # Sort by consistency score
        pivoted = pivoted.sort_values('consistency_score', ascending=False)

        # Extract strategy components
        pivoted[['launch_month', 'trigger_type', 'selection_algo']] = pivoted['strategy_id'].str.split('_', n=2, expand=True)

    return pivoted

