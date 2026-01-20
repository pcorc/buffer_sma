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


def analyze_by_future_regime(results_list, df_forward_regimes):
    """
    Calculate strategy performance based on FUTURE regime at entry.

    For each backtest result:
    1. Identify the strategy entry date (start_date)
    2. Look up what the future regime was at that date (3M and 6M ahead)
    3. Calculate the actual forward returns (3M and 6M)
    4. Attribute performance to that future regime

    Parameters:
      results_list: List of result dicts from run_single_ticker_backtest
      df_forward_regimes: DataFrame from classify_forward_regimes()

    Returns:
      DataFrame with columns:
        - launch_month
        - trigger_type
        - trigger_params
        - selection_algo
        - strategy_intent
        - entry_date (when strategy started)
        - future_regime_3m (what regime followed)
        - future_regime_6m (what regime followed)
        - forward_3m_return (strategy return over next 3 months)
        - forward_6m_return (strategy return over next 6 months)
        - spy_forward_3m_return (SPY return over next 3 months)
        - spy_forward_6m_return (SPY return over next 6 months)
        - excess_3m (strategy vs SPY over 3M)
        - excess_6m (strategy vs SPY over 6M)
    """

    if not results_list:
        print("ERROR: No results to analyze!")
        return pd.DataFrame()

    print("\n" + "=" * 80)
    print("ANALYZING PERFORMANCE BY FUTURE REGIME")
    print("=" * 80)

    future_regime_records = []

    for result in results_list:
        entry_date = result['start_date']

        # Look up future regime at entry date
        future_regime_data = df_forward_regimes[
            df_forward_regimes['Date'] == entry_date
            ]

        if future_regime_data.empty:
            print(f"  ⚠️  No future regime data for {entry_date.date()} - skipping")
            continue

        future_regime_row = future_regime_data.iloc[0]
        future_regime_3m = future_regime_row['Future_Regime_3M']
        future_regime_6m = future_regime_row['Future_Regime_6M']

        # Skip if future regime is unknown (end of dataset)
        if future_regime_3m == 'unknown' and future_regime_6m == 'unknown':
            continue

        # Calculate forward returns from entry date
        # Get daily performance data
        daily_perf = result['daily_performance'].copy()
        daily_perf = daily_perf.sort_values('Date').reset_index(drop=True)

        # Find 3-month and 6-month future dates (approximately)
        entry_nav = daily_perf.iloc[0]['Strategy_NAV']  # Should be 100
        entry_spy_nav = daily_perf.iloc[0]['SPY_NAV']
        entry_bufr_nav = daily_perf.iloc[0]['BUFR_NAV']

        # 3-month forward (63 trading days)
        if len(daily_perf) >= 64:  # Need at least 64 days (0-indexed)
            day_63 = daily_perf.iloc[63]
            forward_3m_strat_nav = day_63['Strategy_NAV']
            forward_3m_spy_nav = day_63['SPY_NAV']
            forward_3m_bufr_nav = day_63['BUFR_NAV']

            forward_3m_return = (forward_3m_strat_nav / entry_nav) - 1
            spy_forward_3m_return = (forward_3m_spy_nav / entry_spy_nav) - 1
            bufr_forward_3m_return = (forward_3m_bufr_nav / entry_bufr_nav) - 1
        else:
            forward_3m_return = np.nan
            spy_forward_3m_return = np.nan
            bufr_forward_3m_return = np.nan

        # 6-month forward (126 trading days)
        if len(daily_perf) >= 127:
            day_126 = daily_perf.iloc[126]
            forward_6m_strat_nav = day_126['Strategy_NAV']
            forward_6m_spy_nav = day_126['SPY_NAV']
            forward_6m_bufr_nav = day_126['BUFR_NAV']

            forward_6m_return = (forward_6m_strat_nav / entry_nav) - 1
            spy_forward_6m_return = (forward_6m_spy_nav / entry_spy_nav) - 1
            bufr_forward_6m_return = (forward_6m_bufr_nav / entry_bufr_nav) - 1
        else:
            forward_6m_return = np.nan
            spy_forward_6m_return = np.nan
            bufr_forward_6m_return = np.nan

        # Create record
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

    print(f"\n✅ Analyzed {len(future_regime_df)} strategy entries with future regime data")

    return future_regime_df


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

    # Flatten column names
    aggregated.columns = [
        'strategy_id', 'launch_month', 'trigger_type', 'selection_algo',
        'strategy_intent', 'future_regime',
        f'{metric}_mean', f'{metric}_median', f'{metric}_std', 'num_observations'
    ]

    # Rank within each regime
    ranked_results = []

    for regime in ['bull', 'bear', 'neutral']:
        regime_data = aggregated[aggregated['future_regime'] == regime].copy()

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

    for regime in ['bull', 'bear', 'neutral']:
        regime_data = ranked[ranked['future_regime'] == regime].head(top_n)

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


def print_future_regime_summary(optimal_strategies, horizon='6M'):
    """
    Print formatted summary of optimal strategies by future regime.

    Parameters:
      optimal_strategies: Dict from summarize_optimal_strategies()
      horizon: '3M' or '6M'
    """
    print(f"\n{'=' * 80}")
    print(f"OPTIMAL STRATEGIES BY FUTURE REGIME ({horizon} HORIZON)")
    print(f"{'=' * 80}\n")

    for regime in ['bull', 'bear', 'neutral']:
        if regime not in optimal_strategies:
            continue

        regime_data = optimal_strategies[regime]

        if regime_data.empty:
            continue

        print(f"\n{regime.upper()} MARKET AHEAD:")
        print("-" * 80)

        for _, row in regime_data.iterrows():
            print(f"  #{int(row['rank'])}. {row['launch_month']} | "
                  f"{row['trigger_type'][:30]} | {row['selection_algo'][:25]}")
            print(f"      Intent: {row['strategy_intent']:8s} | "
                  f"Excess vs BUFR: {row['excess_vs_bufr_mean'] * 100:+6.2f}% (median: {row['excess_vs_bufr_median'] * 100:+6.2f}%) | "
                  f"N={int(row['num_observations'])}")

    print(f"\n{'=' * 80}\n")