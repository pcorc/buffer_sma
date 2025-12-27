"""
Market regime classification based on rolling reference index returns.
"""

import pandas as pd
import numpy as np


def classify_market_regimes(df_spy, window_months=6, bull_threshold=0.10, bear_threshold=-0.10):
    """
    Classify each date as bull, bear, or neutral based on rolling SPY returns.

    Uses a rolling window to calculate returns, then classifies based on thresholds:
    - Bull: Rolling return >= bull_threshold
    - Bear: Rolling return <= bear_threshold
    - Neutral: Everything in between

    Parameters:
      df_spy: DataFrame with 'Date' and 'Ref_Index' (SPY prices) columns
      window_months: Rolling window in months (default 6)
      bull_threshold: Return threshold for bull market (default 0.10 = 10%)
      bear_threshold: Return threshold for bear market (default -0.10 = -10%)

    Returns:
      DataFrame with columns: Date, Regime, Rolling_Return
    """
    print("\n" + "=" * 80)
    print("CLASSIFYING MARKET REGIMES")
    print("=" * 80)
    print(f"Parameters:")
    print(f"  Window: {window_months} months")
    print(f"  Bull threshold: {bull_threshold * 100:+.1f}%")
    print(f"  Bear threshold: {bear_threshold * 100:+.1f}%")

    df = df_spy.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Calculate rolling returns
    window_days = window_months * 21  # Approximate trading days per month
    df['Rolling_Return'] = df['Ref_Index'].pct_change(periods=window_days)

    # Classify regimes
    def classify_regime(rolling_return):
        if pd.isna(rolling_return):
            return 'neutral'
        elif rolling_return >= bull_threshold:
            return 'bull'
        elif rolling_return <= bear_threshold:
            return 'bear'
        else:
            return 'neutral'

    df['Regime'] = df['Rolling_Return'].apply(classify_regime)

    # Summary statistics
    regime_counts = df['Regime'].value_counts()
    total_days = len(df)

    print(f"\nRegime Distribution:")
    for regime in ['bull', 'bear', 'neutral']:
        count = regime_counts.get(regime, 0)
        pct = (count / total_days) * 100
        print(f"  {regime.capitalize():8s}: {count:5d} days ({pct:5.1f}%)")

    # # Calculate average returns by regime
    # print(f"\nAverage Rolling Returns by Regime:")
    # for regime in ['bull', 'bear', 'neutral']:
    #     regime_data = df[df['Regime'] == regime]
    #     if len(regime_data) > 0:
    #         avg_return = regime_data['Rolling_Return'].mean()
    #         print(f"  {regime.capitalize():8s}: {avg_return * 100:+6.2f}%")
    #
    # print("=" * 80 + "\n")

    return df[['Date', 'Regime', 'Rolling_Return']]


def get_regime_transitions(df_regimes):
    """
    Identify dates when regime transitions occur.

    Parameters:
      df_regimes: DataFrame with Date and Regime columns

    Returns:
      DataFrame with regime transition events
    """
    df = df_regimes.copy()
    df = df.sort_values('Date').reset_index(drop=True)

    # Identify transitions
    df['Regime_Shift'] = df['Regime'].shift(1)
    transitions = df[df['Regime'] != df['Regime_Shift']].copy()
    transitions = transitions[['Date', 'Regime_Shift', 'Regime']].rename(
        columns={'Regime_Shift': 'From_Regime', 'Regime': 'To_Regime'}
    )

    return transitions


def analyze_regime_persistence(df_regimes):
    """
    Analyze how long regimes typically persist.

    Parameters:
      df_regimes: DataFrame with Date and Regime columns

    Returns:
      Dict with persistence statistics by regime
    """
    df = df_regimes.copy()
    df = df.sort_values('Date').reset_index(drop=True)

    # Identify regime runs
    df['Regime_ID'] = (df['Regime'] != df['Regime'].shift(1)).cumsum()

    regime_runs = df.groupby(['Regime', 'Regime_ID']).size().reset_index(name='Duration')

    persistence_stats = {}
    for regime in ['bull', 'bear', 'neutral']:
        regime_durations = regime_runs[regime_runs['Regime'] == regime]['Duration']
        if len(regime_durations) > 0:
            persistence_stats[regime] = {
                'avg_duration_days': regime_durations.mean(),
                'median_duration_days': regime_durations.median(),
                'max_duration_days': regime_durations.max(),
                'min_duration_days': regime_durations.min(),
                'num_periods': len(regime_durations)
            }

    return persistence_stats


def filter_by_regime(df, df_regimes, regime_type):
    """
    Filter a DataFrame to only include dates in a specific regime.

    Parameters:
      df: DataFrame with Date column
      df_regimes: DataFrame with Date and Regime columns
      regime_type: 'bull', 'bear', or 'neutral'

    Returns:
      Filtered DataFrame
    """
    df_merged = df.merge(df_regimes[['Date', 'Regime']], on='Date', how='left')
    return df_merged[df_merged['Regime'] == regime_type].drop(columns=['Regime'])