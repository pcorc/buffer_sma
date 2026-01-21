"""
Forward-looking regime classification for strategy optimization.

This module calculates FUTURE returns (T+3 months, T+6 months) and classifies
the upcoming market regime. This enables analysis of which trigger/selection
combinations perform best when entering different future market conditions.

Key difference from regime_classifier.py:
- regime_classifier.py: Uses BACKWARD-looking returns (T-6 to T) to classify current regime
- forward_regime_classifier.py: Uses FORWARD-looking returns (T to T+3, T to T+6) to classify future regime
"""

import pandas as pd
import numpy as np


def classify_forward_regimes(df_spy, window_3m_days=63, window_6m_days=126,
                             bull_threshold=0.10, bear_threshold=-0.10):
    """
    Classify future market regimes based on forward-looking returns.

    For each date T, calculates:
    - Return from T to T+3 months (63 trading days)
    - Return from T to T+6 months (126 trading days)
    - Future regime classification based on those forward returns

    This answers: "If I make a strategy decision on date T, what kind of market
    will I experience over the next 3-6 months?"

    Parameters:
      df_spy: DataFrame with 'Date' and 'Ref_Index' (SPY prices) columns
      window_3m_days: Trading days for 3-month window (default 63 = ~3 months)
      window_6m_days: Trading days for 6-month window (default 126 = ~6 months)
      bull_threshold: Return threshold for bull market (default 0.10 = 10%)
      bear_threshold: Return threshold for bear market (default -0.10 = -10%)

    Returns:
      DataFrame with columns:
        - Date: Trading date
        - Forward_3M_Return: Return from T to T+3 months
        - Forward_6M_Return: Return from T to T+6 months
        - Future_Regime_3M: Regime classification based on 3M forward return
        - Future_Regime_6M: Regime classification based on 6M forward return
        - Ref_Index: SPY price at date T (for reference)
    """
    print("\n" + "=" * 80)
    print("CLASSIFYING FORWARD-LOOKING REGIMES")
    print("=" * 80)
    print(f"Parameters:")
    print(f"  3-month window: {window_3m_days} trading days")
    print(f"  6-month window: {window_6m_days} trading days")
    print(f"  Bull threshold: {bull_threshold * 100:+.1f}%")
    print(f"  Bear threshold: {bear_threshold * 100:+.1f}%")

    df = df_spy.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Calculate forward returns using shift with negative periods
    # shift(-N) moves data N periods earlier, so we're looking ahead
    df['Ref_Index_3M_Ahead'] = df['Ref_Index'].shift(-window_3m_days)
    df['Ref_Index_6M_Ahead'] = df['Ref_Index'].shift(-window_6m_days)

    # Calculate forward returns
    df['Forward_3M_Return'] = (df['Ref_Index_3M_Ahead'] / df['Ref_Index']) - 1
    df['Forward_6M_Return'] = (df['Ref_Index_6M_Ahead'] / df['Ref_Index']) - 1

    # Classify future regimes
    def classify_regime(forward_return):
        if pd.isna(forward_return):
            return 'unknown'  # Can't classify if no future data
        elif forward_return >= bull_threshold:
            return 'bull'
        elif forward_return <= bear_threshold:
            return 'bear'
        else:
            return 'neutral'

    df['Future_Regime_3M'] = df['Forward_3M_Return'].apply(classify_regime)
    df['Future_Regime_6M'] = df['Forward_6M_Return'].apply(classify_regime)

    # Calculate how many dates have valid forward regime data
    valid_3m = df['Future_Regime_3M'].ne('unknown').sum()
    valid_6m = df['Future_Regime_6M'].ne('unknown').sum()
    total_dates = len(df)

    # Summary statistics for 3M forward regimes
    regime_3m_counts = df['Future_Regime_3M'].value_counts()
    print(f"\n3-Month Forward Regime Distribution:")
    for regime in ['bull', 'bear', 'neutral', 'unknown']:
        count = regime_3m_counts.get(regime, 0)
        pct = (count / total_dates) * 100
        print(f"  {regime.capitalize():8s}: {count:5d} days ({pct:5.1f}%)")

    # Summary statistics for 6M forward regimes
    regime_6m_counts = df['Future_Regime_6M'].value_counts()
    print(f"\n6-Month Forward Regime Distribution:")
    for regime in ['bull', 'bear', 'neutral', 'unknown']:
        count = regime_6m_counts.get(regime, 0)
        pct = (count / total_dates) * 100
        print(f"  {regime.capitalize():8s}: {count:5d} days ({pct:5.1f}%)")

    # Calculate average forward returns by regime
    print(f"\nAverage Forward Returns by Future Regime (3M):")
    for regime in ['bull', 'bear', 'neutral']:
        regime_data = df[df['Future_Regime_3M'] == regime]
        if len(regime_data) > 0:
            avg_return = regime_data['Forward_3M_Return'].mean()
            print(f"  {regime.capitalize():8s}: {avg_return * 100:+6.2f}%")

    print(f"\nAverage Forward Returns by Future Regime (6M):")
    for regime in ['bull', 'bear', 'neutral']:
        regime_data = df[df['Future_Regime_6M'] == regime]
        if len(regime_data) > 0:
            avg_return = regime_data['Forward_6M_Return'].mean()
            print(f"  {regime.capitalize():8s}: {avg_return * 100:+6.2f}%")

    print("=" * 80 + "\n")

    # Return cleaned DataFrame with only relevant columns
    return df[[
        'Date',
        'Ref_Index',
        'Forward_3M_Return',
        'Forward_6M_Return',
        'Future_Regime_3M',
        'Future_Regime_6M'
    ]]


def get_future_regime_at_date(df_forward_regimes, date, horizon='6M'):
    """
    Get the future regime classification for a specific date.

    Parameters:
      df_forward_regimes: DataFrame from classify_forward_regimes()
      date: Date to look up
      horizon: '3M' or '6M' for 3-month or 6-month forward regime

    Returns:
      String: 'bull', 'bear', 'neutral', or 'unknown'
    """
    date = pd.to_datetime(date)

    regime_row = df_forward_regimes[df_forward_regimes['Date'] == date]

    if regime_row.empty:
        return 'unknown'

    if horizon == '3M':
        return regime_row.iloc[0]['Future_Regime_3M']
    elif horizon == '6M':
        return regime_row.iloc[0]['Future_Regime_6M']
    else:
        raise ValueError(f"Invalid horizon: {horizon}. Use '3M' or '6M'")


def get_future_return_at_date(df_forward_regimes, date, horizon='6M'):
    """
    Get the actual forward return for a specific date.

    Parameters:
      df_forward_regimes: DataFrame from classify_forward_regimes()
      date: Date to look up
      horizon: '3M' or '6M' for 3-month or 6-month forward return

    Returns:
      Float: Forward return (e.g., 0.15 for +15%), or NaN if not available
    """
    date = pd.to_datetime(date)

    regime_row = df_forward_regimes[df_forward_regimes['Date'] == date]

    if regime_row.empty:
        return np.nan

    if horizon == '3M':
        return regime_row.iloc[0]['Forward_3M_Return']
    elif horizon == '6M':
        return regime_row.iloc[0]['Forward_6M_Return']
    else:
        raise ValueError(f"Invalid horizon: {horizon}. Use '3M' or '6M'")


def analyze_regime_agreement(df_forward_regimes):
    """
    Analyze how often 3M and 6M forward regimes agree.

    Helps understand if short-term and medium-term outlook are aligned.

    Parameters:
      df_forward_regimes: DataFrame from classify_forward_regimes()

    Returns:
      DataFrame with regime agreement statistics
    """
    df = df_forward_regimes.copy()

    # Filter out 'unknown' regimes
    valid_data = df[
        (df['Future_Regime_3M'] != 'unknown') &
        (df['Future_Regime_6M'] != 'unknown')
        ].copy()

    if valid_data.empty:
        print("No valid data for regime agreement analysis")
        return pd.DataFrame()

    # Create agreement flag
    valid_data['Regimes_Agree'] = (
            valid_data['Future_Regime_3M'] == valid_data['Future_Regime_6M']
    )

    # Calculate agreement rate
    agreement_rate = valid_data['Regimes_Agree'].mean()

    print(f"\nForward Regime Agreement Analysis:")
    print(f"  3M and 6M regimes agree: {agreement_rate * 100:.1f}% of the time")

    # Cross-tabulation
    crosstab = pd.crosstab(
        valid_data['Future_Regime_3M'],
        valid_data['Future_Regime_6M'],
        margins=True
    )

    print(f"\nCross-tabulation (3M vs 6M):")
    print(crosstab)

    # Identify disagreement patterns
    disagreements = valid_data[~valid_data['Regimes_Agree']].copy()

    if not disagreements.empty:
        print(f"\nDisagreement Patterns ({len(disagreements)} occurrences):")
        pattern_counts = disagreements.groupby(['Future_Regime_3M', 'Future_Regime_6M']).size()
        for (regime_3m, regime_6m), count in pattern_counts.items():
            pct = count / len(disagreements) * 100
            print(f"  3M={regime_3m:8s} → 6M={regime_6m:8s}: {count:4d} times ({pct:.1f}%)")

    return valid_data[['Date', 'Future_Regime_3M', 'Future_Regime_6M', 'Regimes_Agree']]


def compare_backward_vs_forward_regimes(df_backward_regimes, df_forward_regimes):
    """
    Compare backward-looking (current) regime vs forward-looking (future) regime.

    This shows whether current market conditions predict future conditions.

    Parameters:
      df_backward_regimes: DataFrame from regime_classifier.py (current regime)
      df_forward_regimes: DataFrame from classify_forward_regimes() (future regime)

    Returns:
      DataFrame with both regime classifications merged
    """
    # Merge on Date
    merged = df_backward_regimes[['Date', 'Regime']].merge(
        df_forward_regimes[['Date', 'Future_Regime_6M']],
        on='Date',
        how='inner'
    )

    merged = merged.rename(columns={'Regime': 'Current_Regime'})

    # Filter out unknowns
    valid_data = merged[merged['Future_Regime_6M'] != 'unknown'].copy()

    if valid_data.empty:
        print("No valid data for backward vs forward comparison")
        return pd.DataFrame()

    # Create agreement flag
    valid_data['Same_Regime'] = (
            valid_data['Current_Regime'] == valid_data['Future_Regime_6M']
    )

    persistence_rate = valid_data['Same_Regime'].mean()

    print(f"\nCurrent vs Future Regime Analysis:")
    print(f"  Current regime persists 6M ahead: {persistence_rate * 100:.1f}% of the time")

    # Cross-tabulation
    crosstab = pd.crosstab(
        valid_data['Current_Regime'],
        valid_data['Future_Regime_6M'],
        margins=True
    )

    print(f"\nCross-tabulation (Current vs Future):")
    print(crosstab)

    return valid_data


def identify_regime_turning_points(df_forward_regimes, horizon='6M'):
    """
    Identify dates where the future regime differs from recent past regime.

    These are critical decision points where strategy selection matters most.

    Parameters:
      df_forward_regimes: DataFrame from classify_forward_regimes()
      horizon: '3M' or '6M'

    Returns:
      DataFrame with regime turning points
    """
    df = df_forward_regimes.copy()

    regime_col = f'Future_Regime_{horizon}'

    # Get previous future regime (shifted by same window to avoid overlap)
    if horizon == '3M':
        shift_days = 63
    else:
        shift_days = 126

    df['Previous_Future_Regime'] = df[regime_col].shift(shift_days)

    # Identify turning points
    df['Regime_Change'] = (
            (df[regime_col] != df['Previous_Future_Regime']) &
            (df[regime_col] != 'unknown') &
            (df['Previous_Future_Regime'].notna())
    )

    turning_points = df[df['Regime_Change']].copy()

    if not turning_points.empty:
        print(f"\nRegime Turning Points ({horizon} horizon):")
        print(f"  Found {len(turning_points)} regime transitions")

        for _, row in turning_points.head(10).iterrows():
            print(f"  {row['Date'].strftime('%Y-%m-%d')}: "
                  f"{row['Previous_Future_Regime']} → {row[regime_col]}")

        if len(turning_points) > 10:
            print(f"  ... and {len(turning_points) - 10} more")

    return turning_points[[
        'Date', 'Previous_Future_Regime', regime_col,
        'Forward_3M_Return', 'Forward_6M_Return'
    ]]