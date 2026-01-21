"""
Intent-based performance analysis and best strategy selection.

This module groups strategies by their intent (bullish/bearish/neutral/cost_optimized)
and identifies the best performers within each category using multiple criteria.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from config.strategy_intents import get_strategy_intent, INTENT_DESCRIPTIONS


# =============================================================================
# INTENT CLASSIFICATION
# =============================================================================

def add_strategy_intent_column(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'strategy_intent' column to summary results DataFrame.

    Uses the intent mapping system to classify each strategy combination.

    Parameters:
        summary_df: Consolidated results DataFrame with trigger/selection info

    Returns:
        DataFrame with new 'strategy_intent' column added
    """
    print("\n" + "=" * 80)
    print("ADDING STRATEGY INTENT CLASSIFICATIONS")
    print("=" * 80)

    intents = []
    errors = []

    for idx, row in summary_df.iterrows():
        try:
            # Parse trigger params from string back to dict
            trigger_params = eval(row['trigger_params'])

            intent = get_strategy_intent(
                trigger_type=row['trigger_type'],
                trigger_params=trigger_params,
                selection_algo=row['selection_algo']
            )
            intents.append(intent)

        except Exception as e:
            intents.append('unknown')
            errors.append({
                'index': idx,
                'trigger': row['trigger_type'],
                'selection': row['selection_algo'],
                'error': str(e)
            })

    summary_df['strategy_intent'] = intents

    # Report intent distribution
    intent_counts = summary_df['strategy_intent'].value_counts()
    print(f"\nIntent Distribution:")
    for intent in ['bullish', 'bearish', 'neutral', 'cost_optimized', 'unknown']:
        count = intent_counts.get(intent, 0)
        if count > 0:
            print(f"  {intent.upper():15s}: {count:4d} simulations")

    if errors:
        print(f"\n⚠️  Warning: {len(errors)} strategies could not be classified")
        for error in errors[:3]:
            print(f"  • Index {error['index']}: {error['trigger']} + {error['selection']}")

    print("=" * 80 + "\n")

    return summary_df


# =============================================================================
# INTENT GROUPING
# =============================================================================

def create_intent_groups(summary_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group all results by strategy intent.

    Parameters:
        summary_df: Summary DataFrame with 'strategy_intent' column

    Returns:
        Dict with intent names as keys, DataFrames as values
    """
    print("\n" + "=" * 80)
    print("GROUPING STRATEGIES BY INTENT")
    print("=" * 80)

    intent_groups = {}

    for intent in ['bullish', 'bearish', 'neutral', 'cost_optimized']:
        intent_df = summary_df[summary_df['strategy_intent'] == intent].copy()

        if not intent_df.empty:
            # Sort by vs_bufr_excess descending
            intent_df = intent_df.sort_values('vs_bufr_excess', ascending=False).reset_index(drop=True)
            intent_groups[intent] = intent_df

            print(f"\n{intent.upper():15s}: {len(intent_df)} strategies")
            print(f"  Description: {INTENT_DESCRIPTIONS[intent]}")
            print(f"  Best vs BUFR: {intent_df['vs_bufr_excess'].max() * 100:+.2f}%")
            print(f"  Avg vs BUFR: {intent_df['vs_bufr_excess'].mean() * 100:+.2f}%")
            print(f"  Worst vs BUFR: {intent_df['vs_bufr_excess'].min() * 100:+.2f}%")

    print("\n" + "=" * 80 + "\n")

    return intent_groups


# =============================================================================
# BEST STRATEGY SELECTION
# =============================================================================

def find_best_by_criteria(df: pd.DataFrame, criteria: str) -> pd.Series:
    """
    Find best strategy in DataFrame by specific criteria.

    Parameters:
        df: DataFrame of strategies
        criteria: One of 'sharpe', 'max_dd', 'max_return', 'min_return', 'vs_bufr'

    Returns:
        Series representing the best strategy row
    """
    if df.empty:
        return pd.Series()

    if criteria == 'sharpe':
        # Highest Sharpe ratio
        return df.loc[df['strategy_sharpe'].idxmax()]

    elif criteria == 'max_dd':
        # Least negative drawdown (closest to 0)
        return df.loc[df['strategy_max_dd'].idxmax()]

    elif criteria == 'max_return':
        # Highest total return
        return df.loc[df['strategy_return'].idxmax()]

    elif criteria == 'min_return':
        # Highest minimum (most conservative floor)
        # This requires min return within the strategy, not in this df
        # For now, use strategy with smallest max_dd as proxy
        return df.loc[df['strategy_max_dd'].idxmax()]

    elif criteria == 'vs_bufr':
        # Best vs BUFR excess return
        return df.loc[df['vs_bufr_excess'].idxmax()]

    else:
        raise ValueError(f"Unknown criteria: {criteria}")


def select_best_by_intent(
        summary_df: pd.DataFrame,
        regime_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Select best strategy within each intent based on appropriate criteria.

    Selection logic:
    - Bullish: Best Sharpe ratio in bull markets (if regime data available)
              Otherwise, best overall Sharpe
    - Bearish: Best max drawdown protection in bear markets
              Otherwise, least negative max drawdown overall
    - Neutral: Best Sharpe ratio overall
    - Cost-optimized: Best Sharpe ratio overall

    Parameters:
        summary_df: Summary DataFrame with 'strategy_intent' column
        regime_df: Optional regime-specific analysis DataFrame

    Returns:
        DataFrame with one row per intent (4 rows total)
    """
    print("\n" + "=" * 80)
    print("SELECTING BEST STRATEGY PER INTENT")
    print("=" * 80)

    best_strategies = []

    # Group by intent
    intent_groups = create_intent_groups(summary_df)

    for intent, intent_df in intent_groups.items():
        if intent_df.empty:
            continue

        print(f"\n{intent.upper()} STRATEGIES:")
        print("-" * 80)

        # Determine selection criteria based on intent
        if intent == 'bullish':
            if regime_df is not None and not regime_df.empty:
                # Best Sharpe in bull markets
                bull_regime = regime_df[regime_df['regime'] == 'bull'].copy()
                if not bull_regime.empty:
                    # Create strategy ID and merge
                    bull_regime['strategy_id'] = (
                            bull_regime['launch_month'] + '_' +
                            bull_regime['trigger_type'] + '_' +
                            bull_regime['selection_algo']
                    )
                    intent_df['strategy_id'] = (
                            intent_df['launch_month'] + '_' +
                            intent_df['trigger_type'] + '_' +
                            intent_df['selection_algo']
                    )

                    # Get Sharpe-like metric from bull regime (using returns)
                    bull_perf = bull_regime.groupby('strategy_id')['strategy_return'].mean()
                    intent_df['bull_return'] = intent_df['strategy_id'].map(bull_perf)

                    best = intent_df.loc[intent_df['bull_return'].idxmax()]
                    criteria_used = 'best_return_in_bull_markets'
                else:
                    best = find_best_by_criteria(intent_df, 'sharpe')
                    criteria_used = 'best_sharpe_overall'
            else:
                best = find_best_by_criteria(intent_df, 'sharpe')
                criteria_used = 'best_sharpe_overall'

        elif intent == 'bearish':
            if regime_df is not None and not regime_df.empty:
                # Best drawdown protection in bear markets
                bear_regime = regime_df[regime_df['regime'] == 'bear'].copy()
                if not bear_regime.empty:
                    bear_regime['strategy_id'] = (
                            bear_regime['launch_month'] + '_' +
                            bear_regime['trigger_type'] + '_' +
                            bear_regime['selection_algo']
                    )
                    intent_df['strategy_id'] = (
                            intent_df['launch_month'] + '_' +
                            intent_df['trigger_type'] + '_' +
                            intent_df['selection_algo']
                    )

                    # Get return in bear markets (least negative is best)
                    bear_perf = bear_regime.groupby('strategy_id')['strategy_return'].mean()
                    intent_df['bear_return'] = intent_df['strategy_id'].map(bear_perf)

                    best = intent_df.loc[intent_df['bear_return'].idxmax()]
                    criteria_used = 'best_return_in_bear_markets'
                else:
                    best = find_best_by_criteria(intent_df, 'max_dd')
                    criteria_used = 'least_negative_max_dd'
            else:
                best = find_best_by_criteria(intent_df, 'max_dd')
                criteria_used = 'least_negative_max_dd'

        elif intent == 'neutral':
            best = find_best_by_criteria(intent_df, 'sharpe')
            criteria_used = 'best_sharpe_overall'

        elif intent == 'cost_optimized':
            best = find_best_by_criteria(intent_df, 'sharpe')
            criteria_used = 'best_sharpe_overall'

        else:
            continue

        # Add selection metadata
        best_dict = best.to_dict()
        best_dict['selection_criteria'] = criteria_used
        best_strategies.append(best_dict)

        print(f"Selected Strategy:")
        print(f"  Launch Month: {best['launch_month']}")
        print(f"  Trigger: {best['trigger_type']}")
        print(f"  Selection: {best['selection_algo']}")
        print(f"  Criteria: {criteria_used}")
        print(f"  Return: {best['strategy_return'] * 100:+.2f}%")
        print(f"  Sharpe: {best['strategy_sharpe']:.2f}")
        print(f"  Max DD: {best['strategy_max_dd'] * 100:.2f}%")
        print(f"  vs BUFR: {best['vs_bufr_excess'] * 100:+.2f}%")

    print("\n" + "=" * 80 + "\n")

    best_df = pd.DataFrame(best_strategies)
    return best_df


# =============================================================================
# PERFORMANCE COMPARISON
# =============================================================================

def compare_intent_performance(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare average performance metrics across different intents.

    Parameters:
        summary_df: Summary DataFrame with 'strategy_intent' column

    Returns:
        DataFrame with average metrics by intent
    """
    if 'strategy_intent' not in summary_df.columns:
        raise ValueError("summary_df must have 'strategy_intent' column")

    comparison = summary_df.groupby('strategy_intent').agg({
        'strategy_return': ['mean', 'median', 'std'],
        'strategy_sharpe': ['mean', 'median'],
        'strategy_max_dd': ['mean', 'min', 'max'],
        'vs_bufr_excess': ['mean', 'median'],
        'vs_spy_excess': ['mean', 'median'],
        'num_trades': ['mean', 'median']
    }).round(4)

    # Flatten column names
    comparison.columns = ['_'.join(col).strip() for col in comparison.columns.values]
    comparison = comparison.reset_index()

    return comparison


def create_intent_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create high-level summary table showing key stats per intent.

    Parameters:
        summary_df: Summary DataFrame with 'strategy_intent' column

    Returns:
        DataFrame with one row per intent showing key statistics
    """
    summary_records = []

    for intent in ['bullish', 'bearish', 'neutral', 'cost_optimized']:
        intent_df = summary_df[summary_df['strategy_intent'] == intent]

        if intent_df.empty:
            continue

        record = {
            'intent': intent,
            'count': len(intent_df),
            'avg_return': intent_df['strategy_return'].mean(),
            'avg_sharpe': intent_df['strategy_sharpe'].mean(),
            'avg_max_dd': intent_df['strategy_max_dd'].mean(),
            'avg_vs_bufr': intent_df['vs_bufr_excess'].mean(),
            'pct_beat_bufr': (intent_df['vs_bufr_excess'] > 0).sum() / len(intent_df) * 100,
            'best_return': intent_df['strategy_return'].max(),
            'worst_return': intent_df['strategy_return'].min(),
            'best_sharpe': intent_df['strategy_sharpe'].max(),
            'best_vs_bufr': intent_df['vs_bufr_excess'].max()
        }

        summary_records.append(record)

    summary_table = pd.DataFrame(summary_records)

    return summary_table


# =============================================================================
# REGIME-SPECIFIC INTENT ANALYSIS
# =============================================================================

def analyze_intent_by_regime(
        summary_df: pd.DataFrame,
        regime_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze how each intent performs across different market regimes.

    Parameters:
        summary_df: Summary DataFrame with 'strategy_intent' column
        regime_df: Regime-specific analysis DataFrame

    Returns:
        DataFrame with intent × regime performance matrix
    """
    if regime_df.empty:
        return pd.DataFrame()

    # Add intent to regime_df
    regime_df['strategy_id'] = (
            regime_df['launch_month'] + '_' +
            regime_df['trigger_type'] + '_' +
            regime_df['selection_algo']
    )

    summary_df['strategy_id'] = (
            summary_df['launch_month'] + '_' +
            summary_df['trigger_type'] + '_' +
            summary_df['selection_algo']
    )

    # Merge intent info
    regime_with_intent = regime_df.merge(
        summary_df[['strategy_id', 'strategy_intent']],
        on='strategy_id',
        how='left'
    )

    # Aggregate by intent and regime
    intent_regime_perf = regime_with_intent.groupby(['strategy_intent', 'regime']).agg({
        'strategy_return': ['mean', 'median'],
        'vs_bufr_excess': ['mean', 'median'],
        'num_trades': 'sum'
    }).round(4)

    intent_regime_perf.columns = ['_'.join(col).strip() for col in intent_regime_perf.columns.values]
    intent_regime_perf = intent_regime_perf.reset_index()

    return intent_regime_perf


# =============================================================================
# VALIDATION
# =============================================================================

def validate_intent_performance(
        summary_df: pd.DataFrame,
        regime_df: pd.DataFrame
) -> Dict[str, bool]:
    """
    Validate that strategies perform as intended in their target regimes.

    For example:
    - Do bullish strategies actually outperform in bull markets?
    - Do bearish strategies protect better in bear markets?

    Parameters:
        summary_df: Summary DataFrame with 'strategy_intent' column
        regime_df: Regime-specific analysis DataFrame

    Returns:
        Dict with validation results
    """
    if regime_df.empty:
        return {'validation_skipped': True}

    intent_regime = analyze_intent_by_regime(summary_df, regime_df)

    validation = {}

    # Check bullish strategies in bull markets
    bullish_bull = intent_regime[
        (intent_regime['strategy_intent'] == 'bullish') &
        (intent_regime['regime'] == 'bull')
        ]

    if not bullish_bull.empty:
        bullish_bull_return = bullish_bull['strategy_return_mean'].values[0]

        # Compare to other intents in bull markets
        all_bull = intent_regime[intent_regime['regime'] == 'bull']
        avg_bull_return = all_bull['strategy_return_mean'].mean()

        validation['bullish_in_bull'] = bullish_bull_return >= avg_bull_return

    # Check bearish strategies in bear markets
    bearish_bear = intent_regime[
        (intent_regime['strategy_intent'] == 'bearish') &
        (intent_regime['regime'] == 'bear')
        ]

    if not bearish_bear.empty:
        bearish_bear_return = bearish_bear['strategy_return_mean'].values[0]

        # In bear markets, less negative is better
        all_bear = intent_regime[intent_regime['regime'] == 'bear']
        avg_bear_return = all_bear['strategy_return_mean'].mean()

        validation['bearish_in_bear'] = bearish_bear_return >= avg_bear_return

    return validation


def select_best_by_intent_two_strategies(
        summary_df: pd.DataFrame,
        regime_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Select best strategy for bullish and defensive only (2 strategies).

    Returns:
        DataFrame with 2 rows (bullish and defensive)
    """
    best_strategies = []

    intent_groups = create_intent_groups(summary_df)

    # BULLISH
    if 'bullish' in intent_groups and not intent_groups['bullish'].empty:
        bullish = intent_groups['bullish'].nlargest(1, 'vs_bufr_excess').iloc[0]
        best_strategies.append(bullish.to_dict())

    # DEFENSIVE (use bearish intent, select by best Sharpe)
    if 'bearish' in intent_groups and not intent_groups['bearish'].empty:
        defensive = intent_groups['bearish'].nlargest(1, 'strategy_sharpe').iloc[0]
        best_strategies.append(defensive.to_dict())

    return pd.DataFrame(best_strategies)