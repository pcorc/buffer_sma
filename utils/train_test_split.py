"""
Train/Test split utilities for forward regime analysis validation.

Enables testing whether optimal strategies identified in training period
actually perform well in out-of-sample test period.
"""

import pandas as pd
from datetime import datetime


class TrainTestConfig:
    """Configuration for train/test split."""

    def __init__(self, train_end_date, test_start_date=None):
        """
        Initialize train/test configuration.

        Parameters:
          train_end_date: Last date of training period (str or datetime)
          test_start_date: First date of test period (optional, defaults to day after train_end_date)
        """
        self.train_end_date = pd.to_datetime(train_end_date)

        if test_start_date:
            self.test_start_date = pd.to_datetime(test_start_date)
        else:
            # Default: test starts day after training ends
            self.test_start_date = self.train_end_date + pd.Timedelta(days=1)

    def __repr__(self):
        return (f"TrainTestConfig(train_end={self.train_end_date.date()}, "
                f"test_start={self.test_start_date.date()})")


def split_results_by_period(results_list, train_test_config):
    """
    Split backtest results into training and test periods.

    Parameters:
      results_list: List of result dicts from run_single_ticker_backtest
      train_test_config: TrainTestConfig instance

    Returns:
      Tuple: (train_results, test_results)
        - train_results: List of results where start_date <= train_end_date
        - test_results: List of results where start_date >= test_start_date
    """
    train_results = []
    test_results = []

    for result in results_list:
        start_date = result['start_date']

        if start_date <= train_test_config.train_end_date:
            train_results.append(result)

        if start_date >= train_test_config.test_start_date:
            test_results.append(result)

    return train_results, test_results


def split_dataframe_by_period(df, date_column, train_test_config):
    """
    Split a DataFrame into training and test periods.

    Parameters:
      df: DataFrame with date column
      date_column: Name of date column
      train_test_config: TrainTestConfig instance

    Returns:
      Tuple: (train_df, test_df)
    """
    df[date_column] = pd.to_datetime(df[date_column])

    train_df = df[df[date_column] <= train_test_config.train_end_date].copy()
    test_df = df[df[date_column] >= train_test_config.test_start_date].copy()

    return train_df, test_df


def compare_train_test_optimal_strategies(train_optimal, test_optimal, horizon='6M'):
    """
    Compare optimal strategies identified in training vs test performance.

    Answers: "Did the strategies we identified as 'optimal' in training
    actually perform well in the test period?"

    Parameters:
      train_optimal: Dict from summarize_optimal_strategies() on training data
      test_optimal: Dict from summarize_optimal_strategies() on test data
      horizon: '3M' or '6M'

    Returns:
      DataFrame comparing train vs test performance for each regime
    """
    comparison_records = []

    for regime in ['bull', 'bear', 'neutral']:
        if regime not in train_optimal:
            continue

        train_top = train_optimal[regime]

        if train_top.empty:
            continue

        # For each strategy in training top 10
        for _, train_row in train_top.iterrows():
            strategy_id = (
                f"{train_row['launch_month']}_"
                f"{train_row['trigger_type']}_"
                f"{train_row['selection_algo']}"
            )

            # Find this strategy in test results
            if regime in test_optimal:
                test_top = test_optimal[regime]

                test_match = test_top[
                    (test_top['launch_month'] == train_row['launch_month']) &
                    (test_top['trigger_type'] == train_row['trigger_type']) &
                    (test_top['selection_algo'] == train_row['selection_algo'])
                    ]

                if not test_match.empty:
                    test_row = test_match.iloc[0]

                    record = {
                        'future_regime': regime,
                        'launch_month': train_row['launch_month'],
                        'trigger_type': train_row['trigger_type'],
                        'selection_algo': train_row['selection_algo'],
                        'strategy_intent': train_row['strategy_intent'],
                        'train_rank': int(train_row['rank']),
                        'train_excess_vs_bufr': train_row['excess_vs_bufr_mean'],
                        'train_observations': int(train_row['num_observations']),
                        'test_rank': int(test_row['rank']),
                        'test_excess_vs_bufr': test_row['excess_vs_bufr_mean'],
                        'test_observations': int(test_row['num_observations']),
                        'rank_change': int(test_row['rank']) - int(train_row['rank']),
                        'performance_delta': (
                                test_row['excess_vs_bufr_mean'] -
                                train_row['excess_vs_bufr_mean']
                        )
                    }

                    comparison_records.append(record)

    if comparison_records:
        comparison_df = pd.DataFrame(comparison_records)

        # Sort by training rank (best first)
        comparison_df = comparison_df.sort_values(
            ['future_regime', 'train_rank']
        ).reset_index(drop=True)

        return comparison_df
    else:
        return pd.DataFrame()


def analyze_strategy_stability(comparison_df, top_n=5):
    """
    Analyze how stable top strategies are from train to test.

    Parameters:
      comparison_df: DataFrame from compare_train_test_optimal_strategies()
      top_n: Number of top training strategies to analyze

    Returns:
      Dict with stability metrics by regime
    """
    if comparison_df.empty:
        return {}

    stability_metrics = {}

    for regime in ['bull', 'bear', 'neutral']:
        regime_data = comparison_df[
            (comparison_df['future_regime'] == regime) &
            (comparison_df['train_rank'] <= top_n)
            ].copy()

        if regime_data.empty:
            continue

        # Calculate metrics
        avg_rank_change = regime_data['rank_change'].mean()
        avg_perf_delta = regime_data['performance_delta'].mean()

        # How many top N strategies stayed in top N?
        stayed_top_n = (regime_data['test_rank'] <= top_n).sum()

        # How many improved in test?
        improved_count = (regime_data['test_excess_vs_bufr'] >
                          regime_data['train_excess_vs_bufr']).sum()

        stability_metrics[regime] = {
            'analyzed_count': len(regime_data),
            'stayed_in_top_n': stayed_top_n,
            'stayed_in_top_n_pct': stayed_top_n / len(regime_data) * 100,
            'improved_in_test': improved_count,
            'improved_pct': improved_count / len(regime_data) * 100,
            'avg_rank_change': avg_rank_change,
            'avg_performance_delta': avg_perf_delta
        }

    return stability_metrics


def identify_consistently_optimal_strategies(comparison_df, top_n=10):
    """
    Find strategies that performed well in BOTH train and test periods.

    Parameters:
      comparison_df: DataFrame from compare_train_test_optimal_strategies()
      top_n: Rank threshold (e.g., top 10 in both periods)

    Returns:
      DataFrame with strategies that were top N in both train and test
    """
    if comparison_df.empty:
        return pd.DataFrame()

    # Filter to strategies that were top N in both periods
    consistent = comparison_df[
        (comparison_df['train_rank'] <= top_n) &
        (comparison_df['test_rank'] <= top_n)
        ].copy()

    if consistent.empty:
        return pd.DataFrame()

    # Calculate combined score (lower is better)
    consistent['combined_rank'] = consistent['train_rank'] + consistent['test_rank']
    consistent['avg_excess_vs_bufr'] = (
                                               consistent['train_excess_vs_bufr'] + consistent['test_excess_vs_bufr']
                                       ) / 2

    # Sort by combined rank
    consistent = consistent.sort_values('combined_rank').reset_index(drop=True)

    return consistent


def print_train_test_summary(stability_metrics, horizon='6M'):
    """
    Print formatted summary of train/test stability analysis.

    Parameters:
      stability_metrics: Dict from analyze_strategy_stability()
      horizon: '3M' or '6M'
    """
    print(f"\n{'=' * 80}")
    print(f"TRAIN vs TEST STABILITY ANALYSIS ({horizon} HORIZON)")
    print(f"{'=' * 80}\n")

    for regime in ['bull', 'bear', 'neutral']:
        if regime not in stability_metrics:
            continue

        metrics = stability_metrics[regime]

        print(f"\n{regime.upper()} MARKET:")
        print("-" * 80)
        print(f"  Strategies analyzed: {metrics['analyzed_count']}")
        print(f"  Stayed in top {metrics['analyzed_count']}: "
              f"{metrics['stayed_in_top_n']} ({metrics['stayed_in_top_n_pct']:.1f}%)")
        print(f"  Improved in test: "
              f"{metrics['improved_in_test']} ({metrics['improved_pct']:.1f}%)")
        print(f"  Avg rank change: {metrics['avg_rank_change']:+.1f}")
        print(f"  Avg performance delta: {metrics['avg_performance_delta'] * 100:+.2f}%")

    print(f"\n{'=' * 80}\n")


def get_validation_recommendations(comparison_df, stability_metrics, top_n=5):
    """
    Generate recommendations based on train/test validation.

    Parameters:
      comparison_df: DataFrame from compare_train_test_optimal_strategies()
      stability_metrics: Dict from analyze_strategy_stability()
      top_n: Number of top strategies to recommend

    Returns:
      Dict with recommendations by regime
    """
    recommendations = {}

    for regime in ['bull', 'bear', 'neutral']:
        if regime not in stability_metrics:
            continue

        regime_data = comparison_df[
            comparison_df['future_regime'] == regime
            ].copy()

        if regime_data.empty:
            continue

        # Get consistently top performers
        consistent = regime_data[
            (regime_data['train_rank'] <= top_n) &
            (regime_data['test_rank'] <= top_n)
            ].copy()

        # Strategies to trust (performed well in both periods)
        trust_list = []
        if not consistent.empty:
            for _, row in consistent.head(3).iterrows():
                trust_list.append({
                    'launch_month': row['launch_month'],
                    'trigger_type': row['trigger_type'],
                    'selection_algo': row['selection_algo'],
                    'train_rank': int(row['train_rank']),
                    'test_rank': int(row['test_rank']),
                    'avg_excess': (row['train_excess_vs_bufr'] +
                                   row['test_excess_vs_bufr']) / 2
                })

        # Strategies to be cautious about (good in train, poor in test)
        cautious_list = []
        declining = regime_data[
            (regime_data['train_rank'] <= top_n) &
            (regime_data['test_rank'] > top_n * 2)
            ].copy()

        if not declining.empty:
            for _, row in declining.head(3).iterrows():
                cautious_list.append({
                    'launch_month': row['launch_month'],
                    'trigger_type': row['trigger_type'],
                    'selection_algo': row['selection_algo'],
                    'train_rank': int(row['train_rank']),
                    'test_rank': int(row['test_rank']),
                    'performance_declined': True
                })

        recommendations[regime] = {
            'trust': trust_list,
            'caution': cautious_list,
            'stability_score': stability_metrics[regime]['stayed_in_top_n_pct']
        }

    return recommendations