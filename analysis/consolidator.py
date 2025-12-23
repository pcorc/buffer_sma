"""
Results consolidation - convert raw backtest results into summary tables.
"""

import pandas as pd
import numpy as np


def consolidate_results(results_list):
    """
    Convert list of result dicts into a single summary DataFrame.

    Takes the raw output from batch_runner and creates a clean summary table
    with one row per backtest showing all key metrics.

    Parameters:
      results_list: List of result dicts from run_single_ticker_backtest

    Returns:
      DataFrame with one row per backtest, sorted by vs_bufr_excess (descending)
    """
    print("\n" + "=" * 80)
    print("CONSOLIDATING RESULTS")
    print("=" * 80)

    if not results_list:
        print("ERROR: No results to consolidate!")
        return pd.DataFrame()

    summary_records = []

    for result in results_list:
        # Format trigger params as string
        trigger_params_str = str(result['trigger_params'])

        record = {
            'launch_month': result['launch_month'],
            'trigger_type': result['trigger_type'],
            'trigger_params': trigger_params_str,
            'selection_algo': result['selection_algo'],
            'start_date': result['start_date'],
            'end_date': result['end_date'],
            'num_trades': result['num_trades'],

            'strategy_return': result['strategy_total_return'],
            'strategy_ann_return': result['strategy_ann_return'],
            'strategy_sharpe': result['strategy_sharpe'],
            'strategy_volatility': result['strategy_volatility'],
            'strategy_max_dd': result['strategy_max_dd'],

            'spy_return': result['spy_total_return'],
            'spy_ann_return': result['spy_ann_return'],

            'bufr_return': result['bufr_total_return'],
            'bufr_ann_return': result['bufr_ann_return'],

            'hold_return': result['hold_total_return'],
            'hold_ann_return': result['hold_ann_return'],

            'vs_spy_excess': result['vs_spy_excess'],
            'vs_bufr_excess': result['vs_bufr_excess'],
            'vs_hold_excess': result['vs_hold_excess']
        }

        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)

    # Sort by vs_bufr_excess (descending) to see best performers first
    summary_df = summary_df.sort_values('vs_bufr_excess', ascending=False).reset_index(drop=True)

    print(f"Consolidation complete: {len(summary_df)} backtests summarized")
    print("=" * 80 + "\n")

    return summary_df


def create_performance_summary(summary_df):
    """
    Create high-level performance summary statistics.

    Parameters:
      summary_df: Consolidated results DataFrame

    Returns:
      Dict with summary statistics
    """
    if summary_df.empty:
        return {}

    summary_stats = {
        'total_backtests': len(summary_df),
        'best_vs_bufr': {
            'excess_return': summary_df['vs_bufr_excess'].max(),
            'launch_month': summary_df.loc[summary_df['vs_bufr_excess'].idxmax(), 'launch_month'],
            'trigger': summary_df.loc[summary_df['vs_bufr_excess'].idxmax(), 'trigger_type'],
            'selection': summary_df.loc[summary_df['vs_bufr_excess'].idxmax(), 'selection_algo']
        },
        'avg_vs_bufr': summary_df['vs_bufr_excess'].mean(),
        'median_vs_bufr': summary_df['vs_bufr_excess'].median(),
        'pct_beat_bufr': (summary_df['vs_bufr_excess'] > 0).sum() / len(summary_df) * 100,
        'pct_beat_spy': (summary_df['vs_spy_excess'] > 0).sum() / len(summary_df) * 100,
        'pct_beat_hold': (summary_df['vs_hold_excess'] > 0).sum() / len(summary_df) * 100,
        'avg_sharpe': summary_df['strategy_sharpe'].mean(),
        'avg_trades': summary_df['num_trades'].mean()
    }

    return summary_stats


def summarize_by_launch_month(summary_df):
    """
    Aggregate performance by launch month.

    Parameters:
      summary_df: Consolidated results DataFrame

    Returns:
      DataFrame with performance metrics by launch month
    """
    if summary_df.empty:
        return pd.DataFrame()

    month_summary = summary_df.groupby('launch_month').agg({
        'vs_bufr_excess': ['mean', 'median', 'max', 'min'],
        'strategy_sharpe': 'mean',
        'num_trades': 'mean',
        'strategy_return': 'mean'
    }).round(4)

    month_summary.columns = ['_'.join(col).strip() for col in month_summary.columns.values]
    month_summary = month_summary.reset_index()

    return month_summary


def summarize_by_trigger_type(summary_df):
    """
    Aggregate performance by trigger type.

    Parameters:
      summary_df: Consolidated results DataFrame

    Returns:
      DataFrame with performance metrics by trigger type
    """
    if summary_df.empty:
        return pd.DataFrame()

    trigger_summary = summary_df.groupby('trigger_type').agg({
        'vs_bufr_excess': ['mean', 'median', 'max', 'min'],
        'strategy_sharpe': 'mean',
        'num_trades': 'mean',
        'strategy_return': 'mean'
    }).round(4)

    trigger_summary.columns = ['_'.join(col).strip() for col in trigger_summary.columns.values]
    trigger_summary = trigger_summary.reset_index()

    return trigger_summary


def summarize_by_selection_algo(summary_df):
    """
    Aggregate performance by selection algorithm.

    Parameters:
      summary_df: Consolidated results DataFrame

    Returns:
      DataFrame with performance metrics by selection algorithm
    """
    if summary_df.empty:
        return pd.DataFrame()

    selection_summary = summary_df.groupby('selection_algo').agg({
        'vs_bufr_excess': ['mean', 'median', 'max', 'min'],
        'strategy_sharpe': 'mean',
        'num_trades': 'mean',
        'strategy_return': 'mean'
    }).round(4)

    selection_summary.columns = ['_'.join(col).strip() for col in selection_summary.columns.values]
    selection_summary = selection_summary.reset_index()

    return selection_summary