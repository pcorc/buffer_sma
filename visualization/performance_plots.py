"""
Consolidated Performance Visualization Module

Creates professional charts for backtesting analysis with intelligent plot selection
based on batch type, data availability, and strategy composition.

Master Function: generate_batch_visualizations()
- Auto-detects which plots are relevant
- Handles missing data gracefully
- Adapts to single-intent vs multi-intent scenarios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import ast

# =============================================================================
# COLOR SCHEMES
# =============================================================================

INTENT_COLORS = {
    'bullish': {
        'light': '#81C784',  # Light green for all strategies
        'best': '#2E7D32',   # Dark green for best (thick line)
        'fill': '#A5D6A7'    # For confidence bands if needed
    },
    'bearish': {
        'light': '#E57373',  # Light red
        'best': '#C62828',   # Dark red
        'fill': '#EF9A9A'
    },
    'neutral': {
        'light': '#64B5F6',  # Light blue
        'best': '#1565C0',   # Dark blue
        'fill': '#90CAF9'
    },
    'cost_optimized': {
        'light': '#FFB74D',  # Light orange
        'best': '#F57C00',   # Dark orange
        'fill': '#FFCC80'
    }
}

BENCHMARK_COLORS = {
    'SPY': '#757575',   # Gray
    'BUFR': '#9E9E9E',  # Light gray
    'Hold': '#BDBDBD'   # Lighter gray
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_earliest_benchmark_data(results_list: List[Dict]) -> pd.DataFrame:
    """
    Get benchmark NAV data from the earliest-starting strategy.

    Ensures benchmarks are plotted from the same start date as
    the earliest strategy, avoiding visual misalignment.
    """
    if not results_list:
        return pd.DataFrame()

    # Find result with earliest start date
    earliest_result = min(results_list,
                          key=lambda r: r['daily_performance']['Date'].min())

    earliest_date = earliest_result['daily_performance']['Date'].min()
    print(f"    Using benchmark data from {earliest_date.date()} (earliest strategy)")

    return earliest_result['daily_performance']


def create_strategy_id(row):
    """Create unique ID including threshold parameter."""

    # Extract threshold from trigger_params
    try:
        params = ast.literal_eval(row['trigger_params']) if isinstance(row['trigger_params'], str) else row['trigger_params']
        threshold = params.get('threshold', 'none')
    except:
        threshold = 'none'

    # Include threshold in ID
    return f"{row['launch_month']}_{row['trigger_type']}_{row['selection_algo']}_{threshold}"

def _classify_strategy_intent(selection_algo: str) -> str:
    """
    Fallback intent classification when strategy_intent column not available.

    Used for batch tests that don't have explicit intent assignment.
    """
    bullish_algos = [
        'select_cap_utilization_lowest',
        'select_remaining_cap_highest',
        'select_most_recent_launch'
    ]

    bearish_algos = [
        'select_downside_buffer_highest',
        'select_cap_utilization_highest'
    ]

    if selection_algo in bullish_algos:
        return 'bullish'
    elif selection_algo in bearish_algos:
        return 'bearish'
    else:
        return 'neutral'


def _ensure_intent_column(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure strategy_intent column exists, create if missing."""
    if 'strategy_intent' not in summary_df.columns:
        summary_df = summary_df.copy()
        summary_df['strategy_intent'] = summary_df['selection_algo'].apply(_classify_strategy_intent)
    return summary_df


def _find_best_strategies(summary_df: pd.DataFrame) -> Dict:
    """
    Find best strategies by intent and criteria.

    Handles both single-intent and multi-intent scenarios.
    """
    summary_df = _ensure_intent_column(summary_df)

    best = {}

    # Get unique intents present
    intents = summary_df['strategy_intent'].unique()

    for intent in intents:
        intent_df = summary_df[summary_df['strategy_intent'] == intent]

        if not intent_df.empty:
            # Best by Sharpe ratio
            best[f'{intent}_sharpe'] = intent_df.loc[intent_df['strategy_sharpe'].idxmax()]

            # Best by return
            best[f'{intent}_return'] = intent_df.loc[intent_df['strategy_return'].idxmax()]

            # For bearish, also track best drawdown protection
            if intent == 'bearish':
                best[f'{intent}_drawdown'] = intent_df.loc[intent_df['strategy_max_dd'].idxmax()]

    # Overall best vs BUFR
    if not summary_df.empty:
        best['vs_bufr_best'] = summary_df.loc[summary_df['vs_bufr_excess'].idxmax()]

    return best


def _format_strategy_details(row, include_metrics: bool = True) -> str:
    """
    Format strategy details for text box display.

    Parameters:
        row: Strategy row (Series or dict)
        include_metrics: Include return/sharpe metrics

    Returns:
        Formatted multi-line string
    """
    import ast

    # Extract trigger params
    if isinstance(row.get('trigger_params'), str):
        try:
            params = ast.literal_eval(row['trigger_params'])
        except:
            params = {}
    else:
        params = row.get('trigger_params', {})

    # Format trigger with params
    trigger_type = row['trigger_type']

    if 'threshold' in params:
        threshold_pct = int(params['threshold'] * 100)
        if 'cap_utilization' in trigger_type:
            trigger_str = f"cap_utilization_threshold ({threshold_pct}%)"
        elif 'remaining_cap' in trigger_type:
            trigger_str = f"remaining_cap_threshold ({threshold_pct}%)"
        elif 'downside' in trigger_type:
            trigger_str = f"downside_buffer_threshold ({threshold_pct}%)"
        elif 'ref_asset' in trigger_type:
            trigger_str = f"ref_asset_return_threshold ({threshold_pct:+d}%)"
        else:
            trigger_str = f"{trigger_type} ({threshold_pct}%)"
    elif 'frequency' in params:
        freq = params['frequency']
        trigger_str = f"rebalance_time_period ({freq})"
    else:
        trigger_str = trigger_type

    # Format selection
    selection = row['selection_algo']

    # Build output
    lines = [
        f"Launch: {row['launch_month']}",
        f"  {trigger_str}",
        f"  {selection}"
    ]

    if include_metrics:
        ret = row.get('strategy_return', 0) * 100
        sharpe = row.get('strategy_sharpe', 0)
        lines.append(f"  Return: {ret:+.1f}% | Sharpe: {sharpe:.2f}")

    return '\n'.join(lines)

# =============================================================================
# PLOT 1: ALL STRATEGIES WITH BEST HIGHLIGHTED (BATCH VERSION)
# =============================================================================

def plot_all_strategies_with_best(
        results_list: List[Dict],
        summary_df: pd.DataFrame,
        output_dir: str
) -> str:
    """
    All strategies grouped by intent with best performers highlighted.

    Adapts to single-intent or multi-intent batches.
    """
    print("\n  Generating: All Strategies with Best Highlighted...")

    summary_df = _ensure_intent_column(summary_df)
    best = _find_best_strategies(summary_df)

    fig, ax = plt.subplots(figsize=(16, 9))

    # Track which intents we've plotted (for legend)
    plotted_intents = set()

    # Get best strategy IDs
    best_bullish_id = _create_strategy_id(best['bullish_sharpe']) if 'bullish_sharpe' in best else None
    best_bearish_id = _create_strategy_id(best['bearish_sharpe']) if 'bearish_sharpe' in best else None

    # Plot all strategies
    for result in results_list:
        result_id = _create_strategy_id(result)
        intent = _classify_strategy_intent(result['selection_algo'])

        daily = result['daily_performance']

        is_best_bullish = (result_id == best_bullish_id) if best_bullish_id else False
        is_best_bearish = (result_id == best_bearish_id) if best_bearish_id else False

        if is_best_bullish:
            ax.plot(daily['Date'], daily['Strategy_NAV'],
                    color=INTENT_COLORS['bullish']['best'], linewidth=3, alpha=1.0,
                    label='Best Bullish (Sharpe)', zorder=10)
        elif is_best_bearish:
            ax.plot(daily['Date'], daily['Strategy_NAV'],
                    color=INTENT_COLORS['bearish']['best'], linewidth=3, alpha=1.0,
                    label='Best Bearish (Sharpe)', zorder=10)
        else:
            # Plot as background strategy
            if intent not in plotted_intents:
                label = f'{intent.capitalize()} Strategies'
                plotted_intents.add(intent)
            else:
                label = None

            color = INTENT_COLORS.get(intent, {'light': '#CCCCCC'})['light']
            ax.plot(daily['Date'], daily['Strategy_NAV'],
                    color=color, linewidth=0.5, alpha=0.3,
                    label=label, zorder=1)

    # Plot benchmarks
    if results_list:
        benchmark = results_list[0]['daily_performance']
        ax.plot(benchmark['Date'], benchmark['SPY_NAV'],
                color=BENCHMARK_COLORS['SPY'], linewidth=2, linestyle='--',
                alpha=0.7, label='SPY', zorder=5)
        ax.plot(benchmark['Date'], benchmark['BUFR_NAV'],
                color=BENCHMARK_COLORS['BUFR'], linewidth=2, linestyle=':',
                alpha=0.7, label='BUFR', zorder=5)

    # Formatting
    ax.set_title('All Strategies: Best Performers Highlighted',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('NAV (Normalized to 100)', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    filename = 'all_strategies_with_best.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {filename}")
    return filepath


# =============================================================================
# PLOT 3: THRESHOLD COMPARISON (SPECIALIZED FOR BATCH 0)
# =============================================================================

def plot_threshold_comparison(
        summary_df: pd.DataFrame,
        output_dir: str
) -> Optional[str]:
    """
    Threshold analysis: performance by launch month and average.

    Only relevant for batches testing cap_utilization_threshold.
    """
    import ast

    print("\n  Generating: Threshold Comparison...")

    # Extract threshold values
    def extract_threshold(params_str):
        try:
            params_dict = ast.literal_eval(params_str)
            return params_dict.get('threshold', None)
        except:
            return None

    summary_df_copy = summary_df.copy()
    summary_df_copy['threshold_value'] = summary_df_copy['trigger_params'].apply(extract_threshold)

    # Filter to threshold strategies
    threshold_data = summary_df_copy[
        (summary_df_copy['trigger_type'] == 'cap_utilization_threshold') &
        (summary_df_copy['selection_algo'] == 'select_most_recent_launch') &
        (summary_df_copy['threshold_value'].notna())
    ].copy()

    if threshold_data.empty:
        print("    ⚠ No threshold data - skipping")
        return None

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    unique_thresholds = sorted(threshold_data['threshold_value'].unique())

    # Color mapping
    color_map = {
        0.25: '#2ca02c',
        0.40: '#17becf',
        0.50: '#1f77b4',
        0.70: '#bcbd22',
        0.75: '#ff7f0e',
        0.90: '#d62728'
    }

    # LEFT PANEL: Line chart by month
    for threshold in unique_thresholds:
        thresh_data = threshold_data[threshold_data['threshold_value'] == threshold]
        month_data = thresh_data.groupby('launch_month')['vs_bufr_excess'].mean().sort_index()

        color = color_map.get(threshold, '#333333')
        linewidth = 2.5 if threshold in [0.75, 0.90] else 2
        alpha = 0.9 if threshold in [0.75, 0.90] else 0.7

        ax1.plot(month_data.index, month_data.values * 100,
                 marker='o', linewidth=linewidth, markersize=8,
                 color=color, label=f'{int(threshold * 100)}% Threshold',
                 alpha=alpha, zorder=10 if threshold == 0.75 else 5)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax1.set_xlabel('Launch Month', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Excess Return vs BUFR (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Cap Utilization Threshold Performance by Launch Month',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.2)

    # RIGHT PANEL: Average bars
    avg_by_threshold = threshold_data.groupby('threshold_value')['vs_bufr_excess'].mean().sort_index()
    bar_colors = [color_map.get(t, '#333333') for t in avg_by_threshold.index]

    bars = ax2.barh(range(len(avg_by_threshold)), avg_by_threshold.values * 100,
                    color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_yticks(range(len(avg_by_threshold)))
    ax2.set_yticklabels([f'{int(t * 100)}%' for t in avg_by_threshold.index])
    ax2.set_xlabel('Average Excess vs BUFR (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Performance by Threshold', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax2.grid(True, alpha=0.2, axis='x')

    # Add value labels
    for bar, val in zip(bars, avg_by_threshold.values * 100):
        x_pos = val + 0.3 if val > 0 else val - 0.3
        ha = 'left' if val > 0 else 'right'
        ax2.text(x_pos, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center', ha=ha, fontsize=10, fontweight='bold')

    plt.tight_layout()

    filename = 'threshold_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {filename}")
    return filepath


def plot_threshold_performance_with_table(
        summary_df: pd.DataFrame,
        output_dir: str
) -> Optional[str]:
    """
    Comprehensive threshold analysis with integrated performance table.

    Shows:
    1. Bar chart of average excess vs BUFR by threshold
    2. Integrated table with detailed statistics
    3. Visual ranking highlighting 90% underperformance

    Only relevant for batches testing cap_utilization_threshold.

    Parameters:
        summary_df: Summary DataFrame with all results
        output_dir: Directory to save plot

    Returns:
        Filepath if successful, None otherwise
    """
    import ast

    print("\n  Generating: Threshold Performance Analysis with Table...")

    # Extract threshold values
    def extract_threshold(params_str):
        try:
            params_dict = ast.literal_eval(params_str)
            return params_dict.get('threshold', None)
        except:
            return None

    summary_df_copy = summary_df.copy()
    summary_df_copy['threshold_value'] = summary_df_copy['trigger_params'].apply(extract_threshold)

    # Filter to threshold strategies with correct selection algo
    threshold_data = summary_df_copy[
        (summary_df_copy['trigger_type'] == 'cap_utilization_threshold') &
        (summary_df_copy['selection_algo'] == 'select_most_recent_launch') &
        (summary_df_copy['threshold_value'].notna())
        ].copy()

    if threshold_data.empty:
        print("    ⚠ No threshold data - skipping")
        return None

    # Calculate statistics by threshold
    stats_by_threshold = []

    for threshold in sorted(threshold_data['threshold_value'].unique()):
        thresh_data = threshold_data[threshold_data['threshold_value'] == threshold]

        avg_excess = thresh_data['vs_bufr_excess'].mean()
        std_excess = thresh_data['vs_bufr_excess'].std()
        min_excess = thresh_data['vs_bufr_excess'].min()
        max_excess = thresh_data['vs_bufr_excess'].max()
        num_beating = (thresh_data['vs_bufr_excess'] > 0).sum()
        total_months = len(thresh_data)

        stats_by_threshold.append({
            'threshold': threshold,
            'avg_excess': avg_excess,
            'std_excess': std_excess,
            'min_excess': min_excess,
            'max_excess': max_excess,
            'num_beating': num_beating,
            'total_months': total_months,
            'pct_beating': (num_beating / total_months * 100) if total_months > 0 else 0
        })

    stats_df = pd.DataFrame(stats_by_threshold)
    stats_df = stats_df.sort_values('avg_excess', ascending=False)  # Best to worst

    # Print summary to console
    print("\n  Threshold Performance Summary:")
    print("  " + "=" * 70)
    for _, row in stats_df.iterrows():
        print(f"  {int(row['threshold'] * 100):3d}% | "
              f"Avg: {row['avg_excess'] * 100:+6.2f}% | "
              f"Std: {row['std_excess'] * 100:5.2f}% | "
              f"Beat BUFR: {row['num_beating']:.0f}/{row['total_months']:.0f} months")
    print("  " + "=" * 70)

    # Create figure with adjusted layout
    fig = plt.figure(figsize=(14, 10))

    # Top: Bar chart (70% of figure height)
    ax_bar = plt.subplot2grid((10, 1), (0, 0), rowspan=6)

    # Bottom: Table (30% of figure height)
    ax_table = plt.subplot2grid((10, 1), (7, 0), rowspan=3)
    ax_table.axis('off')

    # =========================================================================
    # BAR CHART
    # =========================================================================

    thresholds_pct = [int(t * 100) for t in stats_df['threshold']]
    avg_excess_pct = stats_df['avg_excess'].values * 100

    # Color coding: highlight 90% as red, best as green
    colors = []
    for i, threshold in enumerate(stats_df['threshold']):
        if i == 0:  # Best performer
            colors.append('#2E7D32')  # Dark green
        elif threshold == 0.90:  # 90% threshold
            colors.append('#D32F2F')  # Red
        else:
            colors.append('#1976D2')  # Blue

    bars = ax_bar.bar(range(len(thresholds_pct)), avg_excess_pct,
                      color=colors, alpha=0.85, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, val in zip(bars, avg_excess_pct):
        height = bar.get_height()
        y_pos = height + 0.15 if height > 0 else height - 0.15
        va = 'bottom' if height > 0 else 'top'

        ax_bar.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{val:+.2f}%',
                    ha='center', va=va, fontsize=12, fontweight='bold')

    # Formatting
    ax_bar.set_xticks(range(len(thresholds_pct)))
    ax_bar.set_xticklabels([f'{t}%' for t in thresholds_pct], fontsize=12, fontweight='bold')
    ax_bar.set_ylabel('Average Excess Return vs BUFR (%)', fontsize=13, fontweight='bold')
    ax_bar.set_title('Cap Utilization Threshold Performance Analysis (Averaged Across 12 Months)',
                     fontsize=15, fontweight='bold', pad=20)
    ax_bar.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax_bar.grid(True, alpha=0.3, axis='y', linestyle=':')

    # Add ranking labels
    for i, (bar, rank) in enumerate(zip(bars, range(1, len(bars) + 1))):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, ax_bar.get_ylim()[1] * 0.95,
                    f'Rank #{rank}',
                    ha='center', va='top', fontsize=10, fontweight='bold',
                    color='white' if i == 0 or stats_df.iloc[i]['threshold'] == 0.90 else 'black',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor=colors[i],
                              edgecolor='black',
                              linewidth=1.5,
                              alpha=0.9))

    # =========================================================================
    # PERFORMANCE TABLE
    # =========================================================================

    # Prepare table data
    table_data = []
    table_data.append(['Threshold', 'Avg vs BUFR', 'Std Dev', 'Min', 'Max', 'Months Beat BUFR'])

    for _, row in stats_df.iterrows():
        table_data.append([
            f"{int(row['threshold'] * 100)}%",
            f"{row['avg_excess'] * 100:+.2f}%",
            f"{row['std_excess'] * 100:.2f}%",
            f"{row['min_excess'] * 100:+.2f}%",
            f"{row['max_excess'] * 100:+.2f}%",
            f"{int(row['num_beating'])}/{int(row['total_months'])}"
        ])

    # Create table
    table = ax_table.table(cellText=table_data,
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0, 1, 1])

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Header row styling
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#366092')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
        cell.set_edgecolor('black')
        cell.set_linewidth(2)

    # Data row styling
    for i in range(1, len(table_data)):
        # Color code by ranking
        row_color = colors[i - 1]
        row_alpha = 0.15

        for j in range(len(table_data[i])):
            cell = table[(i, j)]
            cell.set_facecolor(row_color)
            cell.set_alpha(row_alpha)
            cell.set_edgecolor('black')
            cell.set_linewidth(1)

            # Bold the avg vs BUFR column (index 1)
            if j == 1:
                cell.set_text_props(weight='bold', fontsize=10)

    plt.tight_layout()

    filename = 'threshold_performance_analysis.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {filename}")

    # Also export table as CSV for reference
    stats_df_export = stats_df.copy()
    stats_df_export['threshold'] = (stats_df_export['threshold'] * 100).astype(int).astype(str) + '%'
    stats_df_export['avg_excess'] = (stats_df_export['avg_excess'] * 100).round(2)
    stats_df_export['std_excess'] = (stats_df_export['std_excess'] * 100).round(2)
    stats_df_export['min_excess'] = (stats_df_export['min_excess'] * 100).round(2)
    stats_df_export['max_excess'] = (stats_df_export['max_excess'] * 100).round(2)

    csv_path = os.path.join(output_dir, 'threshold_performance_table.csv')
    stats_df_export.to_csv(csv_path, index=False)
    print(f"    ✓ Saved: threshold_performance_table.csv")

    return filepath


# =============================================================================
# PLOT 4: REGIME PERFORMANCE WITH BUFR
# =============================================================================

def plot_regime_performance_with_bufr(
        future_regime_df: pd.DataFrame,
        optimal_strategies: Dict,
        output_dir: str,
        horizon: str = '6M'  # ✅ NEW PARAMETER: '3M' or '6M'
) -> Optional[str]:
    """
    Strategy performance across bull/neutral/bear markets vs BUFR.

    Shows how bullish and defensive strategies perform in each regime.

    Parameters:
        future_regime_df: Forward regime analysis DataFrame
        optimal_strategies: Dict of optimal strategies by regime
        output_dir: Directory to save plots
        horizon: '3M' or '6M' forward window
    """
    print(f"\n  Generating: Regime Performance with BUFR ({horizon})...")

    if future_regime_df.empty or not optimal_strategies:
        print("    ⚠ No regime data - skipping")
        return None

    # Get top strategies
    bull_strats = optimal_strategies.get('bull')
    bear_strats = optimal_strategies.get('bear')

    if bull_strats is None or bull_strats.empty or bear_strats is None or bear_strats.empty:
        print("    ⚠ Missing bullish or defensive strategy - skipping")
        return None

    strategy_bullish = bull_strats.iloc[0]
    strategy_defensive = bear_strats.iloc[0]

    # ✅ Use horizon parameter to select columns
    if horizon == '3M':
        regime_col = 'future_regime_3m'
        strat_return_col = 'forward_3m_return'
        bufr_return_col = 'bufr_forward_3m_return'
    else:  # 6M
        regime_col = 'future_regime_6m'
        strat_return_col = 'forward_6m_return'
        bufr_return_col = 'bufr_forward_6m_return'

    # Verify columns exist
    if regime_col not in future_regime_df.columns:
        print(f"    ⚠ Column '{regime_col}' not found - skipping")
        return None

    if strat_return_col not in future_regime_df.columns:
        print(f"    ⚠ Column '{strat_return_col}' not found - skipping")
        return None

    if bufr_return_col not in future_regime_df.columns:
        print(f"    ⚠ Column '{bufr_return_col}' not found - skipping")
        return None

    print(f"    Using columns: '{strat_return_col}', '{bufr_return_col}'")

    # Collect performance data
    regimes = ['bull', 'bear']
    bullish_returns = []
    defensive_returns = []
    bufr_returns = []

    for regime in regimes:
        regime_data = future_regime_df[future_regime_df[regime_col] == regime]

        if regime_data.empty:
            bullish_returns.append(0)
            defensive_returns.append(0)
            bufr_returns.append(0)
            continue

        # Get bullish strategy performance
        bullish_data = regime_data[
            (regime_data['launch_month'] == strategy_bullish['launch_month']) &
            (regime_data['trigger_type'] == strategy_bullish['trigger_type']) &
            (regime_data['selection_algo'] == strategy_bullish['selection_algo'])
            ]

        # Get defensive strategy performance
        defensive_data = regime_data[
            (regime_data['launch_month'] == strategy_defensive['launch_month']) &
            (regime_data['trigger_type'] == strategy_defensive['trigger_type']) &
            (regime_data['selection_algo'] == strategy_defensive['selection_algo'])
            ]

        # Extract returns
        if not bullish_data.empty:
            bull_ret = bullish_data[strat_return_col].mean()
            bufr_ret = bullish_data[bufr_return_col].mean()
            bullish_returns.append(bull_ret)
            bufr_returns.append(bufr_ret)
        else:
            bullish_returns.append(0)
            bufr_returns.append(0)

        if not defensive_data.empty:
            def_ret = defensive_data[strat_return_col].mean()
            defensive_returns.append(def_ret)
        else:
            defensive_returns.append(0)

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(regimes))
    width = 0.25

    bars1 = ax.bar(x - width, np.array(bullish_returns) * 100, width,
                   label='Bullish Strategy', color='#2ca02c', alpha=0.85,
                   edgecolor='black', linewidth=1.5)

    bars2 = ax.bar(x, np.array(defensive_returns) * 100, width,
                   label='Defensive Strategy', color='#d62728', alpha=0.85,
                   edgecolor='black', linewidth=1.5)

    bars3 = ax.bar(x + width, np.array(bufr_returns) * 100, width,
                   label='BUFR Benchmark', color='#ff7f0e', alpha=0.7,
                   edgecolor='black', linewidth=1.5)

    # Add value labels
    def add_value_labels(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if abs(height) < 0.5:
                continue

            y_pos = height + 0.5 if height >= 0 else height - 0.5
            va = 'bottom' if height >= 0 else 'top'

            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{val * 100:.1f}%',
                    ha='center', va=va, fontsize=10, fontweight='bold')

    add_value_labels(bars1, bullish_returns)
    add_value_labels(bars2, defensive_returns)
    add_value_labels(bars3, bufr_returns)

    # Formatting
    ax.set_xlabel(f'Market Regime ({horizon} Forward)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Strategy Return (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Strategy Performance Across Market Regimes ({horizon} Forward vs BUFR)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Bull', 'Bear'], fontsize=12)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.2, alpha=0.4)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    plt.tight_layout()

    filename = f'regime_performance_{horizon.lower()}_with_bufr.png'  # ✅ Horizon in filename
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {filename}")
    return filepath



def plot_threshold_regime_performance(
        future_regime_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        output_dir: str,
        horizon: str = '3M',
        trigger_type: str = 'cap_utilization_threshold',
        selection_algo: str = 'select_most_recent_launch'
) -> Optional[str]:
    """
    Show regime performance broken out by threshold level.

    Generic function that works with any threshold-based strategy.
    Shows how each threshold performs in bull vs bear markets.

    Parameters:
        future_regime_df: Forward regime analysis DataFrame
        summary_df: Summary DataFrame with all simulation results
        output_dir: Directory to save plots
        horizon: '3M' or '6M' forward window
        trigger_type: Type of threshold trigger to analyze
                     (e.g., 'cap_utilization_threshold', 'remaining_cap_threshold')
        selection_algo: Selection algorithm to filter for
                       (e.g., 'select_most_recent_launch', 'select_remaining_cap_highest')

    Returns:
        Filepath if successful, None otherwise
    """
    import ast

    print(f"\n  Generating: Threshold Regime Performance ({horizon})...")

    if future_regime_df.empty or summary_df.empty:
        print("    ⚠ No regime data - skipping")
        return None

    # Select columns based on horizon
    if horizon == '3M':
        regime_col = 'future_regime_3m'
        strat_return_col = 'forward_3m_return'
        bufr_return_col = 'bufr_forward_3m_return'
    else:  # 6M
        regime_col = 'future_regime_6m'
        strat_return_col = 'forward_6m_return'
        bufr_return_col = 'bufr_forward_6m_return'

    # Verify columns exist
    if regime_col not in future_regime_df.columns:
        print(f"    ⚠ Column '{regime_col}' not found - skipping")
        return None

    print(f"    Using columns: '{strat_return_col}', '{bufr_return_col}'")
    print(f"    Filtering: {trigger_type} + {selection_algo}")

    # Filter to specified threshold strategies
    threshold_strategies = summary_df[
        (summary_df['trigger_type'] == trigger_type) &
        (summary_df['selection_algo'] == selection_algo)
        ].copy()

    if threshold_strategies.empty:
        print(f"    ⚠ No strategies found with {trigger_type} + {selection_algo} - skipping")
        return None

    # Extract threshold values
    def extract_threshold(params_str):
        try:
            params = ast.literal_eval(params_str)
            return params.get('threshold', None)
        except:
            return None

    threshold_strategies['threshold_value'] = threshold_strategies['trigger_params'].apply(extract_threshold)
    threshold_strategies = threshold_strategies.dropna(subset=['threshold_value'])

    # Get unique thresholds and sort
    unique_thresholds = sorted(threshold_strategies['threshold_value'].unique())

    print(f"    Found {len(unique_thresholds)} thresholds: {[f'{int(t * 100)}%' for t in unique_thresholds]}")

    # Collect performance data for each threshold in each regime
    regimes = ['bull', 'bear']
    threshold_performance = {regime: {} for regime in regimes}
    bufr_performance = {}

    for regime in regimes:
        regime_data = future_regime_df[future_regime_df[regime_col] == regime]

        if regime_data.empty:
            print(f"    ⚠ No data for {regime} regime - skipping")
            continue

        # Get BUFR performance in this regime
        bufr_return = regime_data[bufr_return_col].mean()
        bufr_performance[regime] = bufr_return

        # Get performance for each threshold
        for threshold in unique_thresholds:
            # Find strategies with this threshold
            threshold_strats = threshold_strategies[
                threshold_strategies['threshold_value'] == threshold
                ]

            # Collect returns for all strategies with this threshold in this regime
            threshold_returns = []

            for _, strat_row in threshold_strats.iterrows():
                strat_data = regime_data[
                    (regime_data['launch_month'] == strat_row['launch_month']) &
                    (regime_data['trigger_type'] == strat_row['trigger_type']) &
                    (regime_data['selection_algo'] == strat_row['selection_algo'])
                    ]

                if not strat_data.empty:
                    threshold_returns.append(strat_data[strat_return_col].mean())

            # Average across all strategies with this threshold
            if threshold_returns:
                avg_return = np.mean(threshold_returns)
                threshold_performance[regime][threshold] = avg_return
            else:
                threshold_performance[regime][threshold] = 0

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    # Number of groups (thresholds + BUFR)
    n_thresholds = len(unique_thresholds)
    n_bars = n_thresholds + 1  # +1 for BUFR

    # X positions for each regime
    x = np.arange(len(regimes))
    width = 0.8 / n_bars  # Divide space among all bars

    # Color palette for thresholds (gradient from green to red)
    threshold_colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, n_thresholds))

    # Plot bars for each threshold
    for i, threshold in enumerate(unique_thresholds):
        threshold_pct = int(threshold * 100)

        returns_by_regime = [
            threshold_performance[regime].get(threshold, 0) * 100
            for regime in regimes
        ]

        offset = width * (i - n_bars / 2 + 0.5)

        bars = ax.bar(x + offset, returns_by_regime, width,
                      label=f'{threshold_pct}% Threshold',
                      color=threshold_colors[i], alpha=0.85,
                      edgecolor='black', linewidth=1.2)

        # Add value labels
        for bar, val in zip(bars, returns_by_regime):
            height = bar.get_height()
            if abs(height) < 0.3:  # Skip very small values
                continue

            y_pos = height + 0.3 if height >= 0 else height - 0.3
            va = 'bottom' if height >= 0 else 'top'

            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{val:.1f}%',
                    ha='center', va=va, fontsize=9, fontweight='bold')

    # Plot BUFR benchmark
    bufr_returns = [bufr_performance.get(regime, 0) * 100 for regime in regimes]
    offset = width * (n_thresholds - n_bars / 2 + 0.5)

    bufr_bars = ax.bar(x + offset, bufr_returns, width,
                       label='BUFR Benchmark',
                       color='#ff7f0e', alpha=0.7,
                       edgecolor='black', linewidth=1.2)

    # Add BUFR value labels
    for bar, val in zip(bufr_bars, bufr_returns):
        height = bar.get_height()
        if abs(height) < 0.3:
            continue

        y_pos = height + 0.3 if height >= 0 else height - 0.3
        va = 'bottom' if height >= 0 else 'top'

        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f'{val:.1f}%',
                ha='center', va=va, fontsize=9, fontweight='bold')

    # Create dynamic title based on trigger type
    trigger_display = trigger_type.replace('_threshold', '').replace('_', ' ').title()

    # Formatting
    ax.set_xlabel(f'Market Regime ({horizon} Forward)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Strategy Return (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'{trigger_display} Threshold Performance Across Market Regimes ({horizon} Forward)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Bull', 'Bear'], fontsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, ncol=2)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.2, alpha=0.4)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    plt.tight_layout()

    # Generic filename
    filename = f'threshold_regime_performance_{horizon.lower()}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    for threshold in unique_thresholds:
        bull_ret = threshold_performance['bull'].get(threshold, 0) * 100
        bear_ret = threshold_performance['bear'].get(threshold, 0) * 100
        spread = bull_ret - bear_ret

        print(f"    {int(threshold * 100):>3}%        {bull_ret:>+6.2f}%         {bear_ret:>+6.2f}%         {spread:>+6.2f}%")

    return filepath


# =============================================================================
# PLOT 5: PERFORMANCE METRICS BARS (INTENT-BASED)
# =============================================================================

def plot_performance_metrics_comparison(
        best_strategies_df: pd.DataFrame,
        output_dir: str
) -> Optional[str]:
    """
    Multi-panel bar chart comparing best strategies across metrics.

    Only generates if we have multiple intents to compare.
    """
    print("\n  Generating: Performance Metrics Comparison...")

    if best_strategies_df.empty or len(best_strategies_df) < 2:
        print("    ⚠ Need multiple strategies to compare - skipping")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Performance Metrics: Best Strategies by Intent',
                 fontsize=16, fontweight='bold')

    # Prepare data
    if 'strategy_intent' in best_strategies_df.columns:
        intents = best_strategies_df['strategy_intent'].values
        colors = [INTENT_COLORS[intent]['best'] for intent in intents]
        labels = [f"{intent.title()}\n({row['launch_month']})"
                  for intent, (_, row) in zip(intents, best_strategies_df.iterrows())]
    else:
        colors = ['#1f77b4'] * len(best_strategies_df)
        labels = [f"{row['launch_month']}" for _, row in best_strategies_df.iterrows()]

    # Subplot 1: Returns
    ax1 = axes[0, 0]
    returns = best_strategies_df['strategy_return'].values * 100
    bars1 = ax1.bar(labels, returns, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_title('Total Returns', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Return (%)', fontsize=11)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(axis='y', alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom' if height > 0 else 'top',
                 fontsize=9, fontweight='bold')

    # Subplot 2: Sharpe
    ax2 = axes[0, 1]
    sharpes = best_strategies_df['strategy_sharpe'].values
    bars2 = ax2.bar(labels, sharpes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_title('Sharpe Ratios', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sharpe Ratio', fontsize=11)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom' if height > 0 else 'top',
                 fontsize=9, fontweight='bold')

    # Subplot 3: Max Drawdown
    ax3 = axes[1, 0]
    drawdowns = best_strategies_df['strategy_max_dd'].values * 100
    bars3 = ax3.bar(labels, drawdowns, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Max Drawdown (%)', fontsize=11)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.grid(axis='y', alpha=0.3)

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%',
                 ha='center', va='top' if height < 0 else 'bottom',
                 fontsize=9, fontweight='bold')

    # Subplot 4: vs BUFR
    ax4 = axes[1, 1]
    excess = best_strategies_df['vs_bufr_excess'].values * 100
    bars4 = ax4.bar(labels, excess, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_title('Excess Return vs BUFR', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Excess Return (%)', fontsize=11)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax4.grid(axis='y', alpha=0.3)

    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom' if height > 0 else 'top',
                 fontsize=9, fontweight='bold')

    plt.tight_layout()

    filename = 'performance_metrics_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {filename}")
    return filepath


# =============================================================================
# MASTER FUNCTION: INTELLIGENT BATCH VISUALIZATION
# =============================================================================


def plot_batch8_normalized_nav(
        results_list: List[Dict],
        summary_df: pd.DataFrame,
        output_dir: str,
        top_n: int = 5
) -> Optional[str]:
    """
    Create normalized NAV plot for Batch 8 threshold analysis.

    All strategies start at NAV=100 on Day 0, plotted on relative time axis.
    Shows top 5 performers with detailed strategy information.

    Parameters:
        results_list: List of backtest result dicts
        summary_df: Summary DataFrame with performance metrics
        output_dir: Directory to save plot
        top_n: Number of top strategies to plot (default 5)

    Returns:
        Filepath if successful, None otherwise
    """
    import ast

    print(f"\n  Generating: Batch 8 Normalized NAV Plot (Top {top_n})...")

    if summary_df.empty or not results_list:
        print("    ⚠ No data available - skipping")
        return None

    # Get top N strategies by Sharpe ratio
    top_strategies = summary_df.nlargest(top_n, 'strategy_sharpe')

    # Create strategy IDs for matching
    def create_strategy_id(row):
        return f"{row['launch_month']}_{row['trigger_type']}_{row['selection_algo']}"

    top_strategy_ids = [create_strategy_id(row) for _, row in top_strategies.iterrows()]

    # Color palette for top 5 strategies
    colors = ['#2E7D32', '#1565C0', '#F57C00', '#7B1FA2', '#C62828']

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))

    # Track plotted strategies for legend and details
    plotted_strategies = []

    # Plot each top strategy
    for i, (idx, row) in enumerate(top_strategies.iterrows()):
        strategy_id = create_strategy_id(row)

        # Find matching result
        for result in results_list:
            result_id = create_strategy_id(result)

            if result_id == strategy_id:
                # Get daily performance data
                daily_nav = result['daily_performance'].copy()

                # Normalize to start at 100
                first_nav = daily_nav['Strategy_NAV'].iloc[0]
                daily_nav['Normalized_NAV'] = (daily_nav['Strategy_NAV'] / first_nav) * 100

                # Create relative time axis (days since inception)
                daily_nav['Days_Since_Inception'] = range(len(daily_nav))

                # Extract threshold for label
                try:
                    params = ast.literal_eval(row['trigger_params'])
                    threshold = int(params.get('threshold', 0) * 100)
                except:
                    threshold = 0

                # Create label
                label = f"#{i + 1}: {threshold}% Threshold ({row['launch_month']})"

                # Plot normalized NAV
                ax.plot(daily_nav['Days_Since_Inception'],
                        daily_nav['Normalized_NAV'],
                        color=colors[i], linewidth=2.5, alpha=0.9,
                        label=label, zorder=10 - i)

                # Store for details box
                plotted_strategies.append({
                    'rank': i + 1,
                    'row': row,
                    'color': colors[i],
                    'threshold': threshold,
                    'daily_nav': daily_nav
                })

                print(f"    Plotted Rank #{i + 1}: {threshold}% ({row['launch_month']}) - Sharpe: {row['strategy_sharpe']:.2f}")

                break

    # Plot benchmarks (normalized to same start point as strategies)
    if plotted_strategies:
        first_strategy_id = create_strategy_id(plotted_strategies[0]['row'])

        for result in results_list:
            if create_strategy_id(result) == first_strategy_id:
                benchmark_daily = result['daily_performance'].copy()

                # Normalize benchmarks to start at 100
                first_spy = benchmark_daily['SPY_NAV'].iloc[0]
                first_bufr = benchmark_daily['BUFR_NAV'].iloc[0]

                benchmark_daily['Normalized_SPY'] = (benchmark_daily['SPY_NAV'] / first_spy) * 100
                benchmark_daily['Normalized_BUFR'] = (benchmark_daily['BUFR_NAV'] / first_bufr) * 100
                benchmark_daily['Days_Since_Inception'] = range(len(benchmark_daily))

                # Plot benchmarks
                ax.plot(benchmark_daily['Days_Since_Inception'],
                        benchmark_daily['Normalized_SPY'],
                        color='#757575', linewidth=2, linestyle='--',
                        alpha=0.7, label='SPY', zorder=5)

                ax.plot(benchmark_daily['Days_Since_Inception'],
                        benchmark_daily['Normalized_BUFR'],
                        color='#9E9E9E', linewidth=2, linestyle=':',
                        alpha=0.7, label='BUFR', zorder=5)

                break

    # Formatting
    ax.set_xlabel('Days Since Inception', fontsize=13, fontweight='bold')
    ax.set_ylabel('Normalized NAV (Starting at 100)', fontsize=13, fontweight='bold')
    ax.set_title('Top 5 Cap Utilization Threshold Strategies - Normalized Performance',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.axhline(y=100, color='black', linestyle='-', linewidth=1, alpha=0.3)

    # Add strategy details box
    if plotted_strategies:
        textstr_lines = ['STRATEGY DETAILS\n' + '─' * 50]

        for strat_info in plotted_strategies:
            row = strat_info['row']
            rank = strat_info['rank']
            threshold = strat_info['threshold']

            textstr_lines.append('')  # Blank line
            textstr_lines.append(f'■ Rank #{rank}: {threshold}% Threshold')
            textstr_lines.append(f'  Launch: {row["launch_month"]}')
            textstr_lines.append(f'  Return: {row["strategy_return"] * 100:+.1f}% | Sharpe: {row["strategy_sharpe"]:.2f}')
            textstr_lines.append(f'  vs BUFR: {row["vs_bufr_excess"] * 100:+.1f}% | Trades: {int(row["num_trades"])}')

        textstr = '\n'.join(textstr_lines)

        # Position in bottom-right
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9,
                     edgecolor='black', linewidth=1.5)
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=props, family='monospace')

    plt.tight_layout()

    filename = 'batch8_normalized_nav_top5.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {filename}")
    return filepath


def generate_batch_visualizations(
        results_list: List[Dict],
        summary_df: pd.DataFrame,
        future_regime_df: pd.DataFrame,
        optimal_strategies: Dict,
        output_dir: str,
        batch_number: int = None,
        skip_all_strategies: bool = True
) -> Dict[str, str]:
    """
    Master function: Generate all relevant visualizations based on batch type.

    Intelligently selects which plots to generate based on:
    - Batch number (specialized plots for certain batches)
    - Data availability (regime data, optimal strategies)
    - Strategy composition (single vs multi-intent)

    Parameters:
        results_list: List of backtest result dicts
        summary_df: Summary DataFrame
        future_regime_df: Forward regime analysis DataFrame
        optimal_strategies: Dict of optimal strategies by regime
        output_dir: Directory to save plots
        batch_number: Optional batch identifier for specialized handling

    Returns:
        Dict mapping plot names to filepaths
    """

    os.makedirs(output_dir, exist_ok=True)

    generated_plots = {}

    # =========================================================================
    # CORE PLOTS
    # =========================================================================

    # Plot 1: All strategies with best highlighted (SUPPRESSED BY DEFAULT)
    if not skip_all_strategies:
        filepath = plot_all_strategies_with_best(results_list, summary_df, output_dir)
        if filepath:
            generated_plots['all_strategies'] = filepath

    # Plot 2: Best strategies comparison
    filepath = plot_best_strategies_comparison(results_list, summary_df, output_dir)
    if filepath:
        generated_plots['best_comparison'] = filepath

        # =========================================================================
        # SPECIALIZED PLOTS (Conditional)
        # =========================================================================

        # Threshold comparison (Batch 0 or any batch with threshold testing)
        if batch_number == 8:
            # Threshold performance bar chart + table
            filepath = plot_threshold_performance_with_table(summary_df, output_dir)
            if filepath:
                generated_plots['batch8_threshold_performance'] = filepath

            # Normalized NAV plot (top 5)
            filepath = plot_batch8_normalized_nav_ALIGNED_DETAILED_LEGEND(results_list, summary_df, output_dir, top_n=5)
            if filepath:
                generated_plots['normalized_nav_aligned'] = filepath

            filepath = plot_threshold_regime_performance(
                future_regime_df, summary_df, output_dir,
                horizon='3M',
                trigger_type='cap_utilization_threshold',
                selection_algo='select_most_recent_launch'
            )
            if filepath:
                generated_plots['threshold_regime_3m'] = filepath

            filepath = plot_threshold_regime_performance(
                future_regime_df, summary_df, output_dir,
                horizon='6M',
                trigger_type='cap_utilization_threshold',
                selection_algo='select_most_recent_launch'
            )
            if filepath:
                generated_plots['threshold_regime_6m'] = filepath

        elif batch_number == 0 or _has_threshold_strategies(summary_df):
            # Use original threshold comparison for other batches
            filepath = plot_threshold_comparison(summary_df, output_dir)
            if filepath:
                generated_plots['threshold_comparison'] = filepath

    # Regime performance (if we have regime data and optimal strategies)
    if not future_regime_df.empty and optimal_strategies:
        # Check if we have both bull and bear strategies
        has_bull = 'bull' in optimal_strategies and optimal_strategies['bull'] is not None and not optimal_strategies['bull'].empty
        has_bear = 'bear' in optimal_strategies and optimal_strategies['bear'] is not None and not optimal_strategies['bear'].empty

        if has_bull and has_bear:
            print(f"\n  Found optimal strategies for regime plot:")
            print(f"    Bull strategies: {len(optimal_strategies['bull'])}")
            print(f"    Bear strategies: {len(optimal_strategies['bear'])}")

            # ✅ Generate BOTH 3M and 6M charts
            for horizon in ['3M', '6M']:
                filepath = plot_regime_performance_with_bufr(
                    future_regime_df, optimal_strategies, output_dir, horizon=horizon
                )
                if filepath:
                    generated_plots[f'regime_performance_{horizon.lower()}'] = filepath
        else:
            print(f"\n  ⚠ Skipping regime performance plot:")
            print(f"    Bull strategies available: {has_bull}")
            print(f"    Bear strategies available: {has_bear}")

    # Performance metrics comparison (if multiple intents)
    summary_df_with_intent = _ensure_intent_column(summary_df)
    if summary_df_with_intent['strategy_intent'].nunique() > 1:
        best_strategies = _find_best_strategies(summary_df)
        if len(best_strategies) >= 2:
            # Create mini DataFrame of best strategies
            best_df_list = []
            for key, row in best_strategies.items():
                if key.endswith('_sharpe'):
                    best_df_list.append(row)

            if len(best_df_list) >= 2:
                best_df = pd.DataFrame(best_df_list)
                filepath = plot_performance_metrics_comparison(best_df, output_dir)
                if filepath:
                    generated_plots['metrics_comparison'] = filepath

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "=" * 80)
    print(f"VISUALIZATION COMPLETE: {len(generated_plots)} plots generated")
    print("=" * 80)

    return generated_plots


def _has_threshold_strategies(summary_df: pd.DataFrame) -> bool:
    """Check if summary contains threshold-based strategies."""
    if 'trigger_type' not in summary_df.columns:
        return False

    threshold_triggers = [
        'cap_utilization_threshold',
        'remaining_cap_threshold',
        'downside_before_buffer_threshold'
    ]

    return summary_df['trigger_type'].isin(threshold_triggers).any()


# =============================================================================
# BACKWARD COMPATIBILITY: Original function for main.py
# =============================================================================

def create_all_plots(
        results_list: List[Dict],
        summary_df: pd.DataFrame,
        best_strategies_df: pd.DataFrame,
        output_dir: str
):
    """
    Original function signature for backward compatibility with main.py.

    Delegates to generate_batch_visualizations with appropriate defaults.
    """
    # Convert best_strategies_df to optimal_strategies dict format
    optimal_strategies = {}

    if 'strategy_intent' in best_strategies_df.columns:
        for _, row in best_strategies_df.iterrows():
            intent = row['strategy_intent']
            if intent not in optimal_strategies:
                optimal_strategies[intent] = pd.DataFrame([row])

    return generate_batch_visualizations(
        results_list=results_list,
        summary_df=summary_df,
        future_regime_df=pd.DataFrame(),  # Empty - main.py doesn't use forward regimes
        optimal_strategies=optimal_strategies,
        output_dir=output_dir,
        batch_number=None
    )


# Add to visualization/performance_plots.py

def plot_cross_batch_comparison(
        all_summaries: Dict[int, pd.DataFrame],
        output_dir: str
):
    """
    Compare best strategies across different batches.

    Creates:
    1. Bar chart comparing key metrics across batches
    2. Summary table showing best strategy from each batch

    Parameters:
        all_summaries: Dict mapping batch_num -> summary DataFrame
        output_dir: Directory to save comparison plots
    """
    print("\n" + "=" * 80)
    print("GENERATING CROSS-BATCH COMPARISON")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Find best strategy from each batch (by Sharpe ratio)
    best_by_batch = {}

    for batch_num, summary_df in all_summaries.items():
        if summary_df.empty:
            continue

        best_idx = summary_df['strategy_sharpe'].idxmax()
        best_row = summary_df.loc[best_idx].copy()
        best_by_batch[batch_num] = best_row

        print(f"\nBatch {batch_num}: {best_row['batch_name']}")
        print(f"  Best strategy: {best_row['launch_month']} | {best_row['trigger_type']}")
        print(f"  Sharpe: {best_row['strategy_sharpe']:.2f}")
        print(f"  Return: {best_row['strategy_return'] * 100:.2f}%")
        print(f"  vs BUFR: {best_row['vs_bufr_excess'] * 100:+.2f}%")

    if not best_by_batch:
        print("⚠️ No valid batch results found")
        return

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(best_by_batch).T

    # =========================================================================
    # PLOT: Metrics Comparison Bar Chart
    # =========================================================================

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Best Strategy Comparison Across Batches', fontsize=16, fontweight='bold')

    colors = ['#2E7D32', '#1565C0', '#F57C00', '#E53935']
    batch_nums = list(best_by_batch.keys())
    batch_labels = [f"Batch {n}" for n in batch_nums]

    # Sharpe Ratio
    ax1 = axes[0, 0]
    sharpes = [best_by_batch[n]['strategy_sharpe'] for n in batch_nums]
    bars1 = ax1.bar(batch_labels, sharpes, color=colors[:len(batch_nums)])
    ax1.set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars1, sharpes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # Total Return
    ax2 = axes[0, 1]
    returns = [best_by_batch[n]['strategy_return'] * 100 for n in batch_nums]
    bars2 = ax2.bar(batch_labels, returns, color=colors[:len(batch_nums)])
    ax2.set_title('Total Return', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Return (%)')
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars2, returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Excess vs BUFR
    ax3 = axes[1, 0]
    excess_bufr = [best_by_batch[n]['vs_bufr_excess'] * 100 for n in batch_nums]
    bars3 = ax3.bar(batch_labels, excess_bufr, color=colors[:len(batch_nums)])
    ax3.set_title('Excess Return vs BUFR', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Excess Return (%)')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars3, excess_bufr):
        height = bar.get_height()
        va = 'bottom' if height > 0 else 'top'
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:+.1f}%', ha='center', va=va, fontweight='bold')

    # Max Drawdown
    ax4 = axes[1, 1]
    max_dds = [best_by_batch[n]['strategy_max_dd'] * 100 for n in batch_nums]
    bars4 = ax4.bar(batch_labels, max_dds, color=colors[:len(batch_nums)])
    ax4.set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Max Drawdown (%)')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax4.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars4, max_dds):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.1f}%', ha='center', va='top', fontweight='bold')

    plt.tight_layout()

    filename = 'cross_batch_metrics_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  ✓ Saved: {filename}")

    # =========================================================================
    # EXPORT: Summary Table
    # =========================================================================

    summary_table = comparison_df[[
        'batch_name', 'launch_month', 'trigger_type', 'selection_algo',
        'strategy_return', 'strategy_sharpe', 'strategy_max_dd',
        'vs_bufr_excess', 'vs_spy_excess', 'num_trades'
    ]].copy()

    summary_table.to_excel(
        os.path.join(output_dir, 'cross_batch_summary.xlsx'),
        index_label='Batch'
    )

    print(f"  ✓ Saved: cross_batch_summary.xlsx")

    print("\n" + "=" * 80)
    print(f"✅ CROSS-BATCH COMPARISON COMPLETE")
    print(f"   Output: {output_dir}")
    print("=" * 80 + "\n")


"""
Batch 7 Specialized Visualization: Four-Strategy Direct Comparison

Add this function to visualization/performance_plots.py
"""


def plot_batch_7_four_strategies(
        results_list: List[Dict],
        summary_df: pd.DataFrame,
        output_dir: str
) -> str:
    """
    Batch 7 specialized plot: Four specific strategies vs benchmarks.

    Shows all 4 strategies on one clean chart with benchmarks for direct comparison.
    Each strategy gets a distinct color and style based on its trigger/selection logic.

    Strategy Color Coding:
    - Green: Bullish (Remaining Cap 75% → Highest Cap)
    - Red: Bearish (Cap Util 75% → Highest Util)
    - Blue: Hybrid Aggressive (Cap Util 75% → Lowest Util)
    - Orange: Hybrid Defensive (Downside 50% → Lowest Util)

    Parameters:
        results_list: List of backtest results (should be 4 strategies)
        summary_df: Summary DataFrame
        output_dir: Directory to save plot

    Returns:
        Filepath to saved plot
    """
    print("\n  Generating: Batch 7 Four-Strategy Comparison...")

    if len(results_list) != 4:
        print(f"    ⚠ Expected 4 strategies, got {len(results_list)} - continuing anyway")

    fig, ax = plt.subplots(figsize=(18, 10))

    # Define colors and labels for the 4 strategies
    strategy_styles = [
        {
            'color': '#2E7D32',  # Dark green (bullish)
            'linestyle': '-',
            'linewidth': 3.0,
            'label': 'Strategy 1: RemCap 75% → Highest Cap',
            'short_label': 'RemCap→High'
        },
        {
            'color': '#D32F2F',  # Dark red (bearish)
            'linestyle': '-',
            'linewidth': 3.0,
            'label': 'Strategy 2: CapUtil 75% → Highest Util',
            'short_label': 'CapUtil→High'
        },
        {
            'color': '#1565C0',  # Dark blue (hybrid aggressive)
            'linestyle': '-',
            'linewidth': 3.0,
            'label': 'Strategy 3: CapUtil 75% → Lowest Util',
            'short_label': 'CapUtil→Low'
        },
        {
            'color': '#F57C00',  # Dark orange (hybrid defensive)
            'linestyle': '-',
            'linewidth': 3.0,
            'label': 'Strategy 4: Downside 50% → Lowest Util',
            'short_label': 'Downside→Low'
        }
    ]

    # Sort results to match expected order
    def sort_key(result):
        """Sort by trigger type and selection to match strategy_styles order."""
        trigger = result['trigger_type']
        selection = result['selection_algo']

        if trigger == 'remaining_cap_threshold':
            return 0
        elif trigger == 'cap_utilization_threshold' and selection == 'select_cap_utilization_highest':
            return 1
        elif trigger == 'cap_utilization_threshold' and selection == 'select_cap_utilization_lowest':
            return 2
        elif trigger == 'downside_before_buffer_threshold':
            return 3
        return 99

    sorted_results = sorted(results_list, key=sort_key)

    # Plot each strategy
    plotted_strategies = []

    for i, (result, style) in enumerate(zip(sorted_results, strategy_styles)):
        daily = result['daily_performance']

        # Plot strategy NAV
        ax.plot(daily['Date'], daily['Strategy_NAV'],
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                alpha=0.95,
                label=style['label'],
                zorder=10 + i)

        # Store performance data for text box
        plotted_strategies.append({
            'number': i + 1,
            'label': style['label'],
            'short_label': style['short_label'],
            'color': style['color'],
            'return': result['strategy_total_return'],
            'sharpe': result['strategy_sharpe'],
            'max_dd': result['strategy_max_dd'],
            'vs_bufr': result['vs_bufr_excess'],
            'vs_spy': result['vs_spy_excess'],
            'trades': result['num_trades'],
            'launch': result['launch_month'],
            'trigger': result['trigger_type'],
            'selection': result['selection_algo']
        })

    # Plot benchmarks
    benchmark = _get_earliest_benchmark_data(sorted_results)

    ax.plot(benchmark['Date'], benchmark['SPY_NAV'],
            color=BENCHMARK_COLORS['SPY'], linewidth=2.5, linestyle='--',
            alpha=0.75, label='SPY (Benchmark)', zorder=5)

    ax.plot(benchmark['Date'], benchmark['BUFR_NAV'],
            color=BENCHMARK_COLORS['BUFR'], linewidth=2.5, linestyle=':',
            alpha=0.75, label='BUFR (Benchmark)', zorder=5)

    # Add detailed performance text box
    textstr_lines = ['PERFORMANCE SUMMARY\n' + '═' * 55]

    # Find best performer by Sharpe
    best_sharpe_idx = max(range(len(plotted_strategies)),
                          key=lambda i: plotted_strategies[i]['sharpe'])

    # Find best vs BUFR
    best_bufr_idx = max(range(len(plotted_strategies)),
                        key=lambda i: plotted_strategies[i]['vs_bufr'])

    for i, strat in enumerate(plotted_strategies):
        if i > 0:
            textstr_lines.append('')  # Blank line between strategies

        # Add marker for best performers
        marker = ''
        if i == best_sharpe_idx and i == best_bufr_idx:
            marker = ' ★ BEST OVERALL'
        elif i == best_sharpe_idx:
            marker = ' ★ BEST SHARPE'
        elif i == best_bufr_idx:
            marker = ' ★ BEST VS BUFR'

        textstr_lines.append(f"Strategy {strat['number']}: {strat['short_label']}{marker}")
        textstr_lines.append(f"  Return:   {strat['return'] * 100:+7.2f}%  |  Sharpe: {strat['sharpe']:6.2f}")
        textstr_lines.append(f"  vs BUFR:  {strat['vs_bufr'] * 100:+7.2f}%  |  vs SPY: {strat['vs_spy'] * 100:+7.2f}%")
        textstr_lines.append(f"  Max DD:   {strat['max_dd'] * 100:7.2f}%  |  Trades: {strat['trades']:6.0f}")

    # Add benchmark performance
    textstr_lines.append('\n' + '─' * 55)
    textstr_lines.append('BENCHMARKS:')

    spy_return = (benchmark['SPY_NAV'].iloc[-1] / 100 - 1) * 100
    bufr_return = (benchmark['BUFR_NAV'].iloc[-1] / 100 - 1) * 100

    textstr_lines.append(f"  SPY:  {spy_return:+7.2f}%")
    textstr_lines.append(f"  BUFR: {bufr_return:+7.2f}%")

    textstr = '\n'.join(textstr_lines)

    # Position text box in bottom-right
    props = dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.92,
                 edgecolor='black', linewidth=2)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=props, family='monospace', linespacing=1.4)

    # Formatting
    ax.set_title('Four-Strategy Direct Comparison: Different Trigger/Selection Logic (SEP Launch)',
                 fontsize=17, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('NAV (Normalized to 100)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)

    # Legend with better positioning
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95,
              edgecolor='black', fancybox=True, shadow=True)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add horizontal line at NAV=100
    ax.axhline(y=100, color='gray', linestyle='-', linewidth=0.8, alpha=0.3, zorder=1)

    plt.tight_layout()

    # Save with high DPI for clarity
    filename = 'batch7_four_strategy_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {filename}")

    # Print summary to console as well
    print("\n  Strategy Rankings:")
    ranked = sorted(plotted_strategies, key=lambda s: s['sharpe'], reverse=True)
    for i, strat in enumerate(ranked, 1):
        print(f"    {i}. {strat['short_label']:15s} | "
              f"Sharpe: {strat['sharpe']:5.2f} | "
              f"vs BUFR: {strat['vs_bufr'] * 100:+6.2f}%")

    return filepath


# =============================================================================
# BATCH 8 SPECIALIZED PLOT: Threshold Performance with Table
# =============================================================================

def plot_threshold_performance_with_table(
        summary_df: pd.DataFrame,
        output_dir: str
) -> Optional[str]:
    """
    Comprehensive threshold analysis with integrated performance table.

    Shows:
    1. Bar chart of average excess vs BUFR by threshold
    2. Integrated table with detailed statistics
    3. Visual ranking highlighting 90% underperformance

    Only relevant for batches testing cap_utilization_threshold.

    Parameters:
        summary_df: Summary DataFrame with all results
        output_dir: Directory to save plot

    Returns:
        Filepath if successful, None otherwise
    """
    import ast

    print("\n  Generating: Threshold Performance Analysis with Table...")

    # Extract threshold values
    def extract_threshold(params_str):
        try:
            params_dict = ast.literal_eval(params_str)
            return params_dict.get('threshold', None)
        except:
            return None

    summary_df_copy = summary_df.copy()
    summary_df_copy['threshold_value'] = summary_df_copy['trigger_params'].apply(extract_threshold)

    # Filter to threshold strategies with correct selection algo
    threshold_data = summary_df_copy[
        (summary_df_copy['trigger_type'] == 'cap_utilization_threshold') &
        (summary_df_copy['selection_algo'] == 'select_most_recent_launch') &
        (summary_df_copy['threshold_value'].notna())
        ].copy()

    if threshold_data.empty:
        print("    ⚠ No threshold data - skipping")
        return None

    # Calculate statistics by threshold
    stats_by_threshold = []

    for threshold in sorted(threshold_data['threshold_value'].unique()):
        thresh_data = threshold_data[threshold_data['threshold_value'] == threshold]

        avg_excess = thresh_data['vs_bufr_excess'].mean()
        std_excess = thresh_data['vs_bufr_excess'].std()
        min_excess = thresh_data['vs_bufr_excess'].min()
        max_excess = thresh_data['vs_bufr_excess'].max()
        num_beating = (thresh_data['vs_bufr_excess'] > 0).sum()
        total_months = len(thresh_data)

        stats_by_threshold.append({
            'threshold': threshold,
            'avg_excess': avg_excess,
            'std_excess': std_excess,
            'min_excess': min_excess,
            'max_excess': max_excess,
            'num_beating': num_beating,
            'total_months': total_months,
            'pct_beating': (num_beating / total_months * 100) if total_months > 0 else 0
        })

    stats_df = pd.DataFrame(stats_by_threshold)
    stats_df = stats_df.sort_values('avg_excess', ascending=False)  # Best to worst

    # Print summary to console
    print("\n  Threshold Performance Summary:")
    print("  " + "=" * 70)
    for _, row in stats_df.iterrows():
        print(f"  {int(row['threshold'] * 100):3d}% | "
              f"Avg: {row['avg_excess'] * 100:+6.2f}% | "
              f"Std: {row['std_excess'] * 100:5.2f}% | "
              f"Beat BUFR: {row['num_beating']:.0f}/{row['total_months']:.0f} months")
    print("  " + "=" * 70)

    # Create figure with adjusted layout
    fig = plt.figure(figsize=(14, 10))

    # Top: Bar chart (70% of figure height)
    ax_bar = plt.subplot2grid((10, 1), (0, 0), rowspan=6)

    # Bottom: Table (30% of figure height)
    ax_table = plt.subplot2grid((10, 1), (7, 0), rowspan=3)
    ax_table.axis('off')

    # =========================================================================
    # BAR CHART
    # =========================================================================

    thresholds_pct = [int(t * 100) for t in stats_df['threshold']]
    avg_excess_pct = stats_df['avg_excess'].values * 100

    # Color coding: highlight 90% as red, best as green
    colors = []
    for i, threshold in enumerate(stats_df['threshold']):
        if i == 0:  # Best performer
            colors.append('#2E7D32')  # Dark green
        elif threshold == 0.90:  # 90% threshold
            colors.append('#D32F2F')  # Red
        else:
            colors.append('#1976D2')  # Blue

    bars = ax_bar.bar(range(len(thresholds_pct)), avg_excess_pct,
                      color=colors, alpha=0.85, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, val in zip(bars, avg_excess_pct):
        height = bar.get_height()
        y_pos = height + 0.15 if height > 0 else height - 0.15
        va = 'bottom' if height > 0 else 'top'

        ax_bar.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{val:+.2f}%',
                    ha='center', va=va, fontsize=12, fontweight='bold')

    # Formatting
    ax_bar.set_xticks(range(len(thresholds_pct)))
    ax_bar.set_xticklabels([f'{t}%' for t in thresholds_pct], fontsize=12, fontweight='bold')
    ax_bar.set_ylabel('Average Excess Return vs BUFR (%)', fontsize=13, fontweight='bold')
    ax_bar.set_title('Cap Utilization Threshold Performance Analysis (Averaged Across 12 Months)',
                     fontsize=15, fontweight='bold', pad=20)
    ax_bar.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax_bar.grid(True, alpha=0.3, axis='y', linestyle=':')

    # # Add ranking labels
    # for i, (bar, rank) in enumerate(zip(bars, range(1, len(bars) + 1))):
    #     ax_bar.text(bar.get_x() + bar.get_width() / 2, ax_bar.get_ylim()[1] * 0.95,
    #                 f'Rank #{rank}',
    #                 ha='center', va='top', fontsize=10, fontweight='bold',
    #                 color='white' if i == 0 or stats_df.iloc[i]['threshold'] == 0.90 else 'black',
    #                 bbox=dict(boxstyle='round,pad=0.3',
    #                           facecolor=colors[i],
    #                           edgecolor='black',
    #                           linewidth=1.5,
    #                           alpha=0.9))

    # =========================================================================
    # PERFORMANCE TABLE
    # =========================================================================

    # Prepare table data
    table_data = []
    table_data.append(['Threshold', 'Avg vs BUFR', 'Std Dev', 'Min', 'Max', 'Months Beat BUFR'])

    for _, row in stats_df.iterrows():
        table_data.append([
            f"{int(row['threshold'] * 100)}%",
            f"{row['avg_excess'] * 100:+.2f}%",
            f"{row['std_excess'] * 100:.2f}%",
            f"{row['min_excess'] * 100:+.2f}%",
            f"{row['max_excess'] * 100:+.2f}%",
            f"{int(row['num_beating'])}/{int(row['total_months'])}"
        ])

    # Create table
    table = ax_table.table(cellText=table_data,
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0, 1, 1])

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Header row styling
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#366092')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
        cell.set_edgecolor('black')
        cell.set_linewidth(2)

    # Data row styling
    for i in range(1, len(table_data)):
        # Color code by ranking
        row_color = colors[i - 1]
        row_alpha = 0.15

        for j in range(len(table_data[i])):
            cell = table[(i, j)]
            cell.set_facecolor(row_color)
            cell.set_alpha(row_alpha)
            cell.set_edgecolor('black')
            cell.set_linewidth(1)

            # Bold the avg vs BUFR column (index 1)
            if j == 1:
                cell.set_text_props(weight='bold', fontsize=10)

    plt.tight_layout()

    filename = 'threshold_performance_analysis.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {filename}")

    # Also export table as CSV for reference
    stats_df_export = stats_df.copy()
    stats_df_export['threshold'] = (stats_df_export['threshold'] * 100).astype(int).astype(str) + '%'
    stats_df_export['avg_excess'] = (stats_df_export['avg_excess'] * 100).round(2)
    stats_df_export['std_excess'] = (stats_df_export['std_excess'] * 100).round(2)
    stats_df_export['min_excess'] = (stats_df_export['min_excess'] * 100).round(2)
    stats_df_export['max_excess'] = (stats_df_export['max_excess'] * 100).round(2)

    csv_path = os.path.join(output_dir, 'threshold_performance_table.csv')
    stats_df_export.to_csv(csv_path, index=False)
    print(f"    ✓ Saved: threshold_performance_table.csv")

    return filepath


def plot_best_strategies_comparison(
        results_list: List[Dict],
        summary_df: pd.DataFrame,
        output_dir: str,
        max_strategies: int = None
) -> str:
    """
    Clean comparison of top-performing strategies.

    Dynamically adapts to show all unique strategies (or top N if many strategies).
    Shows actual trigger/selection logic in legend instead of rank numbers.

    Parameters:
        results_list: List of all backtest results
        summary_df: Summary DataFrame
        output_dir: Directory to save plot
        max_strategies: Optional limit on number of strategies to show (default: show all)

    Returns:
        Filepath to saved plot
    """
    print("\n  Generating: Best Strategies Comparison...")

    summary_df = _ensure_intent_column(summary_df)

    # Determine how many unique strategies we have
    num_unique_strategies = len(summary_df.groupby(['trigger_type', 'selection_algo', 'launch_month']).size())

    # If not specified, show all strategies (or top 10 if more than 10)
    if max_strategies is None:
        max_strategies = min(num_unique_strategies, 10)

    print(f"    Total unique strategies: {num_unique_strategies}")
    print(f"    Showing top {max_strategies} by Sharpe ratio")

    # Dynamic figure sizing based on number of strategies
    fig_height = max(8, 6 + max_strategies * 0.3)  # Add height for more strategies
    fig, ax = plt.subplots(figsize=(16, fig_height))

    # Get top N strategies by Sharpe ratio
    top_n = summary_df.nlargest(max_strategies, 'strategy_sharpe')

    # Build strategy info
    strategy_ids = {}
    strategy_rows = {}
    labels = {}

    # Color palette that scales with N
    if max_strategies <= 4:
        color_palette = ['#2E7D32', '#D32F2F', '#1565C0', '#F57C00']  # Green, Red, Blue, Orange
    elif max_strategies <= 7:
        color_palette = ['#2E7D32', '#D32F2F', '#1565C0', '#F57C00', '#7B1FA2', '#00897B', '#C62828']
    else:
        # Use matplotlib colormap for many strategies
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10')
        color_palette = [cmap(i) for i in range(max_strategies)]

    colors = {}

    for i, (idx, row) in enumerate(top_n.iterrows(), 1):
        key = f'top_{i}'
        strategy_ids[key] = create_strategy_id(row)
        strategy_rows[key] = row
        colors[key] = color_palette[i - 1] if i <= len(color_palette) else '#666666'

        # Create concise label showing trigger + selection
        labels[key] = _format_strategy_label_short(row)

    # Plot strategies
    plotted_strategies = []
    for key, strat_id in strategy_ids.items():
        for result in results_list:
            result_id = create_strategy_id(result)

            if result_id == strat_id:
                daily_nav = result['daily_performance']

                color = colors.get(key, '#666666')

                ax.plot(daily_nav['Date'], daily_nav['Strategy_NAV'],
                        color=color, linewidth=2.5, alpha=0.9,
                        label=labels[key], zorder=10)

                plotted_strategies.append((key, strategy_rows[key], color))
                break

    # Plot benchmarks
    if results_list:
        benchmark_nav = _get_earliest_benchmark_data(results_list)

        ax.plot(benchmark_nav['Date'], benchmark_nav['SPY_NAV'],
                color=BENCHMARK_COLORS['SPY'], linewidth=2.0, linestyle='--',
                alpha=0.7, label='SPY', zorder=5)

        ax.plot(benchmark_nav['Date'], benchmark_nav['BUFR_NAV'],
                color=BENCHMARK_COLORS['BUFR'], linewidth=2.0, linestyle=':',
                alpha=0.7, label='BUFR', zorder=5)

    # Add text box with detailed strategy information
    textstr_lines = ['STRATEGY DETAILS\n' + '─' * 50]

    for i, (key, row, color) in enumerate(plotted_strategies):
        if i > 0:
            textstr_lines.append('')  # Blank line between strategies

        # Color indicator
        color_names = {
            '#2E7D32': '■ Green',
            '#D32F2F': '■ Red',
            '#1565C0': '■ Blue',
            '#F57C00': '■ Orange',
            '#7B1FA2': '■ Purple',
            '#00897B': '■ Teal',
            '#C62828': '■ Dark Red',
        }
        color_label = color_names.get(color, '■')

        textstr_lines.append(f'{color_label} Rank #{i + 1}:')
        textstr_lines.append(_format_strategy_details(row, include_metrics=True))

    textstr = '\n'.join(textstr_lines)

    # Position text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85,
                 edgecolor='black', linewidth=1.5)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=props, family='monospace')

    # Formatting
    title = f'Top {max_strategies} Performing Strategies' if max_strategies < num_unique_strategies else 'Top Performing Strategies'
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('NAV (Normalized to 100)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Legend positioning - adjust based on number of items
    num_legend_items = len(plotted_strategies) + 2  # +2 for benchmarks
    if num_legend_items <= 6:
        legend_ncol = 1
        legend_loc = 'upper left'
    else:
        legend_ncol = 2
        legend_loc = 'upper left'

    ax.legend(loc=legend_loc, fontsize=10, framealpha=0.9, ncol=legend_ncol)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    filename = 'best_strategies_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {filename}")
    return filepath


def _format_strategy_label_short(row) -> str:
    """
    Create concise label for legend showing trigger + selection.

    Examples:
    - "RemCap 75% → Highest Cap"
    - "CapUtil 75% → Lowest Util"
    - "Monthly → Most Recent"

    Parameters:
        row: Strategy row (Series or dict)

    Returns:
        Formatted label string
    """
    import ast

    # Extract trigger params
    if isinstance(row.get('trigger_params'), str):
        try:
            params = ast.literal_eval(row['trigger_params'])
        except:
            params = {}
    else:
        params = row.get('trigger_params', {})

    trigger_type = row['trigger_type']
    selection = row['selection_algo']

    # Shorten trigger name
    if 'remaining_cap' in trigger_type:
        if 'threshold' in params:
            threshold_pct = int(params['threshold'] * 100)
            trigger_str = f"RemCap {threshold_pct}%"
        else:
            trigger_str = "RemCap"

    elif 'cap_utilization' in trigger_type:
        if 'threshold' in params:
            threshold_pct = int(params['threshold'] * 100)
            trigger_str = f"CapUtil {threshold_pct}%"
        else:
            trigger_str = "CapUtil"

    elif 'downside' in trigger_type:
        if 'threshold' in params:
            threshold_pct = int(params['threshold'] * 100)
            trigger_str = f"Downside {threshold_pct}%"
        else:
            trigger_str = "Downside"

    elif 'ref_asset' in trigger_type:
        if 'threshold' in params:
            threshold_pct = int(params['threshold'] * 100)
            trigger_str = f"RefAsset {threshold_pct:+d}%"
        else:
            trigger_str = "RefAsset"

    elif 'rebalance_time_period' in trigger_type:
        if 'frequency' in params:
            freq = params['frequency']
            freq_map = {
                'monthly': 'Monthly',
                'quarterly': 'Quarterly',
                'semi_annual': 'Semi-Annual',
                'annual': 'Annual'
            }
            trigger_str = freq_map.get(freq, freq.title())
        else:
            trigger_str = "Time-Based"
    else:
        # Fallback - just use type
        trigger_str = trigger_type.replace('_', ' ').title()

    # Shorten selection name
    selection_map = {
        'select_most_recent_launch': 'Most Recent',
        'select_remaining_cap_highest': 'Highest Cap',
        'select_remaining_cap_lowest': 'Lowest Cap',
        'select_cap_utilization_lowest': 'Lowest Util',
        'select_cap_utilization_highest': 'Highest Util',
        'select_downside_buffer_highest': 'Highest Downside',
        'select_downside_buffer_lowest': 'Lowest Downside',
        'select_cost_analysis': 'Cost Opt',
        'select_highest_outcome_and_cap': 'High Outcome+Cap'
    }

    selection_str = selection_map.get(selection, selection.replace('select_', '').replace('_', ' ').title())

    # Combine with arrow
    return f"{trigger_str} → {selection_str}"


def plot_batch8_normalized_nav_ALIGNED_DETAILED_LEGEND(
        results_list: List[Dict],
        summary_df: pd.DataFrame,
        output_dir: str,
        top_n: int = 5
) -> Optional[str]:
    """
    Create normalized NAV plot with detailed legend showing trigger/selection/threshold.

    Legend format: "#1: Cap Util 75% → Most Recent (SEP)"

    This provides more context than just showing the threshold percentage.

    Parameters:
        results_list: List of backtest result dicts
        summary_df: Summary DataFrame with performance metrics
        output_dir: Directory to save plot
        top_n: Number of top strategies to plot (default 5)

    Returns:
        Filepath if successful, None otherwise
    """
    print(f"\n  Generating: Batch 8 Normalized NAV Plot - Common Date Range (Top {top_n})...")

    if summary_df.empty or not results_list:
        print("    ⚠ No data available - skipping")
        return None

    # Get top N strategies by Sharpe ratio
    top_strategies = summary_df.nlargest(top_n, 'strategy_sharpe')

    def create_strategy_id(row):
        """Create unique ID including threshold parameter."""
        try:
            params = ast.literal_eval(row['trigger_params']) if isinstance(row['trigger_params'], str) else row['trigger_params']
            threshold = params.get('threshold', 'none')
        except:
            threshold = 'none'

        return f"{row['launch_month']}_{row['trigger_type']}_{row['selection_algo']}_{threshold}"

    # ==========================================================================
    # HELPER FUNCTIONS FOR LABEL FORMATTING
    # ==========================================================================

    def format_trigger_name(trigger_type: str) -> str:
        """Convert trigger_type to short readable name."""
        trigger_map = {
            'cap_utilization_threshold': 'Cap Util',
            'remaining_cap_threshold': 'Rem Cap',
            'downside_before_buffer_threshold': 'Downside',
            'ref_asset_return_threshold': 'Ref Return',
            'rebalance_time_period': 'Time'
        }
        return trigger_map.get(trigger_type, trigger_type)

    def format_selection_name(selection_algo: str) -> str:
        """Convert selection_algo to short readable name."""
        selection_map = {
            'select_most_recent_launch': 'Most Recent',
            'select_remaining_cap_highest': 'Highest Cap',
            'select_remaining_cap_lowest': 'Lowest Cap',
            'select_cap_utilization_lowest': 'Lowest Util',
            'select_cap_utilization_highest': 'Highest Util',
            'select_downside_buffer_highest': 'Highest Downside',
            'select_downside_buffer_lowest': 'Lowest Downside',
            'select_cost_analysis': 'Cost Optimized'
        }
        return selection_map.get(selection_algo, selection_algo)

    # ==========================================================================
    # STEP 1: COLLECT ALL STRATEGY DATA AND FIND COMMON DATE RANGE
    # ==========================================================================

    strategy_data = []
    latest_start = None
    earliest_end = None

    print("\n    Collecting strategy data:")

    for i, (idx, row) in enumerate(top_strategies.iterrows()):
        strategy_id = create_strategy_id(row)

        # Find matching result
        for result in results_list:
            result_id = create_strategy_id(result)

            if result_id == strategy_id:
                daily_nav = result['daily_performance'].copy()

                start_date = daily_nav['Date'].min()
                end_date = daily_nav['Date'].max()

                # Track latest start and earliest end
                if latest_start is None or start_date > latest_start:
                    latest_start = start_date

                if earliest_end is None or end_date < earliest_end:
                    earliest_end = end_date

                # Extract threshold
                try:
                    params = ast.literal_eval(row['trigger_params']) if isinstance(row['trigger_params'], str) else row['trigger_params']
                    threshold = params.get('threshold', 0)
                except:
                    threshold = 0

                strategy_data.append({
                    'rank': i + 1,
                    'row': row,
                    'daily_nav': daily_nav,
                    'threshold': threshold,
                    'threshold_pct': int(threshold * 100),
                    'trigger_type': row['trigger_type'],
                    'selection_algo': row['selection_algo'],
                    'launch_month': row['launch_month'],
                    'start_date': start_date,
                    'end_date': end_date
                })

                print(f"      Rank #{i + 1}: {int(threshold * 100)}% ({row['launch_month']}) - "
                      f"{start_date.date()} to {end_date.date()}")

                break

    if not strategy_data:
        print("    ⚠ No matching strategies found - skipping")
        return None

    # ==========================================================================
    # STEP 2: VALIDATE COMMON DATE RANGE
    # ==========================================================================

    if latest_start > earliest_end:
        print(f"    ⚠ No overlapping period! Latest start: {latest_start.date()}, "
              f"Earliest end: {earliest_end.date()}")
        return None

    common_days = (earliest_end - latest_start).days

    print(f"\n    Common date range: {latest_start.date()} to {earliest_end.date()}")
    print(f"    Common period: {common_days} days")

    # ==========================================================================
    # STEP 3: FILTER ALL STRATEGIES TO COMMON PERIOD AND NORMALIZE
    # ==========================================================================

    print("\n    Normalizing to common period:")

    aligned_strategies = []

    for strat_info in strategy_data:
        daily_nav = strat_info['daily_nav']

        # Filter to common date range
        common_nav = daily_nav[
            (daily_nav['Date'] >= latest_start) &
            (daily_nav['Date'] <= earliest_end)
            ].copy().reset_index(drop=True)

        if common_nav.empty:
            print(f"      ⚠ Rank #{strat_info['rank']}: No data in common period - skipping")
            continue

        # Normalize to 100 at START of common period
        first_nav = common_nav['Strategy_NAV'].iloc[0]
        common_nav['Normalized_NAV'] = (common_nav['Strategy_NAV'] / first_nav) * 100

        # Also normalize benchmarks
        first_spy = common_nav['SPY_NAV'].iloc[0]
        first_bufr = common_nav['BUFR_NAV'].iloc[0]
        common_nav['Normalized_SPY'] = (common_nav['SPY_NAV'] / first_spy) * 100
        common_nav['Normalized_BUFR'] = (common_nav['BUFR_NAV'] / first_bufr) * 100

        # Create relative time axis
        common_nav['Days_Since_Common_Start'] = range(len(common_nav))

        strat_info['common_nav'] = common_nav
        aligned_strategies.append(strat_info)

        print(f"      Rank #{strat_info['rank']}: {strat_info['threshold_pct']}% - "
              f"{len(common_nav)} days aligned")

    if not aligned_strategies:
        print("    ⚠ No strategies after alignment - skipping")
        return None

    # ==========================================================================
    # STEP 4: CREATE VISUALIZATION
    # ==========================================================================

    fig, ax = plt.subplots(figsize=(16, 9))

    # Color palette for top 5 strategies
    colors = ['#2E7D32', '#1565C0', '#F57C00', '#7B1FA2', '#C62828']

    # Plot each aligned strategy
    for i, strat_info in enumerate(aligned_strategies):
        common_nav = strat_info['common_nav']

        # ✅ NEW: Detailed legend format
        trigger_short = format_trigger_name(strat_info['trigger_type'])
        selection_short = format_selection_name(strat_info['selection_algo'])
        threshold_pct = strat_info['threshold_pct']
        launch = strat_info['launch_month']

        label = f"#{strat_info['rank']}: {trigger_short} {threshold_pct}% → {selection_short} ({launch})"

        ax.plot(common_nav['Days_Since_Common_Start'],
                common_nav['Normalized_NAV'],
                color=colors[i], linewidth=2.5, alpha=0.9,
                label=label, zorder=10 - i)

    # Plot benchmarks
    first_common_nav = aligned_strategies[0]['common_nav']

    # Average benchmarks across all strategies for smoothing
    spy_avg = []
    bufr_avg = []

    for day_idx in range(len(first_common_nav)):
        spy_vals = [s['common_nav']['Normalized_SPY'].iloc[day_idx] for s in aligned_strategies]
        bufr_vals = [s['common_nav']['Normalized_BUFR'].iloc[day_idx] for s in aligned_strategies]

        spy_avg.append(np.mean(spy_vals))
        bufr_avg.append(np.mean(bufr_vals))

    days_axis = first_common_nav['Days_Since_Common_Start']

    ax.plot(days_axis, spy_avg,
            color='#757575', linewidth=2, linestyle='--',
            alpha=0.7, label='SPY', zorder=5)

    ax.plot(days_axis, bufr_avg,
            color='#9E9E9E', linewidth=2, linestyle=':',
            alpha=0.7, label='BUFR', zorder=5)

    # ==========================================================================
    # STEP 5: FORMATTING
    # ==========================================================================

    ax.set_xlabel('Days Since Common Start Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Normalized NAV (Starting at 100)', fontsize=13, fontweight='bold')
    ax.set_title(f'Example: December Cap Utilization Threshold --> Select Most Recent Launch\n'
                 f'Common Period: {latest_start.strftime("%Y-%m-%d")} to {earliest_end.strftime("%Y-%m-%d")}',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)  # Slightly smaller font for longer labels
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.axhline(y=100, color='black', linestyle='-', linewidth=1, alpha=0.3)

    # ==========================================================================
    # STEP 6: ADD STRATEGY DETAILS BOX (with full names)
    # ==========================================================================

    textstr_lines = ['STRATEGY DETAILS\n' + '─' * 65]
    textstr_lines.append(f'Common Period: {common_days} days')
    textstr_lines.append(f'Start: {latest_start.strftime("%Y-%m-%d")}')
    textstr_lines.append(f'End: {earliest_end.strftime("%Y-%m-%d")}')

    for strat_info in aligned_strategies:
        row = strat_info['row']
        rank = strat_info['rank']
        threshold_pct = strat_info['threshold_pct']

        # Full trigger and selection names for detail box
        trigger_short = format_trigger_name(strat_info['trigger_type'])
        selection_short = format_selection_name(strat_info['selection_algo'])

        textstr_lines.append('')  # Blank line
        textstr_lines.append(f'■ Rank #{rank}: {trigger_short} {threshold_pct}% → {selection_short}')
        textstr_lines.append(f'  Launch: {row["launch_month"]}')
        textstr_lines.append(f'  Return: {row["strategy_return"] * 100:+.1f}% | Sharpe: {row["strategy_sharpe"]:.2f}')
        textstr_lines.append(f'  vs BUFR: {row["vs_bufr_excess"] * 100:+.1f}% | Trades: {int(row["num_trades"])}')

    textstr = '\n'.join(textstr_lines)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9,
                 edgecolor='black', linewidth=1.5)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=props, family='monospace')

    plt.tight_layout()

    # ==========================================================================
    # STEP 7: SAVE
    # ==========================================================================

    filename = 'batch8_normalized_nav_aligned_top5.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n    ✓ Saved: {filename}")
    return filepath


def format_trigger_name(trigger_type: str) -> str:
    """Convert trigger_type to short readable name."""
    trigger_map = {
        'cap_utilization_threshold': 'Cap Util',
        'remaining_cap_threshold': 'Rem Cap',
        'downside_before_buffer_threshold': 'Downside',
        'ref_asset_return_threshold': 'Ref Return'
    }
    return trigger_map.get(trigger_type, trigger_type)

def format_selection_name(selection_algo: str) -> str:
    """Convert selection_algo to short readable name."""
    selection_map = {
        'select_most_recent_launch': 'Most Recent',
        'select_remaining_cap_highest': 'Highest Cap',
        'select_cap_utilization_lowest': 'Lowest Util',
        'select_cost_analysis': 'Cost Optimized'
        # ... etc
    }
    return selection_map.get(selection_algo, selection_algo)