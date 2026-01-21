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

def _create_strategy_id(row) -> str:
    """Create unique strategy identifier from row or dict."""
    if isinstance(row, dict):
        return f"{row['launch_month']}_{row['trigger_type']}_{row['selection_algo']}"
    return f"{row['launch_month']}_{row['trigger_type']}_{row['selection_algo']}"


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
# PLOT 2: BEST STRATEGIES COMPARISON
# =============================================================================

def plot_best_strategies_comparison(
        results_list: List[Dict],
        summary_df: pd.DataFrame,
        output_dir: str
) -> str:
    """
    Clean comparison of top-performing strategies only.

    Includes detailed text box showing strategy specifications.
    """
    print("\n  Generating: Best Strategies Comparison...")

    summary_df = _ensure_intent_column(summary_df)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Determine what to plot
    intents = summary_df['strategy_intent'].unique()

    if len(intents) > 1:
        # Multi-intent: show best of each
        best = _find_best_strategies(summary_df)

        strategy_ids = {}
        strategy_rows = {}
        for key, row in best.items():
            if key.endswith('_sharpe'):  # Only plot Sharpe-based best
                strategy_ids[key] = _create_strategy_id(row)
                strategy_rows[key] = row

        labels = {
            'bullish_sharpe': 'Bullish (Best Sharpe)',
            'bearish_sharpe': 'Bearish (Best Sharpe)',
            'neutral_sharpe': 'Neutral (Best Sharpe)',
            'cost_optimized_sharpe': 'Cost-Opt (Best Sharpe)'
        }

    else:
        # Single intent: show top 3 by different criteria
        best = {}
        best['best_sharpe'] = summary_df.loc[summary_df['strategy_sharpe'].idxmax()]
        best['best_return'] = summary_df.loc[summary_df['strategy_return'].idxmax()]
        best['best_vs_bufr'] = summary_df.loc[summary_df['vs_bufr_excess'].idxmax()]

        strategy_ids = {k: _create_strategy_id(v) for k, v in best.items()}
        strategy_rows = best

        labels = {
            'best_sharpe': 'Best Sharpe Ratio',
            'best_return': 'Best Total Return',
            'best_vs_bufr': 'Best vs BUFR'
        }

    # Plot strategies
    plotted_strategies = []
    for key, strat_id in strategy_ids.items():
        for result in results_list:
            result_id = _create_strategy_id(result)

            if result_id == strat_id:
                daily_nav = result['daily_performance']

                # Color based on intent or criteria
                if 'bullish' in key:
                    color = INTENT_COLORS['bullish']['best']
                elif 'bearish' in key:
                    color = INTENT_COLORS['bearish']['best']
                elif 'neutral' in key:
                    color = INTENT_COLORS['neutral']['best']
                elif key == 'best_sharpe':
                    color = '#1f77b4'
                elif key == 'best_return':
                    color = '#2ca02c'
                else:
                    color = '#ff7f0e'

                ax.plot(daily_nav['Date'], daily_nav['Strategy_NAV'],
                        color=color, linewidth=2.5, alpha=0.9,
                        label=labels[key], zorder=10)

                plotted_strategies.append((key, strategy_rows[key]))
                break

    # Plot benchmarks
    if results_list:
        benchmark_nav = results_list[0]['daily_performance']

        ax.plot(benchmark_nav['Date'], benchmark_nav['SPY_NAV'],
                color=BENCHMARK_COLORS['SPY'], linewidth=2.0, linestyle='--',
                alpha=0.7, label='SPY', zorder=5)

        ax.plot(benchmark_nav['Date'], benchmark_nav['BUFR_NAV'],
                color=BENCHMARK_COLORS['BUFR'], linewidth=2.0, linestyle=':',
                alpha=0.7, label='BUFR', zorder=5)

    # Add text box with strategy details
    textstr_lines = ['STRATEGY DETAILS\n' + '─' * 40]

    for i, (key, row) in enumerate(plotted_strategies):
        if i > 0:
            textstr_lines.append('')  # Blank line between strategies

        # Get intent for label
        if 'bullish' in key:
            label = 'Bullish:'
        elif 'bearish' in key:
            label = 'Bearish:'
        elif 'neutral' in key:
            label = 'Neutral:'
        elif key == 'best_sharpe':
            label = 'Best Sharpe:'
        elif key == 'best_return':
            label = 'Best Return:'
        else:
            label = 'Best vs BUFR:'

        textstr_lines.append(label)
        textstr_lines.append(_format_strategy_details(row, include_metrics=True))

    textstr = '\n'.join(textstr_lines)

    # Position in bottom-right
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='black', linewidth=1.5)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=props, family='monospace')

    # Formatting
    ax.set_title('Top Performing Strategies', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('NAV (Normalized to 100)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

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


# =============================================================================
# PLOT 4: REGIME PERFORMANCE WITH BUFR
# =============================================================================

def plot_regime_performance_with_bufr(
        future_regime_df: pd.DataFrame,
        optimal_strategies: Dict,
        output_dir: str
) -> Optional[str]:
    """
    Strategy performance across bull/neutral/bear markets vs BUFR.

    Shows how bullish and defensive strategies perform in each regime.
    """
    print("\n  Generating: Regime Performance with BUFR...")

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

    # DIAGNOSTIC: Print available columns
    print(f"\n    DEBUG: future_regime_df columns:")
    print(f"    {list(future_regime_df.columns)[:10]}...")  # First 10 columns
    print(f"    Total columns: {len(future_regime_df.columns)}")

    # Detect horizon
    if 'future_regime_3m' in future_regime_df.columns:
        regime_col = 'future_regime_3m'
        return_col_suffix = '_3m'
        horizon = '3M'
    elif 'future_regime_6m' in future_regime_df.columns:
        regime_col = 'future_regime_6m'
        return_col_suffix = '_6m'
        horizon = '6M'
    else:
        print("    ⚠ No regime classification column - skipping")
        return None

    # Column names based on actual DataFrame structure
    strat_return_col = f'forward{return_col_suffix}_return'
    bufr_return_col = f'bufr_forward{return_col_suffix}_return'

    print(f"    Using columns: '{strat_return_col}', '{bufr_return_col}'")

    if strat_return_col not in future_regime_df.columns:
        print(f"    ⚠ Column '{strat_return_col}' not found - skipping")
        print(f"    Available return columns: {[c for c in future_regime_df.columns if 'return' in c.lower()]}")
        return None

    if bufr_return_col not in future_regime_df.columns:
        print(f"    ⚠ Column '{bufr_return_col}' not found - skipping")
        return None

    # Collect performance data
    regimes = ['bull', 'neutral', 'bear']
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
            bull_ret = 0
            bufr_ret = 0

        if not defensive_data.empty:
            def_ret = defensive_data[strat_return_col].mean()
            defensive_returns.append(def_ret)
        else:
            defensive_returns.append(0)
            def_ret = 0

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
    ax.set_title('Strategy Performance Across Market Regimes (vs BUFR)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Bull', 'Neutral', 'Bear'], fontsize=12)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.2, alpha=0.4)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    plt.tight_layout()

    filename = 'regime_performance_with_bufr.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {filename}")
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
    if batch_number == 0 or _has_threshold_strategies(summary_df):
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

            filepath = plot_regime_performance_with_bufr(
                future_regime_df, optimal_strategies, output_dir
            )
            if filepath:
                generated_plots['regime_performance'] = filepath
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