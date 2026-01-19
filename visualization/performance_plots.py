"""
Performance visualization with intent-grouped charts.

This module creates professional charts showing:
1. All strategies grouped by intent with best highlighted
2. Comparison of best strategies across intents
3. Performance metrics comparisons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import List, Dict, Any
import os

# =============================================================================
# COLOR SCHEMES
# =============================================================================

INTENT_COLORS = {
    'bullish': {
        'light': '#81C784',  # Light green for all strategies
        'best': '#2E7D32',  # Dark green for best (thick line)
        'fill': '#A5D6A7'  # For confidence bands if needed
    },
    'bearish': {
        'light': '#E57373',  # Light red
        'best': '#C62828',  # Dark red
        'fill': '#EF9A9A'
    },
    'neutral': {
        'light': '#64B5F6',  # Light blue
        'best': '#1565C0',  # Dark blue
        'fill': '#90CAF9'
    },
    'cost_optimized': {
        'light': '#FFB74D',  # Light orange
        'best': '#F57C00',  # Dark orange
        'fill': '#FFCC80'
    }
}

BENCHMARK_COLORS = {
    'SPY': '#757575',  # Gray
    'BUFR': '#9E9E9E',  # Light gray
    'Hold': '#BDBDBD'  # Lighter gray
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_daily_nav_from_result(result: Dict) -> pd.DataFrame:
    """Extract daily NAV DataFrame from result dict."""
    if 'daily_performance' not in result:
        return pd.DataFrame()

    df = result['daily_performance'].copy()

    # Add strategy metadata
    df['launch_month'] = result['launch_month']
    df['trigger_type'] = result['trigger_type']
    df['selection_algo'] = result['selection_algo']

    return df


def _create_strategy_id(row) -> str:
    """Create unique strategy identifier."""
    return f"{row['launch_month']}_{row['trigger_type']}_{row['selection_algo']}"


# =============================================================================
# PLOT 1: ALL STRATEGIES BY INTENT (MULTI-PANEL)
# =============================================================================

def plot_strategies_by_intent(
        results_list: List[Dict],
        summary_df: pd.DataFrame,
        best_strategies_df: pd.DataFrame,
        output_dir: str
):
    """
    Create multi-panel plot with one subplot per strategy intent.

    Each subplot shows:
    - All strategies in that intent (thin light lines, alpha=0.3)
    - Best performer in that intent (thick dark line, alpha=1.0)
    - Benchmark lines (SPY, BUFR) in gray

    Parameters:
        results_list: List of backtest result dicts
        summary_df: Summary DataFrame with 'strategy_intent' column
        best_strategies_df: DataFrame with best strategy per intent
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 80)
    print("GENERATING: All Strategies by Intent (Multi-Panel)")
    print("=" * 80)

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Strategy Performance by Intent', fontsize=16, fontweight='bold')

    intents = ['bullish', 'bearish', 'neutral', 'cost_optimized']
    intent_labels = {
        'bullish': 'Bullish Strategies',
        'bearish': 'Bearish Strategies',
        'neutral': 'Neutral Strategies',
        'cost_optimized': 'Cost-Optimized Strategies'
    }

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    for idx, intent in enumerate(intents):
        ax = axes_flat[idx]

        # Filter strategies by intent
        intent_strategies = summary_df[summary_df['strategy_intent'] == intent]

        if intent_strategies.empty:
            ax.text(0.5, 0.5, f'No {intent} strategies',
                    ha='center', va='center', fontsize=14)
            ax.set_title(intent_labels[intent])
            continue

        print(f"\nPlotting {intent.upper()}: {len(intent_strategies)} strategies")

        # Get best strategy for this intent
        best_strategy = best_strategies_df[best_strategies_df['strategy_intent'] == intent]

        if not best_strategy.empty:
            best_id = _create_strategy_id(best_strategy.iloc[0])
        else:
            best_id = None

        # Plot all strategies in this intent
        plotted_count = 0
        best_plotted = False

        for result in results_list:
            result_id = _create_strategy_id(result)

            # Check if this result belongs to current intent
            matching = intent_strategies[
                (intent_strategies['launch_month'] == result['launch_month']) &
                (intent_strategies['trigger_type'] == result['trigger_type']) &
                (intent_strategies['selection_algo'] == result['selection_algo'])
                ]

            if matching.empty:
                continue

            # Get NAV data
            daily_nav = result['daily_performance']

            # Determine if this is the best strategy
            is_best = (result_id == best_id)

            if is_best:
                # Plot best strategy (thick, dark)
                ax.plot(daily_nav['Date'], daily_nav['Strategy_NAV'],
                        color=INTENT_COLORS[intent]['best'],
                        linewidth=2.5,
                        alpha=1.0,
                        label='Best Strategy',
                        zorder=10)
                best_plotted = True
            else:
                # Plot regular strategy (thin, light)
                ax.plot(daily_nav['Date'], daily_nav['Strategy_NAV'],
                        color=INTENT_COLORS[intent]['light'],
                        linewidth=0.5,
                        alpha=0.3,
                        zorder=1)

            plotted_count += 1

        # Plot benchmarks (use first result for benchmark data)
        if results_list:
            benchmark_nav = results_list[0]['daily_performance']

            ax.plot(benchmark_nav['Date'], benchmark_nav['SPY_NAV'],
                    color=BENCHMARK_COLORS['SPY'],
                    linewidth=1.5,
                    linestyle='--',
                    alpha=0.7,
                    label='SPY',
                    zorder=5)

            ax.plot(benchmark_nav['Date'], benchmark_nav['BUFR_NAV'],
                    color=BENCHMARK_COLORS['BUFR'],
                    linewidth=1.5,
                    linestyle=':',
                    alpha=0.7,
                    label='BUFR',
                    zorder=5)

        # Formatting
        ax.set_title(f"{intent_labels[intent]} (n={len(intent_strategies)})",
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('NAV (Normalized to 100)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        print(f"  Plotted {plotted_count} strategies (best highlighted: {best_plotted})")

    plt.tight_layout()

    # Save plot
    filename = 'strategies_by_intent_multipanel.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# PLOT 2: BEST STRATEGIES COMPARISON
# =============================================================================

def plot_best_strategies_comparison(
        results_list: List[Dict],
        best_strategies_df: pd.DataFrame,
        output_dir: str
):
    """
    Single clean chart showing only the 4 best strategies + benchmarks.

    Parameters:
        results_list: List of all backtest results
        best_strategies_df: DataFrame with best strategy per intent
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 80)
    print("GENERATING: Best Strategies Comparison")
    print("=" * 80)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create strategy IDs for best strategies
    best_ids = {}
    for _, row in best_strategies_df.iterrows():
        intent = row['strategy_intent']
        strategy_id = _create_strategy_id(row)
        best_ids[strategy_id] = intent

    # Plot best strategies
    plotted_strategies = []

    for result in results_list:
        result_id = _create_strategy_id(result)

        if result_id in best_ids:
            intent = best_ids[result_id]
            daily_nav = result['daily_performance']

            label = f"{intent.title()}: {result['launch_month']}"

            ax.plot(daily_nav['Date'], daily_nav['Strategy_NAV'],
                    color=INTENT_COLORS[intent]['best'],
                    linewidth=2.5,
                    alpha=0.9,
                    label=label,
                    zorder=10)

            plotted_strategies.append(intent)
            print(f"  Plotted {intent.upper()}: {result['launch_month']} | "
                  f"{result['trigger_type']} | {result['selection_algo']}")

    # Plot benchmarks
    if results_list:
        benchmark_nav = results_list[0]['daily_performance']

        ax.plot(benchmark_nav['Date'], benchmark_nav['SPY_NAV'],
                color=BENCHMARK_COLORS['SPY'],
                linewidth=2.0,
                linestyle='--',
                alpha=0.7,
                label='SPY',
                zorder=5)

        ax.plot(benchmark_nav['Date'], benchmark_nav['BUFR_NAV'],
                color=BENCHMARK_COLORS['BUFR'],
                linewidth=2.0,
                linestyle=':',
                alpha=0.7,
                label='BUFR',
                zorder=5)

    # Formatting
    ax.set_title('Top Performing Strategies by Intent', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('NAV (Normalized to 100)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save
    filename = 'best_strategies_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# PLOT 3: PERFORMANCE METRICS COMPARISON (BAR CHARTS)
# =============================================================================

def plot_performance_metrics_comparison(
        best_strategies_df: pd.DataFrame,
        output_dir: str
):
    """
    Multi-panel bar chart comparing the 4 best strategies across metrics.

    4 subplots showing:
    1. Total Returns (%)
    2. Sharpe Ratios
    3. Max Drawdowns (%)
    4. Excess vs BUFR (%)

    Parameters:
        best_strategies_df: DataFrame with best strategy per intent
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 80)
    print("GENERATING: Performance Metrics Comparison")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Performance Metrics: Best Strategies by Intent',
                 fontsize=16, fontweight='bold')

    # Prepare data
    intents = best_strategies_df['strategy_intent'].values
    colors = [INTENT_COLORS[intent]['best'] for intent in intents]
    labels = [f"{intent.title()}\n({row['launch_month']})"
              for intent, (_, row) in zip(intents, best_strategies_df.iterrows())]

    # Subplot 1: Total Returns
    ax1 = axes[0, 0]
    returns = best_strategies_df['strategy_return'].values * 100
    bars1 = ax1.bar(labels, returns, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_title('Total Returns', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Return (%)', fontsize=11)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom' if height > 0 else 'top',
                 fontsize=9, fontweight='bold')

    # Subplot 2: Sharpe Ratios
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

    # Subplot 3: Max Drawdowns
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

    # Subplot 4: Excess vs BUFR
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

    # Rotate x-axis labels
    for ax in axes.flat:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    plt.tight_layout()

    # Save
    filename = 'best_strategies_metrics.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# PLOT 4: INTENT PERFORMANCE DISTRIBUTION (BOX PLOTS)
# =============================================================================

def plot_intent_performance_distribution(
        summary_df: pd.DataFrame,
        output_dir: str
):
    """
    Box plots showing distribution of returns within each intent category.

    Parameters:
        summary_df: Summary DataFrame with 'strategy_intent' column
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 80)
    print("GENERATING: Intent Performance Distribution")
    print("=" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Performance Distribution by Intent', fontsize=16, fontweight='bold')

    intents = ['bullish', 'bearish', 'neutral', 'cost_optimized']
    intent_labels = [intent.replace('_', ' ').title() for intent in intents]
    colors = [INTENT_COLORS[intent]['best'] for intent in intents]

    # Subplot 1: Total Returns
    ax1 = axes[0]
    returns_data = [summary_df[summary_df['strategy_intent'] == intent]['strategy_return'].values * 100
                    for intent in intents]

    bp1 = ax1.boxplot(returns_data, labels=intent_labels, patch_artist=True,
                      showmeans=True, meanline=True)

    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_title('Total Return Distribution', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Return (%)', fontsize=11)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')

    # Subplot 2: vs BUFR Excess
    ax2 = axes[1]
    excess_data = [summary_df[summary_df['strategy_intent'] == intent]['vs_bufr_excess'].values * 100
                   for intent in intents]

    bp2 = ax2.boxplot(excess_data, labels=intent_labels, patch_artist=True,
                      showmeans=True, meanline=True)

    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_title('Excess vs BUFR Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Excess Return (%)', fontsize=11)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')

    plt.tight_layout()

    # Save
    filename = 'intent_performance_distribution.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# MASTER PLOTTING FUNCTION
# =============================================================================

def create_all_plots(
        results_list: List[Dict],
        summary_df: pd.DataFrame,
        best_strategies_df: pd.DataFrame,
        output_dir: str
):
    """
    Generate all visualization plots.

    Parameters:
        results_list: List of all backtest results
        summary_df: Summary DataFrame with 'strategy_intent' column
        best_strategies_df: DataFrame with best strategy per intent
        output_dir: Directory to save plots
    """
    print("\n" + "#" * 80)
    print("# GENERATING ALL VISUALIZATIONS")
    print("#" * 80 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Multi-panel by intent
    plot_strategies_by_intent(results_list, summary_df, best_strategies_df, output_dir)

    # Plot 2: Best strategies comparison
    plot_best_strategies_comparison(results_list, best_strategies_df, output_dir)

    # Plot 3: Metrics comparison
    plot_performance_metrics_comparison(best_strategies_df, output_dir)

    # Plot 4: Distribution box plots
    plot_intent_performance_distribution(summary_df, output_dir)

    print("\n" + "#" * 80)
    print("# ALL VISUALIZATIONS COMPLETE")
    print("#" * 80 + "\n")