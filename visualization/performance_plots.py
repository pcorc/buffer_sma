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


def plot_regime_performance_with_bufr(
        future_regime_df: pd.DataFrame,
        optimal_strategies: dict,
        output_dir: str
):
    """
    Chart: Strategy performance across bull/neutral/bear markets with BUFR benchmark.

    Shows performance of bullish and defensive strategies vs BUFR in each regime.
    Creates stacked bar chart showing absolute returns for comparison.

    Parameters:
        future_regime_df: DataFrame with regime-specific performance data
        optimal_strategies: Dict with 'bull', 'bear', 'neutral' keys containing top strategies
        output_dir: Directory to save chart
    """
    import matplotlib.pyplot as plt
    import numpy as np

    print("\nGenerating regime performance comparison chart...")

    if future_regime_df.empty or not optimal_strategies:
        print("  ⚠ No regime data available for plotting")
        return

    # Get top strategies
    bull_strats = optimal_strategies.get('bull')
    bear_strats = optimal_strategies.get('bear')

    if bull_strats is None or bull_strats.empty:
        print("  ⚠ No bullish strategy found")
        return

    if bear_strats is None or bear_strats.empty:
        print("  ⚠ No defensive strategy found")
        return

    strategy_bullish = bull_strats.iloc[0]
    strategy_defensive = bear_strats.iloc[0]

    # Print strategy details
    print(f"\n  Bullish Strategy:")
    print(f"    {strategy_bullish['launch_month']} | {strategy_bullish['trigger_type']}")
    print(f"    {strategy_bullish['selection_algo']}")

    print(f"\n  Defensive Strategy:")
    print(f"    {strategy_defensive['launch_month']} | {strategy_defensive['trigger_type']}")
    print(f"    {strategy_defensive['selection_algo']}")

    # Detect which columns to use (3M or 6M)
    if 'future_regime_3m' in future_regime_df.columns:
        regime_col = 'future_regime_3m'
        return_col_suffix = '_3m'
        horizon = '3M'
    elif 'future_regime_6m' in future_regime_df.columns:
        regime_col = 'future_regime_6m'
        return_col_suffix = '_6m'
        horizon = '6M'
    else:
        print("  ⚠ No regime classification column found")
        return

    # Build full column names
    strat_return_col = f'avg_return{return_col_suffix}'
    bufr_return_col = f'avg_bufr_return{return_col_suffix}'

    # Verify columns exist
    if strat_return_col not in future_regime_df.columns:
        print(f"  ⚠ Column '{strat_return_col}' not found")
        print(f"  Available columns: {list(future_regime_df.columns)}")
        return

    print(f"\n  Using {horizon} forward-looking regime analysis")
    print(f"  Columns: {regime_col}, {strat_return_col}, {bufr_return_col}")

    # Collect performance data
    regimes = ['bull', 'neutral', 'bear']
    regime_labels = ['Bull', 'Neutral', 'Bear']

    bullish_returns = []
    defensive_returns = []
    bufr_returns = []

    print("\n  Performance by Regime:")
    print("  " + "=" * 70)

    for regime in regimes:
        regime_data = future_regime_df[future_regime_df[regime_col] == regime]

        if regime_data.empty:
            print(f"  {regime.upper():8s}: No data")
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

        # Print to terminal
        print(f"  {regime.upper():8s}:")
        print(f"    Bullish:   {bull_ret * 100:+6.1f}%")
        print(f"    Defensive: {def_ret * 100:+6.1f}%")
        print(f"    BUFR:      {bufr_ret * 100:+6.1f}%")
        print(f"    Bullish vs BUFR: {(bull_ret - bufr_ret) * 100:+6.1f}%")
        print(f"    Defensive vs BUFR: {(def_ret - bufr_ret) * 100:+6.1f}%")

    print("  " + "=" * 70)

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(regimes))
    width = 0.25

    # Create bars
    bars1 = ax.bar(x - width, np.array(bullish_returns) * 100, width,
                   label='Bullish Strategy', color='#2ca02c', alpha=0.85,
                   edgecolor='black', linewidth=1.5)

    bars2 = ax.bar(x, np.array(defensive_returns) * 100, width,
                   label='Defensive Strategy', color='#d62728', alpha=0.85,
                   edgecolor='black', linewidth=1.5)

    bars3 = ax.bar(x + width, np.array(bufr_returns) * 100, width,
                   label='BUFR Benchmark', color='#ff7f0e', alpha=0.7,
                   edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    def add_value_labels(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if abs(height) < 0.5:  # Skip very small values
                continue

            # Position label above or below bar depending on sign
            if height >= 0:
                y_pos = height + 0.5
                va = 'bottom'
            else:
                y_pos = height - 0.5
                va = 'top'

            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{val * 100:.1f}%',
                    ha='center', va=va, fontsize=10, fontweight='bold')

    add_value_labels(bars1, bullish_returns)
    add_value_labels(bars2, defensive_returns)
    add_value_labels(bars3, bufr_returns)

    # Add text box with strategy descriptions
    textstr = ('Bullish strategy captures upside in Bull markets\n'
               'Defensive strategy preserves capital in Bear markets\n'
               'Both strategies compared against BUFR benchmark')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Formatting
    ax.set_xlabel(f'Market Regime ({horizon} Forward)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Strategy Return (%)', fontsize=13, fontweight='bold')
    ax.set_title('Strategy Performance Across Market Regimes (vs BUFR)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels, fontsize=12)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.2, alpha=0.4)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    # Add padding to y-axis
    y_min, y_max = ax.get_ylim()
    padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - padding, y_max + padding)

    plt.tight_layout()

    # Save figure
    filepath = os.path.join(output_dir, 'regime_performance_with_bufr.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  ✓ Chart saved: regime_performance_with_bufr.png")
    print(f"  Location: {output_dir}")


def plot_threshold_comparison(summary_df: pd.DataFrame, output_dir: str):
    """
    Chart: Threshold comparison showing performance by launch month and averages.

    Left panel: Line chart by month for different thresholds
    Right panel: Bar chart showing average performance by threshold
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import ast

    # Extract threshold values from trigger_params
    def extract_threshold(params_str):
        try:
            params_dict = ast.literal_eval(params_str)
            return params_dict.get('threshold', None)
        except:
            return None

    summary_df_copy = summary_df.copy()
    summary_df_copy['threshold_value'] = summary_df_copy['trigger_params'].apply(extract_threshold)

    # Filter to cap_utilization_threshold with select_most_recent_launch
    threshold_data = summary_df_copy[
        (summary_df_copy['trigger_type'] == 'cap_utilization_threshold') &
        (summary_df_copy['selection_algo'] == 'select_most_recent_launch') &
        (summary_df_copy['threshold_value'].notna())
        ].copy()

    if threshold_data.empty:
        print("  ⚠ No threshold data found for plotting")
        return

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # LEFT PANEL: Line chart by launch month
    unique_thresholds = sorted(threshold_data['threshold_value'].unique())

    # Color mapping for thresholds
    color_map = {
        0.25: '#2ca02c',  # Green
        0.40: '#17becf',  # Cyan
        0.50: '#1f77b4',  # Blue
        0.70: '#bcbd22',  # Yellow-green
        0.75: '#ff7f0e',  # Orange (recommended)
        0.90: '#d62728'  # Red (current)
    }

    for threshold in unique_thresholds:
        thresh_data = threshold_data[threshold_data['threshold_value'] == threshold]

        # Group by launch month
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

    # Add annotation box
    ax1.text(0.02, 0.98, 'Current: 90% (Red)\nRecommended: 75% (Orange)',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # RIGHT PANEL: Average performance bars
    avg_by_threshold = threshold_data.groupby('threshold_value')['vs_bufr_excess'].mean().sort_index()

    # Color bars based on performance
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
    filepath = os.path.join(output_dir, 'threshold_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Threshold comparison chart saved")