"""
Batch performance visualization with best strategy highlighting.

Creates three key plots:
1. All strategies grouped by intent (bullish vs bearish) with best highlighted
2. Best bullish vs best bearish vs benchmarks
3. Performance by forward regime
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# Color schemes
BULLISH_COLOR = '#2E7D32'  # Dark green
BULLISH_LIGHT = '#81C784'  # Light green
BEARISH_COLOR = '#C62828'  # Dark red
BEARISH_LIGHT = '#E57373'  # Light red
BENCHMARK_COLORS = {
    'SPY': '#757575',
    'BUFR': '#9E9E9E'
}


def classify_strategy_intent(selection_algo):
    """
    Classify strategy as bullish or bearish based on selection algorithm.

    Bullish: Seeking upside capture
    Bearish: Seeking protection/defensive positioning
    """
    bullish_algos = [
        'select_cap_utilization_lowest',
        'select_remaining_cap_highest',
        'select_downside_buffer_lowest'
    ]

    bearish_algos = [
        'select_downside_buffer_highest',
        'select_downside_buffer_lowest',
        'select_cap_utilization_lowest'
    ]

    if selection_algo in bullish_algos:
        return 'bullish'
    elif selection_algo in bearish_algos:
        return 'bearish'
    else:
        return 'neutral'


def find_best_strategies(summary_df):
    """
    Find best bullish and bearish strategies by multiple criteria.

    Returns dict with available best strategies (handles missing intents gracefully)
    """
    # Add intent classification
    summary_df['intent'] = summary_df['selection_algo'].apply(classify_strategy_intent)

    bullish_df = summary_df[summary_df['intent'] == 'bullish']
    bearish_df = summary_df[summary_df['intent'] == 'bearish']

    best = {}

    # Only add if strategies exist
    if not bullish_df.empty:
        best['bullish_sharpe'] = bullish_df.loc[bullish_df['strategy_sharpe'].idxmax()]
        best['bullish_return'] = bullish_df.loc[bullish_df['strategy_return'].idxmax()]

    if not bearish_df.empty:
        best['bearish_sharpe'] = bearish_df.loc[bearish_df['strategy_sharpe'].idxmax()]
        best['bearish_drawdown'] = bearish_df.loc[bearish_df['strategy_max_dd'].idxmax()]

    # Overall best vs BUFR
    if not summary_df.empty:
        best['vs_bufr_excess'] = summary_df.loc[summary_df['vs_bufr_excess'].idxmax()]

    return best


def plot_1_all_strategies_with_best(results_list, summary_df, output_dir):
    """
    Plot 1: All strategies grouped by intent with best highlighted.

    - All bullish strategies (thin light green lines)
    - Best bullish (thick dark green)
    - All bearish strategies (thin light red lines)
    - Best bearish (thick dark red)
    - Benchmarks (gray dashed)
    """
    print("\nGenerating Plot 1: All Strategies with Best Highlighted...")

    fig, ax = plt.subplots(figsize=(16, 9))

    # Find best strategies
    best = find_best_strategies(summary_df)

    # Create strategy ID for matching
    def make_strategy_id(row):
        return f"{row['launch_month']}_{row['trigger_type']}_{row['selection_algo']}"

    summary_df['strategy_id'] = summary_df.apply(make_strategy_id, axis=1)

    # Only create IDs if strategies exist
    best_bullish_id = make_strategy_id(best['bullish_sharpe']) if 'bullish_sharpe' in best else None
    best_bearish_id = make_strategy_id(best['bearish_sharpe']) if 'bearish_sharpe' in best else None

    # Plot all strategies
    bullish_plotted = False
    bearish_plotted = False

    for result in results_list:
        result_id = f"{result['launch_month']}_{result['trigger_type']}_{result['selection_algo']}"
        intent = classify_strategy_intent(result['selection_algo'])

        daily = result['daily_performance']

        is_best_bullish = (result_id == best_bullish_id) if best_bullish_id else False
        is_best_bearish = (result_id == best_bearish_id) if best_bearish_id else False

        if intent == 'bullish':
            if is_best_bullish:
                ax.plot(daily['Date'], daily['Strategy_NAV'],
                        color=BULLISH_COLOR, linewidth=3, alpha=1.0,
                        label='Best Bullish (Sharpe)', zorder=10)
            else:
                if not bullish_plotted:
                    ax.plot(daily['Date'], daily['Strategy_NAV'],
                            color=BULLISH_LIGHT, linewidth=0.5, alpha=0.3,
                            label='Bullish Strategies', zorder=1)
                    bullish_plotted = True
                else:
                    ax.plot(daily['Date'], daily['Strategy_NAV'],
                            color=BULLISH_LIGHT, linewidth=0.5, alpha=0.3, zorder=1)

        elif intent == 'bearish':
            if is_best_bearish:
                ax.plot(daily['Date'], daily['Strategy_NAV'],
                        color=BEARISH_COLOR, linewidth=3, alpha=1.0,
                        label='Best Bearish (Sharpe)', zorder=10)
            else:
                if not bearish_plotted:
                    ax.plot(daily['Date'], daily['Strategy_NAV'],
                            color=BEARISH_LIGHT, linewidth=0.5, alpha=0.3,
                            label='Bearish Strategies', zorder=1)
                    bearish_plotted = True
                else:
                    ax.plot(daily['Date'], daily['Strategy_NAV'],
                            color=BEARISH_LIGHT, linewidth=0.5, alpha=0.3, zorder=1)

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
    ax.set_title('All Strategies: Bullish vs Bearish (Best Highlighted)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('NAV (Normalized to 100)', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save
    filename = 'plot1_all_strategies_with_best.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

    # Show in PyCharm
    plt.show()

    print(f"  ✓ Saved: {filename}")

    return filepath


def plot_2_best_comparison(results_list, summary_df, output_dir):
    """
    Plot 2: Best bullish vs best bearish vs benchmarks.

    Clean comparison of top performers by different criteria:
    - Best bullish (Sharpe)
    - Best bullish (Return)
    - Best bearish (Sharpe)
    - Best bearish (Drawdown)
    - SPY, BUFR
    """
    print("\nGenerating Plot 2: Best Strategy Comparison...")

    fig, ax = plt.subplots(figsize=(16, 9))

    # Find best strategies
    best = find_best_strategies(summary_df)

    # Create strategy ID for matching
    def make_strategy_id(row):
        return f"{row['launch_month']}_{row['trigger_type']}_{row['selection_algo']}"

    best_ids = {
        'bullish_sharpe': make_strategy_id(best['bullish_sharpe']),
        'bullish_return': make_strategy_id(best['bullish_return']),
        'bearish_sharpe': make_strategy_id(best['bearish_sharpe']),
        'bearish_drawdown': make_strategy_id(best['bearish_drawdown'])
    }

    # Plot best strategies
    for result in results_list:
        result_id = f"{result['launch_month']}_{result['trigger_type']}_{result['selection_algo']}"
        daily = result['daily_performance']

        if result_id == best_ids['bullish_sharpe']:
            ax.plot(daily['Date'], daily['Strategy_NAV'],
                    color=BULLISH_COLOR, linewidth=2.5, alpha=0.9,
                    label=f"Best Bullish (Sharpe: {best['bullish_sharpe']['strategy_sharpe']:.2f})",
                    zorder=10)

        elif result_id == best_ids['bullish_return']:
            ax.plot(daily['Date'], daily['Strategy_NAV'],
                    color=BULLISH_LIGHT, linewidth=2.5, alpha=0.9, linestyle='--',
                    label=f"Best Bullish (Return: {best['bullish_return']['strategy_return'] * 100:.2f}%)",
                    zorder=10)

        elif result_id == best_ids['bearish_sharpe']:
            ax.plot(daily['Date'], daily['Strategy_NAV'],
                    color=BEARISH_COLOR, linewidth=2.5, alpha=0.9,
                    label=f"Best Bearish (Sharpe: {best['bearish_sharpe']['strategy_sharpe']:.2f})",
                    zorder=10)

        elif result_id == best_ids['bearish_drawdown']:
            ax.plot(daily['Date'], daily['Strategy_NAV'],
                    color=BEARISH_LIGHT, linewidth=2.5, alpha=0.9, linestyle='--',
                    label=f"Best Bearish (Max DD: {best['bearish_drawdown']['strategy_max_dd'] * 100:.2f}%)",
                    zorder=10)

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
    ax.set_title('Best Strategies Comparison: Bullish vs Bearish',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('NAV (Normalized to 100)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save
    filename = 'plot2_best_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

    # Show in PyCharm
    plt.show()

    print(f"  ✓ Saved: {filename}")

    return filepath


def plot_3_best_by_forward_regime(results_list, summary_df, future_regime_df, optimal_6m, output_dir):
    """
    Plot 3: Best strategy for each forward regime.

    Shows which strategies perform best when entering different market regimes.
    """
    print("\nGenerating Plot 3: Best by Forward Regime...")

    if not optimal_6m or future_regime_df.empty:
        print("  ⚠ Skipping - no forward regime data available")
        return None

    fig, ax = plt.subplots(figsize=(16, 9))

    # Get best strategy for each forward regime from optimal_6m
    best_by_regime = {}

    for regime in ['bull', 'bear', 'neutral']:
        if regime in optimal_6m and not optimal_6m[regime].empty:
            best_row = optimal_6m[regime].iloc[0]  # Top performer
            best_by_regime[regime] = {
                'launch_month': best_row['launch_month'],
                'trigger_type': best_row['trigger_type'],
                'selection_algo': best_row['selection_algo']
            }

    # Plot best for each regime
    regime_colors = {
        'bull': BULLISH_COLOR,
        'bear': BEARISH_COLOR,
        'neutral': '#1565C0'  # Blue
    }

    for regime, info in best_by_regime.items():
        # Find matching result
        for result in results_list:
            if (result['launch_month'] == info['launch_month'] and
                    result['trigger_type'] == info['trigger_type'] and
                    result['selection_algo'] == info['selection_algo']):
                daily = result['daily_performance']
                ax.plot(daily['Date'], daily['Strategy_NAV'],
                        color=regime_colors[regime], linewidth=2.5, alpha=0.9,
                        label=f"Best for {regime.capitalize()} Entry: {info['launch_month']}",
                        zorder=10)
                break

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
    ax.set_title('Best Strategies by Forward Regime (6M)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('NAV (Normalized to 100)', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save
    filename = 'plot3_best_by_forward_regime.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

    # Show in PyCharm
    plt.show()

    print(f"  ✓ Saved: {filename}")

    return filepath


def create_all_batch_plots(results_list, summary_df, future_regime_df, optimal_6m, output_dir):
    """
    Generate all three plots for batch analysis.
    """
    print("\n" + "=" * 80)
    print("GENERATING BATCH PERFORMANCE PLOTS")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    plot_1_all_strategies_with_best(results_list, summary_df, output_dir)
    plot_2_best_comparison(results_list, summary_df, output_dir)
    plot_3_best_by_forward_regime(results_list, summary_df, future_regime_df, optimal_6m, output_dir)

    print("\n" + "=" * 80)
    print("ALL PLOTS COMPLETE")
    print("=" * 80 + "\n")