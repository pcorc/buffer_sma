"""
Export utilities for saving backtest results to files.
"""

import os
import pandas as pd
from datetime import datetime


def export_results(summary_df, regime_df, output_dir):
    """
    Export consolidated results and regime analysis to CSV files.
    Appends to existing files if they exist.

    Parameters:
      summary_df: Consolidated summary DataFrame
      regime_df: Regime-specific analysis DataFrame
      output_dir: Directory path for output files
    """
    print(f"\n{'=' * 80}")
    print(f"EXPORTING RESULTS")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    # =============================================================================
    # EXPORT SUMMARY - APPEND MODE
    # =============================================================================

    # Path for cumulative results file
    cumulative_summary_path = os.path.join(output_dir, 'backtest_results_cumulative.csv')

    # Add run timestamp to each row
    summary_df['run_timestamp'] = timestamp

    if os.path.exists(cumulative_summary_path):
        # Append to existing file
        existing_df = pd.read_csv(cumulative_summary_path)
        combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
        combined_df.to_csv(cumulative_summary_path, index=False)

    else:
        # Create new file
        summary_df.to_csv(cumulative_summary_path, index=False)


    # Also save timestamped version for backup
    summary_filename = f'backtest_results_{timestamp}.csv'
    summary_path = os.path.join(output_dir, summary_filename)
    summary_df.to_csv(summary_path, index=False)

    # Also save as "latest" for easy access
    latest_summary_path = os.path.join(output_dir, 'backtest_results_latest.csv')
    summary_df.to_csv(latest_summary_path, index=False)

    # =============================================================================
    # EXPORT REGIME ANALYSIS - APPEND MODE
    # =============================================================================

    cumulative_regime_path = os.path.join(output_dir, 'regime_analysis_cumulative.csv')

    # Add run timestamp
    regime_df['run_timestamp'] = timestamp

    if os.path.exists(cumulative_regime_path):
        # Append to existing file
        existing_regime = pd.read_csv(cumulative_regime_path)
        combined_regime = pd.concat([existing_regime, regime_df], ignore_index=True)
        combined_regime.to_csv(cumulative_regime_path, index=False)

    else:
        # Create new file
        regime_df.to_csv(cumulative_regime_path, index=False)

    # Save timestamped and latest versions
    regime_filename = f'regime_analysis_{timestamp}.csv'
    regime_path = os.path.join(output_dir, regime_filename)
    regime_df.to_csv(regime_path, index=False)

    latest_regime_path = os.path.join(output_dir, 'regime_analysis_latest.csv')
    regime_df.to_csv(latest_regime_path, index=False)


    return summary_path, regime_path

def export_trade_logs(results_list, output_dir):
    """
    Export detailed trade history for all backtests.

    Parameters:
      results_list: List of result dicts from backtests
      output_dir: Directory path for output files
    """
    print(f"Exporting trade logs to {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    all_trades = []

    for result in results_list:
        if result['trade_history'].empty:
            continue

        trades = result['trade_history'].copy()
        trades['launch_month'] = result['launch_month']
        trades['trigger_type'] = result['trigger_type']
        trades['selection_algo'] = result['selection_algo']

        all_trades.append(trades)

    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'trade_logs_{timestamp}.csv'
        filepath = os.path.join(output_dir, filename)

        combined_trades.to_csv(filepath, index=False)
        print(f"  ✓ Saved: {filename}")
        print(f"    Total trades: {len(combined_trades)}")
    else:
        print(f"  ⚠ No trades to export")


def export_daily_performance(results_list, output_dir, save_individual=False):
    """
    Export daily NAV series for all backtests.

    Parameters:
      results_list: List of result dicts from backtests
      output_dir: Directory path for output files
      save_individual: If True, save separate CSV for each backtest
    """
    print(f"Exporting daily performance to {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    if save_individual:
        # Save each backtest as separate file
        for result in results_list:
            filename = f"daily_nav_{result['launch_month']}_{result['trigger_type']}_{result['selection_algo']}.csv"
            filepath = os.path.join(output_dir, filename)
            result['daily_performance'].to_csv(filepath, index=False)

        print(f"  ✓ Saved {len(results_list)} individual daily NAV files")

    else:
        # Combine all into one file
        all_daily = []

        for result in results_list:
            daily = result['daily_performance'].copy()
            daily['launch_month'] = result['launch_month']
            daily['trigger_type'] = result['trigger_type']
            daily['selection_algo'] = result['selection_algo']

            all_daily.append(daily)

        if all_daily:
            combined_daily = pd.concat(all_daily, ignore_index=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'daily_performance_{timestamp}.csv'
            filepath = os.path.join(output_dir, filename)

            combined_daily.to_csv(filepath, index=False)
            print(f"  ✓ Saved: {filename}")
            print(f"    Total rows: {len(combined_daily)}")


def export_summary_stats(summary_df, regime_df, output_dir):
    """
    Export high-level summary statistics as a formatted text report.

    Parameters:
      summary_df: Consolidated summary DataFrame
      regime_df: Regime-specific analysis DataFrame
      output_dir: Directory path for output files
    """
    from analysis.consolidator import create_performance_summary
    from analysis.regime_analyzer import compare_regime_performance, find_best_by_regime

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'summary_report_{timestamp}.txt'
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BACKTEST SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        # Overall statistics
        perf_summary = create_performance_summary(summary_df)

        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total backtests: {perf_summary['total_backtests']}\n")
        f.write(f"Average vs BUFR: {perf_summary['avg_vs_bufr']*100:+.2f}%\n")
        f.write(f"Median vs BUFR: {perf_summary['median_vs_bufr']*100:+.2f}%\n")
        f.write(f"% beating BUFR: {perf_summary['pct_beat_bufr']:.1f}%\n")
        f.write(f"% beating SPY: {perf_summary['pct_beat_spy']:.1f}%\n")
        f.write(f"% beating Hold: {perf_summary['pct_beat_hold']:.1f}%\n")
        f.write(f"Average Sharpe: {perf_summary['avg_sharpe']:.2f}\n")
        f.write(f"Average trades: {perf_summary['avg_trades']:.1f}\n\n")

        # Best performer
        f.write("BEST PERFORMER (vs BUFR)\n")
        f.write("-"*80 + "\n")
        best = perf_summary['best_vs_bufr']
        f.write(f"Launch month: {best['launch_month']}\n")
        f.write(f"Trigger: {best['trigger']}\n")
        f.write(f"Selection: {best['selection']}\n")
        f.write(f"Excess return: {best['excess_return']*100:+.2f}%\n\n")

        # Top 10 performers
        f.write("TOP 10 PERFORMERS (vs BUFR)\n")
        f.write("-"*80 + "\n")
        top_10 = summary_df.nlargest(10, 'vs_bufr_excess')
        for idx, row in top_10.iterrows():
            f.write(f"{idx+1:2d}. {row['launch_month']:3s} | {row['trigger_type']:30s} | {row['selection_algo']:30s}\n")
            f.write(f"    vs BUFR: {row['vs_bufr_excess']*100:+6.2f}% | Sharpe: {row['strategy_sharpe']:5.2f} | Trades: {row['num_trades']:3.0f}\n")
        f.write("\n")

        # Regime comparison
        if not regime_df.empty:
            regime_comp = compare_regime_performance(regime_df)

            f.write("PERFORMANCE BY REGIME\n")
            f.write("-"*80 + "\n")
            for _, row in regime_comp.iterrows():
                f.write(f"{row['regime'].capitalize():8s}:\n")
                f.write(f"  Avg Strategy Return: {row['strategy_return']*100:+6.2f}%\n")
                f.write(f"  Avg vs BUFR: {row['vs_bufr_excess']*100:+6.2f}%\n")
                f.write(f"  Total days: {row['days_in_regime']:.0f}\n")
                f.write(f"  Total trades: {row['num_trades']:.0f}\n\n")

            # Best by regime
            best_by_regime = find_best_by_regime(regime_df)

            f.write("BEST STRATEGY BY REGIME\n")
            f.write("-"*80 + "\n")
            for regime, best in best_by_regime.items():
                f.write(f"{regime.capitalize():8s}: {best['launch_month']} | {best['trigger_type']} | {best['selection_algo']}\n")
                f.write(f"  vs BUFR: {best['vs_bufr_excess']*100:+6.2f}%\n\n")

        f.write("="*80 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")

    print(f"  ✓ Saved: {filename}")


def display_top_performers(summary_df, n=5):
    """
    Display top N performers in console with formatting.

    Parameters:
      summary_df: Consolidated summary DataFrame
      n: Number of top performers to display
    """
    print(f"\n{'='*80}")
    print(f"TOP {n} PERFORMERS (vs BUFR)")
    print(f"{'='*80}\n")

    if summary_df.empty:
        print("No results to display!")
        return

    top_n = summary_df.nlargest(n, 'vs_bufr_excess')

    for idx, row in top_n.iterrows():
        print(f"{idx+1}. {row['launch_month']} | {row['trigger_type']} | {row['selection_algo']}")
        print(f"   vs BUFR: {row['vs_bufr_excess']*100:+6.2f}% | vs SPY: {row['vs_spy_excess']*100:+6.2f}% | vs Hold: {row['vs_hold_excess']*100:+6.2f}%")
        print(f"   Return: {row['strategy_return']*100:+6.2f}% | Sharpe: {row['strategy_sharpe']:5.2f} | Max DD: {row['strategy_max_dd']*100:6.2f}%")
        print(f"   Trades: {row['num_trades']:3.0f} | Period: {row['start_date'].date()} to {row['end_date'].date()}")
        print()

    print(f"{'='*80}\n")