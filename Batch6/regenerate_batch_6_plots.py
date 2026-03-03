"""
Regenerate Batch 6 Visualizations from Existing Results

This script reads the saved Excel workbook from a completed Batch 6 run
and regenerates only the visualization plots without re-running simulations.

Usage:
    1. Find your Batch 6 Excel file in: output/backtest_results/batch_6/
    2. Update EXCEL_FILE_PATH below with the actual filename
    3. Run: python regenerate_batch_6_plots.py
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from visualization.performance_plots import (
    plot_ecr_regime_comparison,
    plot_ecr_by_trigger_type,
    generate_batch_visualizations
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# UPDATE THIS PATH to match your actual Excel file
BATCH_6_DIR = 'output/backtest_results/batch_6'


# The script will auto-detect the most recent Excel file in the directory
# Or you can specify it manually:
# EXCEL_FILE_PATH = 'output/backtest_results/batch_6/batch6_Enhanced_Cost_Ratio_Testing_YYYYMMDD_HHMMSS.xlsx'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_latest_excel_file(directory):
    """Find the most recent Excel file in the directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    excel_files = [f for f in os.listdir(directory) if f.endswith('.xlsx') and f.startswith('batch6')]

    if not excel_files:
        raise FileNotFoundError(f"No Batch 6 Excel files found in {directory}")

    # Sort by modification time, get most recent
    excel_files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)

    return os.path.join(directory, excel_files[0])


def load_summary_from_excel(excel_path):
    """Load the Summary tab from the Excel workbook."""
    print(f"\nLoading results from: {excel_path}")

    try:
        summary_df = pd.read_excel(excel_path, sheet_name='Summary')
        print(f"✓ Loaded {len(summary_df)} simulation results")
        return summary_df
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        raise


def reconstruct_results_list_minimal(summary_df):
    """
    Create a minimal results_list structure for plotting.

    Note: We can't fully reconstruct daily_performance data from the Excel,
    so we'll only generate plots that don't require it.
    """
    # For ECR plots that only need summary_df, we don't need results_list
    # But generate_batch_visualizations expects it, so return empty list
    return []


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""

    print("\n" + "=" * 80)
    print("BATCH 6 PLOT REGENERATION")
    print("=" * 80)

    # Step 1: Find Excel file
    try:
        excel_path = find_latest_excel_file(BATCH_6_DIR)
        print(f"\nFound Excel file: {os.path.basename(excel_path)}")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nPlease ensure:")
        print("  1. Batch 6 has been run successfully")
        print("  2. Excel file exists in output/backtest_results/batch_6/")
        return

    # Step 2: Load summary data
    try:
        summary_df = load_summary_from_excel(excel_path)
    except Exception as e:
        print(f"\n❌ Failed to load Excel file: {e}")
        return

    # Step 3: Verify it's ECR data
    ecr_mask = summary_df['selection_algo'].str.contains('enhanced_cost_ratio', na=False)
    ecr_count = ecr_mask.sum()

    if ecr_count == 0:
        print("\n❌ No Enhanced Cost Ratio strategies found in this file!")
        print("   Are you sure this is a Batch 6 results file?")
        return

    print(f"✓ Found {ecr_count} Enhanced Cost Ratio strategies")

    # Step 4: Create output directory for plots
    output_dir = os.path.dirname(excel_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Step 5: Generate ECR-specific plots
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    generated_plots = {}

    # Plot 1: ECR Regime Comparison
    try:
        print("\n  Generating: ECR Regime Weighting Comparison...")
        filepath = plot_ecr_regime_comparison(summary_df, output_dir)
        if filepath:
            generated_plots['ecr_regime_comparison'] = filepath
            print(f"    ✓ Saved: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"    ❌ Error: {e}")

    # Plot 2: ECR Performance by Trigger Type
    try:
        print("\n  Generating: ECR Performance by Trigger Type...")
        filepath = plot_ecr_by_trigger_type(summary_df, output_dir)
        if filepath:
            generated_plots['ecr_by_trigger_type'] = filepath
            print(f"    ✓ Saved: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"    ❌ Error: {e}")

    # Note: plot_ecr_top_performers requires results_list with daily_performance data
    # which we can't reconstruct from Excel. Skip it or load from pickle if available.
    print("\n  ⊘ Skipping: ECR Top Performers (requires full results data)")
    print("     To generate this plot, you'll need to re-run the full batch")

    # Step 6: Summary
    print("\n" + "=" * 80)
    print(f"PLOT GENERATION COMPLETE")
    print("=" * 80)
    print(f"Generated {len(generated_plots)} plots:")
    for plot_name, filepath in generated_plots.items():
        print(f"  ✓ {plot_name}: {os.path.basename(filepath)}")

    print(f"\nPlots saved to: {output_dir}")
    print("=" * 80 + "\n")

    # Step 7: Display key findings from summary
    print("\n" + "=" * 80)
    print("BATCH 6 KEY FINDINGS (from Excel)")
    print("=" * 80)

    # Top 5 strategies by Sharpe
    top_5 = summary_df.nlargest(5, 'strategy_sharpe')
    print("\nTop 5 Strategies by Sharpe Ratio:")
    print("-" * 80)
    for i, (idx, row) in enumerate(top_5.iterrows(), 1):
        print(f"{i}. {row['selection_algo']:40s} | "
              f"Sharpe: {row['strategy_sharpe']:5.2f} | "
              f"Return: {row['strategy_return'] * 100:+6.2f}% | "
              f"vs BUFR: {row['vs_bufr_excess'] * 100:+5.2f}%")

    # Performance by regime
    ecr_df = summary_df[ecr_mask].copy()

    def extract_regime(selection_algo):
        if 'bullish' in selection_algo:
            return 'Bullish'
        elif 'bearish' in selection_algo:
            return 'Bearish'
        elif 'neutral' in selection_algo:
            return 'Neutral'
        return 'Unknown'

    ecr_df['ecr_regime'] = ecr_df['selection_algo'].apply(extract_regime)

    regime_stats = ecr_df.groupby('ecr_regime').agg({
        'strategy_return': 'mean',
        'strategy_sharpe': 'mean',
        'vs_bufr_excess': 'mean'
    }).round(4)

    print("\n\nPerformance by ECR Regime Weighting:")
    print("-" * 80)
    print(f"{'Regime':<12} {'Avg Return':<15} {'Avg Sharpe':<15} {'Avg vs BUFR':<15}")
    print("-" * 80)
    for regime, row in regime_stats.iterrows():
        print(f"{regime:<12} {row['strategy_return'] * 100:>6.2f}%        "
              f"{row['strategy_sharpe']:>6.2f}         "
              f"{row['vs_bufr_excess'] * 100:>+6.2f}%")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()