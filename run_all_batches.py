"""
Run all batches sequentially and generate cross-batch comparison.

This script:
1. Runs Batch 0, 1, 2, 3 in sequence
2. Stores results from each batch
3. Generates cross-batch comparison at the end
"""

import os
import sys
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configuration
from config import settings
from config.settings import SERIES, REGIME_WINDOW_MONTHS, REGIME_BULL_THRESHOLD, REGIME_BEAR_THRESHOLD

# Data handling
from data.loader import load_fund_data, load_benchmark_data, load_roll_dates
from data.preprocessor import preprocess_fund_data
from utils.validators import validate_fund_data, validate_benchmark_data

# Core
from core.regime_classifier import classify_market_regimes
from core.forward_regime_classifier import classify_forward_regimes

# Backtesting
from backtesting.batch_runner import run_all_single_ticker_tests

# Analysis
from analysis.consolidator import consolidate_results
from analysis.forward_regime_analyzer import analyze_by_future_regime, summarize_optimal_strategies

# Export
from utils.excel_exporter import export_main_consolidated_workbook

# Visualization
from visualization.performance_plots import generate_batch_visualizations, plot_cross_batch_comparison

# Batch configurations
from run_batch_tests import (
    get_batch_0_configs,
    get_batch_1_configs,
    get_batch_2_configs,
    get_batch_3_configs,
    get_batch_4_configs
)

BATCH_CONFIGS = {
    0: get_batch_0_configs,
    1: get_batch_1_configs,
    2: get_batch_2_configs,
    3: get_batch_3_configs,
    4: get_batch_4_configs,
}

BATCH_DESCRIPTIONS = {
    0: "Threshold Testing",
    1: "Time-Based Rebalancing",
    2: "Cap Utilization Tactical",
    3: "Remaining Cap Tactical",
    4: "Market-Responsive (Ref Asset + Buffer)",
}


def load_and_prepare_data():
    """Load and prepare all data needed for backtesting (one time)."""

    print("\n" + "=" * 80)
    print("LOADING AND PREPARING DATA")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading data...")
    df_raw = load_fund_data(settings.DATA_FILE, series=SERIES)
    df_benchmarks = load_benchmark_data(settings.BENCHMARK_FILE)
    roll_dates_dict = load_roll_dates(settings.ROLL_DATES_FILE)

    # Validate
    print("\n[2/5] Validating data...")
    is_valid, errors, df_raw = validate_fund_data(df_raw, series=SERIES)
    if not is_valid:
        raise ValueError("❌ Data validation failed")

    is_valid, errors, df_benchmarks = validate_benchmark_data(df_benchmarks)
    if not is_valid:
        raise ValueError("❌ Benchmark validation failed")

    # Preprocess
    print("\n[3/5] Preprocessing...")
    df_enriched = preprocess_fund_data(df_raw, roll_dates_dict)

    # Classify regimes
    print("\n[4/5] Classifying historical regimes...")
    df_spy_for_regime = df_benchmarks[['Date', 'SPY']].copy()
    df_spy_for_regime.rename(columns={'SPY': 'Ref_Index'}, inplace=True)

    df_regimes = classify_market_regimes(
        df_spy_for_regime,
        window_months=REGIME_WINDOW_MONTHS,
        bull_threshold=REGIME_BULL_THRESHOLD,
        bear_threshold=REGIME_BEAR_THRESHOLD
    )

    print("\n[5/5] Classifying forward regimes...")
    df_forward_regimes = classify_forward_regimes(
        df_spy_for_regime,
        window_3m_days=63,
        window_6m_days=126,
        bull_threshold=REGIME_BULL_THRESHOLD,
        bear_threshold=REGIME_BEAR_THRESHOLD
    )

    print("\n✅ Data preparation complete")

    return df_enriched, df_benchmarks, roll_dates_dict, df_regimes, df_forward_regimes


def run_single_batch(
        batch_number: int,
        df_enriched,
        df_benchmarks,
        roll_dates_dict,
        df_regimes,
        df_forward_regimes
):
    """
    Run a single batch and return results.

    Returns:
        Tuple of (results_list, summary_df, future_regime_df, optimal_3m, optimal_6m)
    """

    print("\n" + "#" * 80)
    print(f"# BATCH {batch_number}: {BATCH_DESCRIPTIONS[batch_number]}")
    print("#" * 80)

    # Get batch configurations
    batch_configs = BATCH_CONFIGS[batch_number]()

    # Calculate total simulations
    total_sims = sum(len(config['launch_months']) for config in batch_configs)
    print(f"\nThis batch: {len(batch_configs)} trigger/selection combinations")
    print(f"Total simulations: {total_sims}")
    print(f"Estimated time: ~{total_sims * 0.3:.0f} minutes\n")

    batch_start = datetime.now()

    # Run backtests
    print(f"{'=' * 80}")
    print(f"RUNNING BATCH {batch_number} BACKTESTS")
    print(f"{'=' * 80}\n")

    results_list = run_all_single_ticker_tests(
        df_enriched=df_enriched,
        df_benchmarks=df_benchmarks,
        roll_dates_dict=roll_dates_dict,
        trigger_selection_combos=batch_configs,
        series=SERIES
    )

    if not results_list:
        print(f"❌ Batch {batch_number}: No results generated")
        return None, None, None, None, None

    # Analyze
    print("\nAnalyzing results...")
    summary_df = consolidate_results(results_list)
    summary_df['batch'] = batch_number
    summary_df['batch_name'] = BATCH_DESCRIPTIONS[batch_number]

    # Forward regime analysis
    future_regime_df = analyze_by_future_regime(
        results_list,
        df_forward_regimes,
        entry_frequency='quarterly'
    )

    if not future_regime_df.empty:
        optimal_3m = summarize_optimal_strategies(future_regime_df, horizon='3M', top_n=10)
        optimal_6m = summarize_optimal_strategies(future_regime_df, horizon='6M', top_n=10)
    else:
        optimal_3m = {}
        optimal_6m = {}

    # Export
    print("\nExporting results...")
    output_dir = os.path.join(settings.RESULTS_DIR, f'batch_{batch_number}')
    os.makedirs(output_dir, exist_ok=True)

    workbook_path = export_main_consolidated_workbook(
        results_list=results_list,
        summary_df=summary_df,
        output_dir=output_dir,
        run_name=f'batch{batch_number}_{BATCH_DESCRIPTIONS[batch_number].replace(" ", "_")}',
        df_forward_regimes=df_forward_regimes,
        future_regime_df=future_regime_df,
        optimal_6m=optimal_6m,
        optimal_3m=optimal_3m,
        intent_vs_regime_6m=None,
        intent_vs_regime_3m=None,
        robust_strategies_6m=None,
        robust_strategies_3m=None,
        ranked_6m_vs_spy=None,
        ranked_6m_vs_bufr=None
    )

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_batch_visualizations(
        results_list=results_list,
        summary_df=summary_df,
        future_regime_df=future_regime_df,
        optimal_strategies=optimal_3m,
        output_dir=output_dir,
        batch_number=batch_number
    )

    batch_duration = datetime.now() - batch_start

    print(f"\n{'=' * 80}")
    print(f"✅ BATCH {batch_number} COMPLETE")
    print(f"{'=' * 80}")
    print(f"Duration: {batch_duration}")
    print(f"Simulations: {len(results_list)}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 80}\n")

    return results_list, summary_df, future_regime_df, optimal_3m, optimal_6m


def main():
    """Run all batches sequentially and generate cross-batch comparison."""

    overall_start = datetime.now()

    print("\n" + "=" * 80)
    print("RUNNING ALL BATCHES SEQUENTIALLY")
    print("=" * 80)
    print(f"Start time: {overall_start.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data once (shared across all batches)
    try:
        df_enriched, df_benchmarks, roll_dates_dict, df_regimes, df_forward_regimes = load_and_prepare_data()
    except Exception as e:
        print(f"\n❌ Data preparation failed: {e}")
        return

    # Storage for cross-batch comparison
    all_batch_summaries = {}
    all_batch_results = {}

    # Run each batch
    batches_to_run = [1, 2, 3]

    for batch_num in batches_to_run:
        try:
            results_list, summary_df, future_regime_df, optimal_3m, optimal_6m = run_single_batch(
                batch_number=batch_num,
                df_enriched=df_enriched,
                df_benchmarks=df_benchmarks,
                roll_dates_dict=roll_dates_dict,
                df_regimes=df_regimes,
                df_forward_regimes=df_forward_regimes
            )

            if summary_df is not None:
                all_batch_summaries[batch_num] = summary_df
                all_batch_results[batch_num] = results_list

        except Exception as e:
            print(f"\n❌ Batch {batch_num} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate cross-batch comparison
    if all_batch_summaries:

        output_dir = os.path.join(settings.RESULTS_DIR, 'cross_batch_comparison')
        os.makedirs(output_dir, exist_ok=True)

        try:
            plot_cross_batch_comparison(all_batch_summaries, output_dir)
        except Exception as e:
            print(f"\n⚠️  Cross-batch comparison failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  No batch results available for cross-batch comparison")

    # Final summary
    overall_duration = datetime.now() - overall_start

    print("\n" + "=" * 80)
    print("ALL BATCHES COMPLETE")
    print("=" * 80)
    print(f"Total duration: {overall_duration}")
    print(f"Batches completed: {len(all_batch_summaries)}/{len(batches_to_run)}")
    print(f"Results directory: {settings.RESULTS_DIR}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()