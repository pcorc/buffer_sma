"""
Main execution script for single ticker backtest framework.

This orchestrates the entire backtest process:
1. Load and validate data
2. Preprocess fund data with derived metrics
3. Classify market regimes
4. Run all backtests
5. Consolidate and analyze results
6. Export to CSV files

Usage:
    python main.py
"""

import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import configuration
from config.settings import (
    DATA_FILE, BENCHMARK_FILE, ROLL_DATES_FILE,
    RESULTS_DIR, REGIME_DIR, TRADE_LOG_DIR,
    SERIES, LAUNCH_MONTHS, COMBO_CONFIGS,
    REGIME_WINDOW_MONTHS, REGIME_BULL_THRESHOLD, REGIME_BEAR_THRESHOLD
)

# Import data handling
from data.loader import (
    load_fund_data, load_benchmark_data, load_roll_dates,
    validate_data_alignment
)
from data.preprocessor import preprocess_fund_data

# Import core functionality
from core.regime_classifier import classify_market_regimes
from core.selections import get_selection_function

# Import backtesting
from backtesting.batch_runner import run_all_single_ticker_tests

# Import analysis
from analysis.consolidator import (
    consolidate_results, create_performance_summary,
    summarize_by_launch_month, summarize_by_trigger_type,
    summarize_by_selection_algo
)
from analysis.regime_analyzer import (
    analyze_by_regime, compare_regime_performance,
    find_best_by_regime, calculate_capture_ratios
)

# Import utilities
from utils.exporters import (
    export_results, export_trade_logs, export_daily_performance,
    export_summary_stats, display_top_performers
)
from utils.validators import (
    validate_fund_data, validate_benchmark_data,
    validate_roll_dates, print_validation_results
)


def main():
    """
    Main execution function.
    """
    print("\n" + "#" * 80)
    print("# SINGLE TICKER BACKTEST FRAMEWORK")
    print("# Phase 1: Quarterly Rebalance, All 12 Launch Months, F-Series")
    print("#" * 80 + "\n")

    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================

    print("STEP 1: Loading data files...")
    print("-" * 80)

    try:
        df_raw = load_fund_data(DATA_FILE)
        df_benchmarks = load_benchmark_data(BENCHMARK_FILE)
        roll_dates_dict = load_roll_dates(ROLL_DATES_FILE)
    except Exception as e:
        print(f"\n❌ ERROR loading data: {str(e)}")
        print("Please check file paths in config/settings.py")
        return

    # =========================================================================
    # STEP 2: VALIDATE DATA
    # =========================================================================

    print("\nSTEP 2: Validating data...")
    print("-" * 80)

    # Validate fund data
    is_valid, errors = validate_fund_data(df_raw)
    print_validation_results("Fund Data", is_valid, errors)
    if not is_valid:
        print("⚠️  Proceeding despite validation errors...")

    # Validate benchmark data
    is_valid, errors = validate_benchmark_data(df_benchmarks)
    print_validation_results("Benchmark Data", is_valid, errors)
    if not is_valid:
        print("❌ Cannot proceed with invalid benchmark data")
        return

    # Validate roll dates
    is_valid, errors = validate_roll_dates(roll_dates_dict)
    print_validation_results("Roll Dates", is_valid, errors)
    if not is_valid:
        print("❌ Cannot proceed with invalid roll dates")
        return

    # Validate data alignment
    is_valid, errors, common_start, common_end = validate_data_alignment(
        df_raw, df_benchmarks
    )
    print_validation_results("Data Alignment", is_valid, errors)
    if not is_valid:
        print("❌ Cannot proceed without data alignment")
        return

    print(f"✅ All validations passed!")
    print(f"Common date range: {common_start.date()} to {common_end.date()}")

    # =========================================================================
    # STEP 3: PREPROCESS DATA
    # =========================================================================

    print("\nSTEP 3: Preprocessing fund data...")
    print("-" * 80)

    try:
        df_enriched = preprocess_fund_data(df_raw)
    except Exception as e:
        print(f"\n❌ ERROR during preprocessing: {str(e)}")
        return

    # =========================================================================
    # STEP 4: CLASSIFY MARKET REGIMES
    # =========================================================================

    print("\nSTEP 4: Classifying market regimes...")
    print("-" * 80)

    # Prepare SPY data for regime classification
    df_spy_for_regime = df_benchmarks[['Date', 'SPY']].copy()
    df_spy_for_regime.rename(columns={'SPY': 'Ref_Index'}, inplace=True)

    try:
        df_regimes = classify_market_regimes(
            df_spy_for_regime,
            window_months=REGIME_WINDOW_MONTHS,
            bull_threshold=REGIME_BULL_THRESHOLD,
            bear_threshold=REGIME_BEAR_THRESHOLD
        )
    except Exception as e:
        print(f"\n❌ ERROR during regime classification: {str(e)}")
        return

    # =========================================================================
    # STEP 5: PREPARE TRIGGER/SELECTION COMBINATIONS
    # =========================================================================

    print("\nSTEP 5: Preparing trigger/selection combinations...")
    print("-" * 80)

    # Convert config format to execution format
    trigger_selection_combos = []
    for config in COMBO_CONFIGS:
        combo = {
            'trigger_type': config['trigger_type'],
            'trigger_params': config['trigger_params'],
            'selection_func_name': config['selection_func_name']
        }
        trigger_selection_combos.append(combo)

    print(f"Prepared {len(trigger_selection_combos)} combinations:")
    for i, combo in enumerate(trigger_selection_combos, 1):
        print(f"  {i}. {combo['trigger_type']} + {combo['selection_func_name']}")

    print(f"\nTotal backtests: {len(LAUNCH_MONTHS)} months × {len(trigger_selection_combos)} combos = {len(LAUNCH_MONTHS) * len(trigger_selection_combos)}")

    # =========================================================================
    # STEP 6: RUN ALL BACKTESTS
    # =========================================================================

    print("\nSTEP 6: Running all backtests...")
    print("-" * 80)

    try:
        results_list = run_all_single_ticker_tests(
            df_enriched=df_enriched,
            df_benchmarks=df_benchmarks,
            roll_dates_dict=roll_dates_dict,
            trigger_selection_combos=trigger_selection_combos,
            launch_months=LAUNCH_MONTHS,
            series=SERIES
        )
    except Exception as e:
        print(f"\n❌ ERROR during backtesting: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    if not results_list:
        print("\n❌ No results generated. Check error messages above.")
        return

    # =========================================================================
    # STEP 7: CONSOLIDATE RESULTS
    # =========================================================================

    print("\nSTEP 7: Consolidating results...")
    print("-" * 80)

    try:
        summary_df = consolidate_results(results_list)

        # Create additional summary tables
        month_summary = summarize_by_launch_month(summary_df)
        trigger_summary = summarize_by_trigger_type(summary_df)
        selection_summary = summarize_by_selection_algo(summary_df)

        print(f"\n📊 Summary Statistics:")
        perf_summary = create_performance_summary(summary_df)
        print(f"  Total backtests: {perf_summary['total_backtests']}")
        print(f"  Avg vs BUFR: {perf_summary['avg_vs_bufr'] * 100:+.2f}%")
        print(f"  % beating BUFR: {perf_summary['pct_beat_bufr']:.1f}%")
        print(f"  % beating SPY: {perf_summary['pct_beat_spy']:.1f}%")

    except Exception as e:
        print(f"\n❌ ERROR during consolidation: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 8: ANALYZE BY REGIME
    # =========================================================================

    print("\nSTEP 8: Analyzing performance by regime...")
    print("-" * 80)

    try:
        regime_df = analyze_by_regime(results_list, df_regimes)

        regime_comparison = compare_regime_performance(regime_df)
        print("\n📈 Performance by Regime:")
        for _, row in regime_comparison.iterrows():
            print(f"  {row['regime'].capitalize():8s}: {row['vs_bufr_excess'] * 100:+6.2f}% vs BUFR (avg)")

        best_by_regime = find_best_by_regime(regime_df)
        print("\n🏆 Best Strategy by Regime:")
        for regime, best in best_by_regime.items():
            print(f"  {regime.capitalize():8s}: {best['launch_month']} + {best['trigger_type'][:20]}")

        capture_ratios = calculate_capture_ratios(regime_df)

    except Exception as e:
        print(f"\n❌ ERROR during regime analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        regime_df = pd.DataFrame()  # Continue with empty regime results

    # =========================================================================
    # STEP 9: EXPORT RESULTS
    # =========================================================================

    print("\nSTEP 9: Exporting results...")
    print("-" * 80)

    try:
        # Export main results
        summary_path, regime_path = export_results(summary_df, regime_df, RESULTS_DIR)

        # Export additional summaries
        month_summary.to_csv(os.path.join(RESULTS_DIR, 'summary_by_month.csv'), index=False)
        trigger_summary.to_csv(os.path.join(RESULTS_DIR, 'summary_by_trigger.csv'), index=False)
        selection_summary.to_csv(os.path.join(RESULTS_DIR, 'summary_by_selection.csv'), index=False)

        if not capture_ratios.empty:
            capture_ratios.to_csv(os.path.join(REGIME_DIR, 'capture_ratios.csv'), index=False)

        # Export trade logs
        export_trade_logs(results_list, TRADE_LOG_DIR)

        # Export summary report
        export_summary_stats(summary_df, regime_df, RESULTS_DIR)

        print(f"\n✅ All results exported to: {RESULTS_DIR}")

    except Exception as e:
        print(f"\n❌ ERROR during export: {str(e)}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # STEP 10: DISPLAY TOP PERFORMERS
    # =========================================================================

    display_top_performers(summary_df, n=10)

    # =========================================================================
    # COMPLETION
    # =========================================================================

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "#" * 80)
    print("# BACKTEST COMPLETE")
    print("#" * 80)
    print(f"Start time:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:      {duration}")
    print(f"Results saved: {RESULTS_DIR}")
    print("#" * 80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()