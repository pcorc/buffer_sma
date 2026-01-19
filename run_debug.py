"""
Test execution script - use this for debugging and validation.

Runs minimal subset of simulations for quick testing.
"""

import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configuration
from config.settings import (
    DATA_FILE, BENCHMARK_FILE, ROLL_DATES_FILE, RESULTS_DIR, SERIES,
    REGIME_WINDOW_MONTHS, REGIME_BULL_THRESHOLD, REGIME_BEAR_THRESHOLD,
    COMBO_CONFIGS
)

# Test configurations
from config.test_config import (
    get_minimal_test_set,
    get_small_test_set,
    get_medium_test_set,
    print_test_set_info,
    filter_by_intent
)

# Data handling
from data.loader import load_fund_data, load_benchmark_data, load_roll_dates
from data.preprocessor import preprocess_fund_data

# Core functionality
from core.regime_classifier import classify_market_regimes

# Backtesting (sequential for debugging)
from backtesting.batch_runner import run_all_single_ticker_tests

# Analysis
from analysis.consolidator import consolidate_results
from analysis.regime_analyzer import analyze_by_regime
from analysis.intent_analyzer import (
    add_strategy_intent_column,
    create_intent_groups,
    select_best_by_intent,
    create_intent_summary_table
)

# Visualization
from visualization.performance_plots import create_all_plots

# Excel export
from utils.excel_consolidator import create_comprehensive_workbook

# Validation
from utils.validators import (
    validate_fund_data, validate_benchmark_data,
    validate_roll_dates, validate_data_alignment
)


def main():
    """Test execution function."""

    print("\n" + "=" * 80)
    print("BUFFER ETF BACKTEST - TESTING MODE")
    print("=" * 80)

    # =========================================================================
    # SELECT TEST CONFIGURATION
    # =========================================================================

    # Choose one:
    # test_combos, test_months = get_minimal_test_set()    # 1 simulation
    # test_combos, test_months = get_small_test_set()  # 8 simulations
    # test_combos, test_months = get_medium_test_set()    # ~112 simulations


    # Test one threshold type at 3 different levels
    # test_combos = [
    #     {
    #         'trigger_type': 'cap_utilization_threshold',
    #         'trigger_params': {'threshold': 0.15},  # 15% utilized
    #         'selection_func_name': 'select_most_recent_launch',
    #         'description': 'Switch at 15% utilization'
    #     },
    #     # {
    #     #     'trigger_type': 'cap_utilization_threshold',
    #     #     'trigger_params': {'threshold': 0.50},  # 50% utilized
    #     #     'selection_func_name': 'select_most_recent_launch',
    #     #     'description': 'Switch at 50% utilization'
    #     # },
    #     # {
    #     #     'trigger_type': 'cap_utilization_threshold',
    #     #     'trigger_params': {'threshold': 0.85},  # 85% utilized
    #     #     'selection_func_name': 'select_most_recent_launch',
    #     #     'description': 'Switch at 85% utilization'
    #     # }
    # ]
    # test_months = ['MAR']
    # print_test_set_info(test_combos, test_months, "Selected Test Set")
    # Total: 3 simulations

    # Define threshold levels to test
    THRESHOLD_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90]

    # Define selection algorithms to test with each trigger
    SELECTION_ALGOS = [
        'select_most_recent_launch',
        'select_remaining_cap_highest',
        'select_remaining_cap_lowest',
        'select_cap_utilization_lowest',
        'select_cap_utilization_highest',
    ]

    # Build test combinations dynamically
    test_combos = []

    # Remaining cap threshold combinations
    for threshold in THRESHOLD_LEVELS:
        for selection in SELECTION_ALGOS:
            test_combos.append({
                'trigger_type': 'remaining_cap_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'description': f'Remaining cap {threshold:.0%} → {selection}'
            })

    # Cap utilization threshold combinations
    for threshold in THRESHOLD_LEVELS:
        for selection in SELECTION_ALGOS:
            test_combos.append({
                'trigger_type': 'cap_utilization_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'description': f'Cap utilization {threshold:.0%} → {selection}'
            })

    test_months = ['MAR', 'JUN', 'SEP', 'DEC']
    test_months = ['SEP']

    # =========================================================================
    # DEBUG EXPORT CONFIGURATION
    # =========================================================================

    DEBUG_EXPORT_ENABLED = True   # Toggle debug time series exports
    DEBUG_EXPORT_ALL = True       # If False, only export first N configs
    DEBUG_EXPORT_N = 3            # Number of configs to export (if DEBUG_EXPORT_ALL = False)
    DEBUG_EXPORT_DIR = os.path.join(RESULTS_DIR, 'debug_time_series')

    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # =========================================================================
    # STEP 1: LOAD & VALIDATE DATA
    # =========================================================================

    print("[1/10] Loading and validating data...")

    try:
        df_raw = load_fund_data(DATA_FILE)
        df_benchmarks = load_benchmark_data(BENCHMARK_FILE)
        roll_dates_dict = load_roll_dates(ROLL_DATES_FILE)

        is_valid, _, df_raw = validate_fund_data(df_raw, series=SERIES)
        if not is_valid:
            print("❌ Fund data validation failed")
            return

        is_valid, _, df_benchmarks = validate_benchmark_data(df_benchmarks)
        if not is_valid:
            print("❌ Benchmark data validation failed")
            return

        is_valid, _ = validate_roll_dates(roll_dates_dict)
        if not is_valid:
            print("❌ Roll dates validation failed")
            return

        is_valid, _, common_start, common_end = validate_data_alignment(df_raw, df_benchmarks)
        if not is_valid:
            print("❌ Data alignment validation failed")
            return

        print(f"   ✓ Data validated: {common_start.date()} to {common_end.date()}")

    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 2: PREPROCESS DATA
    # =========================================================================

    print("\n[2/10] Preprocessing fund data...")

    try:
        df_enriched = preprocess_fund_data(df_raw, roll_dates_dict)
        print(f"   ✓ Preprocessed {df_enriched['Outcome_Period_ID'].nunique()} outcome periods")
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 3: CLASSIFY MARKET REGIMES
    # =========================================================================

    print("\n[3/10] Classifying market regimes...")

    try:
        df_spy_for_regime = df_benchmarks[['Date', 'SPY']].copy()
        df_spy_for_regime.rename(columns={'SPY': 'Ref_Index'}, inplace=True)

        df_regimes = classify_market_regimes(
            df_spy_for_regime,
            window_months=REGIME_WINDOW_MONTHS,
            bull_threshold=REGIME_BULL_THRESHOLD,
            bear_threshold=REGIME_BEAR_THRESHOLD
        )

        regime_counts = df_regimes['Regime'].value_counts()
        print(f"   ✓ Bull: {regime_counts.get('bull', 0)} days | "
              f"Bear: {regime_counts.get('bear', 0)} days | "
              f"Neutral: {regime_counts.get('neutral', 0)} days")

    except Exception as e:
        print(f"❌ Error during regime classification: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 4: RUN BACKTESTS (SEQUENTIAL FOR DEBUGGING)
    # =========================================================================

    print(f"\n[4/10] Running backtests (SEQUENTIAL for debugging)...")

    try:
        results_list = run_all_single_ticker_tests(
            df_enriched=df_enriched,
            df_benchmarks=df_benchmarks,
            roll_dates_dict=roll_dates_dict,
            trigger_selection_combos=test_combos,
            launch_months=test_months,
            series=SERIES
        )

        if not results_list:
            print("❌ No results generated")
            return

        print(f"   ✓ Completed {len(results_list)} backtests")

    except Exception as e:
        print(f"❌ Error during backtesting: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 5: CONSOLIDATE RESULTS
    # =========================================================================

    print("\n[5/10] Consolidating results...")

    try:
        summary_df = consolidate_results(results_list)
        print(f"   ✓ Consolidated {len(summary_df)} simulation results")

    except Exception as e:
        print(f"❌ Error during consolidation: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 5.5: DEBUG EXPORT (OPTIONAL)
    # =========================================================================

    if DEBUG_EXPORT_ENABLED:
        print("\n[5.5/10] Exporting debug time series...")

        try:
            from utils.debug_exporter import export_debug_time_series

            # Determine which results to export
            if DEBUG_EXPORT_ALL:
                results_to_export = results_list
                print(f"   Exporting ALL {len(results_to_export)} configurations")
            else:
                results_to_export = results_list[:DEBUG_EXPORT_N]
                print(f"   Exporting first {len(results_to_export)} of {len(results_list)} configurations")

            export_debug_time_series(
                results_list=results_to_export,
                output_dir=DEBUG_EXPORT_DIR,
                df_regimes=df_regimes
            )

            print(f"   ✓ Exported {len(results_to_export)} time series to {DEBUG_EXPORT_DIR}")

        except Exception as e:
            print(f"⚠️  Warning during debug export: {str(e)}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # STEP 6: INTENT CLASSIFICATION & ANALYSIS
    # =========================================================================

    print("\n[6/10] Analyzing by strategy intent...")

    try:
        summary_df = add_strategy_intent_column(summary_df)
        intent_groups = create_intent_groups(summary_df)
        intent_summary = create_intent_summary_table(summary_df)

        for intent in ['bullish', 'bearish', 'neutral', 'cost_optimized']:
            if intent in intent_groups and not intent_groups[intent].empty:
                count = len(intent_groups[intent])
                avg_return = intent_groups[intent]['strategy_return'].mean()
                print(f"   {intent.capitalize():15s}: {count:3d} strategies | Avg return: {avg_return * 100:+6.2f}%")

    except Exception as e:
        print(f"❌ Error during intent analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 7: REGIME ANALYSIS
    # =========================================================================

    print("\n[7/10] Analyzing by market regime...")

    try:
        regime_df = analyze_by_regime(results_list, df_regimes)
        print(f"   ✓ Analyzed {len(regime_df)} regime-specific results")

    except Exception as e:
        print(f"⚠️  Warning during regime analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        regime_df = pd.DataFrame()

    # =========================================================================
    # STEP 8: SELECT BEST STRATEGIES
    # =========================================================================

    print("\n[8/10] Selecting best strategies per intent...")

    try:
        best_strategies_df = select_best_by_intent(summary_df, regime_df)

        if not best_strategies_df.empty:
            print("\n   Top Performers:")
            for _, row in best_strategies_df.iterrows():
                print(f"   {row['strategy_intent'].upper():15s}: "
                      f"{row['launch_month']} | "
                      f"Return: {row['strategy_return'] * 100:+6.2f}% | "
                      f"Sharpe: {row['strategy_sharpe']:5.2f}")
        else:
            print("   ⚠️  No best strategies selected (may need more intent diversity)")

    except Exception as e:
        print(f"❌ Error selecting best strategies: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 9: GENERATE OUTPUTS
    # =========================================================================

    print("\n[9/10] Generating outputs...")

    try:
        # Only generate visualizations if we have best strategies
        if not best_strategies_df.empty:
            create_all_plots(
                results_list=results_list,
                summary_df=summary_df,
                best_strategies_df=best_strategies_df,
                output_dir=RESULTS_DIR
            )
            print("   ✓ Generated visualizations")
        else:
            print("   ⚠️  Skipping visualizations (need best strategies)")

        # Always create Excel workbook
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workbook_path = create_comprehensive_workbook(
            summary_df=summary_df,
            best_strategies_df=best_strategies_df if not best_strategies_df.empty else pd.DataFrame(),
            intent_groups=intent_groups,
            regime_df=regime_df,
            intent_summary=intent_summary,
            output_dir=RESULTS_DIR,
            timestamp=f"TEST_{timestamp}"
        )
        print(f"   ✓ Created Excel workbook")

    except Exception as e:
        print(f"❌ Error generating outputs: {str(e)}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # COMPLETION
    # =========================================================================

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"Duration: {duration}")
    print(f"Results: {RESULTS_DIR}")
    if DEBUG_EXPORT_ENABLED:
        print(f"Debug exports: {DEBUG_EXPORT_DIR}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    import pandas as pd  # Need this for DataFrame()

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()