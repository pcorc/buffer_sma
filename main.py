"""
Main execution script for buffer ETF backtest framework.

Orchestrates the complete backtesting workflow:
1. Data loading and validation
2. Preprocessing and regime classification
3. Parallel backtest execution
4. Intent-based analysis
5. Visualization and Excel export

Usage:
    python main.py
"""

import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configuration
from config.settings import (
    DATA_FILE, BENCHMARK_FILE, ROLL_DATES_FILE, RESULTS_DIR,
    SERIES, LAUNCH_MONTHS, COMBO_CONFIGS,
    REGIME_WINDOW_MONTHS, REGIME_BULL_THRESHOLD, REGIME_BEAR_THRESHOLD
)

# Data handling
from data.loader import load_fund_data, load_benchmark_data, load_roll_dates
from data.preprocessor import preprocess_fund_data

# Core functionality
from core.regime_classifier import classify_market_regimes

# Backtesting (with parallel processing)
from backtesting.batch_runner import run_all_single_ticker_tests_parallel

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
    """Main execution function."""

    print("\n" + "=" * 80)
    print("BUFFER ETF BACKTEST FRAMEWORK")
    print("=" * 80)

    start_time = datetime.now()
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # =========================================================================
    # STEP 1: LOAD & VALIDATE DATA
    # =========================================================================

    print("\n[1/9] Loading and validating data...")

    try:
        # Load data
        df_raw = load_fund_data(DATA_FILE)
        df_benchmarks = load_benchmark_data(BENCHMARK_FILE)
        roll_dates_dict = load_roll_dates(ROLL_DATES_FILE)

        # Validate
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
        return

    # =========================================================================
    # STEP 2: PREPROCESS DATA
    # =========================================================================

    print("\n[2/9] Preprocessing fund data...")

    try:
        df_enriched = preprocess_fund_data(df_raw, roll_dates_dict)
        print(f"   ✓ Preprocessed {df_enriched['Outcome_Period_ID'].nunique()} outcome periods")
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        return

    # =========================================================================
    # STEP 3: CLASSIFY MARKET REGIMES
    # =========================================================================

    print("\n[3/9] Classifying market regimes...")

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
        return

    # =========================================================================
    # STEP 4: RUN BACKTESTS (PARALLEL)
    # =========================================================================

    print(f"\n[4/9] Running backtests...")
    print(f"   Strategies: {len(COMBO_CONFIGS)}")
    print(f"   Launch months: {len(LAUNCH_MONTHS)}")
    print(f"   Total simulations: {len(COMBO_CONFIGS) * len(LAUNCH_MONTHS)}")

    try:
        results_list = run_all_single_ticker_tests_parallel(
            df_enriched=df_enriched,
            df_benchmarks=df_benchmarks,
            roll_dates_dict=roll_dates_dict,
            trigger_selection_combos=COMBO_CONFIGS,
            launch_months=LAUNCH_MONTHS,
            series=SERIES,
            show_progress=True
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

    print("\n[5/9] Consolidating results...")

    try:
        summary_df = consolidate_results(results_list)
        print(f"   ✓ Consolidated {len(summary_df)} simulation results")

    except Exception as e:
        print(f"❌ Error during consolidation: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 6: INTENT CLASSIFICATION & ANALYSIS
    # =========================================================================

    print("\n[6/9] Analyzing by strategy intent...")

    try:
        # Add intent classification
        summary_df = add_strategy_intent_column(summary_df)

        # Group by intent
        intent_groups = create_intent_groups(summary_df)

        # Create intent summary
        intent_summary = create_intent_summary_table(summary_df)

        # Print intent distribution
        for intent in ['bullish', 'bearish', 'neutral', 'cost_optimized']:
            if intent in intent_groups:
                count = len(intent_groups[intent])
                avg_return = intent_groups[intent]['strategy_return'].mean()
                print(f"   {intent.capitalize():15s}: {count:3d} strategies | Avg return: {avg_return*100:+6.2f}%")

    except Exception as e:
        print(f"❌ Error during intent analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 7: REGIME ANALYSIS
    # =========================================================================

    print("\n[7/9] Analyzing by market regime...")

    try:
        regime_df = analyze_by_regime(results_list, df_regimes)
        print(f"   ✓ Analyzed {len(regime_df)} regime-specific results")

    except Exception as e:
        print(f"⚠️  Warning during regime analysis: {str(e)}")
        regime_df = pd.DataFrame()

    # =========================================================================
    # STEP 8: SELECT BEST STRATEGIES
    # =========================================================================

    print("\n[8/9] Selecting best strategies per intent...")

    try:
        best_strategies_df = select_best_by_intent(summary_df, regime_df)

        print("\n   Top Performers:")
        for _, row in best_strategies_df.iterrows():
            print(f"   {row['strategy_intent'].upper():15s}: "
                  f"{row['launch_month']} | "
                  f"Return: {row['strategy_return']*100:+6.2f}% | "
                  f"Sharpe: {row['strategy_sharpe']:5.2f}")

    except Exception as e:
        print(f"❌ Error selecting best strategies: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 9: GENERATE OUTPUTS
    # =========================================================================

    print("\n[9/9] Generating visualizations and Excel workbook...")

    try:
        # Generate all plots
        create_all_plots(
            results_list=results_list,
            summary_df=summary_df,
            best_strategies_df=best_strategies_df,
            output_dir=RESULTS_DIR
        )
        print("   ✓ Generated 4 performance charts")

        # Create Excel workbook
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workbook_path = create_comprehensive_workbook(
            summary_df=summary_df,
            best_strategies_df=best_strategies_df,
            intent_groups=intent_groups,
            regime_df=regime_df,
            intent_summary=intent_summary,
            output_dir=RESULTS_DIR,
            timestamp=timestamp
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
    print("BACKTEST COMPLETE")
    print("=" * 80)
    print(f"Duration: {duration}")
    print(f"Results: {RESULTS_DIR}")
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