"""
Single fund test script for validation and debugging.

Tests one launch month across all trigger/selection combinations
to validate calculations before running full backtest suite.

Usage:
    python main_single_fund_test.py
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import traceback

# Import configuration
from config.settings import (
    DATA_FILE, BENCHMARK_FILE, ROLL_DATES_FILE,
    RESULTS_DIR, SERIES, COMBO_CONFIGS,
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
from backtesting.engine import run_single_ticker_backtest

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
from utils.validators import (
    validate_fund_data, validate_benchmark_data,
    validate_roll_dates, print_validation_results
)

# Export consolidated Excel workbook
from utils.excel_exporter import export_consolidated_workbook


def main():
    """
    Main execution function for single fund testing.
    """

    # =============================================================================
    # CONFIGURATION FOR SINGLE FUND TEST
    # =============================================================================

    TEST_LAUNCH_MONTH = 'JUL'  # Change this to test different months
    TEST_SERIES = 'F'  # Fund series to test

    # Optional: Filter to specific trigger/selection combos for faster testing
    # Set to None to test all combinations
    FILTER_TRIGGER_TYPES = None  # Example: ['rebalance_time_period']
    FILTER_SELECTION_ALGOS = None  # Example: ['select_most_recent_launch']

    print("\n" + "#" * 80)
    print("# SINGLE FUND TEST MODE" + f"Testing: {TEST_SERIES}{TEST_LAUNCH_MONTH}")

    # =============================================================================
    # STEP 1: LOAD DATA
    # =============================================================================

    print("STEP 1: Loading data files...")
    print("-" * 80)

    try:
        df_raw = load_fund_data(DATA_FILE, series=TEST_SERIES)
        df_benchmarks = load_benchmark_data(BENCHMARK_FILE)
        roll_dates_dict = load_roll_dates(ROLL_DATES_FILE)
    except Exception as e:
        print(f"\n❌ ERROR loading data: {str(e)}")
        return

    # =============================================================================
    # STEP 2: VALIDATE DATA
    # =============================================================================

    print("\nSTEP 2: Validating data...")
    print("-" * 80)

    is_valid, errors, df_raw = validate_fund_data(df_raw, series=TEST_SERIES)
    print_validation_results("Fund Data", is_valid, errors)
    if not is_valid:
        print("❌ Cannot proceed with invalid fund data")
        return

    is_valid, errors, df_benchmarks = validate_benchmark_data(df_benchmarks)
    print_validation_results("Benchmark Data", is_valid, errors)
    if not is_valid:
        print("❌ Cannot proceed with invalid benchmark data")
        return

    is_valid, errors = validate_roll_dates(roll_dates_dict)
    print_validation_results("Roll Dates", is_valid, errors)
    if not is_valid:
        print("❌ Cannot proceed with invalid roll dates")
        return

    is_valid, errors, common_start, common_end = validate_data_alignment(
        df_raw, df_benchmarks
    )
    print_validation_results("Data Alignment", is_valid, errors)
    if not is_valid:
        print("❌ Cannot proceed without data alignment")
        return

    # =============================================================================
    # STEP 3: PREPROCESS DATA
    # =============================================================================

    print("\nSTEP 3: Preprocessing fund data...")
    print("-" * 80)

    try:
        df_enriched = preprocess_fund_data(df_raw, roll_dates_dict)
    except Exception as e:
        print(f"\n❌ ERROR during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # =============================================================================
    # DETAILED INSPECTION OF TEST FUND
    # =============================================================================

    print(f"INSPECTING {TEST_SERIES}{TEST_LAUNCH_MONTH} DATA")
    print("=" * 80)

    test_fund = TEST_SERIES + TEST_LAUNCH_MONTH
    fund_data = df_enriched[df_enriched['Fund'] == test_fund].copy()

    if fund_data.empty:
        print(f"❌ ERROR: No data found for {test_fund}")
        return
    #
    # print(f"\nData Summary:")
    # print(f"  Date range: {fund_data['Date'].min().date()} to {fund_data['Date'].max().date()}")
    # print(f"  Total observations: {len(fund_data):,}")
    # print(f"  Unique outcome periods: {fund_data['Outcome_Period_ID'].nunique()}")

    # Show outcome periods
    period_summary = fund_data.groupby('Outcome_Period_ID').agg({
        'Date': ['min', 'max', 'count'],
        'Roll_Date': 'first',
        'Original_Cap': 'first'
    }).reset_index()

    for idx, row in period_summary.iterrows():
        period_id = row[('Outcome_Period_ID', '')]
        start_date = row[('Date', 'min')]
        end_date = row[('Date', 'max')]
        days = row[('Date', 'count')]
        roll_date = row[('Roll_Date', 'first')]
        orig_cap = row[('Original_Cap', 'first')]
        # print(f"  {period_id}:")
        # print(f"    Roll Date: {roll_date.date()}")
        # print(f"    Period: {start_date.date()} to {end_date.date()} ({days} days)")
        # print(f"    Original Cap: {orig_cap:.2%}")

    # Show sample of derived metrics
    sample_cols = [
        'Date', 'Fund Value (USD)', 'Original_Cap', 'Current_Remaining_Cap',
        'Cap_Utilization', 'Cap_Remaining_Pct', 'Outcome_Period_ID'
    ]

    # DIAGNOSTIC: Check specific roll dates
    print("\n" + "=" * 80)
    print("DIAGNOSTIC: Checking specific roll dates")
    print("=" * 80)

    test_dates = ['2024-06-21', '2024-09-20']

    for test_date_str in test_dates:
        test_date = pd.to_datetime(test_date_str)

        # Check if this date exists in the data
        date_rows = fund_data[fund_data['Date'] == test_date]

        if not date_rows.empty:
            row = date_rows.iloc[0]
            print(f"\n{test_date_str}:")
            print(f"  Outcome_Period_ID: {row['Outcome_Period_ID']}")
            print(f"  Roll_Date: {row['Roll_Date']}")
            print(f"  Original_Cap: {row['Original_Cap']}")
            print(f"  Starting_Fund_Value: {row['Starting_Fund_Value']}")
            print(f"  Starting_Ref_Asset_Value: {row['Starting_Ref_Asset_Value']}")
            print(f"  Fund Value (USD): {row['Fund Value (USD)']}")
            print(f"  Remaining Cap: {row['Remaining Cap']}")
        else:
            print(f"\n{test_date_str}: NOT FOUND in data")

    print("=" * 80 + "\n")


    # =============================================================================
    # STEP 4: CLASSIFY MARKET REGIMES
    # =============================================================================

    print("STEP 4: Classifying market regimes...")
    print("-" * 80)

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

    # =============================================================================
    # STEP 5: PREPARE TRIGGER/SELECTION COMBINATIONS
    # =============================================================================

    print("STEP 5: Preparing trigger/selection combinations...")
    print("-" * 80)

    trigger_selection_combos = []
    for config in COMBO_CONFIGS:
        # Apply filters if specified
        if FILTER_TRIGGER_TYPES and config['trigger_type'] not in FILTER_TRIGGER_TYPES:
            continue
        if FILTER_SELECTION_ALGOS and config['selection_func_name'] not in FILTER_SELECTION_ALGOS:
            continue

        combo = {
            'trigger_type': config['trigger_type'],
            'trigger_params': config['trigger_params'],
            'selection_func_name': config['selection_func_name']
        }
        trigger_selection_combos.append(combo)

    print(f"Testing {len(trigger_selection_combos)} combinations:")
    for i, combo in enumerate(trigger_selection_combos, 1):
        print(f"  {i}. {combo['trigger_type']} + {combo['selection_func_name']}")

    # =============================================================================
    # STEP 6: RUN BACKTESTS FOR TEST FUND
    # =============================================================================

    print(f"\nSTEP 6: Running backtests for {test_fund}...")
    print("-" * 80)

    results_list = []

    for i, combo in enumerate(trigger_selection_combos, 1):

        try:
            selection_func = get_selection_function(combo['selection_func_name'])

            result = run_single_ticker_backtest(
                df_enriched=df_enriched,
                df_benchmarks=df_benchmarks,
                launch_month=TEST_LAUNCH_MONTH,
                trigger_config={
                    'type': combo['trigger_type'],
                    'params': combo['trigger_params']
                },
                selection_func=selection_func,
                roll_dates_dict=roll_dates_dict,
                series=TEST_SERIES
            )

            if result:
                results_list.append(result)
            else:
                print(f"⚠️  No result returned")

        except Exception as e:
            print(f"\n❌ ERROR in backtest:" + f"   Trigger: {combo['trigger_type']}" +f"   Selection: {combo['selection_func_name']}")
            print(f"   Error: {str(e)}")

    if not results_list:
        print("\n❌ No successful backtests. Exiting.")
        return

    # =============================================================================
    # STEP 7: ANALYZE RESULTS
    # =============================================================================
    print("\n" + "=" * 80)
    print(f"ANALYZING RESULTS FOR {test_fund}")
    print("=" * 80)

    try:
        # Consolidate results
        summary_df = consolidate_results(results_list)

        # =======================================================================
        # REGIME ANALYSIS
        # =======================================================================

        regime_df = analyze_by_regime(results_list, df_regimes)
        # Calculate capture ratios
        capture_ratios = calculate_capture_ratios(regime_df)

        # if not regime_df.empty:
        #     # Show regime breakdown
        #     regime_comparison = compare_regime_performance(regime_df)
        #
        #     print("\n📈 Performance by Regime (Average across all tests):")
        #     for _, row in regime_comparison.iterrows():
        #         print(f"  {row['regime'].capitalize():8s}: "
        #               f"Strategy {row['strategy_return'] * 100:+6.2f}% | "
        #               f"vs BUFR {row['vs_bufr_excess'] * 100:+6.2f}% | "
        #               f"{row['days_in_regime']:.0f} days")
        #
        #     # Find best strategy for each regime
        #     best_by_regime = find_best_by_regime(regime_df)
        #
        #     print("\n🏆 Best Strategy by Regime:")
        #     for regime, best in best_by_regime.items():
        #         print(f"  {regime.capitalize():8s}: {best['trigger_type'][:25]} + {best['selection_algo'][:25]}")
        #         print(f"             vs BUFR: {best['vs_bufr_excess'] * 100:+6.2f}%")
        #
        #     if not capture_ratios.empty:
        #         print("\n📊 Capture Ratios (Strategy Return / SPY Return):")
        #         for _, row in capture_ratios.iterrows():
        #             upside = row['upside_capture']
        #             downside = row['downside_capture']
        #             print(f"  {row['trigger_type'][:20]} + {row['selection_algo'][:20]}")
        #             if pd.notna(upside):
        #                 print(f"    Upside:   {upside:.2f}x")
        #             else:
        #                 print(f"    Upside:   N/A")
        #             if pd.notna(downside):
        #                 print(f"    Downside: {downside:.2f}x")
        #             else:
        #                 print(f"    Downside: N/A")
        # else:
        #     regime_df = pd.DataFrame()
        #     capture_ratios = pd.DataFrame()
        #     print("  ⚠️  No regime data available")

        # =======================================================================
        # EXPORT TO EXCEL
        # =======================================================================

        print("\nExporting consolidated workbook...")

        # Export directly to main RESULTS_DIR without fund-specific subfolder
        # Use timestamp-based filename instead of fund name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare aggregated summaries if multiple combos tested
        trigger_summary = pd.DataFrame()
        selection_summary = pd.DataFrame()
        month_summary = pd.DataFrame()  # Not used in single fund test, but needed for export function

        if len(summary_df) > 1:
            trigger_summary = summarize_by_trigger_type(summary_df)
            selection_summary = summarize_by_selection_algo(summary_df)

        # Export consolidated Excel workbook with ALL data
        workbook_path = export_consolidated_workbook(
            results_list=results_list,
            summary_df=summary_df,
            output_dir=RESULTS_DIR,
            run_name=f'{test_fund.lower()}_test',
            df_regimes=df_regimes,
            regime_df=regime_df,
            capture_ratios=capture_ratios,
            trigger_summary=trigger_summary,
            selection_summary=selection_summary,
            month_summary=month_summary  # Pass empty DataFrame
        )

    except Exception as e:
        print(f"\n❌ ERROR during analysis/export: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # =============================================================================
    # STEP 8: SHOW BEST PERFORMER
    # =============================================================================


    # print("\n" + "=" * 80)
    # print(f"RESULTS SUMMARY FOR {test_fund}")
    # print("=" * 80)
    #
    # # Display detailed results
    # print(f"\n{len(summary_df)} successful backtests completed\n")
    #
    # for idx, row in summary_df.iterrows():
    #     print(f"{idx + 1}. {row['trigger_type']} + {row['selection_algo']}")
    #     print(f"   Period: {row['start_date'].date()} to {row['end_date'].date()}")
    #     print(f"   Strategy Return: {row['strategy_return'] * 100:+6.2f}% (Ann: {row['strategy_ann_return'] * 100:+6.2f}%)")
    #     print(f"   vs SPY:   {row['vs_spy_excess'] * 100:+6.2f}%")
    #     print(f"   vs BUFR:  {row['vs_bufr_excess'] * 100:+6.2f}%")
    #     print(f"   vs Hold:  {row['vs_hold_excess'] * 100:+6.2f}%")
    #     print(f"   Sharpe:   {row['strategy_sharpe']:6.2f}")
    #     print(f"   Max DD:   {row['strategy_max_dd'] * 100:6.2f}%")
    #     print(f"   Trades:   {row['num_trades']:.0f}")
    #     print()

    # print("\n" + "=" * 80)
    # print("BEST PERFORMER (vs BUFR)")
    # print("=" * 80)
    #
    # best_idx = summary_df['vs_bufr_excess'].idxmax()
    # best = summary_df.loc[best_idx]
    #
    # print(f"\nTrigger:   {best['trigger_type']}")
    # print(f"Selection: {best['selection_algo']}")
    # print(f"vs BUFR:   {best['vs_bufr_excess'] * 100:+.2f}%")
    # print(f"Return:    {best['strategy_return'] * 100:+.2f}%")
    # print(f"Sharpe:    {best['strategy_sharpe']:.2f}")
    # print(f"Trades:    {best['num_trades']:.0f}")

    # =============================================================================
    # COMPLETION
    # =============================================================================

    print("\n" + "#" * 80)
    print("# TEST COMPLETE")



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        traceback.print_exc()