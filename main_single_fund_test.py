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

# Import configuration
from config.settings import (
    DATA_FILE, BENCHMARK_FILE, ROLL_DATES_FILE,
    RESULTS_DIR, REGIME_DIR, TRADE_LOG_DIR,
    SERIES, COMBO_CONFIGS,
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
from analysis.consolidator import consolidate_results

# Import utilities
from utils.validators import (
    validate_fund_data, validate_benchmark_data,
    validate_roll_dates, print_validation_results
)


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
    print("# SINGLE FUND TEST MODE")
    print("#" * 80)
    print(f"Testing: {TEST_SERIES}{TEST_LAUNCH_MONTH}")
    print("#" * 80 + "\n")

    start_time = datetime.now()

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

    print(f"✅ Data range: {common_start.date()} to {common_end.date()}")

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

    print("\n" + "=" * 80)
    print(f"INSPECTING {TEST_SERIES}{TEST_LAUNCH_MONTH} DATA")
    print("=" * 80)

    test_fund = TEST_SERIES + TEST_LAUNCH_MONTH
    fund_data = df_enriched[df_enriched['Fund'] == test_fund].copy()

    if fund_data.empty:
        print(f"❌ ERROR: No data found for {test_fund}")
        return

    print(f"\nData Summary:")
    print(f"  Date range: {fund_data['Date'].min().date()} to {fund_data['Date'].max().date()}")
    print(f"  Total observations: {len(fund_data):,}")
    print(f"  Unique outcome periods: {fund_data['Outcome_Period_ID'].nunique()}")

    # Show outcome periods
    print(f"\nOutcome Periods:")
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

        print(f"  {period_id}:")
        print(f"    Roll Date: {roll_date.date()}")
        print(f"    Period: {start_date.date()} to {end_date.date()} ({days} days)")
        print(f"    Original Cap: {orig_cap:.2%}")

    # Show sample of derived metrics
    print(f"\nSample Derived Metrics (first 5 rows):")
    sample_cols = [
        'Date', 'Fund Value (USD)', 'Original_Cap', 'Current_Remaining_Cap',
        'Cap_Utilization', 'Cap_Remaining_Pct', 'Outcome_Period_ID'
    ]
    print(fund_data[sample_cols].head().to_string(index=False))

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
        print(f"\n{'─' * 80}")
        print(f"Test {i}/{len(trigger_selection_combos)}")
        print(f"{'─' * 80}")

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
            print(f"\n❌ ERROR in backtest:")
            print(f"   Trigger: {combo['trigger_type']}")
            print(f"   Selection: {combo['selection_func_name']}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()

    if not results_list:
        print("\n❌ No successful backtests. Exiting.")
        return

    # =============================================================================
    # STEP 7: ANALYZE RESULTS
    # =============================================================================

    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY FOR {test_fund}")
    print("=" * 80)

    summary_df = consolidate_results(results_list)

    # Display detailed results
    print(f"\n{len(summary_df)} successful backtests completed\n")

    for idx, row in summary_df.iterrows():
        print(f"{idx + 1}. {row['trigger_type']} + {row['selection_algo']}")
        print(f"   Period: {row['start_date'].date()} to {row['end_date'].date()}")
        print(f"   Strategy Return: {row['strategy_return'] * 100:+6.2f}% (Ann: {row['strategy_ann_return'] * 100:+6.2f}%)")
        print(f"   vs SPY:   {row['vs_spy_excess'] * 100:+6.2f}%")
        print(f"   vs BUFR:  {row['vs_bufr_excess'] * 100:+6.2f}%")
        print(f"   vs Hold:  {row['vs_hold_excess'] * 100:+6.2f}%")
        print(f"   Sharpe:   {row['strategy_sharpe']:6.2f}")
        print(f"   Max DD:   {row['strategy_max_dd'] * 100:6.2f}%")
        print(f"   Trades:   {row['num_trades']:.0f}")
        print()

    # =============================================================================
    # STEP 8: EXPORT DETAILED RESULTS
    # =============================================================================

    print("\nSTEP 8: Exporting detailed results...")
    print("-" * 80)

    # Create test output directory
    test_output_dir = os.path.join(RESULTS_DIR, f'test_{test_fund.lower()}')
    os.makedirs(test_output_dir, exist_ok=True)

    # Export summary
    summary_path = os.path.join(test_output_dir, f'{test_fund.lower()}_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Saved summary: {summary_path}")

    # Export daily performance for each backtest
    for result in results_list:
        trigger_short = result['trigger_type'].replace('_', '')[:10]
        selection_short = result['selection_algo'].replace('_', '')[:10]
        filename = f"{test_fund.lower()}_daily_{trigger_short}_{selection_short}.csv"
        filepath = os.path.join(test_output_dir, filename)
        result['daily_performance'].to_csv(filepath, index=False)
        print(f"✅ Saved daily NAV: {filename}")

    # Export trade logs
    for result in results_list:
        if not result['trade_history'].empty:
            trigger_short = result['trigger_type'].replace('_', '')[:10]
            selection_short = result['selection_algo'].replace('_', '')[:10]
            filename = f"{test_fund.lower()}_trades_{trigger_short}_{selection_short}.csv"
            filepath = os.path.join(test_output_dir, filename)
            result['trade_history'].to_csv(filepath, index=False)
            print(f"✅ Saved trades: {filename}")

    # =============================================================================
    # STEP 9: SHOW BEST PERFORMER
    # =============================================================================

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

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "#" * 80)
    print("# TEST COMPLETE")



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()