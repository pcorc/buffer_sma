"""
Main execution script with train/test split for forward regime validation.

This extends main.py to:
1. Run backtests on FULL dataset
2. Split results into training and test periods
3. Identify optimal strategies in training period
4. Validate performance in test period
5. Export comprehensive comparison

Usage:
    python main_with_train_test.py
"""

import os
import sys
import traceback
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import configuration
from config.settings import (
    DATA_FILE, BENCHMARK_FILE, ROLL_DATES_FILE,
    RESULTS_DIR,
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
from core.forward_regime_classifier import classify_forward_regimes
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
from analysis.forward_regime_analyzer import (
    analyze_by_future_regime, rank_strategies_by_future_regime,
    summarize_optimal_strategies, compare_strategy_intent_vs_future_regime,
    identify_robust_strategies
)

# Import train/test utilities
from utils.train_test_split import (
    TrainTestConfig, split_results_by_period, split_dataframe_by_period,
    compare_train_test_optimal_strategies, analyze_strategy_stability,
    identify_consistently_optimal_strategies, print_train_test_summary,
    get_validation_recommendations
)

# Import utilities
from utils.excel_exporter import export_main_consolidated_workbook
from utils.validators import (
    validate_fund_data, validate_benchmark_data,
    validate_roll_dates, print_validation_results
)


def main():
    """
    Main execution function with train/test split.
    """

    start_time = datetime.now()

    # =========================================================================
    # CONFIGURATION: TRAIN/TEST SPLIT
    # =========================================================================

    # Training: 2020-01-01 through 2023-12-31 (4 years)
    # Testing: 2024-01-01 through 2024-12-31+ (1 year+)
    TRAIN_END_DATE = '2023-12-31'
    TEST_START_DATE = '2024-01-01'

    train_test_config = TrainTestConfig(
        train_end_date=TRAIN_END_DATE,
        test_start_date=TEST_START_DATE
    )

    print("\n" + "#" * 80)
    print("# BACKTEST WITH TRAIN/TEST VALIDATION")
    print("#" * 80)
    print(f"\nTrain/Test Configuration:")
    print(f"  Training Period: Through {train_test_config.train_end_date.date()}")
    print(f"  Test Period: From {train_test_config.test_start_date.date()}")
    print("#" * 80 + "\n")

    # =========================================================================
    # STEP 1-7: SAME AS ORIGINAL main.py
    # =========================================================================

    # Load data
    print("\nSTEP 1: Loading data files...")
    print("-" * 80)

    try:
        df_raw = load_fund_data(DATA_FILE, series=SERIES)
        df_benchmarks = load_benchmark_data(BENCHMARK_FILE)
        roll_dates_dict = load_roll_dates(ROLL_DATES_FILE)
    except Exception as e:
        print(f"\n❌ ERROR loading data: {str(e)}")
        return

    # Validate data
    print("\nSTEP 2: Validating data...")
    print("-" * 80)

    is_valid, errors, df_raw = validate_fund_data(df_raw, series=SERIES)
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

    # Preprocess
    print("\nSTEP 3: Preprocessing fund data...")
    print("-" * 80)

    try:
        df_enriched = preprocess_fund_data(df_raw, roll_dates_dict)
    except Exception as e:
        print(f"\n❌ ERROR during preprocessing: {str(e)}")
        traceback.print_exc()
        return

    # Classify regimes
    print("\nSTEP 4: Classifying market regimes...")
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

        df_forward_regimes = classify_forward_regimes(
            df_spy_for_regime,
            window_3m_days=63,
            window_6m_days=126,
            bull_threshold=REGIME_BULL_THRESHOLD,
            bear_threshold=REGIME_BEAR_THRESHOLD
        )
    except Exception as e:
        print(f"\n❌ ERROR during regime classification: {str(e)}")
        traceback.print_exc()
        return

    # Prepare combinations
    print("\nSTEP 5: Preparing trigger/selection combinations...")
    print("-" * 80)

    trigger_selection_combos = []
    for config in COMBO_CONFIGS:
        combo = {
            'trigger_type': config['trigger_type'],
            'trigger_params': config['trigger_params'],
            'selection_func_name': config['selection_func_name'],
            'launch_months': config.get('launch_months', LAUNCH_MONTHS),
            'strategy_intent': config.get('strategy_intent', 'neutral')
        }
        trigger_selection_combos.append(combo)

    # Run backtests
    print("\nSTEP 6: Running all backtests...")
    print("-" * 80)

    try:
        results_list = run_all_single_ticker_tests(
            df_enriched=df_enriched,
            df_benchmarks=df_benchmarks,
            roll_dates_dict=roll_dates_dict,
            trigger_selection_combos=trigger_selection_combos,
            series=SERIES
        )
    except Exception as e:
        print(f"\n❌ ERROR during backtesting: {str(e)}")
        traceback.print_exc()
        return

    if not results_list:
        print("\n❌ No results generated. Check error messages above.")
        return

    # Consolidate
    print("\nSTEP 7: Consolidating results...")
    print("-" * 80)

    try:
        summary_df = consolidate_results(results_list)
        month_summary = summarize_by_launch_month(summary_df)
        trigger_summary = summarize_by_trigger_type(summary_df)
        selection_summary = summarize_by_selection_algo(summary_df)

        perf_summary = create_performance_summary(summary_df)
        print(f"  Total backtests: {perf_summary['total_backtests']}")
    except Exception as e:
        print(f"\n❌ ERROR during consolidation: {str(e)}")
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 8: SPLIT INTO TRAIN AND TEST PERIODS
    # =========================================================================

    print("\nSTEP 8: Splitting results into train and test periods...")
    print("-" * 80)

    try:
        train_results, test_results = split_results_by_period(
            results_list, train_test_config
        )

        print(f"  Training results: {len(train_results)}")
        print(f"  Test results: {len(test_results)}")

        if not train_results:
            print("❌ No training results found!")
            return

        if not test_results:
            print("⚠️  No test results found - will analyze training only")

    except Exception as e:
        print(f"\n❌ ERROR during train/test split: {str(e)}")
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 9: ANALYZE TRAINING PERIOD
    # =========================================================================

    print("\nSTEP 9: Analyzing TRAINING period...")
    print("-" * 80)

    try:
        # Forward regime analysis on training data
        future_regime_train = analyze_by_future_regime(
            train_results, df_forward_regimes
        )

        if not future_regime_train.empty:
            # Get optimal strategies from training
            optimal_train_6m = summarize_optimal_strategies(
                future_regime_train, horizon='6M', top_n=10
            )

        else:
            print("  ⚠️  No training future regime data")
            optimal_train_6m = {}

    except Exception as e:
        print(f"\n❌ ERROR during training analysis: {str(e)}")
        traceback.print_exc()
        future_regime_train = pd.DataFrame()
        optimal_train_6m = {}

    # =========================================================================
    # STEP 10: ANALYZE TEST PERIOD
    # =========================================================================

    print("\nSTEP 10: Analyzing TEST period...")
    print("-" * 80)

    try:
        if test_results:
            # Forward regime analysis on test data
            future_regime_test = analyze_by_future_regime(
                test_results, df_forward_regimes
            )

            if not future_regime_test.empty:
                # Get optimal strategies from test
                optimal_test_6m = summarize_optimal_strategies(
                    future_regime_test, horizon='6M', top_n=10
                )

            else:
                print("  ⚠️  No test future regime data")
                optimal_test_6m = {}
        else:
            future_regime_test = pd.DataFrame()
            optimal_test_6m = {}

    except Exception as e:
        print(f"\n❌ ERROR during test analysis: {str(e)}")
        traceback.print_exc()
        future_regime_test = pd.DataFrame()
        optimal_test_6m = {}

    # =========================================================================
    # STEP 11: COMPARE TRAIN VS TEST
    # =========================================================================

    print("\nSTEP 11: Comparing train vs test performance...")
    print("-" * 80)

    try:
        if optimal_train_6m and optimal_test_6m:
            # Compare optimal strategies
            train_test_comparison = compare_train_test_optimal_strategies(
                optimal_train_6m, optimal_test_6m, horizon='6M'
            )

            if not train_test_comparison.empty:
                # Analyze stability
                stability_metrics = analyze_strategy_stability(
                    train_test_comparison, top_n=5
                )

                print_train_test_summary(stability_metrics, horizon='6M')

                # Get consistently optimal strategies
                consistent_optimal = identify_consistently_optimal_strategies(
                    train_test_comparison, top_n=10
                )

                # Get recommendations
                recommendations = get_validation_recommendations(
                    train_test_comparison, stability_metrics, top_n=5
                )

                print("\n📋 VALIDATION RECOMMENDATIONS:")
                print("=" * 80)
                for regime, rec in recommendations.items():
                    print(f"\n{regime.upper()} MARKET:")
                    print(f"  Stability Score: {rec['stability_score']:.1f}%")

                    if rec['trust']:
                        print(f"\n  ✅ TRUSTED (consistent in both periods):")
                        for strat in rec['trust']:
                            print(f"    • {strat['launch_month']} + "
                                  f"{strat['trigger_type'][:20]} + "
                                  f"{strat['selection_algo'][:20]}")
                            print(f"      Train rank: #{strat['train_rank']}, "
                                  f"Test rank: #{strat['test_rank']}, "
                                  f"Avg excess: {strat['avg_excess'] * 100:+.2f}%")

                    if rec['caution']:
                        print(f"\n  ⚠️  CAUTION (declined in test):")
                        for strat in rec['caution']:
                            print(f"    • {strat['launch_month']} + "
                                  f"{strat['trigger_type'][:20]} + "
                                  f"{strat['selection_algo'][:20]}")
                            print(f"      Train rank: #{strat['train_rank']}, "
                                  f"Test rank: #{strat['test_rank']}")

            else:
                print("  ⚠️  No overlapping strategies to compare")
                train_test_comparison = pd.DataFrame()
                stability_metrics = {}
                consistent_optimal = pd.DataFrame()
                recommendations = {}
        else:
            print("  ⚠️  Missing train or test optimal strategies")
            train_test_comparison = pd.DataFrame()
            stability_metrics = {}
            consistent_optimal = pd.DataFrame()
            recommendations = {}

    except Exception as e:
        print(f"\n❌ ERROR during train/test comparison: {str(e)}")
        traceback.print_exc()
        train_test_comparison = pd.DataFrame()
        stability_metrics = {}
        consistent_optimal = pd.DataFrame()
        recommendations = {}

    # =========================================================================
    # STEP 12: EXPORT WITH TRAIN/TEST COMPARISON
    # =========================================================================

    print("\nSTEP 12: Exporting results with train/test comparison...")
    print("-" * 80)

    try:
        # Export with train/test tabs
        workbook_path = export_train_test_workbook(
            results_list=results_list,
            summary_df=summary_df,
            train_results=train_results,
            test_results=test_results,
            future_regime_train=future_regime_train,
            future_regime_test=future_regime_test,
            optimal_train_6m=optimal_train_6m,
            optimal_test_6m=optimal_test_6m,
            train_test_comparison=train_test_comparison,
            consistent_optimal=consistent_optimal,
            output_dir=RESULTS_DIR,
            train_test_config=train_test_config
        )

        print(f"\n✅ Train/Test workbook exported:")
        print(f"   {workbook_path}")

    except Exception as e:
        print(f"\n❌ ERROR during export: {str(e)}")
        traceback.print_exc()

    # =========================================================================
    # COMPLETION
    # =========================================================================

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'=' * 80}")
    print(f"BACKTEST COMPLETE")
    print(f"Duration: {duration}")
    print(f"{'=' * 80}")


def export_train_test_workbook(
        results_list, summary_df, train_results, test_results,
        future_regime_train, future_regime_test,
        optimal_train_6m, optimal_test_6m,
        train_test_comparison, consistent_optimal,
        output_dir, train_test_config
):
    """
    Export workbook with train/test comparison tabs.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'train_test_analysis_{timestamp}.xlsx'
    filepath = os.path.join(output_dir, filename)

    from openpyxl.styles import Font, PatternFill, Alignment

    header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
    header_font = Font(bold=True, color='FFFFFF')

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:

        # Tab 1: Full Summary
        summary_df.to_excel(writer, sheet_name='Full Summary', index=False)
        _format_sheet(writer.sheets['Full Summary'], header_fill, header_font)

        # Tab 2: Train Optimal (Bull)
        if optimal_train_6m and 'bull' in optimal_train_6m:
            optimal_train_6m['bull'].to_excel(writer, sheet_name='Train-Optimal Bull', index=False)
            _format_sheet(writer.sheets['Train-Optimal Bull'], header_fill, header_font)

        # Tab 3: Test Optimal (Bull)
        if optimal_test_6m and 'bull' in optimal_test_6m:
            optimal_test_6m['bull'].to_excel(writer, sheet_name='Test-Optimal Bull', index=False)
            _format_sheet(writer.sheets['Test-Optimal Bull'], header_fill, header_font)

        # Tab 4: Train vs Test Comparison
        if not train_test_comparison.empty:
            train_test_comparison.to_excel(writer, sheet_name='Train vs Test', index=False)
            _format_sheet(writer.sheets['Train vs Test'], header_fill, header_font)

        # Tab 5: Consistent Winners
        if not consistent_optimal.empty:
            consistent_optimal.to_excel(writer, sheet_name='Consistent Winners', index=False)
            _format_sheet(writer.sheets['Consistent Winners'], header_fill, header_font)

    return filepath


def _format_sheet(worksheet, header_fill, header_font):
    """Apply formatting."""
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        worksheet.column_dimensions[column_letter].width = adjusted_width

    worksheet.freeze_panes = 'A2'


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        traceback.print_exc()