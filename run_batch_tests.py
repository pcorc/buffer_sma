"""
Batch Test Runner for Forward Regime Analysis

Runs focused batches of ~50 simulations each to systematically identify
optimal trigger/selection combinations for bull/bear/neutral regimes.

Usage:
    1. Set BATCH_NUMBER below (1-6)
    2. Run: python run_batch_tests.py
    3. Review results in Excel output
    4. Move to next batch

Each batch takes ~15 minutes and tests specific strategy types.
"""

import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ============================================================================
# CONFIGURATION: SELECT BATCH TO RUN
# ============================================================================

BATCH_NUMBER = 1  # Change this to run different batches (1-6)


# ============================================================================
# BATCH DEFINITIONS
# ============================================================================

def get_batch_1_configs():
    """
    BATCH 1: Time-Based Rebalancing (Neutral Strategies)
    ~48 simulations, ~15 minutes

    Tests: Monthly, Quarterly, Semi-Annual, Annual rebalancing
    Launch Months: Quarterly (MAR, JUN, SEP, DEC)
    Selections: 4 selection algorithms
    Intent: Neutral (systematic rebalancing)
    """
    configs = []

    frequencies = [
        ('monthly', ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN']),
        ('quarterly', ['MAR', 'JUN', 'SEP', 'DEC']),
        ('semi_annual', ['MAR', 'SEP']),
        ('annual', ['JAN'])
    ]

    selections = [
        'select_most_recent_launch',
        'select_remaining_cap',
        'select_cap_utilization',
        'select_highest_outcome_and_cap'
    ]

    for freq, months in frequencies:
        for selection in selections:
            configs.append({
                'trigger_type': 'rebalance_time_period',
                'trigger_params': {'frequency': freq},
                'selection_func_name': selection,
                'launch_months': months,
                'strategy_intent': 'neutral'
            })

    return configs


def get_batch_2_configs():
    """
    BATCH 2: Cap Utilization Thresholds (Bearish/Defensive)
    ~48 simulations, ~15 minutes

    Tests: Cap utilization 25%, 50%, 75%, 90%
    Launch Months: Quarterly (MAR, JUN, SEP, DEC)
    Selections: 4 selection algorithms
    Intent: Bearish (preserve protection)
    """
    configs = []

    thresholds = [0.25, 0.50, 0.75, 0.90]
    months = ['MAR', 'JUN', 'SEP', 'DEC']

    selections = [
        'select_most_recent_launch',
        'select_remaining_cap',
        'select_cap_utilization',
        'select_highest_outcome_and_cap'
    ]

    for threshold in thresholds:
        for selection in selections:
            configs.append({
                'trigger_type': 'cap_utilization_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months,
                'strategy_intent': 'bearish'
            })

    return configs


def get_batch_3_configs():
    """
    BATCH 3: Remaining Cap Thresholds (Bearish/Defensive)
    ~48 simulations, ~15 minutes

    Tests: Remaining cap 10%, 25%, 50%, 75%
    Launch Months: Quarterly (MAR, JUN, SEP, DEC)
    Selections: 4 selection algorithms
    Intent: Bearish (protect when cap depleted)
    """
    configs = []

    thresholds = [0.10, 0.25, 0.50, 0.75]
    months = ['MAR', 'JUN', 'SEP', 'DEC']

    selections = [
        'select_most_recent_launch',
        'select_remaining_cap',
        'select_cap_utilization',
        'select_highest_outcome_and_cap'
    ]

    for threshold in thresholds:
        for selection in selections:
            configs.append({
                'trigger_type': 'remaining_cap_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months,
                'strategy_intent': 'bearish'
            })

    return configs


def get_batch_4_configs():
    """
    BATCH 4: Reference Asset Return Thresholds (Directional)
    ~60 simulations, ~18 minutes

    Tests: SPY returns at -10%, -5%, +5%, +10%, +15%
    Launch Months: Quarterly (MAR, JUN, SEP, DEC)
    Selections: 3 selection algorithms
    Intent: Mixed (bullish for positive, bearish for negative)
    """
    configs = []

    # Bearish thresholds (negative returns)
    bearish_thresholds = [-0.10, -0.05]
    # Bullish thresholds (positive returns)
    bullish_thresholds = [0.05, 0.10, 0.15]

    months = ['MAR', 'JUN', 'SEP', 'DEC']

    selections = [
        'select_most_recent_launch',
        'select_remaining_cap',
        'select_highest_outcome_and_cap'
    ]

    for threshold in bearish_thresholds:
        for selection in selections:
            configs.append({
                'trigger_type': 'ref_asset_return_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months,
                'strategy_intent': 'bearish'
            })

    for threshold in bullish_thresholds:
        for selection in selections:
            configs.append({
                'trigger_type': 'ref_asset_return_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months,
                'strategy_intent': 'bullish'
            })

    return configs


def get_batch_5_configs():
    """
    BATCH 5: Downside Before Buffer Thresholds (Defensive)
    ~36 simulations, ~12 minutes

    Tests: Downside -5%, -2%, 0% (in buffer)
    Launch Months: Quarterly (MAR, JUN, SEP, DEC)
    Selections: 3 selection algorithms
    Intent: Bearish (react to buffer breach)
    """
    configs = []

    thresholds = [-0.05, -0.02, 0.0]
    months = ['MAR', 'JUN', 'SEP', 'DEC']

    selections = [
        'select_most_recent_launch',
        'select_remaining_cap',
        'select_highest_outcome_and_cap'
    ]

    for threshold in thresholds:
        for selection in selections:
            configs.append({
                'trigger_type': 'downside_before_buffer_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months,
                'strategy_intent': 'bearish'
            })

    return configs


def get_batch_6_configs():
    """
    BATCH 6: Best-of-Batch Cross-Validation
    ~48 simulations, ~15 minutes

    After reviewing batches 1-5, select top 2-3 from each category
    and test with ALL 12 launch months for comprehensive coverage.

    This batch should be customized based on learnings from batches 1-5.
    """
    configs = []

    # Example: Top performers from each batch (customize after reviewing results)

    # From Batch 1: Best time-based
    configs.append({
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],
        'strategy_intent': 'neutral'
    })

    # From Batch 2: Best cap utilization
    configs.append({
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.75},
        'selection_func_name': 'select_remaining_cap',
        'launch_months': ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],
        'strategy_intent': 'bearish'
    })

    # From Batch 3: Best remaining cap
    configs.append({
        'trigger_type': 'remaining_cap_threshold',
        'trigger_params': {'threshold': 0.25},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],
        'strategy_intent': 'bearish'
    })

    # From Batch 4: Best directional (customize based on results)
    configs.append({
        'trigger_type': 'ref_asset_return_threshold',
        'trigger_params': {'threshold': 0.10},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],
        'strategy_intent': 'bullish'
    })

    return configs


# ============================================================================
# BATCH SELECTOR
# ============================================================================

BATCH_CONFIGS = {
    1: get_batch_1_configs,
    2: get_batch_2_configs,
    3: get_batch_3_configs,
    4: get_batch_4_configs,
    5: get_batch_5_configs,
    6: get_batch_6_configs
}

BATCH_DESCRIPTIONS = {
    1: "Time-Based Rebalancing (Neutral)",
    2: "Cap Utilization Thresholds (Bearish)",
    3: "Remaining Cap Thresholds (Bearish)",
    4: "Reference Asset Return Thresholds (Directional)",
    5: "Downside Before Buffer Thresholds (Defensive)",
    6: "Best-of-Batch Cross-Validation"
}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the selected batch."""

    if BATCH_NUMBER not in BATCH_CONFIGS:
        print(f"❌ Invalid BATCH_NUMBER: {BATCH_NUMBER}")
        print(f"   Please set BATCH_NUMBER to 1-6")
        return

    print("\n" + "=" * 80)
    print(f"BATCH {BATCH_NUMBER}: {BATCH_DESCRIPTIONS[BATCH_NUMBER]}")
    print("=" * 80 + "\n")

    # Get batch configurations
    batch_configs = BATCH_CONFIGS[BATCH_NUMBER]()

    print(f"This batch will run {len(batch_configs)} trigger/selection combinations")

    # Calculate total simulations (configs * launch_months)
    total_sims = sum(len(config['launch_months']) for config in batch_configs)
    print(f"Total simulations: {total_sims}")
    print(f"Estimated time: ~{total_sims * 0.3:.0f} minutes ({total_sims * 0.3 / 60:.1f} hours)")

    print("\n" + "-" * 80)
    input("Press ENTER to start batch execution...")
    print("-" * 80 + "\n")

    # Import and run
    from config import settings
    from data.loader import load_fund_data, load_benchmark_data, load_roll_dates
    from data.preprocessor import preprocess_fund_data
    from core.regime_classifier import classify_market_regimes
    from core.forward_regime_classifier import classify_forward_regimes
    from backtesting.batch_runner import run_all_single_ticker_tests
    from analysis.consolidator import consolidate_results
    from analysis.forward_regime_analyzer import (
        analyze_by_future_regime, summarize_optimal_strategies,
        print_future_regime_summary
    )
    from utils.excel_exporter import export_main_consolidated_workbook
    from utils.validators import validate_fund_data, validate_benchmark_data, validate_roll_dates

    start_time = datetime.now()

    # Load data
    print("Loading data...")
    df_raw = load_fund_data(settings.DATA_FILE, series=settings.SERIES)
    df_benchmarks = load_benchmark_data(settings.BENCHMARK_FILE)
    roll_dates_dict = load_roll_dates(settings.ROLL_DATES_FILE)

    # Validate
    print("Validating data...")
    is_valid, errors, df_raw = validate_fund_data(df_raw, series=settings.SERIES)
    if not is_valid:
        print("❌ Data validation failed")
        return

    is_valid, errors, df_benchmarks = validate_benchmark_data(df_benchmarks)
    if not is_valid:
        print("❌ Benchmark validation failed")
        return

    # Preprocess
    print("Preprocessing...")
    df_enriched = preprocess_fund_data(df_raw, roll_dates_dict)

    # Classify regimes
    print("Classifying regimes...")
    df_spy_for_regime = df_benchmarks[['Date', 'SPY']].copy()
    df_spy_for_regime.rename(columns={'SPY': 'Ref_Index'}, inplace=True)

    df_regimes = classify_market_regimes(
        df_spy_for_regime,
        window_months=settings.REGIME_WINDOW_MONTHS,
        bull_threshold=settings.REGIME_BULL_THRESHOLD,
        bear_threshold=settings.REGIME_BEAR_THRESHOLD
    )

    df_forward_regimes = classify_forward_regimes(
        df_spy_for_regime,
        window_3m_days=63,
        window_6m_days=126,
        bull_threshold=settings.REGIME_BULL_THRESHOLD,
        bear_threshold=settings.REGIME_BEAR_THRESHOLD
    )

    # Run backtests
    print(f"\n{'=' * 80}")
    print(f"RUNNING BATCH {BATCH_NUMBER} BACKTESTS")
    print(f"{'=' * 80}\n")

    results_list = run_all_single_ticker_tests(
        df_enriched=df_enriched,
        df_benchmarks=df_benchmarks,
        roll_dates_dict=roll_dates_dict,
        trigger_selection_combos=batch_configs,
        series=settings.SERIES
    )

    if not results_list:
        print("❌ No results generated")
        return

    # Analyze
    print("\nAnalyzing results...")
    summary_df = consolidate_results(results_list)

    # Forward regime analysis
    future_regime_df = analyze_by_future_regime(results_list, df_forward_regimes)

    if not future_regime_df.empty:
        optimal_6m = summarize_optimal_strategies(future_regime_df, horizon='6M', top_n=10)
        print_future_regime_summary(optimal_6m, horizon='6M')
    else:
        optimal_6m = {}

    # Export
    print("\nExporting results...")
    output_dir = os.path.join(settings.RESULTS_DIR, f'batch_{BATCH_NUMBER}')
    os.makedirs(output_dir, exist_ok=True)

    workbook_path = export_main_consolidated_workbook(
        results_list=results_list,
        summary_df=summary_df,
        output_dir=output_dir,
        run_name=f'batch{BATCH_NUMBER}_{BATCH_DESCRIPTIONS[BATCH_NUMBER].replace(" ", "_")}',
        df_forward_regimes=df_forward_regimes,
        future_regime_df=future_regime_df,
        optimal_6m=optimal_6m,
        optimal_3m={},
        intent_vs_regime_6m=None,
        intent_vs_regime_3m=None,
        robust_strategies_6m=None,
        robust_strategies_3m=None,
        ranked_6m_vs_spy=None,
        ranked_6m_vs_bufr=None
    )

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'=' * 80}")
    print(f"BATCH {BATCH_NUMBER} COMPLETE")
    print(f"{'=' * 80}")
    print(f"Duration: {duration}")
    print(f"Total simulations: {len(results_list)}")
    print(f"Output: {workbook_path}")
    print(f"\n{'=' * 80}")
    print(f"NEXT STEPS:")
    print(f"1. Review {workbook_path}")
    print(f"2. Check 'Optimal-Bull (6M)', 'Optimal-Bear (6M)', 'Optimal-Neutral (6M)' tabs")
    print(f"3. Set BATCH_NUMBER = {BATCH_NUMBER + 1} and run again")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Batch execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()