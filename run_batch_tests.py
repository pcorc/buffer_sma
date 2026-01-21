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

BATCH_NUMBER = 0  # Change this to run different batches (1-6)


def get_batch_0_configs():
    """
    BATCH 0: Quick Test - 6 Strategies (3 Bullish + 3 Bearish)
    ~24 simulations, ~7 minutes

    Used to verify regime classification and Excel export tabs
    """
    configs = []

    months = ['MAR','DEC']

    # # BULLISH STRATEGIES (3)
    # configs.append({
    #     'trigger_type': 'rebalance_time_period',
    #     'trigger_params': {'frequency': 'quarterly'},
    #     'selection_func_name': 'select_cap_utilization_lowest',  # Bullish
    #     'launch_months': months
    # })
    #
    # configs.append({
    #     'trigger_type': 'cap_utilization_threshold',
    #     'trigger_params': {'threshold': 0.75},
    #     'selection_func_name': 'select_remaining_cap_highest',  # Bullish
    #     'launch_months': months
    # })
    #
    # configs.append({
    #     'trigger_type': 'remaining_cap_threshold',
    #     'trigger_params': {'threshold': 0.50},
    #     'selection_func_name': 'select_downside_buffer_lowest',  # Bullish
    #     'launch_months': months
    # })

    # BEARISH STRATEGIES (3)
    configs.append({
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_downside_buffer_highest',  # Bearish
        'launch_months': months
    })

    configs.append({
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.75},
        'selection_func_name': 'select_downside_buffer_lowest',  # Bearish (overlaps with bullish)
        'launch_months': months
    })

    # configs.append({
    #     'trigger_type': 'remaining_cap_threshold',
    #     'trigger_params': {'threshold': 0.50},
    #     'selection_func_name': 'select_cap_utilization_lowest',  # Bearish (overlaps with bullish)
    #     'launch_months': months
    # })

    return configs

# ============================================================================
# BATCH DEFINITIONS
# ============================================================================
def get_batch_1_configs():
    """
    BATCH 1: Time-Based Systematic Rebalancing
    ~48 simulations, ~15 minutes

    Tests systematic rebalancing with bullish vs bearish selection intent
    """
    configs = []

    frequencies = ['quarterly', 'semi_annual', 'annual']
    months = ['MAR', 'JUN', 'SEP', 'DEC']

    bullish_selections = [
        'select_cap_utilization_lowest',
        'select_remaining_cap_highest',
        'select_downside_buffer_lowest'
    ]

    bearish_selections = [
        'select_downside_buffer_highest',
        'select_downside_buffer_lowest',
        'select_cap_utilization_lowest'
    ]

    for freq in frequencies:
        # Bullish strategies
        for selection in bullish_selections:
            configs.append({
                'trigger_type': 'rebalance_time_period',
                'trigger_params': {'frequency': freq},
                'selection_func_name': selection,
                'launch_months': months
            })

        # Bearish strategies
        for selection in bearish_selections:
            configs.append({
                'trigger_type': 'rebalance_time_period',
                'trigger_params': {'frequency': freq},
                'selection_func_name': selection,
                'launch_months': months
            })

    return configs


def get_batch_2_configs():
    """
    BATCH 2: Cap Utilization Tactical Triggers
    ~48 simulations, ~15 minutes

    Tests rotation based on cap consumption
    """
    configs = []

    thresholds = [0.50, 0.75]
    months = ['MAR', 'JUN', 'SEP', 'DEC']

    bullish_selections = [
        'select_cap_utilization_lowest',
        'select_remaining_cap_highest',
        'select_downside_buffer_lowest'
    ]

    bearish_selections = [
        'select_downside_buffer_highest',
        'select_downside_buffer_lowest',
        'select_cap_utilization_lowest'
    ]

    for threshold in thresholds:
        # Bullish strategies
        for selection in bullish_selections:
            configs.append({
                'trigger_type': 'cap_utilization_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months
            })

        # Bearish strategies
        for selection in bearish_selections:
            configs.append({
                'trigger_type': 'cap_utilization_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months
            })

    return configs


def get_batch_3_configs():
    """
    BATCH 3: Remaining Cap Tactical Triggers
    ~48 simulations, ~15 minutes

    Tests rotation based on cap depletion
    """
    configs = []

    thresholds = [0.25, 0.50]
    months = ['MAR', 'JUN', 'SEP', 'DEC']

    bullish_selections = [
        'select_cap_utilization_lowest',
        'select_remaining_cap_highest',
        'select_downside_buffer_lowest'
    ]

    bearish_selections = [
        'select_downside_buffer_highest',
        'select_downside_buffer_lowest',
        'select_cap_utilization_lowest'
    ]

    for threshold in thresholds:
        # Bullish strategies
        for selection in bullish_selections:
            configs.append({
                'trigger_type': 'remaining_cap_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months
            })

        # Bearish strategies
        for selection in bearish_selections:
            configs.append({
                'trigger_type': 'remaining_cap_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months
            })

    return configs


def get_batch_4_configs():
    """
    BATCH 4: Market-Responsive Triggers (Ref Asset + Buffer)
    ~60 simulations, ~18 minutes

    Tests rotation based on market performance and buffer proximity
    """
    configs = []

    months = ['MAR', 'JUN', 'SEP', 'DEC']

    bullish_selections = [
        'select_cap_utilization_lowest',
        'select_remaining_cap_highest',
        'select_downside_buffer_lowest'
    ]

    bearish_selections = [
        'select_downside_buffer_highest',
        'select_downside_buffer_lowest',
        'select_cap_utilization_lowest'
    ]

    # Reference Asset Return Thresholds
    ref_thresholds = [-0.10, 0.0, 0.10]

    for threshold in ref_thresholds:
        # Bullish strategies
        for selection in bullish_selections:
            configs.append({
                'trigger_type': 'ref_asset_return_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months
            })

        # Bearish strategies
        for selection in bearish_selections:
            configs.append({
                'trigger_type': 'ref_asset_return_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months
            })

    # Downside Before Buffer Thresholds
    buffer_thresholds = [-0.02, 0.0]

    for threshold in buffer_thresholds:
        # Bullish strategies
        for selection in bullish_selections:
            configs.append({
                'trigger_type': 'downside_before_buffer_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months
            })

        # Bearish strategies
        for selection in bearish_selections:
            configs.append({
                'trigger_type': 'downside_before_buffer_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months
            })

    return configs

# ============================================================================
# BATCH SELECTOR
# ============================================================================

BATCH_CONFIGS = {
    0: get_batch_0_configs,  # ← Add this
    1: get_batch_1_configs,
    2: get_batch_2_configs,
    3: get_batch_3_configs,
    4: get_batch_4_configs,
}

BATCH_DESCRIPTIONS = {
    0: "Quick Test (3 Strategies)",  # ← Add this
    1: "Time-Based Systematic (Bullish vs Bearish)",
    2: "Cap Utilization Tactical",
    3: "Remaining Cap Tactical",
    4: "Market-Responsive (Ref Asset + Buffer)"
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
        analyze_by_future_regime, summarize_optimal_strategies
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

    # ADD THIS DIAGNOSTIC:
    print("\n" + "=" * 80)
    print("FORWARD REGIME DISTRIBUTION CHECK")
    print("=" * 80)
    print(f"Bull threshold: {settings.REGIME_BULL_THRESHOLD * 100:+.1f}%")
    print(f"Bear threshold: {settings.REGIME_BEAR_THRESHOLD * 100:+.1f}%")
    print("\n3M Forward Regimes:")
    print(df_forward_regimes['Future_Regime_3M'].value_counts())
    print("\n6M Forward Regimes:")
    print(df_forward_regimes['Future_Regime_6M'].value_counts())
    print("=" * 80 + "\n")

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
        # Use 3M horizon instead of 6M
        optimal_3m = summarize_optimal_strategies(future_regime_df, horizon='3M', top_n=10)

        # Also keep 6M for comparison (optional)
        optimal_6m = summarize_optimal_strategies(future_regime_df, horizon='6M', top_n=10)
    else:
        optimal_3m = {}
        optimal_6m = {}

    # DIAGNOSTIC - Add after optimal_3m and optimal_6m creation
    print("\n" + "=" * 80)
    print("OPTIMAL STRATEGIES DIAGNOSTIC")
    print("=" * 80)

    print("\noptimal_6m keys:", list(optimal_6m.keys()) if optimal_6m else "None")
    if optimal_6m:
        for regime, df in optimal_6m.items():
            print(f"  {regime}: {len(df) if df is not None and not df.empty else 0} strategies")

    print("\noptimal_3m keys:", list(optimal_3m.keys()) if optimal_3m else "None")
    if optimal_3m:
        for regime, df in optimal_3m.items():
            print(f"  {regime}: {len(df) if df is not None and not df.empty else 0} strategies")

    # Check what's in future_regime_df
    print("\nfuture_regime_df shape:", future_regime_df.shape if not future_regime_df.empty else "EMPTY")
    if not future_regime_df.empty:
        print("\n6M regime distribution in future_regime_df:")
        if 'future_regime_6m' in future_regime_df.columns:
            print(future_regime_df['future_regime_6m'].value_counts())
        print("\n3M regime distribution in future_regime_df:")
        if 'future_regime_3m' in future_regime_df.columns:
            print(future_regime_df['future_regime_3m'].value_counts())

    print("=" * 80 + "\n")


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
        optimal_3m=optimal_3m,
        intent_vs_regime_6m=None,
        intent_vs_regime_3m=None,
        robust_strategies_6m=None,
        robust_strategies_3m=None,
        ranked_6m_vs_spy=None,
        ranked_6m_vs_bufr=None
    )

    # ========================================================================
    # ADD PLOTTING HERE
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    from visualization.batch_performance_plots import create_all_batch_plots

    create_all_batch_plots(
        results_list=results_list,
        summary_df=summary_df,
        future_regime_df=future_regime_df,
        optimal_6m=optimal_3m,  # Use 3M as primary analysis
        output_dir=output_dir
    )

    # ========================================================================
    # END PLOTTING
    # ========================================================================

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