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

# Generate all relevant plots based on batch type and data
from visualization.performance_plots import generate_batch_visualizations
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

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ============================================================================
# CONFIGURATION: SELECT BATCH TO RUN
# ============================================================================


# TESTER

def get_batch_0_configs():
    """
    BATCH 0: Threshold Testing - Cap Utilization Analysis
    Tests: 6 thresholds × 12 months = 72 simulations
    Estimated time: ~22 minutes

    Purpose: Identify optimal cap utilization threshold for switching funds.
    Tests thresholds: 25%, 40%, 50%, 70%, 75%, 90%
    Compares current 90% threshold against alternatives.

    Generates:
    - Threshold comparison chart (line + bar)
    - Regime performance analysis
    """
    configs = []

    # Test these specific thresholds
    threshold_levels = [0.15, 0.50, 0.85]

    # All launch months for fair comparison
    months = ['SEP']

    for threshold in threshold_levels:
        # configs.append({
        #     'trigger_type': 'rebalance_time_period',
        #     'trigger_params': {'frequency': 'quarterly'},
        #     'selection_func_name': 'select_downside_buffer_lowest',
        #     'launch_months': months,
        # })

        configs.append({
            'trigger_type': 'remaining_buffer_threshold',
            'trigger_params': {'threshold': threshold},
            'selection_func_name': 'select_downside_buffer_lowest',
            'launch_months': months,
        })

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
    months = ['SEP']

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

    thresholds = [0.50, 0.75, 0.90]
    months = ['SEP']

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

    thresholds = [0.50, 0.75, 0.90]
    months = ['JAN']

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

    months = ['SEP']

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
    ref_thresholds = [-0.07, 0.0, 0.07]

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

# ALL
def get_comprehensive_regime_batch_configs():
    """
    COMPREHENSIVE BATCH: Optimized for Finding Best Strategy per Regime
    180 simulations, ~54 minutes

    Refinements:
    - 6 launch months (MAR, MAY, JUL, SEP, NOV, JAN) for better coverage
    - Quarterly only for time-based (no monthly/semi-annual/annual)
    - Expanded regime thresholds: ±2%, ±3%, ±5%, ±7%
    - Strategic design for bull/bear/neutral optimization
    """
    configs = []

    # =========================================================================
    # SECTION 1: BULLISH STRATEGIES (60 simulations)
    # Goal: Maximum upside capture in bull markets
    # =========================================================================

    # GROUP 1A: Aggressive Time-Based - Quarterly Only (12 sims)
    # Strategy: Frequent rebalancing to fresh upside
    bullish_selections_aggressive = [
        'select_cap_utilization_lowest',
        'select_remaining_cap_highest'
    ]

    for selection in bullish_selections_aggressive:
        configs.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': selection,
            'strategy_intent': 'bullish'
        })

    # GROUP 1B: Low Cap Utilization Triggers (18 sims)
    # Strategy: Rotate when cap depletes, seeking fresh gains
    for threshold in [0.25, 0.50, 0.75]:
        for selection in bullish_selections_aggressive:
            configs.append({
                'trigger_type': 'cap_utilization_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'strategy_intent': 'bullish'
            })

    # GROUP 1C: Low Remaining Cap Triggers (18 sims)
    # Strategy: Switch when cap running low, find fresh caps
    for threshold in [0.25, 0.50, 0.75]:
        for selection in bullish_selections_aggressive:
            configs.append({
                'trigger_type': 'remaining_cap_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'strategy_intent': 'bullish'
            })

    # GROUP 1D: Positive Momentum Triggers - EXPANDED (12 sims)
    # Strategy: Ride momentum when market is strong
    # Test: +2%, +3%, +5%, +7% return thresholds
    for threshold in [0.03, 0.05]:
        for selection in bullish_selections_aggressive:
            configs.append({
                'trigger_type': 'ref_asset_return_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'strategy_intent': 'bullish'
            })

    # =========================================================================
    # SECTION 2: BEARISH STRATEGIES (60 simulations)
    # Goal: Capital preservation and downside protection
    # =========================================================================

    # GROUP 2A: Defensive Time-Based - Quarterly Only (12 sims)
    # Strategy: Regular rotation to highest buffer protection
    bearish_selections_defensive = [
        'select_downside_buffer_highest',
        'select_cap_utilization_lowest'
    ]

    for selection in bearish_selections_defensive:
        configs.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': selection,
            'strategy_intent': 'bearish'
        })

    # GROUP 2B: Buffer Proximity Triggers - EXPANDED (24 sims)
    # Strategy: Rotate when approaching buffer zone
    # Test: -2%, -3%, -5%, -7% from buffer
    for threshold in [-0.07, -0.05, -0.03]:
        for selection in bearish_selections_defensive:
            configs.append({
                'trigger_type': 'downside_before_buffer_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'strategy_intent': 'bearish'
            })

    # GROUP 2C: Negative Momentum Triggers - EXPANDED (12 sims)
    # Strategy: Rotate to protection when market weakens
    # Test: -2%, -3%, -5%, -7% returns
    for threshold in [-0.07, -0.05, -0.03]:
        for selection in bearish_selections_defensive:
            configs.append({
                'trigger_type': 'ref_asset_return_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'strategy_intent': 'bearish'
            })

    # GROUP 2D: High Cap Utilization Triggers (12 sims)
    # Strategy: When cap is exhausted, rotate to fresh protection
    for threshold in [0.75, 0.90]:
        for selection in bearish_selections_defensive:
            configs.append({
                'trigger_type': 'cap_utilization_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'strategy_intent': 'bearish'
            })

    # =========================================================================
    # SECTION 3: NEUTRAL STRATEGIES (60 simulations)
    # Goal: Consistency across all market conditions
    # =========================================================================

    # GROUP 3A: Balanced Time-Based - Quarterly Only (18 sims)
    # Strategy: Systematic rebalancing with balanced selections
    neutral_selections_balanced = [
        'select_remaining_cap_highest',
        'select_downside_buffer_highest',
        'select_cap_utilization_lowest'
    ]

    for selection in neutral_selections_balanced:
        configs.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': selection,
            'strategy_intent': 'neutral'
        })

    # GROUP 3B: Moderate Cap Thresholds (18 sims)
    # Strategy: Mid-range triggers for balanced rotation
    for threshold in [0.40, 0.70]:
        for selection in neutral_selections_balanced:
            configs.append({
                'trigger_type': 'cap_utilization_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'strategy_intent': 'neutral'
            })

    # GROUP 3C: Moderate Remaining Cap (18 sims)
    # Strategy: Switch at moderate depletion
    for threshold in [0.40, 0.70]:
        for selection in neutral_selections_balanced:
            configs.append({
                'trigger_type': 'remaining_cap_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'strategy_intent': 'neutral'
            })

    # GROUP 3D: Zero-Threshold Triggers (6 sims)
    # Strategy: React to any directional move
    for selection in ['select_downside_buffer_highest']:
        configs.append({
            'trigger_type': 'ref_asset_return_threshold',
            'trigger_params': {'threshold': 0.0},
            'selection_func_name': selection,
            'strategy_intent': 'neutral'
        })


    return configs

# ALL CALLER
def get_batch_5_configs():
    """
    BATCH 5: Comprehensive Regime-Optimized Testing
    180 simulations, ~54 minutes
    """
    configs = get_comprehensive_regime_batch_configs()

    # Add launch_months to each config
    months = ['JAN', 'MAR', 'MAY', 'JUL', 'SEP', 'NOV']
    for config in configs:
        config['launch_months'] = months

    return configs


# random
def get_batch_7_configs():
    """
    BATCH 7: Four Specific Strategy Comparison (SEP Month Only)
    Tests: 4 strategies × 1 month = 4 simulations
    Estimated time: ~1-2 minutes

    Purpose: Direct comparison of 4 specific trigger/selection combinations
    to evaluate how different entry/exit logic and selection criteria interact.

    Strategies:
    1. Remaining Cap (75%) → Highest Remaining Cap
       - Exit: When 75% cap remaining (early/bullish exit)
       - Select: Fund with most upside potential (bullish selection)
       - Intent: Double bullish (capture upside aggressively)

    2. Cap Utilization (75%) → Highest Utilization
       - Exit: When 75% cap used (late/bearish exit)
       - Select: Fund with least upside remaining (bearish selection)
       - Intent: Double bearish (conservative positioning)

    3. Cap Utilization (75%) → Lowest Utilization
       - Exit: When 75% cap used (late/bearish exit)
       - Select: Fund with most upside remaining (bullish selection)
       - Intent: Hybrid (patient exit, aggressive selection)

    4. Downside Buffer (50%) → Lowest Utilization
       - Exit: When 50% before buffer (defensive exit)
       - Select: Fund with most upside remaining (bullish selection)
       - Intent: Hybrid (risk-aware exit, aggressive selection)
    """
    configs = []

    # Only SEP month for direct comparison
    months = ['SEP']

    # =========================================================================
    # Strategy 1: Remaining Cap 75% → Highest Remaining Cap
    # =========================================================================
    configs.append({
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_remaining_cap_highest',
        'launch_months': months,    })

    # =========================================================================
    # Strategy 2: Cap Utilization 75% → Highest Utilization
    # =========================================================================
    configs.append({
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.65},  # Switch when 75% cap utilized
        'selection_func_name': 'select_cap_utilization_lowest',
        'launch_months': months,    })


    configs.append({
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_remaining_buffer_lowest',
        'launch_months': months,
    })

    configs.append({
        'trigger_type': 'remaining_buffer_threshold',
        'trigger_params': {'threshold': 0.85},
        'selection_func_name': 'select_downside_buffer_lowest',
        'launch_months': months,
    })

    return configs

# AUSTIN SCHULTZ CURRENT
def get_batch_8_configs():
    """
    BATCH 8: Comprehensive Threshold Analysis
    Tests: 4 thresholds × 12 months = 48 simulations
    Estimated time: ~15 minutes

    Purpose: Demonstrate that 90% threshold underperforms compared to alternatives.
    Tests thresholds: 25%, 40%, 75%, 90%
    Strategy: cap_utilization_threshold + select_most_recent_launch

    Generates:
    - Bar chart with performance table
    - Clear ranking showing 90% as suboptimal
    """
    configs = []

    # Test these specific thresholds
    threshold_levels = [0.55, 0.65, 0.75, 0.90]
    #threshold_levels = [0.65, 0.90]

    # All 12 launch months for comprehensive averaging
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
              'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    months = ['DEC',]

    for threshold in threshold_levels:
        configs.append({
            'trigger_type': 'cap_utilization_threshold',
            'trigger_params': {'threshold': threshold},
            'selection_func_name': 'select_most_recent_launch',
            'launch_months': months
        })

    return configs


def get_batch_9_configs():
    """
    BATCH 9: Remaining Buffer Testing (New Feature Validation)
    Tests: 16 strategies × 1 month = 16 simulations
    Estimated time: ~5 minutes

    Purpose: Validate new remaining buffer selection and trigger functions.

    Two strategy groups:
    1. GROUP 13: Time-based + select_remaining_buffer_lowest (4 strategies)
       - Tests: Monthly, Quarterly, Semi-annual, Annual frequencies
       - All paired with new remaining_buffer_lowest selection

    2. GROUP 14: Buffer threshold triggers (12 strategies)
       - Tests: 3 thresholds (15%, 50%, 85%) × 4 selections
       - Thresholds indicate when to rotate based on buffer depletion
       - Selections: remaining_buffer_lowest, downside_buffer_lowest,
                     most_recent_launch, cap_utilization_lowest

    All strategies are BEARISH (defensive capital preservation focus).
    Uses SEP month only for quick validation.

    Expected Results:
    - Strategies should trade more in bear markets
    - Lower threshold (15%) = more sensitive/frequent rotation
    - Higher threshold (85%) = conservative/infrequent rotation
    - Double bearish (buffer threshold + buffer selection) should show
      strongest defensive characteristics
    """
    configs = []

    # Use single month for quick testing
    months = ['SEP']

    # =========================================================================
    # GROUP 13: Time-Based + Remaining Buffer Selection
    # =========================================================================

    frequencies = ['quarterly']

    for freq in frequencies:
        configs.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': freq},
            'selection_func_name': 'select_remaining_buffer_lowest',
            'launch_months': months,
            'strategy_intent': 'bearish',
            'description': f'{freq.title()} → lowest buffer (GROUP 13)'
        })

    # =========================================================================
    # GROUP 14: Buffer Threshold Triggers
    # =========================================================================

    thresholds = [0.15, 0.50, 0.85]

    selections = [
        'select_remaining_buffer_lowest',  # Double bearish
    ]

    for threshold in thresholds:
        for selection in selections:
            configs.append({
                'trigger_type': 'remaining_buffer_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months,
                'strategy_intent': 'bearish',
                'description': f'Buffer {threshold * 100:.0f}% → {selection} (GROUP 14)'
            })

    return configs


# ============================================================================
# BATCH SELECTOR
# ============================================================================

BATCH_NUMBER = 7  # Change this to run different batches (1-6)


BATCH_CONFIGS = {
    0: get_batch_0_configs,
    1: get_batch_1_configs,
    2: get_batch_2_configs,
    3: get_batch_3_configs,
    4: get_batch_4_configs,
    5: get_batch_5_configs,
    7: get_batch_7_configs,
    8: get_batch_8_configs,
    9: get_batch_9_configs

}

BATCH_DESCRIPTIONS = {
    0: "Quick Test (3 Strategies)",
    1: "Time-Based Systematic (Bullish vs Bearish)",
    2: "Cap Utilization Tactical",
    3: "Remaining Cap Tactical",
    4: "Market-Responsive (Ref Asset + Buffer)",
    5: "Comprehensive Regime-Optimized (180 sims, 6 months, expanded thresholds)",
    7: "Random Assortment",
    8: "Schultz",
    9: "Remaining Buffer Testing (16 sims, validation)"

}



# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Run the selected batch."""

    if BATCH_NUMBER not in BATCH_CONFIGS:
        print(f"❌ Invalid BATCH_NUMBER: {BATCH_NUMBER}")
        return

    print("\n" + "=" * 80)
    print(f"BATCH {BATCH_NUMBER}: {BATCH_DESCRIPTIONS[BATCH_NUMBER]}")

    # Get batch configurations
    batch_configs = BATCH_CONFIGS[BATCH_NUMBER]()

    # Calculate total simulations (configs * launch_months)
    total_sims = sum(len(config['launch_months']) for config in batch_configs)
    print(f"Total simulations: {total_sims}")
    print(f"Estimated time: ~{total_sims * 0.3:.0f} minutes ({total_sims * 0.3 / 60:.1f} hours)")

    print("\n" + "-" * 80)
    input("Press ENTER to start batch execution...")
    print("-" * 80 + "\n")

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
    summary_df = consolidate_results(results_list)

    # Forward regime analysis
    future_regime_df = analyze_by_future_regime(
        results_list,
        df_forward_regimes,
        entry_frequency='quarterly'  # This creates multiple entry points
    )

    if not future_regime_df.empty:
        optimal_3m = summarize_optimal_strategies(future_regime_df, horizon='3M', top_n=10)
        optimal_6m = summarize_optimal_strategies(future_regime_df, horizon='6M', top_n=10)
    else:
        optimal_3m = {}
        optimal_6m = {}

    if optimal_6m:
        for regime, df in optimal_6m.items():
            print(f"  {regime}: {len(df) if df is not None and not df.empty else 0} strategies")

    if optimal_3m:
        for regime, df in optimal_3m.items():
            print(f"  {regime}: {len(df) if df is not None and not df.empty else 0} strategies")

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

    generate_batch_visualizations(
        results_list=results_list,
        summary_df=summary_df,
        future_regime_df=future_regime_df,
        optimal_strategies=optimal_3m,
        output_dir=output_dir,
        batch_number=BATCH_NUMBER
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