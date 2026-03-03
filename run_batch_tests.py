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
import pandas as pd


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


def extract_and_export_daily_nav(results_list, output_dir, batch_number):
    """
    Extract daily NAV data from results_list and export to CSV.

    Properly identifies ECR V2 and Existing 90% strategies.
    """
    from pathlib import Path

    print(f"\n{'=' * 80}")
    print(f"EXTRACTING DAILY TIME SERIES DATA")
    print(f"{'=' * 80}\n")

    if not results_list:
        print("❌ No results to extract")
        return

    # Collect all daily dataframes
    all_daily_data = {}

    for idx, result in enumerate(results_list, 1):
        # Get strategy identification
        launch = result.get('launch_month', 'UNK')
        trigger = result.get('trigger_type', 'unknown')
        selection = result.get('selection_algo', 'unknown')
        trigger_params = result.get('trigger_params', {})

        print(f"\nResult {idx}:")
        print(f"  Launch: {launch}")
        print(f"  Trigger: {trigger}")
        print(f"  Trigger Params: {trigger_params}")
        print(f"  Selection: {selection}")

        # GET THE DAILY DATA
        daily_df = result.get('daily_performance', None)

        if daily_df is None or daily_df.empty:
            print(f"  ⚠️  No daily_performance data")
            continue

        # IDENTIFY STRATEGY LABEL (include launch month prefix)
        label = None

        # ECR variants: rebalance_time_period + quarterly + various ECR functions
        if trigger == 'rebalance_time_period':
            if trigger_params.get('frequency') == 'quarterly':
                # Identify which ECR variant
                if 'select_ecr_v2_equal' in selection:
                    label = f'{launch}_ECR_V2_equal'
                    print(f"  ✅ Identified as: {launch} ECR Equal")
                elif 'select_ecr_v2_cap_balanced' in selection:
                    label = f'{launch}_ECR_V2_cap_balanced'
                    print(f"  ✅ Identified as: {launch} ECR Cap-Balanced")
                elif 'select_ecr_v2_cap_moderate' in selection:
                    label = f'{launch}_ECR_V2_cap_moderate'
                    print(f"  ✅ Identified as: {launch} ECR Cap-Moderate")
                elif 'select_ecr_v2_cap_dominant' in selection:
                    label = f'{launch}_ECR_V2_cap_dominant'
                    print(f"  ✅ Identified as: {launch} ECR Cap-Dominant")
                elif 'select_ecr_v2_protection' in selection:
                    label = f'{launch}_ECR_V2_protection'
                    print(f"  ✅ Identified as: {launch} ECR Protection")
                elif 'enhanced_cost_ratio_neutral_v2' in selection:
                    # Legacy/old naming
                    label = f'{launch}_ECR_V2'
                    print(f"  ✅ Identified as: {launch} ECR V2")

        # Existing 90%: cap_utilization_threshold + 0.9 + most_recent_launch
        if trigger == 'cap_utilization_threshold':
            if trigger_params.get('threshold') == 0.9 or trigger_params.get('threshold') == 0.90:
                if 'most_recent_launch' in selection:
                    label = f'{launch}_Existing_90'
                    print(f"  ✅ Identified as: {launch} Existing 90%")

        if label is None:
            # Fallback: use combination with month
            label = f"{launch}_{trigger[:10]}_{selection[:10]}"
            print(f"  ⚠️  Using fallback label: {label}")

        all_daily_data[label] = daily_df
        print(f"  ✅ Stored data: {label} ({len(daily_df)} rows)")

    if not all_daily_data:
        print("\n❌ No daily data found in any results")
        return

    print(f"\n{'=' * 80}")
    print(f"MERGING {len(all_daily_data)} STRATEGIES")
    print(f"{'=' * 80}")

    base_df = None
    benchmarks_added = False

    for label, daily_df in all_daily_data.items():
        print(f"\nProcessing: {label}")

        # Show available columns
        print(f"  Columns: {list(daily_df.columns)[:10]}...")

        # Required: Date column
        if 'Date' not in daily_df.columns:
            print(f"  ❌ No Date column")
            continue

        # Strategy NAV column
        strategy_nav_col = None
        for possible_col in ['Strategy_NAV', 'Strategy NAV', 'NAV', 'Strategy_Value']:
            if possible_col in daily_df.columns:
                strategy_nav_col = possible_col
                break

        if strategy_nav_col is None:
            print(f"  ❌ No Strategy NAV column found")
            print(f"     Available columns: {list(daily_df.columns)}")
            continue

        # Build dataframe for this strategy
        temp_df = pd.DataFrame()
        temp_df['Date'] = pd.to_datetime(daily_df['Date'])
        temp_df[f'{label}_NAV'] = daily_df[strategy_nav_col]

        print(f"  ✅ Using column '{strategy_nav_col}' as {label}_NAV")

        # Add benchmarks (only once)
        if not benchmarks_added:
            # Check for SPY
            spy_col = None
            for possible_col in ['SPY_NAV', 'SPY NAV', 'SPY', 'Benchmark_SPY']:
                if possible_col in daily_df.columns:
                    spy_col = possible_col
                    break

            if spy_col:
                temp_df['SPY_NAV'] = daily_df[spy_col]
                print(f"  ✅ Added SPY from column '{spy_col}'")

            # Check for BUFR
            bufr_col = None
            for possible_col in ['BUFR_NAV', 'BUFR NAV', 'BUFR', 'Benchmark_BUFR']:
                if possible_col in daily_df.columns:
                    bufr_col = possible_col
                    break

            if bufr_col:
                temp_df['BUFR_NAV'] = daily_df[bufr_col]
                print(f"  ✅ Added BUFR from column '{bufr_col}'")

            benchmarks_added = True

        # Merge or initialize
        if base_df is None:
            base_df = temp_df
        else:
            base_df = base_df.merge(
                temp_df[['Date', f'{label}_NAV']],
                on='Date',
                how='outer'
            )

    if base_df is None or base_df.empty:
        print("\n❌ Failed to merge data")
        return

    # Sort by date
    base_df.sort_values('Date', inplace=True)
    base_df.reset_index(drop=True, inplace=True)

    # Calculate returns
    print(f"\n{'=' * 80}")
    print(f"CALCULATING RETURNS")
    print(f"{'=' * 80}")

    for col in base_df.columns:
        if col.endswith('_NAV') and col != 'Date':
            return_col = col.replace('_NAV', '_Return')
            base_df[return_col] = base_df[col].pct_change()
            print(f"  ✅ {return_col}")

    # Reorder columns: Date, NAVs, then Returns
    nav_cols = sorted([c for c in base_df.columns if c.endswith('_NAV')])
    return_cols = sorted([c for c in base_df.columns if c.endswith('_Return')])
    base_df = base_df[['Date'] + nav_cols + return_cols]

    # Export
    output_path = Path(output_dir) / f'batch_{batch_number}_daily_time_series.csv'
    base_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 80}")
    print(f"EXPORT COMPLETE")
    print(f"{'=' * 80}")
    print(f"✅ File: {output_path}")
    print(f"   Rows: {len(base_df):,}")
    print(f"   Columns: {len(base_df.columns)}")

    print(f"\nColumns included:")
    for i, col in enumerate(base_df.columns, 1):
        print(f"  {i:2d}. {col}")

    # Show sample
    print(f"\n{'=' * 80}")
    print(f"SAMPLE DATA (first 5 rows)")
    print(f"{'=' * 80}")

    # Show just NAV columns for clarity
    sample_cols = ['Date'] + [c for c in base_df.columns if c.endswith('_NAV')]
    print(base_df[sample_cols].head().to_string(index=False))

    print(f"\n{'=' * 80}")

    return base_df



def extract_and_export_daily_nav(results_list, output_dir, batch_number):
    """
    Extract daily NAV data with common start date alignment.

    Key improvements:
    - Detects both old and normalized ECR function names
    - Aligns all strategies to common start date (no early starts)
    - Ensures fair comparison across all strategies
    """
    import pandas as pd
    from pathlib import Path

    print(f"\n{'=' * 80}")
    print(f"EXTRACTING DAILY TIME SERIES DATA")
    print(f"{'=' * 80}\n")

    if not results_list:
        print("❌ No results to extract")
        return

    # Collect all daily dataframes
    all_daily_data = {}

    for idx, result in enumerate(results_list, 1):
        # Get strategy identification
        launch = result.get('launch_month', 'UNK')
        trigger = result.get('trigger_type', 'unknown')
        selection = result.get('selection_algo', 'unknown')
        trigger_params = result.get('trigger_params', {})

        print(f"\nResult {idx}:")
        print(f"  Launch: {launch}")
        print(f"  Trigger: {trigger}")
        print(f"  Selection: {selection}")

        # GET THE DAILY DATA
        daily_df = result.get('daily_performance', None)

        if daily_df is None or daily_df.empty:
            print(f"  ⚠️  No daily_performance data")
            continue

        # IDENTIFY STRATEGY LABEL (include launch month prefix)
        label = None

        # ECR variants: rebalance_time_period + quarterly + various ECR functions
        if trigger == 'rebalance_time_period':
            if trigger_params.get('frequency') == 'quarterly':
                # Check for normalized versions FIRST (more specific)
                if 'select_ecr_v2_equal_normalized' in selection:
                    label = f'{launch}_ECR_V2_equal'
                    print(f"  ✅ Identified as: {launch} ECR Equal (Normalized)")
                elif 'select_ecr_v2_cap_balanced_normalized' in selection:
                    label = f'{launch}_ECR_V2_cap_balanced'
                    print(f"  ✅ Identified as: {launch} ECR Cap-Balanced (Normalized)")
                elif 'select_ecr_v2_cap_moderate_normalized' in selection:
                    label = f'{launch}_ECR_V2_cap_moderate'
                    print(f"  ✅ Identified as: {launch} ECR Cap-Moderate (Normalized)")
                elif 'select_ecr_v2_cap_dominant_normalized' in selection:
                    label = f'{launch}_ECR_V2_cap_dominant'
                    print(f"  ✅ Identified as: {launch} ECR Cap-Dominant (Normalized)")
                elif 'select_ecr_v2_protection_normalized' in selection:
                    label = f'{launch}_ECR_V2_protection'
                    print(f"  ✅ Identified as: {launch} ECR Protection (Normalized)")
                # Then check for old non-normalized versions
                elif 'select_ecr_v2_equal' in selection:
                    label = f'{launch}_ECR_V2_equal'
                    print(f"  ✅ Identified as: {launch} ECR Equal (Old)")
                elif 'select_ecr_v2_cap_balanced' in selection:
                    label = f'{launch}_ECR_V2_cap_balanced'
                    print(f"  ✅ Identified as: {launch} ECR Cap-Balanced (Old)")
                elif 'select_ecr_v2_cap_moderate' in selection:
                    label = f'{launch}_ECR_V2_cap_moderate'
                    print(f"  ✅ Identified as: {launch} ECR Cap-Moderate (Old)")
                elif 'select_ecr_v2_cap_dominant' in selection:
                    label = f'{launch}_ECR_V2_cap_dominant'
                    print(f"  ✅ Identified as: {launch} ECR Cap-Dominant (Old)")
                elif 'select_ecr_v2_protection' in selection:
                    label = f'{launch}_ECR_V2_protection'
                    print(f"  ✅ Identified as: {launch} ECR Protection (Old)")
                elif 'enhanced_cost_ratio_neutral_v2' in selection:
                    # Legacy naming
                    label = f'{launch}_ECR_V2'
                    print(f"  ✅ Identified as: {launch} ECR V2 (Legacy)")

        # Existing 90%: cap_utilization_threshold + 0.9 + most_recent_launch
        if trigger == 'cap_utilization_threshold':
            if trigger_params.get('threshold') == 0.9 or trigger_params.get('threshold') == 0.90:
                if 'most_recent_launch' in selection:
                    label = f'{launch}_Existing_90'
                    print(f"  ✅ Identified as: {launch} Existing 90%")

        if label is None:
            # Fallback: use combination with month
            label = f"{launch}_{trigger[:10]}_{selection[:10]}"
            print(f"  ⚠️  Using fallback label: {label}")

        all_daily_data[label] = daily_df
        print(f"  ✅ Stored data: {label} ({len(daily_df)} rows)")

    if not all_daily_data:
        print("\n❌ No daily data found in any results")
        return

    print(f"\n{'=' * 80}")
    print(f"MERGING {len(all_daily_data)} STRATEGIES WITH COMMON START DATE")
    print(f"{'=' * 80}")

    # =========================================================================
    # STEP 1: Find common start date (latest inception across ALL strategies)
    # =========================================================================

    common_start_date = None

    for label, daily_df in all_daily_data.items():
        if 'Date' in daily_df.columns:
            df_start = pd.to_datetime(daily_df['Date']).min()
            if common_start_date is None or df_start > common_start_date:
                common_start_date = df_start
                print(f"  New latest start: {df_start.strftime('%Y-%m-%d')} from {label}")

    if common_start_date is None:
        print("❌ Could not determine common start date")
        return

    print(f"\n✅ Common start date: {common_start_date.strftime('%Y-%m-%d')}")
    print(f"   All strategies will begin on this date for fair comparison\n")

    # =========================================================================
    # STEP 2: Merge strategies with alignment to common start date
    # =========================================================================

    base_df = None
    benchmarks_added = False

    for label, daily_df in all_daily_data.items():
        print(f"\nProcessing: {label}")

        # Required: Date column
        if 'Date' not in daily_df.columns:
            print(f"  ❌ No Date column")
            continue

        # Convert to datetime and filter to common start date
        daily_df = daily_df.copy()
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        daily_df = daily_df[daily_df['Date'] >= common_start_date].copy()

        if daily_df.empty:
            print(f"  ⚠️  No data after {common_start_date.strftime('%Y-%m-%d')}")
            continue

        print(f"  Filtered to {len(daily_df)} rows (from {common_start_date.strftime('%Y-%m-%d')})")

        # Strategy NAV column
        strategy_nav_col = None
        for possible_col in ['Strategy_NAV', 'Strategy NAV', 'NAV', 'Strategy_Value']:
            if possible_col in daily_df.columns:
                strategy_nav_col = possible_col
                break

        if strategy_nav_col is None:
            print(f"  ❌ No Strategy NAV column found")
            continue

        # Build dataframe for this strategy
        temp_df = pd.DataFrame()
        temp_df['Date'] = daily_df['Date']
        temp_df[f'{label}_NAV'] = daily_df[strategy_nav_col]

        # Normalize to 100 at common start date
        first_valid_nav = temp_df[f'{label}_NAV'].dropna().iloc[0]
        temp_df[f'{label}_NAV'] = (temp_df[f'{label}_NAV'] / first_valid_nav) * 100

        print(f"  ✅ Using column '{strategy_nav_col}' as {label}_NAV")
        print(f"     Normalized to 100.0 at start")

        # Add benchmarks (only once)
        if not benchmarks_added:
            # Check for SPY
            spy_col = None
            for possible_col in ['SPY_NAV', 'SPY NAV', 'SPY', 'Benchmark_SPY']:
                if possible_col in daily_df.columns:
                    spy_col = possible_col
                    break

            if spy_col:
                temp_df['SPY_NAV'] = daily_df[spy_col]
                # Normalize SPY
                first_spy = temp_df['SPY_NAV'].dropna().iloc[0]
                temp_df['SPY_NAV'] = (temp_df['SPY_NAV'] / first_spy) * 100
                print(f"  ✅ Added SPY (normalized to 100.0)")

            # Check for BUFR
            bufr_col = None
            for possible_col in ['BUFR_NAV', 'BUFR NAV', 'BUFR', 'Benchmark_BUFR']:
                if possible_col in daily_df.columns:
                    bufr_col = possible_col
                    break

            if bufr_col:
                temp_df['BUFR_NAV'] = daily_df[bufr_col]
                # Normalize BUFR
                first_bufr = temp_df['BUFR_NAV'].dropna().iloc[0]
                temp_df['BUFR_NAV'] = (temp_df['BUFR_NAV'] / first_bufr) * 100
                print(f"  ✅ Added BUFR (normalized to 100.0)")

            benchmarks_added = True

        # Merge or initialize
        if base_df is None:
            base_df = temp_df
        else:
            base_df = base_df.merge(
                temp_df[['Date'] + [c for c in temp_df.columns if c.endswith('_NAV')]],
                on='Date',
                how='outer'
            )

    if base_df is None or base_df.empty:
        print("\n❌ Failed to merge data")
        return

    # Sort by date
    base_df.sort_values('Date', inplace=True)
    base_df.reset_index(drop=True, inplace=True)

    # Calculate returns
    print(f"\n{'=' * 80}")
    print(f"CALCULATING RETURNS")
    print(f"{'=' * 80}")

    for col in base_df.columns:
        if col.endswith('_NAV') and col != 'Date':
            return_col = col.replace('_NAV', '_Return')
            base_df[return_col] = base_df[col].pct_change()
            print(f"  ✅ {return_col}")

    # Reorder columns: Date, NAVs, then Returns
    nav_cols = sorted([c for c in base_df.columns if c.endswith('_NAV')])
    return_cols = sorted([c for c in base_df.columns if c.endswith('_Return')])
    base_df = base_df[['Date'] + nav_cols + return_cols]

    # Export
    output_path = Path(output_dir) / f'batch_{batch_number}_daily_time_series.csv'
    base_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 80}")
    print(f"EXPORT COMPLETE")
    print(f"{'=' * 80}")
    print(f"✅ File: {output_path}")
    print(f"   Rows: {len(base_df):,}")
    print(f"   Date range: {base_df['Date'].min().strftime('%Y-%m-%d')} to {base_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"   All strategies start on: {base_df['Date'].min().strftime('%Y-%m-%d')}")
    print(f"   Columns: {len(base_df.columns)}")

    print(f"\nColumns included:")
    for i, col in enumerate(base_df.columns, 1):
        print(f"  {i:2d}. {col}")

    # Show sample
    print(f"\n{'=' * 80}")
    print(f"SAMPLE DATA (first 5 rows)")
    print(f"{'=' * 80}")

    # Show just NAV columns for clarity
    sample_cols = ['Date'] + [c for c in base_df.columns if c.endswith('_NAV')][:6]
    print(base_df[sample_cols].head().to_string(index=False))

    print(f"\n{'=' * 80}")

    return base_df

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
    threshold_levels = [0.90]
    #threshold_levels = [0.65, 0.90]

    # All 12 launch months for comprehensive averaging

    months = ['SEP',]

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



# ECR
def select_most_recent_launch(df_universe, current_date, series='F'):
    """Select the most recently launched fund."""
    if df_universe.empty:
        return None

    # Filter to the specified series
    df_series = df_universe[df_universe['Fund'].str.startswith(series)].copy()

    if df_series.empty:
        return None

    # Get the fund with the most recent roll date (inception)
    if 'Roll_Date' in df_series.columns:
        most_recent = df_series.loc[df_series['Roll_Date'].idxmax()]
    else:
        # Fallback: just return first fund
        most_recent = df_series.iloc[0]

    return most_recent['Fund']


def get_batch_6a_configs():
    """
    BATCH 6: Enhanced Cost Ratio Testing (FULL)
    ~90 simulations, ~27 minutes

    Tests all 3 Enhanced Cost Ratio regime weightings:
    - Bullish (w_DBB=0.2, w_Buffer=0.1, w_Cap=0.7)
    - Bearish (w_DBB=0.4, w_Buffer=0.4, w_Cap=0.2)
    - Neutral (w_DBB=0.333, w_Buffer=0.333, w_Cap=0.333)

    Structure:
    - GROUP 1: Time-Based Triggers (36 simulations)
    - GROUP 2: Cap Utilization Triggers (27 simulations)
    - GROUP 3: Remaining Cap Triggers (27 simulations)
    """
    configs = []

    # Launch months for testing
    months = ['JAN', 'MAR', 'SEP']

    # =========================================================================
    # GROUP 1: Enhanced Cost Ratio with Time-Based Triggers
    # =========================================================================
    # Test all 3 regime versions with quarterly, semi-annual, annual, and monthly rebalancing

    frequencies = ['quarterly', 'semi_annual']
    ecr_selections = [
        'select_enhanced_cost_ratio_bullish',
        'select_enhanced_cost_ratio_bearish',
        'select_enhanced_cost_ratio_neutral'
    ]

    for freq in frequencies:
        for selection in ecr_selections:
            configs.append({
                'trigger_type': 'rebalance_time_period',
                'trigger_params': {'frequency': freq},
                'selection_func_name': selection,
                'launch_months': months,
                'strategy_intent': 'cost_optimized',
                'description': f'{freq.title()} rebalancing → {selection.replace("select_enhanced_cost_ratio_", "ECR ").title()} (GROUP 1)'
            })

    # =========================================================================
    # GROUP 2: Enhanced Cost Ratio with Cap Utilization Triggers
    # =========================================================================
    # Test all 3 regime versions with cap utilization thresholds

    cap_util_thresholds = [0.50, 0.75, 0.90]

    for threshold in cap_util_thresholds:
        for selection in ecr_selections:
            configs.append({
                'trigger_type': 'cap_utilization_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months,
                'strategy_intent': 'cost_optimized',
                'description': f'Cap Util {threshold * 100:.0f}% → {selection.replace("select_enhanced_cost_ratio_", "ECR ").title()} (GROUP 2)'
            })

    # =========================================================================
    # GROUP 3: Enhanced Cost Ratio with Remaining Cap Triggers
    # =========================================================================
    # Test all 3 regime versions with remaining cap thresholds

    remaining_cap_thresholds = [0.50, 0.75, 0.90]

    for threshold in remaining_cap_thresholds:
        for selection in ecr_selections:
            configs.append({
                'trigger_type': 'remaining_cap_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months,
                'strategy_intent': 'cost_optimized',
                'description': f'Remaining Cap {threshold * 100:.0f}% → {selection.replace("select_enhanced_cost_ratio_", "ECR ").title()} (GROUP 3)'
            })

    print(f"\n{'=' * 80}")
    print(f"BATCH 6 FULL CONFIGURATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total configurations: {len(configs)}")
    print(f"Total simulations: {len(configs) * len(months)} ({len(configs)} configs × {len(months)} months)")
    print(f"Estimated time: ~{(len(configs) * len(months)) * 0.3:.0f} minutes")
    print(f"\nBreakdown:")
    print(f"  GROUP 1 (Time-Based): {len([c for c in configs if 'GROUP 1' in c['description']])} configs × {len(months)} months = {len([c for c in configs if 'GROUP 1' in c['description']]) * len(months)} sims")
    print(f"  GROUP 2 (Cap Util): {len([c for c in configs if 'GROUP 2' in c['description']])} configs × {len(months)} months = {len([c for c in configs if 'GROUP 2' in c['description']]) * len(months)} sims")
    print(f"  GROUP 3 (Remaining Cap): {len([c for c in configs if 'GROUP 3' in c['description']])} configs × {len(months)} months = {len([c for c in configs if 'GROUP 3' in c['description']]) * len(months)} sims")
    print(f"{'=' * 80}\n")

    return configs


def get_batch_6b_configs():
    """
    BATCH 6B: Enhanced Cost Ratio - Extreme Weight Testing
    54 simulations, ~16 minutes

    Tests extreme component weightings to identify what actually drives performance:
    - Pure component strategies (100% one component)
    - Heavily skewed strategies (80-90% one component)
    - Zero-weight ablations (test by exclusion)
    """

    # We'll need to create NEW selection functions with these weights
    # For now, document the weight configurations to implement

    extreme_weights = [
        # =====================================================================
        # PURE COMPONENT TESTS (100% focus)
        # =====================================================================
        {
            'name': 'pure_cap',
            'weights': (0.0, 0.0, 1.0),  # DBB, Buffer, Cap
            'description': 'Pure Cap Focus - 100% cap, ignore everything else',
            'hypothesis': 'If this wins, cap is all that matters'
        },
        {
            'name': 'pure_buffer',
            'weights': (0.0, 1.0, 0.0),
            'description': 'Pure Buffer Focus - 100% buffer integrity',
            'hypothesis': 'If this wins, buffer preservation is key'
        },
        {
            'name': 'pure_dbb',
            'weights': (1.0, 0.0, 0.0),
            'description': 'Pure DBB Focus - 100% downside before buffer',
            'hypothesis': 'If this wins, safety margin is paramount'
        },

        # =====================================================================
        # EXTREME SKEWS (80-90% focus)
        # =====================================================================
        {
            'name': 'ultra_cap',
            'weights': (0.05, 0.05, 0.90),
            'description': 'Ultra Cap-Heavy - 90% cap, minimal else',
            'hypothesis': 'More extreme than Bullish (70% cap)'
        },
        {
            'name': 'ultra_protection',
            'weights': (0.45, 0.45, 0.10),
            'description': 'Ultra Protection - 90% DBB+Buffer, minimal cap',
            'hypothesis': 'More extreme than Bearish (80% protection)'
        },
        {
            'name': 'dbb_dominant',
            'weights': (0.80, 0.10, 0.10),
            'description': 'DBB Dominant - 80% downside focus',
            'hypothesis': 'Maximize safety margin above all'
        },
        {
            'name': 'buffer_dominant',
            'weights': (0.10, 0.80, 0.10),
            'description': 'Buffer Dominant - 80% buffer integrity',
            'hypothesis': 'Preserve buffer at all costs'
        },

        # =====================================================================
        # ABLATION TESTS (test by exclusion)
        # =====================================================================
        {
            'name': 'no_cap',
            'weights': (0.50, 0.50, 0.0),
            'description': 'No Cap Component - equal DBB/Buffer only',
            'hypothesis': 'Can we ignore cap entirely?'
        },
        {
            'name': 'no_buffer',
            'weights': (0.50, 0.0, 0.50),
            'description': 'No Buffer Component - equal DBB/Cap only',
            'hypothesis': 'Is buffer scaling just noise?'
        },
    ]

    configs = []
    months = ['JAN', 'MAR', 'SEP']

    # Test configurations
    test_scenarios = [
        {
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'label': 'Quarterly'
        },
        {
            'trigger_type': 'cap_utilization_threshold',
            'trigger_params': {'threshold': 0.75},
            'label': 'CapUtil75%'
        }
    ]

    # NOTE: You'll need to create selection functions for each weight combo
    # See implementation instructions below

    for weight_config in extreme_weights:
        for scenario in test_scenarios:
            configs.append({
                'trigger_type': scenario['trigger_type'],
                'trigger_params': scenario['trigger_params'],
                'selection_func_name': f"select_ecr_{weight_config['name']}",
                'launch_months': months,
                'strategy_intent': 'cost_optimized',
                'description': f"{scenario['label']} → ECR {weight_config['name'].replace('_', ' ').title()}",
                'weight_config': weight_config['weights'],
                'hypothesis': weight_config['hypothesis']
            })

    print(f"\n{'=' * 80}")
    print(f"BATCH 6B: EXTREME WEIGHT TESTING")
    print(f"{'=' * 80}")
    print(f"Total configurations: {len(configs)}")
    print(f"Total simulations: {len(configs) * len(months)} ({len(configs)} configs × {len(months)} months)")
    print(f"Estimated time: ~{(len(configs) * len(months)) * 0.3:.0f} minutes")
    print(f"\nExtreme Weight Configurations to Test:")
    print(f"{'=' * 80}")

    for i, wc in enumerate(extreme_weights, 1):
        dbb, buf, cap = wc['weights']
        print(f"\n{i}. {wc['name'].upper()}")
        print(f"   Weights: DBB={dbb:.1f}, Buffer={buf:.1f}, Cap={cap:.1f}")
        print(f"   {wc['description']}")
        print(f"   Hypothesis: {wc['hypothesis']}")

    print(f"\n{'=' * 80}")
    print(f"IMPLEMENTATION REQUIRED:")
    print(f"{'=' * 80}")
    print(f"Before running, create 9 new selection functions in core/selections.py:")
    print(f"  - select_ecr_pure_cap()")
    print(f"  - select_ecr_pure_buffer()")
    print(f"  - select_ecr_pure_dbb()")
    print(f"  - select_ecr_ultra_cap()")
    print(f"  - select_ecr_ultra_protection()")
    print(f"  - select_ecr_dbb_dominant()")
    print(f"  - select_ecr_buffer_dominant()")
    print(f"  - select_ecr_no_cap()")
    print(f"  - select_ecr_no_buffer()")
    print(f"\nOr: Modify existing functions to accept weight parameters.")
    print(f"{'=' * 80}\n")

    return configs


def get_batch_6c_configs():
    """
    BATCH 6C: Enhanced Cost Ratio - Component Mechanics Testing
    ~72 simulations, ~22 minutes

    Tests ECR calculation variations while keeping neutral weights (0.333 each):
    - 4 Buffer Integrity scaling ranges
    - 5 Time scaling methods
    - 2 DBB scaling approaches

    Structure:
    - GROUP 1: Buffer Integrity Scaling Variations (24 sims)
    - GROUP 2: Time Scaling Method Variations (30 sims)
    - GROUP 3: DBB Scaling Variations (18 sims)
    """

    configs = []
    months = ['JAN', 'MAR', 'SEP']

    # Base trigger to use for all tests (most common from Batch 6)
    base_trigger = {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.75}
    }

    # =========================================================================
    # GROUP 1: Buffer Integrity Scaling Range Testing
    # =========================================================================
    # Question: Does the [0.5, 1.0] range matter or is any range equally good?

    buffer_scaling_variants = [
        'select_ecr_buffer_scale_full',  # [0.0, 1.0] - Full range
        'select_ecr_buffer_scale_low_floor',  # [0.3, 1.0] - Lower floor
        'select_ecr_buffer_scale_default',  # [0.5, 1.0] - Current default
        'select_ecr_buffer_scale_high_floor'  # [0.7, 1.0] - Higher floor (less penalty)
    ]

    for selection in buffer_scaling_variants:
        configs.append({
            'trigger_type': base_trigger['trigger_type'],
            'trigger_params': base_trigger['trigger_params'],
            'selection_func_name': selection,
            'launch_months': months,
            'strategy_intent': 'cost_optimized',
            'description': f'Cap Util 75% → {selection.replace("select_ecr_", "").replace("_", " ").title()} (GROUP 1)'
        })

        # Also test with quarterly time-based
        configs.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': selection,
            'launch_months': months,
            'strategy_intent': 'cost_optimized',
            'description': f'Quarterly → {selection.replace("select_ecr_", "").replace("_", " ").title()} (GROUP 1)'
        })

    # =========================================================================
    # GROUP 2: Time Scaling Method Testing
    # =========================================================================
    # Question: Is logarithmic time scaling optimal or would linear/sqrt/none work better?

    time_scaling_variants = [
        'select_ecr_time_log',  # 1 - ln(days_remaining/original_days) - Current
        'select_ecr_time_linear',  # days_remaining / original_days
        'select_ecr_time_sqrt',  # sqrt(days_remaining / original_days)
        'select_ecr_time_inverse',  # 1 - (days_remaining / original_days)
        'select_ecr_time_none'  # No time scaling (constant = 1)
    ]

    for selection in time_scaling_variants:
        configs.append({
            'trigger_type': base_trigger['trigger_type'],
            'trigger_params': base_trigger['trigger_params'],
            'selection_func_name': selection,
            'launch_months': months,
            'strategy_intent': 'cost_optimized',
            'description': f'Cap Util 75% → {selection.replace("select_ecr_", "").replace("_", " ").title()} (GROUP 2)'
        })

        # Also test with quarterly time-based
        configs.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': selection,
            'launch_months': months,
            'strategy_intent': 'cost_optimized',
            'description': f'Quarterly → {selection.replace("select_ecr_", "").replace("_", " ").title()} (GROUP 2)'
        })

    # =========================================================================
    # GROUP 3: DBB Scaling Testing
    # =========================================================================
    # Question: Should DBB be scaled like Buffer Integrity or left raw?

    dbb_scaling_variants = [
        'select_ecr_dbb_raw',  # 1 + DBB_decimal (current - no scaling)
        'select_ecr_dbb_scaled'  # MinMaxScaler like Buffer Integrity
    ]

    for selection in dbb_scaling_variants:
        configs.append({
            'trigger_type': base_trigger['trigger_type'],
            'trigger_params': base_trigger['trigger_params'],
            'selection_func_name': selection,
            'launch_months': months,
            'strategy_intent': 'cost_optimized',
            'description': f'Cap Util 75% → {selection.replace("select_ecr_", "").replace("_", " ").title()} (GROUP 3)'
        })

        # Time-based
        configs.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': selection,
            'launch_months': months,
            'strategy_intent': 'cost_optimized',
            'description': f'Quarterly → {selection.replace("select_ecr_", "").replace("_", " ").title()} (GROUP 3)'
        })

        # Remaining cap threshold
        configs.append({
            'trigger_type': 'remaining_cap_threshold',
            'trigger_params': {'threshold': 0.75},
            'selection_func_name': selection,
            'launch_months': months,
            'strategy_intent': 'cost_optimized',
            'description': f'Remaining Cap 75% → {selection.replace("select_ecr_", "").replace("_", " ").title()} (GROUP 3)'
        })

    print(f"\n{'=' * 80}")
    print(f"BATCH 6C: ECR COMPONENT MECHANICS TESTING")
    print(f"{'=' * 80}")
    print(f"Total configurations: {len(configs)}")
    print(f"Total simulations: {len(configs) * len(months)} ({len(configs)} configs × {len(months)} months)")
    print(f"Estimated time: ~{(len(configs) * len(months)) * 0.3:.0f} minutes")
    print(f"\nBreakdown:")
    print(f"  GROUP 1 (Buffer Scaling): 8 configs × {len(months)} months = {8 * len(months)} sims")
    print(f"  GROUP 2 (Time Scaling): 10 configs × {len(months)} months = {10 * len(months)} sims")
    print(f"  GROUP 3 (DBB Scaling): 6 configs × {len(months)} months = {6 * len(months)} sims")
    print(f"\nTesting Component Mechanics (not weights):")
    print(f"  4 Buffer Integrity scaling ranges")
    print(f"  5 Time scaling methods")
    print(f"  2 DBB scaling approaches")
    print(f"{'=' * 80}\n")

    return configs



def get_batch_6d_configs():
    """
    BATCH 6D EXPANDED: ECR Weight Sensitivity + Multiple Launch Months

    Tests:
    - 5 ECR weight scenarios
    - 1 Existing (90% threshold)
    - 3 launch months (JAN, MAR, SEP)

    Total: 6 strategies × 3 months = 18 simulations (~5 minutes)
    """

    configs = []

    # Test all 3 launch months for robustness
    launch_months = [['JAN'], ['MAR'], ['SEP']]

    # ECR weight scenarios to test
    ecr_variants = [
        ('select_ecr_v2_equal_normalized', 'Equal (0.33/0.33/0.33)'),
        ('select_ecr_v2_cap_balanced_normalized', 'Cap-Balanced (0.15/0.35/0.50)'),
        ('select_ecr_v2_cap_moderate_normalized', 'Cap-Moderate (0.25/0.25/0.50)'),
        ('select_ecr_v2_cap_dominant_normalized', 'Cap-Dominant (0.15/0.15/0.70)'),
        ('select_ecr_v2_protection_normalized', 'Protection (0.40/0.40/0.20)'),
    ]

    # Create configs for each ECR variant × each month
    for month_list in launch_months:
        month = month_list[0]

        # ECR variants
        for selection_func, description in ecr_variants:
            configs.append({
                'trigger_type': 'rebalance_time_period',
                'trigger_params': {'frequency': 'quarterly'},
                'selection_func_name': selection_func,
                'launch_months': month_list,
                'strategy_intent': 'cost_optimized',
                'description': f'{month}: ECR V2 {description}'
            })

        # Existing (90% threshold) - one per month for comparison
        configs.append({
            'trigger_type': 'cap_utilization_threshold',
            'trigger_params': {'threshold': 0.90},
            'selection_func_name': 'select_most_recent_launch',
            'launch_months': month_list,
            'strategy_intent': 'bullish',
            'description': f'{month}: Existing 90% Cap Threshold'
        })

    print(f"\n{'=' * 80}")
    print(f"BATCH 6D EXPANDED: WEIGHT SENSITIVITY ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Total configurations: {len(configs)}")
    print(f"ECR variants: {len(ecr_variants)}")
    print(f"Launch months: {len(launch_months)}")
    print(f"Total simulations: {len(configs)}")
    print(f"Estimated time: ~{len(configs) * 0.3:.0f} minutes")
    print(f"\n{'=' * 80}\n")

    return configs


# ============================================================================
# BATCH SELECTOR
# ============================================================================

BATCH_NUMBER = 6  # Change this to run different batches (1-6)


BATCH_CONFIGS = {
    0: get_batch_0_configs,
    1: get_batch_1_configs,
    2: get_batch_2_configs,
    3: get_batch_3_configs,
    4: get_batch_4_configs,
    5: get_batch_5_configs,
    6: get_batch_6d_configs,
    7: get_batch_7_configs,
    8: get_batch_8_configs,
    9: get_batch_9_configs,
    10: get_batch_6b_configs,  # Add this line

}


BATCH_DESCRIPTIONS = {
    0: "Quick Test (3 Strategies)",
    1: "Time-Based Systematic (Bullish vs Bearish)",
    2: "Cap Utilization Tactical",
    3: "Remaining Cap Tactical",
    4: "Market-Responsive (Ref Asset + Buffer)",
    5: "Comprehensive Regime-Optimized (180 sims, 6 months, expanded thresholds)",
    6: "ECR V2 vs Existing (90% Threshold) Comparison",  # ← UPDATE THIS
    7: "Random Assortment",
    8: "Schultz",
    9: "Remaining Buffer Testing (16 sims, validation)",
    10: "Enhanced Cost Ratio - Extreme Weight Testing",  # Add this line

}



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

    # ADD THIS RIGHT BEFORE extract_and_export_daily_nav():

    print("\n" + "=" * 80)
    print("DIAGNOSTIC: Checking results_list structure")
    print("=" * 80)

    if results_list:
        first_result = results_list[0]

        print(f"\nFound {len(results_list)} results")
        print(f"\nKeys in first result:")
        for i, key in enumerate(first_result.keys(), 1):
            value = first_result[key]
            value_type = type(value).__name__

            if isinstance(value, pd.DataFrame):
                print(f"  {i:2d}. {key:<30} → DataFrame ({len(value)} rows, {len(value.columns)} cols)")
                print(f"      Columns: {list(value.columns)[:5]}...")  # Show first 5 columns
            else:
                print(f"  {i:2d}. {key:<30} → {value_type}")

        print("\n" + "=" * 80)
    else:
        print("❌ results_list is empty")

    # Extract and export daily time series to CSV
    extract_and_export_daily_nav(
        results_list=results_list,
        output_dir=output_dir,
        batch_number=BATCH_NUMBER
    )


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

    # Around line 1020-1030
    try:
        generate_batch_visualizations(
            results_list=results_list,
            summary_df=summary_df,
            future_regime_df=future_regime_df,
            optimal_strategies=optimal_3m,
            output_dir=output_dir,
            batch_number=BATCH_NUMBER
        )
    except KeyError as e:
        print(f"\n⚠️  Visualization error (non-critical): {e}")
        print("    Batch results are still saved in Excel workbook")

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