"""
Configuration settings for backtest framework.
All parameters in one place for easy modification.

This file now generates 68 unique strategy combinations:
- GROUP 1-4: Time-based rebalancing (20 strategies)
- GROUP 5-10: Threshold-based triggers (48 strategies)
- Total: 68 strategies × 12 launch months = 816 simulations
"""

import os

# =============================================================================
# FILE PATHS
# =============================================================================

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'input_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Input files
DATA_FILE = os.path.join(INPUT_DIR, 'data.csv')
BENCHMARK_FILE = os.path.join(INPUT_DIR, 'benchmark_ts.csv')
ROLL_DATES_FILE = os.path.join(INPUT_DIR, 'roll_dates.csv')

# Output subdirectories
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'backtest_results')
REGIME_DIR = os.path.join(OUTPUT_DIR, 'regime_analysis')
TRADE_LOG_DIR = os.path.join(OUTPUT_DIR, 'trade_logs')

# Create output directories if they don't exist
for directory in [RESULTS_DIR, REGIME_DIR, TRADE_LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# =============================================================================
# BACKTEST PARAMETERS
# =============================================================================

# Fund series
SERIES = 'F'  # F-series = 10% buffer

MONTH_MAP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}

# Launch months
LAUNCH_MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# For testing, can use subset
# LAUNCH_MONTHS = ['MAR', 'JUN', 'SEP', 'DEC']

# =============================================================================
# MARKET REGIME PARAMETERS
# =============================================================================

REGIME_WINDOW_MONTHS = 6
REGIME_BULL_THRESHOLD = 0.03   # 5% gain = bull
REGIME_BEAR_THRESHOLD = -0.03  # -5% loss = bear  ← More sensitive

# =============================================================================
# REBALANCING FREQUENCIES
# =============================================================================

REBALANCE_FREQUENCIES = ['monthly', 'quarterly', 'semi_annual', 'annual']

# =============================================================================
# THRESHOLD PARAMETERS
# =============================================================================

# Standard thresholds for threshold-based triggers
THRESHOLD_LEVELS = [0.15, 0.50, 0.85]  # 15%, 50%, 85%

# =============================================================================
# TRIGGER/SELECTION COMBINATIONS TO TEST
# =============================================================================

def generate_combo_configs():
    """
    Generate all 68 strategy combinations programmatically.

    Returns:
        List of dicts with 'trigger_type', 'trigger_params', 'selection_func_name'
    """
    combos = []

    # =========================================================================
    # GROUP 1: Time-Based Rebalancing + Most Recent Launch (NEUTRAL)
    # Count: 4 strategies
    # =========================================================================

    for freq in REBALANCE_FREQUENCIES:
        combos.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': freq},
            'selection_func_name': 'select_most_recent_launch',
            'group': 'GROUP_1_TIME_NEUTRAL',
            'description': f'{freq.title()} rebalance to most recent launch'
        })

    # =========================================================================
    # GROUP 2: Time-Based + Remaining Cap Selection (DIRECTIONAL)
    # Count: 8 strategies (4 bullish + 4 bearish)
    # =========================================================================

    # GROUP 2A: BULLISH (Highest Cap)
    for freq in REBALANCE_FREQUENCIES:
        combos.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': freq},
            'selection_func_name': 'select_remaining_cap_highest',
            'group': 'GROUP_2A_TIME_CAP_BULLISH',
            'description': f'{freq.title()} rebalance to highest remaining cap'
        })

    # GROUP 2B: BEARISH (Lowest Cap)
    for freq in REBALANCE_FREQUENCIES:
        combos.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': freq},
            'selection_func_name': 'select_remaining_cap_lowest',
            'group': 'GROUP_2B_TIME_CAP_BEARISH',
            'description': f'{freq.title()} rebalance to lowest remaining cap'
        })

    # =========================================================================
    # GROUP 3: Time-Based + Downside Before Buffer Selection (DIRECTIONAL)
    # Count: 8 strategies (4 bullish + 4 bearish)
    # =========================================================================

    # GROUP 3A: BULLISH (Highest Downside Buffer)
    for freq in REBALANCE_FREQUENCIES:
        combos.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': freq},
            'selection_func_name': 'select_downside_buffer_highest',
            'group': 'GROUP_3A_TIME_DOWNSIDE_BULLISH',
            'description': f'{freq.title()} rebalance to highest downside before buffer'
        })

    # GROUP 3B: BEARISH (Lowest Downside Buffer)
    for freq in REBALANCE_FREQUENCIES:
        combos.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': freq},
            'selection_func_name': 'select_downside_buffer_lowest',
            'group': 'GROUP_3B_TIME_DOWNSIDE_BEARISH',
            'description': f'{freq.title()} rebalance to lowest downside before buffer'
        })

    # =========================================================================
    # GROUP 4: Time-Based + Cap Utilization Selection (DIRECTIONAL)
    # Count: 8 strategies (4 bullish + 4 bearish)
    # =========================================================================

    # GROUP 4A: BULLISH (Lowest Cap Utilization)
    for freq in REBALANCE_FREQUENCIES:
        combos.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': freq},
            'selection_func_name': 'select_cap_utilization_lowest',
            'group': 'GROUP_4A_TIME_UTIL_BULLISH',
            'description': f'{freq.title()} rebalance to lowest cap utilization'
        })

    # GROUP 4B: BEARISH (Highest Cap Utilization)
    for freq in REBALANCE_FREQUENCIES:
        combos.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': freq},
            'selection_func_name': 'select_cap_utilization_highest',
            'group': 'GROUP_4B_TIME_UTIL_BEARISH',
            'description': f'{freq.title()} rebalance to highest cap utilization'
        })

    # =========================================================================
    # GROUP 5: Remaining Cap Threshold + Most Recent Launch (BULLISH)
    # Count: 3 strategies
    # =========================================================================

    for threshold in THRESHOLD_LEVELS:
        combos.append({
            'trigger_type': 'remaining_cap_threshold',
            'trigger_params': {'threshold': threshold},
            'selection_func_name': 'select_most_recent_launch',
            'group': 'GROUP_5_CAP_THRESHOLD_BULLISH',
            'description': f'Switch at {threshold*100:.0f}% remaining cap to most recent'
        })

    # =========================================================================
    # GROUP 6: Remaining Cap Threshold + Ranked Cap Selection (DIRECTIONAL)
    # Count: 6 strategies (3 bullish + 3 neutral)
    # =========================================================================

    # GROUP 6A: BULLISH² (Threshold + Highest Cap)
    for threshold in THRESHOLD_LEVELS:
        combos.append({
            'trigger_type': 'remaining_cap_threshold',
            'trigger_params': {'threshold': threshold},
            'selection_func_name': 'select_remaining_cap_highest',
            'group': 'GROUP_6A_CAP_THRESHOLD_CAP_BULLISH',
            'description': f'Switch at {threshold*100:.0f}% remaining cap to highest cap'
        })

    # GROUP 6B: NEUTRAL (Threshold + Lowest Cap)
    for threshold in THRESHOLD_LEVELS:
        combos.append({
            'trigger_type': 'remaining_cap_threshold',
            'trigger_params': {'threshold': threshold},
            'selection_func_name': 'select_remaining_cap_lowest',
            'group': 'GROUP_6B_CAP_THRESHOLD_CAP_NEUTRAL',
            'description': f'Switch at {threshold*100:.0f}% remaining cap to lowest cap'
        })

    # =========================================================================
    # GROUP 7: Cap Utilization Threshold + Most Recent Launch (BULLISH)
    # Count: 3 strategies
    # =========================================================================

    for threshold in THRESHOLD_LEVELS:
        combos.append({
            'trigger_type': 'cap_utilization_threshold',
            'trigger_params': {'threshold': threshold},
            'selection_func_name': 'select_most_recent_launch',
            'group': 'GROUP_7_UTIL_THRESHOLD_BULLISH',
            'description': f'Switch at {threshold*100:.0f}% cap utilization to most recent'
        })

    # =========================================================================
    # GROUP 8: Cap Utilization Threshold + Ranked Utilization (DIRECTIONAL)
    # Count: 6 strategies (3 bullish + 3 neutral)
    # =========================================================================

    # GROUP 8A: BULLISH² (Threshold + Lowest Utilization)
    for threshold in THRESHOLD_LEVELS:
        combos.append({
            'trigger_type': 'cap_utilization_threshold',
            'trigger_params': {'threshold': threshold},
            'selection_func_name': 'select_cap_utilization_lowest',
            'group': 'GROUP_8A_UTIL_THRESHOLD_UTIL_BULLISH',
            'description': f'Switch at {threshold*100:.0f}% utilization to lowest utilization'
        })

    # GROUP 8B: NEUTRAL (Threshold + Highest Utilization)
    for threshold in THRESHOLD_LEVELS:
        combos.append({
            'trigger_type': 'cap_utilization_threshold',
            'trigger_params': {'threshold': threshold},
            'selection_func_name': 'select_cap_utilization_highest',
            'group': 'GROUP_8B_UTIL_THRESHOLD_UTIL_NEUTRAL',
            'description': f'Switch at {threshold*100:.0f}% utilization to highest utilization'
        })

    # =========================================================================
    # GROUP 9: Downside Before Buffer Threshold + Most Recent (BEARISH)
    # Count: 3 strategies
    # =========================================================================

    for threshold in THRESHOLD_LEVELS:
        combos.append({
            'trigger_type': 'downside_before_buffer_threshold',
            'trigger_params': {'threshold': threshold},
            'selection_func_name': 'select_most_recent_launch',
            'group': 'GROUP_9_DOWNSIDE_THRESHOLD_BEARISH',
            'description': f'Switch at {threshold*100:.0f}% downside to most recent'
        })

    # =========================================================================
    # GROUP 10: Downside Threshold + Ranked Downside Selection (DIRECTIONAL)
    # Count: 6 strategies (3 neutral + 3 bearish)
    # =========================================================================

    # GROUP 10A: NEUTRAL (Threshold + Highest Downside)
    for threshold in THRESHOLD_LEVELS:
        combos.append({
            'trigger_type': 'downside_before_buffer_threshold',
            'trigger_params': {'threshold': threshold},
            'selection_func_name': 'select_downside_buffer_highest',
            'group': 'GROUP_10A_DOWNSIDE_THRESHOLD_DOWNSIDE_NEUTRAL',
            'description': f'Switch at {threshold*100:.0f}% downside to highest downside'
        })

    # GROUP 10B: BEARISH² (Threshold + Lowest Downside)
    for threshold in THRESHOLD_LEVELS:
        combos.append({
            'trigger_type': 'downside_before_buffer_threshold',
            'trigger_params': {'threshold': threshold},
            'selection_func_name': 'select_downside_buffer_lowest',
            'group': 'GROUP_10B_DOWNSIDE_THRESHOLD_DOWNSIDE_BEARISH',
            'description': f'Switch at {threshold*100:.0f}% downside to lowest downside'
        })

    # =========================================================================
    # GROUP 11: Cost Analysis (Time-Based) - COST-OPTIMIZED
    # Count: 4 strategies
    # =========================================================================

    for freq in REBALANCE_FREQUENCIES:
        combos.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': freq},
            'selection_func_name': 'select_cost_analysis',
            'group': 'GROUP_11_TIME_COST',
            'description': f'{freq.title()} rebalance to best cost per day'
        })

    # =========================================================================
    # GROUP 12: Cost Analysis (Threshold-Based) - COST-OPTIMIZED
    # Count: 9 strategies
    # =========================================================================

    # Remaining cap threshold + cost analysis
    for threshold in THRESHOLD_LEVELS:
        combos.append({
            'trigger_type': 'remaining_cap_threshold',
            'trigger_params': {'threshold': threshold},
            'selection_func_name': 'select_cost_analysis',
            'group': 'GROUP_12A_CAP_THRESHOLD_COST',
            'description': f'Switch at {threshold*100:.0f}% remaining cap to best cost'
        })

    # Cap utilization threshold + cost analysis
    for threshold in THRESHOLD_LEVELS:
        combos.append({
            'trigger_type': 'cap_utilization_threshold',
            'trigger_params': {'threshold': threshold},
            'selection_func_name': 'select_cost_analysis',
            'group': 'GROUP_12B_UTIL_THRESHOLD_COST',
            'description': f'Switch at {threshold*100:.0f}% utilization to best cost'
        })

    # Downside threshold + cost analysis
    for threshold in THRESHOLD_LEVELS:
        combos.append({
            'trigger_type': 'downside_before_buffer_threshold',
            'trigger_params': {'threshold': threshold},
            'selection_func_name': 'select_cost_analysis',
            'group': 'GROUP_12C_DOWNSIDE_THRESHOLD_COST',
            'description': f'Switch at {threshold*100:.0f}% downside to best cost'
        })

    return combos


# Generate all combinations
COMBO_CONFIGS = generate_combo_configs()

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

def print_config_summary():
    """
    Print summary of all strategy combinations.
    """
    print("\n" + "=" * 80)
    print("STRATEGY CONFIGURATION SUMMARY")
    print("=" * 80)

    total_combos = len(COMBO_CONFIGS)
    total_sims = total_combos * len(LAUNCH_MONTHS)

    print(f"\nTotal strategy combinations: {total_combos}")
    print(f"Launch months: {len(LAUNCH_MONTHS)}")
    print(f"Total simulations: {total_sims}")

    # Group counts
    from collections import defaultdict
    group_counts = defaultdict(int)

    for combo in COMBO_CONFIGS:
        group = combo['group']
        group_counts[group] += 1

    print("\n" + "-" * 80)
    print("BREAKDOWN BY GROUP:")
    print("-" * 80)

    group_order = [
        'GROUP_1_TIME_NEUTRAL',
        'GROUP_2A_TIME_CAP_BULLISH',
        'GROUP_2B_TIME_CAP_BEARISH',
        'GROUP_3A_TIME_DOWNSIDE_BULLISH',
        'GROUP_3B_TIME_DOWNSIDE_BEARISH',
        'GROUP_4A_TIME_UTIL_BULLISH',
        'GROUP_4B_TIME_UTIL_BEARISH',
        'GROUP_5_CAP_THRESHOLD_BULLISH',
        'GROUP_6A_CAP_THRESHOLD_CAP_BULLISH',
        'GROUP_6B_CAP_THRESHOLD_CAP_NEUTRAL',
        'GROUP_7_UTIL_THRESHOLD_BULLISH',
        'GROUP_8A_UTIL_THRESHOLD_UTIL_BULLISH',
        'GROUP_8B_UTIL_THRESHOLD_UTIL_NEUTRAL',
        'GROUP_9_DOWNSIDE_THRESHOLD_BEARISH',
        'GROUP_10A_DOWNSIDE_THRESHOLD_DOWNSIDE_NEUTRAL',
        'GROUP_10B_DOWNSIDE_THRESHOLD_DOWNSIDE_BEARISH',
        'GROUP_11_TIME_COST',
        'GROUP_12A_CAP_THRESHOLD_COST',
        'GROUP_12B_UTIL_THRESHOLD_COST',
        'GROUP_12C_DOWNSIDE_THRESHOLD_COST'
    ]

    for group in group_order:
        if group in group_counts:
            count = group_counts[group]
            sims = count * len(LAUNCH_MONTHS)
            print(f"{group:50s}: {count:2d} combos × 12 months = {sims:3d} sims")

    print("-" * 80)
    print(f"{'TOTAL':50s}: {total_combos:2d} combos × 12 months = {total_sims:3d} sims")
    print("=" * 80 + "\n")


# =============================================================================
# BUFFER LEVELS BY SERIES
# =============================================================================

BUFFER_LEVELS = {
    'F': 0.10,  # 10% buffer
    'G': 0.15,  # 15% buffer
    'D': 0.05   # 5% deductible, then up to 30% buffer
}

# =============================================================================
# PERFORMANCE METRICS SETTINGS
# =============================================================================

ANNUALIZATION_FACTOR = 252  # Trading days per year
RISK_FREE_RATE = 0.0        # For Sharpe ratio calculation

