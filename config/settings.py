"""
Configuration settings for backtest framework.
All parameters in one place for easy modification.
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

# Create output directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# BACKTEST PARAMETERS
# =============================================================================

# Fund series
SERIES = 'F'  # F-series = 10% buffer

# Launch months
LAUNCH_MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# For testing, can use subset
# LAUNCH_MONTHS = ['MAR', 'JUN', 'SEP', 'DEC']

MONTH_MAP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}

# =============================================================================
# MARKET REGIME PARAMETERS
# =============================================================================

REGIME_WINDOW_MONTHS = 6
REGIME_BULL_THRESHOLD = 0.10   # 10% over 6 months
REGIME_BEAR_THRESHOLD = -0.10  # -10% over 6 months

# =============================================================================
# TRIGGER/SELECTION COMBINATIONS TO TEST
# =============================================================================

# These will be imported and used in main.py
# Format: List of dicts with 'trigger_type', 'trigger_params', 'selection_func_name'

COMBO_CONFIGS = [
    # ==================================================================
    # TIME-BASED REBALANCING STRATEGIES
    # ==================================================================

    # Quarterly rebalancing with different selection methods
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'neutral'
    },
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_remaining_cap',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'  # Maximizes protection
    },
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_cap_utilization',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'  # Seeks fresh protection
    },
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_highest_outcome_and_cap',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'neutral'  # Balanced approach
    },

    # Monthly rebalancing - more frequent adjustments
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'monthly'},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN'],
        'strategy_intent': 'neutral'
    },
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'monthly'},
        'selection_func_name': 'select_remaining_cap',
        'launch_months': ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN'],
        'strategy_intent': 'bearish'
    },

    # Semi-annual rebalancing - less frequent, lower transaction costs
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'semi_annual'},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['MAR', 'SEP'],
        'strategy_intent': 'neutral'
    },
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'semi_annual'},
        'selection_func_name': 'select_remaining_cap',
        'launch_months': ['MAR', 'SEP'],
        'strategy_intent': 'bearish'
    },

    # Annual rebalancing - minimal turnover
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'annual'},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['JAN'],
        'strategy_intent': 'neutral'
    },

    # ==================================================================
    # CAP UTILIZATION THRESHOLD STRATEGIES (DEFENSIVE)
    # ==================================================================

    # Switch when 50% of cap is used (early rotation)
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.50},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'  # Proactive protection preservation
    },
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.50},
        'selection_func_name': 'select_remaining_cap',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'
    },

    # Switch when 75% of cap is used (moderate rotation)
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.75},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'
    },
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.75},
        'selection_func_name': 'select_cap_utilization',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'
    },

    # Switch when 90% of cap is used (late rotation, near cap exhaustion)
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.90},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'
    },

    # ==================================================================
    # REMAINING CAP THRESHOLD STRATEGIES (DEFENSIVE)
    # ==================================================================

    # Switch when remaining cap falls below 50% of original
    {
        'trigger_type': 'remaining_cap_threshold',
        'trigger_params': {'threshold': 0.50},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'
    },
    {
        'trigger_type': 'remaining_cap_threshold',
        'trigger_params': {'threshold': 0.50},
        'selection_func_name': 'select_remaining_cap',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'
    },

    # Switch when remaining cap falls below 25% of original (critical threshold)
    {
        'trigger_type': 'remaining_cap_threshold',
        'trigger_params': {'threshold': 0.25},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'
    },
    {
        'trigger_type': 'remaining_cap_threshold',
        'trigger_params': {'threshold': 0.25},
        'selection_func_name': 'select_cap_utilization',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'
    },

    # Switch when remaining cap falls below 10% (last resort)
    {
        'trigger_type': 'remaining_cap_threshold',
        'trigger_params': {'threshold': 0.10},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'
    },

    # ==================================================================
    # DOWNSIDE BEFORE BUFFER THRESHOLD (PROTECTION MONITORING)
    # ==================================================================

    # Switch when fund enters buffer zone (downside = 0%)
    {
        'trigger_type': 'downside_before_buffer_threshold',
        'trigger_params': {'threshold': 0.0},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'  # Switch immediately when buffer is touched
    },
    {
        'trigger_type': 'downside_before_buffer_threshold',
        'trigger_params': {'threshold': 0.0},
        'selection_func_name': 'select_remaining_cap',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'
    },

    # Switch when close to buffer (2% cushion remaining)
    {
        'trigger_type': 'downside_before_buffer_threshold',
        'trigger_params': {'threshold': -0.02},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'  # Proactive before hitting buffer
    },

    # Switch when moderate drawdown (5% from buffer)
    {
        'trigger_type': 'downside_before_buffer_threshold',
        'trigger_params': {'threshold': -0.05},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'
    },

    # ==================================================================
    # REFERENCE ASSET RETURN THRESHOLD (MARKET-TIMING STRATEGIES)
    # ==================================================================

    # Switch when SPY is down 5% (mild bearish signal)
    {
        'trigger_type': 'ref_asset_return_threshold',
        'trigger_params': {'threshold': -0.05},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'  # React to market weakness
    },
    {
        'trigger_type': 'ref_asset_return_threshold',
        'trigger_params': {'threshold': -0.05},
        'selection_func_name': 'select_remaining_cap',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'
    },

    # Switch when SPY is down 10% (correction territory)
    {
        'trigger_type': 'ref_asset_return_threshold',
        'trigger_params': {'threshold': -0.10},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bearish'
    },

    # Switch when SPY is up 10% (bullish momentum - rotate to capture more upside)
    {
        'trigger_type': 'ref_asset_return_threshold',
        'trigger_params': {'threshold': 0.10},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bullish'  # Capitalize on uptrend
    },

    # Switch when SPY is up 15% (strong bull market)
    {
        'trigger_type': 'ref_asset_return_threshold',
        'trigger_params': {'threshold': 0.15},
        'selection_func_name': 'select_most_recent_launch',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'bullish'
    },

    # ==================================================================
    # HYBRID STRATEGIES - TEST ACROSS ALL MONTHS
    # ==================================================================

    # Cost-conscious: Select based on cost efficiency
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_cost_analysis',
        'launch_months': ['MAR', 'JUN', 'SEP', 'DEC'],
        'strategy_intent': 'neutral'  # Optimize for cost efficiency
    },

    # Aggressive cap preservation with monthly monitoring
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.60},
        'selection_func_name': 'select_remaining_cap',
        'launch_months': ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN'],
        'strategy_intent': 'bearish'
    },
]

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