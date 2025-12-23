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

# Launch months
LAUNCH_MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# For testing, can use subset
# LAUNCH_MONTHS = ['MAR', 'JUN', 'SEP', 'DEC']

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
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_most_recent_launch'
    },
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_remaining_cap'
    },
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_cap_utilization'
    },
    {
        'trigger_type': 'remaining_cap_threshold',
        'trigger_params': {'threshold': 0.25},
        'selection_func_name': 'select_most_recent_launch'
    },
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.75},
        'selection_func_name': 'select_most_recent_launch'
    },
    {
        'trigger_type': 'downside_before_buffer_threshold',
        'trigger_params': {'threshold': 0.0},
        'selection_func_name': 'select_most_recent_launch'
    }
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