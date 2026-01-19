# =========================================================================
# SINGLE
# =========================================================================


# Test a single threshold strategy
test_combos = [{
    'trigger_type': 'remaining_cap_threshold',
    'trigger_params': {'threshold': 0.50},  # Switch when cap drops to 50%
    'selection_func_name': 'select_most_recent_launch',
    'description': 'Switch at 50% cap to most recent (bullish)'
}]
test_months = ['MAR']
# Total: 1 simulation


# =========================================================================
# Test one threshold type at 3 different levels
# =========================================================================


test_combos = [
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.15},  # 15% utilized
        'selection_func_name': 'select_most_recent_launch',
        'description': 'Switch at 15% utilization'
    },
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.50},  # 50% utilized
        'selection_func_name': 'select_most_recent_launch',
        'description': 'Switch at 50% utilization'
    },
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.85},  # 85% utilized
        'selection_func_name': 'select_most_recent_launch',
        'description': 'Switch at 85% utilization'
    }
]
test_months = ['MAR']
# Total: 3 simulations


# =========================================================================
# Test all 3 threshold types at the mid-level (50%)
# =========================================================================



test_combos = [
    # Remaining cap threshold
    {
        'trigger_type': 'remaining_cap_threshold',
        'trigger_params': {'threshold': 0.50},
        'selection_func_name': 'select_most_recent_launch',
        'description': 'Cap threshold 50%'
    },
    {
        'trigger_type': 'remaining_cap_threshold',
        'trigger_params': {'threshold': 0.50},
        'selection_func_name': 'select_remaining_cap_highest',
        'description': 'Cap threshold 50% → highest cap'
    },
    {
        'trigger_type': 'remaining_cap_threshold',
        'trigger_params': {'threshold': 0.50},
        'selection_func_name': 'select_remaining_cap_lowest',
        'description': 'Cap threshold 50% → lowest cap'
    },

    # Cap utilization threshold
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.50},
        'selection_func_name': 'select_most_recent_launch',
        'description': 'Utilization threshold 50%'
    },
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.50},
        'selection_func_name': 'select_cap_utilization_lowest',
        'description': 'Utilization threshold 50% → lowest util'
    },
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.50},
        'selection_func_name': 'select_cap_utilization_highest',
        'description': 'Utilization threshold 50% → highest util'
    },

    # Downside before buffer threshold
    {
        'trigger_type': 'downside_before_buffer_threshold',
        'trigger_params': {'threshold': 0.50},
        'selection_func_name': 'select_most_recent_launch',
        'description': 'Downside threshold 50%'
    },
    {
        'trigger_type': 'downside_before_buffer_threshold',
        'trigger_params': {'threshold': 0.50},
        'selection_func_name': 'select_downside_buffer_highest',
        'description': 'Downside threshold 50% → highest downside'
    },
    {
        'trigger_type': 'downside_before_buffer_threshold',
        'trigger_params': {'threshold': 0.50},
        'selection_func_name': 'select_downside_buffer_lowest',
        'description': 'Downside threshold 50% → lowest downside'
    }
]
test_months = ['MAR']
# Total: 9 simulations







# =========================================================================

# Comparing Threshold Levels
# =========================================================================

test_combos = [
    # Conservative: Switch early when only 15% cap is used
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.15},
        'selection_func_name': 'select_most_recent_launch',
        'description': 'Early switch (15% utilized)'
    },

    # Moderate: Switch at mid-point when 50% cap is used
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.50},
        'selection_func_name': 'select_most_recent_launch',
        'description': 'Mid switch (50% utilized)'
    },

    # Aggressive: Switch late when 85% cap is used
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.85},
        'selection_func_name': 'select_most_recent_launch',
        'description': 'Late switch (85% utilized)'
    }
]
test_months = ['MAR', 'JUN', 'SEP', 'DEC']  # Test across year
# Total: 3 thresholds × 4 quarters = 12 simulations





# =========================================================================

# Time-Based Triggers (use cadence)

# Example 1: Test All 4 Cadences
# =========================================================================
# Compare monthly vs quarterly vs semi-annual vs annual
test_combos = [
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'monthly'},
        'selection_func_name': 'select_most_recent_launch',
        'description': 'Monthly rebalance'
    },
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_most_recent_launch',
        'description': 'Quarterly rebalance'
    },
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'semi_annual'},
        'selection_func_name': 'select_most_recent_launch',
        'description': 'Semi-annual rebalance'
    },
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'annual'},
        'selection_func_name': 'select_most_recent_launch',
        'description': 'Annual rebalance'
    }
]
test_months = ['MAR']
# Total: 4 simulations (one per cadence)






# =========================================================================
# Time-Based Triggers (use cadence)

# Example 2: Test One Cadence with Different Selections
# =========================================================================
# Quarterly rebalance with different selection strategies
test_combos = [
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_most_recent_launch',
        'description': 'Quarterly → most recent'
    },
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_remaining_cap_highest',
        'description': 'Quarterly → highest cap (bullish)'
    },
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_remaining_cap_lowest',
        'description': 'Quarterly → lowest cap (bearish)'
    },
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_cost_analysis',
        'description': 'Quarterly → cost optimized'
    }
]
test_months = ['MAR']
# Total: 4 simulations




# =========================================================================
# MIX
# =========================================================================

test_combos = [
    # Time-based with cadence
    {
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},  # Has cadence
        'selection_func_name': 'select_most_recent_launch',
        'description': 'Quarterly time-based'
    },

    # Threshold-based (no cadence - checks daily)
    {
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.50},  # No cadence
        'selection_func_name': 'select_most_recent_launch',
        'description': 'Cap threshold (daily check)'
    }
]
test_months = ['MAR']
# Total: 2 simulations