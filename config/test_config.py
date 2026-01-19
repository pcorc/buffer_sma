"""
Test configurations for debugging and validation.

Use these minimal configs to test the framework before running full simulations.
"""

# =============================================================================
# MINIMAL TEST CONFIGURATIONS
# =============================================================================

# Test with just 1 launch month
TEST_LAUNCH_MONTHS_MINIMAL = ['MAR']

# Test with 2 launch months (quarterly pattern)
TEST_LAUNCH_MONTHS_SMALL = ['MAR', 'SEP']

# Test with 4 launch months (full year quarterly)
TEST_LAUNCH_MONTHS_QUARTERLY = ['MAR', 'JUN', 'SEP', 'DEC']


# =============================================================================
# STRATEGY SUBSET FILTERS
# =============================================================================

def filter_by_intent(combo_configs, intent='bullish'):
    """
    Filter configurations to only one strategy intent.

    Parameters:
        combo_configs: List of combo config dicts
        intent: 'bullish', 'bearish', 'neutral', or 'cost_optimized'

    Returns:
        Filtered list of configs
    """
    from config.strategy_intents import get_strategy_intent

    filtered = []
    for combo in combo_configs:
        try:
            combo_intent = get_strategy_intent(
                combo['trigger_type'],
                combo['trigger_params'],
                combo['selection_func_name']
            )
            if combo_intent == intent:
                filtered.append(combo)
        except:
            pass

    return filtered


def filter_by_trigger_type(combo_configs, trigger_type='rebalance_time_period'):
    """
    Filter configurations to only one trigger type.

    Parameters:
        combo_configs: List of combo config dicts
        trigger_type: Type of trigger to filter for

    Returns:
        Filtered list of configs
    """
    return [c for c in combo_configs if c['trigger_type'] == trigger_type]


def filter_by_selection(combo_configs, selection_func='select_most_recent_launch'):
    """
    Filter configurations to only one selection function.

    Parameters:
        combo_configs: List of combo config dicts
        selection_func: Selection function name to filter for

    Returns:
        Filtered list of configs
    """
    return [c for c in combo_configs if c['selection_func_name'] == selection_func]


def get_one_per_intent(combo_configs):
    """
    Get exactly one strategy from each intent category.

    Returns:
        List with 4 configs (one per intent)
    """
    from config.strategy_intents import get_strategy_intent

    intent_samples = {}

    for combo in combo_configs:
        try:
            intent = get_strategy_intent(
                combo['trigger_type'],
                combo['trigger_params'],
                combo['selection_func_name']
            )
            if intent not in intent_samples:
                intent_samples[intent] = combo

            if len(intent_samples) == 4:
                break
        except:
            pass

    return list(intent_samples.values())


def get_minimal_test_set():
    """
    Get absolute minimal test set: 1 strategy + 1 launch month = 1 simulation.

    Returns:
        Tuple of (combo_configs_list, launch_months_list)
    """
    minimal_combo = [{
        'trigger_type': 'rebalance_time_period',
        'trigger_params': {'frequency': 'quarterly'},
        'selection_func_name': 'select_most_recent_launch',
        'group': 'TEST_MINIMAL',
        'description': 'Minimal test: Quarterly rebalance to most recent'
    }]

    minimal_months = ['MAR']

    return minimal_combo, minimal_months


def get_small_test_set():
    """
    Get small test set: 4 strategies (one per intent) + 2 launch months = 8 simulations.

    Returns:
        Tuple of (combo_configs_list, launch_months_list)
    """
    small_combos = [
        {
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': 'select_most_recent_launch',
            'description': 'Neutral: Quarterly + most recent'
        },
        {
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': 'select_remaining_cap_highest',
            'description': 'Bullish: Quarterly + highest cap'
        },
        {
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': 'select_remaining_cap_lowest',
            'description': 'Bearish: Quarterly + lowest cap'
        },
        {
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': 'select_cost_analysis',
            'description': 'Cost-optimized: Quarterly + cost analysis'
        }
    ]

    small_months = ['MAR', 'SEP']

    return small_combos, small_months


def get_medium_test_set():
    """
    Get medium test set: All time-based strategies + 4 launch months = ~112 simulations.

    Returns:
        Tuple of (combo_configs_list, launch_months_list)
    """
    # Import full config
    from config.settings import COMBO_CONFIGS

    # Filter to only time-based rebalancing (no thresholds)
    medium_combos = filter_by_trigger_type(COMBO_CONFIGS, 'rebalance_time_period')

    medium_months = ['MAR', 'JUN', 'SEP', 'DEC']

    return medium_combos, medium_months


# =============================================================================
# QUICK TEST FUNCTIONS
# =============================================================================

def print_test_set_info(combos, months, set_name="Test Set"):
    """Print information about a test configuration."""
    print("\n" + "=" * 80)
    print(f"{set_name.upper()}")
    print("=" * 80)
    print(f"Combinations: {len(combos)}")
    print(f"Launch months: {len(months)} - {months}")
    print(f"Total simulations: {len(combos) * len(months)}")

    if combos:
        print("\nStrategies:")
        for i, combo in enumerate(combos[:5], 1):  # Show first 5
            desc = combo.get('description', f"{combo['trigger_type']} + {combo['selection_func_name']}")
            print(f"  {i}. {desc}")

        if len(combos) > 5:
            print(f"  ... and {len(combos) - 5} more")

    print("=" * 80 + "\n")