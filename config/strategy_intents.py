"""
Strategy intent classification for all trigger/selection combinations.

This module provides explicit mapping of each strategy combination to its
intended market behavior (bullish/bearish/neutral/cost_optimized).

Intent Definitions:
- BULLISH: Strategies designed to maximize upside capture in rising markets
- BEARISH: Strategies designed to preserve capital and minimize drawdowns
- NEUTRAL: Balanced strategies with no directional bias
- COST_OPTIMIZED: Strategies focused on efficiency and risk-adjusted returns
"""

from typing import Dict, Tuple

# =============================================================================
# INTENT DESCRIPTIONS
# =============================================================================

INTENT_DESCRIPTIONS = {
    'bullish': 'Strategies designed to maximize upside capture in rising markets',
    'bearish': 'Strategies designed to preserve capital and minimize drawdowns',
    'neutral': 'Balanced strategies with no directional bias',
    'cost_optimized': 'Strategies focused on efficiency and risk-adjusted returns'
}


# =============================================================================
# EXPLICIT INTENT MAPPING
# =============================================================================

def build_intent_map() -> Dict[Tuple[str, str], str]:
    """
    Build explicit mapping of (trigger_type, selection_algo) -> intent.

    This function creates all 68 combinations and assigns intent based on
    the strategic logic of each trigger/selection pair.

    Returns:
        Dict with keys as (trigger_type, selection_algo) tuples,
        values as intent strings ('bullish', 'bearish', 'neutral', 'cost_optimized')
    """
    intent_map = {}

    # =========================================================================
    # GROUP 1: Time-Based Rebalancing + Most Recent Launch
    # Intent: NEUTRAL - Mechanical rebalancing with no directional bias
    # =========================================================================

    for freq in ['monthly', 'quarterly', 'semi_annual', 'annual']:
        key = (f'rebalance_time_period|{freq}', 'select_most_recent_launch')
        intent_map[key] = 'neutral'

    # =========================================================================
    # GROUP 2: Time-Based + Remaining Cap Selection
    # =========================================================================

    # GROUP 2A: BULLISH - Highest cap = maximum upside potential
    for freq in ['monthly', 'quarterly', 'semi_annual', 'annual']:
        key = (f'rebalance_time_period|{freq}', 'select_remaining_cap_highest')
        intent_map[key] = 'bullish'

    # GROUP 2B: BEARISH - Lowest cap = conservative positioning
    for freq in ['monthly', 'quarterly', 'semi_annual', 'annual']:
        key = (f'rebalance_time_period|{freq}', 'select_remaining_cap_lowest')
        intent_map[key] = 'bearish'

    # =========================================================================
    # GROUP 3: Time-Based + Downside Before Buffer Selection
    # =========================================================================

    # GROUP 3A: BULLISH - Highest downside = most room to fall (aggressive)
    for freq in ['monthly', 'quarterly', 'semi_annual', 'annual']:
        key = (f'rebalance_time_period|{freq}', 'select_downside_buffer_highest')
        intent_map[key] = 'bullish'

    # GROUP 3B: BEARISH - Lowest downside = in/near buffer (defensive)
    for freq in ['monthly', 'quarterly', 'semi_annual', 'annual']:
        key = (f'rebalance_time_period|{freq}', 'select_downside_buffer_lowest')
        intent_map[key] = 'bearish'

    # =========================================================================
    # GROUP 4: Time-Based + Cap Utilization Selection
    # =========================================================================

    # GROUP 4A: BULLISH - Lowest utilization = most cap remaining
    for freq in ['monthly', 'quarterly', 'semi_annual', 'annual']:
        key = (f'rebalance_time_period|{freq}', 'select_cap_utilization_lowest')
        intent_map[key] = 'bullish'

    # GROUP 4B: BEARISH - Highest utilization = nearly capped out
    for freq in ['monthly', 'quarterly', 'semi_annual', 'annual']:
        key = (f'rebalance_time_period|{freq}', 'select_cap_utilization_highest')
        intent_map[key] = 'bearish'

    # =========================================================================
    # GROUP 5: Remaining Cap Threshold + Most Recent Launch
    # Intent: BULLISH - Switch when cap depletes, seeking fresh upside
    # =========================================================================

    for threshold in [0.15, 0.50, 0.85]:
        key = (f'remaining_cap_threshold|{threshold}', 'select_most_recent_launch')
        intent_map[key] = 'bullish'

    # =========================================================================
    # GROUP 6: Remaining Cap Threshold + Ranked Cap Selection
    # =========================================================================

    # GROUP 6A: BULLISH² - Cap threshold + highest cap (double bullish)
    for threshold in [0.15, 0.50, 0.85]:
        key = (f'remaining_cap_threshold|{threshold}', 'select_remaining_cap_highest')
        intent_map[key] = 'bullish'

    # GROUP 6B: NEUTRAL - Cap threshold + lowest cap (conflicting signals)
    for threshold in [0.15, 0.50, 0.85]:
        key = (f'remaining_cap_threshold|{threshold}', 'select_remaining_cap_lowest')
        intent_map[key] = 'neutral'

    # =========================================================================
    # GROUP 7: Cap Utilization Threshold + Most Recent Launch
    # Intent: BULLISH - Switch when cap consumed, seeking fresh upside
    # =========================================================================

    for threshold in [0.15, 0.50, 0.85]:
        key = (f'cap_utilization_threshold|{threshold}', 'select_most_recent_launch')
        intent_map[key] = 'bullish'

    # =========================================================================
    # GROUP 8: Cap Utilization Threshold + Ranked Utilization Selection
    # =========================================================================

    # GROUP 8A: BULLISH² - Utilization threshold + lowest utilization
    for threshold in [0.15, 0.50, 0.85]:
        key = (f'cap_utilization_threshold|{threshold}', 'select_cap_utilization_lowest')
        intent_map[key] = 'bullish'

    # GROUP 8B: NEUTRAL - Utilization threshold + highest utilization
    for threshold in [0.15, 0.50, 0.85]:
        key = (f'cap_utilization_threshold|{threshold}', 'select_cap_utilization_highest')
        intent_map[key] = 'neutral'

    # =========================================================================
    # GROUP 9: Downside Before Buffer Threshold + Most Recent Launch
    # Intent: BEARISH - Switch when approaching buffer (defensive)
    # =========================================================================

    for threshold in [0.15, 0.50, 0.85]:
        key = (f'downside_before_buffer_threshold|{threshold}', 'select_most_recent_launch')
        intent_map[key] = 'bearish'

    # =========================================================================
    # GROUP 10: Downside Threshold + Ranked Downside Selection
    # =========================================================================

    # GROUP 10A: NEUTRAL - Downside threshold + highest downside
    for threshold in [0.15, 0.50, 0.85]:
        key = (f'downside_before_buffer_threshold|{threshold}', 'select_downside_buffer_highest')
        intent_map[key] = 'neutral'

    # GROUP 10B: BEARISH² - Downside threshold + lowest downside (double bearish)
    for threshold in [0.15, 0.50, 0.85]:
        key = (f'downside_before_buffer_threshold|{threshold}', 'select_downside_buffer_lowest')
        intent_map[key] = 'bearish'

    # =========================================================================
    # GROUP 11: Cost Analysis (Time-Based)
    # Intent: COST_OPTIMIZED - Efficiency focus
    # =========================================================================

    for freq in ['monthly', 'quarterly', 'semi_annual', 'annual']:
        key = (f'rebalance_time_period|{freq}', 'select_cost_analysis')
        intent_map[key] = 'cost_optimized'

    # =========================================================================
    # GROUP 12: Cost Analysis (Threshold-Based)
    # Intent: COST_OPTIMIZED - Efficiency with tactical triggers
    # =========================================================================

    # 12A: Cap threshold + cost analysis
    for threshold in [0.15, 0.50, 0.85]:
        key = (f'remaining_cap_threshold|{threshold}', 'select_cost_analysis')
        intent_map[key] = 'cost_optimized'

    # 12B: Utilization threshold + cost analysis
    for threshold in [0.15, 0.50, 0.85]:
        key = (f'cap_utilization_threshold|{threshold}', 'select_cost_analysis')
        intent_map[key] = 'cost_optimized'

    # 12C: Downside threshold + cost analysis
    for threshold in [0.15, 0.50, 0.85]:
        key = (f'downside_before_buffer_threshold|{threshold}', 'select_cost_analysis')
        intent_map[key] = 'cost_optimized'

    return intent_map


# Build the intent map once at module load
STRATEGY_INTENT_MAP = build_intent_map()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_strategy_intent(trigger_type: str, trigger_params: dict, selection_algo: str) -> str:
    """
    Get the intent classification for a given strategy combination.

    Parameters:
        trigger_type: Type of trigger (e.g., 'rebalance_time_period')
        trigger_params: Dict of trigger parameters (e.g., {'frequency': 'quarterly'})
        selection_algo: Selection algorithm name (e.g., 'select_most_recent_launch')

    Returns:
        Intent string: 'bullish', 'bearish', 'neutral', or 'cost_optimized'

    Raises:
        KeyError: If combination is not found in intent map
    """
    # Build composite key
    if trigger_type == 'rebalance_time_period':
        freq = trigger_params.get('frequency', 'quarterly')
        trigger_key = f'{trigger_type}|{freq}'
    elif 'threshold' in trigger_params:
        threshold = trigger_params['threshold']
        trigger_key = f'{trigger_type}|{threshold}'
    else:
        trigger_key = trigger_type

    key = (trigger_key, selection_algo)

    if key not in STRATEGY_INTENT_MAP:
        raise KeyError(
            f"Strategy combination not found in intent map: {key}\n"
            f"Trigger: {trigger_type}, Params: {trigger_params}, Selection: {selection_algo}"
        )

    return STRATEGY_INTENT_MAP[key]


def get_intent_description(intent: str) -> str:
    """
    Get the description for a given intent.

    Parameters:
        intent: Intent string ('bullish', 'bearish', 'neutral', 'cost_optimized')

    Returns:
        Description string
    """
    return INTENT_DESCRIPTIONS.get(intent, 'Unknown intent')


def get_strategies_by_intent(intent: str) -> list:
    """
    Get all strategy combinations for a given intent.

    Parameters:
        intent: Intent string to filter by

    Returns:
        List of (trigger_key, selection_algo) tuples matching the intent
    """
    return [
        key for key, val in STRATEGY_INTENT_MAP.items()
        if val == intent
    ]


def print_intent_summary():
    """
    Print summary of intent distribution across all strategies.
    """
    from collections import Counter

    print("\n" + "=" * 80)
    print("STRATEGY INTENT DISTRIBUTION")
    print("=" * 80)

    intent_counts = Counter(STRATEGY_INTENT_MAP.values())
    total = len(STRATEGY_INTENT_MAP)

    print(f"\nTotal strategy combinations: {total}")
    print("\n" + "-" * 80)

    for intent in ['bullish', 'bearish', 'neutral', 'cost_optimized']:
        count = intent_counts[intent]
        pct = (count / total) * 100
        print(f"{intent.upper():15s}: {count:2d} strategies ({pct:5.1f}%) × 12 months = {count * 12:3d} sims")
        print(f"                {INTENT_DESCRIPTIONS[intent]}")
        print()

    print("-" * 80)
    print(f"{'TOTAL':15s}: {total:2d} strategies (100.0%) × 12 months = {total * 12:3d} sims")
    print("=" * 80 + "\n")


def print_intent_breakdown_by_group():
    """
    Print detailed breakdown showing which groups map to which intents.
    """
    print("\n" + "=" * 80)
    print("INTENT BREAKDOWN BY STRATEGY GROUP")
    print("=" * 80)

    groups = {
        'GROUP 1: Time + Most Recent': [
            ('rebalance_time_period|monthly', 'select_most_recent_launch'),
            ('rebalance_time_period|quarterly', 'select_most_recent_launch'),
        ],
        'GROUP 2A: Time + Highest Cap': [
            ('rebalance_time_period|monthly', 'select_remaining_cap_highest'),
            ('rebalance_time_period|quarterly', 'select_remaining_cap_highest'),
        ],
        'GROUP 2B: Time + Lowest Cap': [
            ('rebalance_time_period|monthly', 'select_remaining_cap_lowest'),
            ('rebalance_time_period|quarterly', 'select_remaining_cap_lowest'),
        ],
        'GROUP 3A: Time + Highest Downside': [
            ('rebalance_time_period|monthly', 'select_downside_buffer_highest'),
        ],
        'GROUP 3B: Time + Lowest Downside': [
            ('rebalance_time_period|monthly', 'select_downside_buffer_lowest'),
        ],
        'GROUP 4A: Time + Lowest Utilization': [
            ('rebalance_time_period|monthly', 'select_cap_utilization_lowest'),
        ],
        'GROUP 4B: Time + Highest Utilization': [
            ('rebalance_time_period|monthly', 'select_cap_utilization_highest'),
        ],
        'GROUP 5: Cap Threshold + Most Recent': [
            ('remaining_cap_threshold|0.15', 'select_most_recent_launch'),
            ('remaining_cap_threshold|0.5', 'select_most_recent_launch'),
        ],
        'GROUP 6A: Cap Threshold + Highest Cap': [
            ('remaining_cap_threshold|0.15', 'select_remaining_cap_highest'),
        ],
        'GROUP 6B: Cap Threshold + Lowest Cap': [
            ('remaining_cap_threshold|0.15', 'select_remaining_cap_lowest'),
        ],
        'GROUP 7: Utilization Threshold + Most Recent': [
            ('cap_utilization_threshold|0.15', 'select_most_recent_launch'),
        ],
        'GROUP 8A: Utilization Threshold + Lowest Util': [
            ('cap_utilization_threshold|0.15', 'select_cap_utilization_lowest'),
        ],
        'GROUP 8B: Utilization Threshold + Highest Util': [
            ('cap_utilization_threshold|0.15', 'select_cap_utilization_highest'),
        ],
        'GROUP 9: Downside Threshold + Most Recent': [
            ('downside_before_buffer_threshold|0.15', 'select_most_recent_launch'),
        ],
        'GROUP 10A: Downside Threshold + Highest Downside': [
            ('downside_before_buffer_threshold|0.15', 'select_downside_buffer_highest'),
        ],
        'GROUP 10B: Downside Threshold + Lowest Downside': [
            ('downside_before_buffer_threshold|0.15', 'select_downside_buffer_lowest'),
        ],
        'GROUP 11: Time + Cost Analysis': [
            ('rebalance_time_period|monthly', 'select_cost_analysis'),
        ],
        'GROUP 12A: Cap Threshold + Cost': [
            ('remaining_cap_threshold|0.15', 'select_cost_analysis'),
        ],
        'GROUP 12B: Utilization Threshold + Cost': [
            ('cap_utilization_threshold|0.15', 'select_cost_analysis'),
        ],
        'GROUP 12C: Downside Threshold + Cost': [
            ('downside_before_buffer_threshold|0.15', 'select_cost_analysis'),
        ],
    }

    for group_name, sample_keys in groups.items():
        if sample_keys:
            intent = STRATEGY_INTENT_MAP[sample_keys[0]]
            print(f"\n{group_name:50s} → {intent.upper()}")


# =============================================================================
# VALIDATION
# =============================================================================

def validate_intent_map():
    """
    Validate that all expected combinations are in the intent map.

    Returns:
        Tuple of (is_valid, missing_combinations)
    """
    expected_count = 68
    actual_count = len(STRATEGY_INTENT_MAP)

    is_valid = (actual_count == expected_count)

    if not is_valid:
        print(f"WARNING: Expected {expected_count} combinations, found {actual_count}")

    return is_valid, []

