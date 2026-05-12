"""
Batch Configuration Module

Defines all batch testing configurations for buffer ETF rotation strategies.
Each batch tests specific strategy combinations across different market conditions.

Usage:
    from config.batch_configs import BATCH_CONFIGS, BATCH_DESCRIPTIONS
    
    configs = BATCH_CONFIGS[batch_number]()
"""

# ============================================================================
# BATCH 0: Quick Testing
# ============================================================================

def get_batch_0_configs():
    """
    BATCH 0: Remaining Buffer Threshold Testing
    Tests: 3 thresholds × 1 month = 3 simulations
    Estimated time: ~1 minute
    """
    configs = []
    threshold_levels = [0.15, 0.50, 0.85]
    months = ['SEP']

    for threshold in threshold_levels:
        configs.append({
            'trigger_type': 'remaining_buffer_threshold',
            'trigger_params': {'threshold': threshold},
            'selection_func_name': 'select_downside_buffer_lowest',
            'launch_months': months,
        })

    return configs


# ============================================================================
# BATCH 1-4: Traditional Strategy Testing
# ============================================================================

def get_batch_1_configs():
    """
    BATCH 1: Time-Based Systematic Rebalancing
    ~18 simulations, ~5 minutes
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
        for selection in bullish_selections:
            configs.append({
                'trigger_type': 'rebalance_time_period',
                'trigger_params': {'frequency': freq},
                'selection_func_name': selection,
                'launch_months': months
            })

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
    ~18 simulations, ~5 minutes
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
        for selection in bullish_selections:
            configs.append({
                'trigger_type': 'cap_utilization_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months
            })

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
    ~18 simulations, ~5 minutes
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
        for selection in bullish_selections:
            configs.append({
                'trigger_type': 'remaining_cap_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months
            })

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
    ~30 simulations, ~10 minutes
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
        for selection in bullish_selections:
            configs.append({
                'trigger_type': 'ref_asset_return_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months
            })

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
        for selection in bullish_selections:
            configs.append({
                'trigger_type': 'downside_before_buffer_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months
            })

        for selection in bearish_selections:
            configs.append({
                'trigger_type': 'downside_before_buffer_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'launch_months': months
            })

    return configs


# ============================================================================
# BATCH 5: Comprehensive Regime Testing
# ============================================================================


def get_batch_5_configs():
    """
    BATCH 5: Comprehensive Regime-Optimized Testing
    ~60 simulations, ~20 minutes
    """
    configs = []
    months = ['JAN', 'MAR', 'MAY', 'JUL', 'SEP', 'NOV']

    # Bullish strategies
    bullish_selections = ['select_cap_utilization_lowest', 'select_remaining_cap_highest']
    
    for selection in bullish_selections:
        configs.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': selection,
            'strategy_intent': 'bullish',
            'launch_months': months
        })

    for threshold in [0.25, 0.50, 0.75]:
        for selection in bullish_selections:
            configs.append({
                'trigger_type': 'cap_utilization_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'strategy_intent': 'bullish',
                'launch_months': months
            })

    # Bearish strategies
    bearish_selections = ['select_downside_buffer_highest', 'select_cap_utilization_lowest']
    
    for selection in bearish_selections:
        configs.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': selection,
            'strategy_intent': 'bearish',
            'launch_months': months
        })

    for threshold in [-0.07, -0.05, -0.03]:
        for selection in bearish_selections:
            configs.append({
                'trigger_type': 'downside_before_buffer_threshold',
                'trigger_params': {'threshold': threshold},
                'selection_func_name': selection,
                'strategy_intent': 'bearish',
                'launch_months': months
            })

    # Neutral strategies
    neutral_selections = ['select_remaining_cap_highest', 'select_downside_buffer_highest']
    
    for selection in neutral_selections:
        configs.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': selection,
            'strategy_intent': 'neutral',
            'launch_months': months
        })

    return configs


# ============================================================================
# BATCH 6: Enhanced Cost Ratio (ECR) Testing
# ============================================================================

def get_batch_6_configs():
    """
    BATCH 6: ECR V2 Weight Sensitivity + Existing 90% Comparison
    Tests: 5 ECR variants + 1 Existing × 4 months = 24 simulations
    Estimated time: ~7 minutes
    """
    configs = []

    launch_months = [
        ['JAN'], ['FEB'], ['MAR'], ['APR']
    ]

    ecr_variants = [
        ('select_ecr_v2_equal_normalized', 'Equal (0.33/0.33/0.33)'),
        ('select_ecr_v2_cap_balanced_normalized', 'Cap-Balanced (0.15/0.35/0.50)'),
        ('select_ecr_v2_cap_moderate_normalized', 'Cap-Moderate (0.25/0.25/0.50)'),
        ('select_ecr_v2_cap_dominant_normalized', 'Cap-Dominant (0.15/0.15/0.70)'),
        ('select_ecr_v2_protection_normalized', 'Protection (0.40/0.40/0.20)'),
    ]

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

        # Existing 90% threshold baseline
        configs.append({
            'trigger_type': 'cap_utilization_threshold',
            'trigger_params': {'threshold': 0.90},
            'selection_func_name': 'select_most_recent_launch',
            'launch_months': month_list,
            'strategy_intent': 'bullish',
            'description': f'{month}: Existing 90% Cap Threshold'
        })

    print(f"\n{'=' * 80}")
    print(f"BATCH 6: WEIGHT SENSITIVITY ANALYSIS")
    print(f"{'=' * 80}")
    print(f"ECR variants: {len(ecr_variants)}")
    print(f"Launch months: {len(launch_months)}")
    print(f"Total simulations: {len(configs)}")
    print(f"Estimated time: ~{len(configs) * 0.3:.0f} minutes")
    print(f"{'=' * 80}\n")

    return configs


def get_batch_8_configs():
    """
    BATCH 8: Comprehensive Threshold Analysis
    Tests: 1 threshold × 1 month = 1 simulation
    """
    configs = []
    threshold_levels = [0.90]
    months = ['SEP']

    for threshold in threshold_levels:
        configs.append({
            'trigger_type': 'cap_utilization_threshold',
            'trigger_params': {'threshold': threshold},
            'selection_func_name': 'select_most_recent_launch',
            'launch_months': months
        })

    return configs


def get_batch_10b_configs():
    """
    BATCH 10B: ECR Percentile Trigger — Reducing Turnover via Daily Spread Evaluation
    ==================================================================================
    Tests: 7 weight configs × 6 percentile thresholds = 42 strategy combinations
    Plus:  1 existing 90% baseline for comparison
    Total: 43 configs × launch months

    Hypothesis:
        Quarterly rebalancing forces trades even when the ECR spread between the
        current fund and the best available fund is negligible. By evaluating daily
        and only firing when the spread exceeds a rolling percentile threshold, we
        reduce unnecessary turnover and only rotate when the opportunity is meaningful.

    Trigger: ecr_percentile_trigger
        - Evaluates daily spread: (best_ecr - current_ecr) / best_ecr
        - Fires when spread >= Nth percentile of rolling 252-day spread history
        - Minimum holding period of 21 days enforced before trigger can fire

    Weight configs tested (same as Batch 10):
        111 = Equal weights         (baseline)
        211 = DBB heavy             (bearish)
        121 = Buffer heavy          (defensive)
        112 = Cap heavy             (bullish)
        101 = DBB + Cap only        (mixed)
        221 = Buffer equal          (most defensive)
        110 = Buffer only           (pure defense)

    Percentile thresholds tested:
        50th = fires when spread is above median    (most active)
        60th = moderate-high spread required
        70th = high spread required
        75th = upper quartile spread required
        80th = top 20% spread required
        90th = top 10% spread only  (most selective)

    First pass: min_holding_days=21, rolling_window=252
    Expand min_holding_days in Batch 10C once optimal percentile range identified.
    """

    weight_configs = ['111', '211', '121', '112', '101', '221', '110']
    weight_configs = ['111','211', '121', '112',]
    percentile_thresholds = [ 0.80, 0.90]
    min_holding_days = 22
    rolling_window = 252

    launch_months = ['JAN', 'MAR', 'MAY', 'JUL', 'SEP', 'NOV', 'DEC', 'FEB']

    configs = []

    # -------------------------------------------------------------------------
    # ECR Percentile configs — 7 weights × 6 thresholds = 42 strategies
    # -------------------------------------------------------------------------
    for weight_code in weight_configs:
        w_dbb    = int(weight_code[0])
        w_buffer = int(weight_code[1])
        w_cap    = int(weight_code[2])

        for pct in percentile_thresholds:
            pct_label = int(pct * 100)
            configs.append({
                'trigger_type': 'ecr_percentile_trigger',
                'trigger_params': {
                    'weight_code': weight_code,
                    'w_dbb': w_dbb,
                    'w_buffer': w_buffer,
                    'w_cap': w_cap,
                    'percentile_threshold': pct,
                    'min_holding_days': min_holding_days,
                    'rolling_window': rolling_window,
                },
                'selection_func_name': f'select_highest_new_ecr_composite_{weight_code}',
                'selection_params': {},
                'launch_months': launch_months,
                'strategy_name': f'ECR_Pct_w{weight_code}_p{pct_label}_hold{min_holding_days}d',
                'strategy_intent': 'neutral',
            })

    # -------------------------------------------------------------------------
    # Existing 90% baseline — included for direct comparison
    # -------------------------------------------------------------------------
    configs.append({
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.90},
        'selection_func_name': 'select_most_recent_launch',
        'selection_params': {},
        'launch_months': launch_months,
        'strategy_name': 'Cap_Util_90% → Most Recent (Baseline)',
        'strategy_intent': 'neutral',
        'min_absolute_spread': 0.05,  # ← ADD: 5% minimum spread
    })

    return configs

def get_batch_10c_configs():
    """
    BATCH 10C: ECR Score Threshold Trigger
    =======================================
    Tests: 7 weight configs × 6 score thresholds = 42 strategies
    Plus:  1 existing 90% baseline

    Hypothesis:
        Instead of percentile-based firing, switch when the current fund's
        ECR score (without time scaling) drops below an absolute threshold.
        Scores without time scaling cluster between 2.47-2.97 for selected funds.
        Thresholds test where the optimal floor is.

    Score thresholds tested:
        2.50 = very selective, only fire on deep deterioration
        2.55
        2.60
        2.65
        2.70
        2.75 = most active, fires early on modest deterioration

    Min holding: 63 days (one quarter floor)
    """
    weight_configs = ['111']#'112', '101', '221', '110']
    score_thresholds = [1.5,]
    min_holding_days = 22
    launch_months = ['JAN', 'MAR', 'MAY', 'JUL', 'SEP', 'NOV', 'DEC', 'FEB', 'JUN', 'OCT', 'APR', 'AUG'] #

    configs = []

    for weight_code in weight_configs:
        w_dbb    = int(weight_code[0])
        w_buffer = int(weight_code[1])
        w_cap    = int(weight_code[2])

        for thresh in score_thresholds:
            thresh_label = str(thresh).replace('.', '_')
            configs.append({
                'trigger_type': 'ecr_score_threshold',
                'trigger_params': {
                    'weight_code': weight_code,
                    'score_threshold': thresh,
                    'min_holding_days': min_holding_days,
                },
                'selection_func_name': f'select_highest_new_ecr_composite_{weight_code}',
                'selection_params': {},
                'launch_months': launch_months,
                'strategy_name': f'ECR_Score_w{weight_code}_t{thresh_label}_hold{min_holding_days}d',
                'strategy_intent': 'neutral',
            })

    # Baseline
    configs.append({
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.90},
        'selection_func_name': 'select_most_recent_launch',
        'selection_params': {},
        'launch_months': launch_months,
        'strategy_name': 'Util_90% → Most Recent',
        'strategy_intent': 'neutral',
    })

    return configs


def get_batch_10d_configs():
    """
    BATCH 10D: ECR Percentile Rank Trigger
    =======================================
    Tests: weight configs × percentile thresholds

    Hypothesis:
        Instead of an absolute score floor (Batch 10c), fire when the current
        fund's ECR score falls below a given percentile rank within the live
        universe on that date.

        Example: 50th percentile fires when the current fund is below the median
        of all 12 funds available today. This is self-calibrating — it adjusts
        to the current score environment rather than relying on a fixed number.

    Percentile thresholds tested:
        25 = bottom quartile  — very patient, only rotate when fund is deeply ranked
        33 = bottom third
        50 = below median     — moderate activity
        67 = below top third  — more active

    Min holding: 22 days (one month floor)
    """
    weight_configs        = ['111']
    percentile_thresholds = [25, 50]
    min_holding_days      = 22
    launch_months         = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                              'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    configs = []

    for weight_code in weight_configs:
        for pct in percentile_thresholds:
            configs.append({
                'trigger_type': 'ecr_percentile_rank',
                'trigger_params': {
                    'weight_code':          weight_code,
                    'percentile_threshold': pct,
                    'min_holding_days':     min_holding_days,
                },
                'selection_func_name': f'select_highest_new_ecr_composite_{weight_code}',
                'selection_params':    {},
                'launch_months':       launch_months,
                'strategy_name':       f'ECR_PctRank_w{weight_code}_p{pct}_hold{min_holding_days}d',
                'strategy_intent':     'neutral',
            })

    # Baseline
    configs.append({
        'trigger_type':        'cap_utilization_threshold',
        'trigger_params':      {'threshold': 0.90},
        'selection_func_name': 'select_most_recent_launch',
        'selection_params':    {},
        'launch_months':       launch_months,
        'strategy_name':       'Util_90% → Most Recent',
        'strategy_intent':     'neutral',
    })

    return configs

def get_batch_11_configs():
    """
    BATCH 11: Par Proximity ECR Selection
    ======================================
    Tests: 12 par_prox variants + 1 baseline = 13 strategies

    Hypothesis:
        The v1 ECR composite has a blind spot for funds whose structure has
        reset to near-par mid-life. These funds have intact buffer/cap and
        relatively low time remaining — the buffer/cap will be realized into
        NAV the fastest. We add a par_proximity bonus that fires only when
        SPY is near the prev roll level AND meaningful time has elapsed.

    Trigger: ecr_score_threshold
        - Fires when current fund's v1 ECR score drops below 1.5 (absolute floor)
        - Trigger uses v1 ECR for the score floor (Option A — trigger/selection decoupled)
        - Selection uses v3 par_prox composite to pick the replacement fund

    Selection variants:
        12 par_prox configs (w_par × time_shape):
            w_par in   {0.5, 1.0, 1.5, 2.0}
            time_shape in {linear, sqrt, sq}
        + 1 baseline (v1 ECR, equal weights 1,1,1)

    Composite formula (par_prox selection):
        composite = dbb_score + buffer_integrity + cap_integrity
                  + w_par * par_proximity * time_elapsed
        (No time_scaling division — that role moves to time_elapsed in par bonus.)
    """

    launch_months = ['FEB', 'JUN', 'SEP', ]
    launch_months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                     'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    min_holding_days = 22
    score_threshold = 1.5

    common_trigger = {
        'trigger_type': 'ecr_score_threshold',
        'trigger_params': {
            'weight_code': '111',
            'score_threshold': score_threshold,
            'min_holding_days': min_holding_days,
        },
    }

    score_label = str(score_threshold).replace('.', 'p')  # 1.5 → '1p5'

    configs = []

    # --- Baseline: v1 ECR with equal weights (no par bonus) ---
    configs.append({
        **common_trigger,
        'selection_func_name': 'select_highest_new_ecr_composite_111',
        'selection_params': {},
        'launch_months': launch_months,
        'strategy_name': f'Baseline_ECRv1_w111_score{score_label}',
        'strategy_intent': 'neutral',
    })

    # --- 12 par_prox variants ---
    w_par_values = [(0.5, '0p5'), (1.0, '1'), (1.5, '1p5'), (2.0, '2')]
    time_shapes = ['lin', 'sqrt', 'sq']

    for w_par_val, w_par_label in w_par_values:
        for shape in time_shapes:
            configs.append({
                **common_trigger,
                'selection_func_name': f'select_ecr_par_prox_w{w_par_label}_{shape}',
                'selection_params': {},
                'launch_months': launch_months,
                'strategy_name': f'ParProx_w{w_par_label}_{shape}_score{score_label}',
                'strategy_intent': 'neutral',
            })

    return configs


def get_batch_11_w2p5_configs():
    """
    BATCH 11 — w_par = 2.5 only (incremental run)
    ==============================================
    Runs only the 3 new par_prox configs for w_par = 2.5 across all 12 launch months.
    Designed to be merged manually with prior Batch 11 results (3 months × 13 configs).

    Strategies tested:
        select_ecr_par_prox_w2p5_lin
        select_ecr_par_prox_w2p5_sqrt
        select_ecr_par_prox_w2p5_sq

    Trigger: ecr_score_threshold (score_threshold=1.5, min_holding_days=22)
    Total simulations: 3 configs × 12 launch months = 36

    NOTE: No baseline included — already covered by prior Batch 11 run.
    """

    launch_months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                     'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    min_holding_days = 22
    score_threshold = 1.5

    common_trigger = {
        'trigger_type': 'ecr_score_threshold',
        'trigger_params': {
            'weight_code': '111',
            'score_threshold': score_threshold,
            'min_holding_days': min_holding_days,
        },
    }

    score_label = str(score_threshold).replace('.', 'p')

    configs = []

    # --- 3 par_prox variants for w_par = 2.5 ---
    w_par_label = '2p5'
    time_shapes = ['lin', 'sqrt', 'sq']

    for shape in time_shapes:
        configs.append({
            **common_trigger,
            'selection_func_name': f'select_ecr_par_prox_w{w_par_label}_{shape}',
            'selection_params': {},
            'launch_months': launch_months,
            'strategy_name': f'ParProx_w{w_par_label}_{shape}_score{score_label}',
            'strategy_intent': 'neutral',
        })

    return configs

BATCH_CONFIGS = {
    0: get_batch_0_configs,
    1: get_batch_1_configs,
    2: get_batch_2_configs,
    3: get_batch_3_configs,
    4: get_batch_4_configs,
    5: get_batch_5_configs,
    6: get_batch_6_configs,
    8: get_batch_8_configs,
    '10b': get_batch_10b_configs,
    '10c': get_batch_10c_configs,
    '10d': get_batch_10d_configs,
    '11': get_batch_11_configs,
    '11w2p5': get_batch_11_w2p5_configs,  # ← NEW

}

BATCH_DESCRIPTIONS = {
    0: "Quick Test - Remaining Buffer Thresholds",
    1: "Time-Based Systematic (Bullish vs Bearish)",
    2: "Cap Utilization Tactical",
    3: "Remaining Cap Tactical",
    4: "Market-Responsive (Ref Asset + Buffer)",
    5: "Comprehensive Regime-Optimized",
    6: "ECR V2 vs Existing (90% Threshold) Comparison",
    7: "Regime-Adaptive ECR vs Fixed Weights",
    8: "Schultz - 90% Threshold Analysis",
    9: "Fixed-Weight ECR Performance by Regime",
    '10b': 'ECR Percentile Trigger - Turnover Reduction',
    '10c': 'ECR Score Threshold Trigger - Absolute Floor',
    '10d': 'ECR Percentile Rank Trigger - Universe Relative',
    '11': 'Par Proximity - Mid-Life Near-Par Bonus',
    '11w2p5': 'Par Proximity w_par=2.5 incremental',  # ← NEW
}
