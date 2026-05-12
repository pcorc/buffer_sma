"""
REGIME-ADAPTIVE ECR STRATEGY
=============================

Contains:
1. Regime data loading
2. Regime-adaptive selection function
3. Batch 7 configuration (3 strategies × 12 months)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler



def load_regime_data(regime_file_path='data/sp500_regimes.csv'):
    """
    Load S&P 500 regime classifications.
    
    Returns DataFrame with:
    - Date: datetime
    - Regimes: int (-1=bearish, 0=neutral, 1=bullish)
    """
    print(f"\n{'='*80}")
    print(f"Loading regime data from: {regime_file_path}")
    print(f"{'='*80}")
    
    regime_path = Path(regime_file_path)
    if not regime_path.exists():
        print(f"❌ ERROR: Regime file not found: {regime_file_path}")
        return None
    
    df_regimes = pd.read_csv(regime_file_path)
    df_regimes.columns = df_regimes.columns.str.strip()
    df_regimes['Date'] = pd.to_datetime(df_regimes['Date'])
    df_regimes = df_regimes[['Date', 'Regimes']].copy()
    df_regimes = df_regimes[df_regimes['Regimes'].notna()].copy()
    df_regimes['Regimes'] = df_regimes['Regimes'].astype(int)
    
    print(f"✅ Loaded {len(df_regimes):,} days of regime data")
    print(f"  Bearish (-1): {(df_regimes['Regimes'] == -1).sum():,} days")
    print(f"  Neutral (0):  {(df_regimes['Regimes'] == 0).sum():,} days")
    print(f"  Bullish (1):  {(df_regimes['Regimes'] == 1).sum():,} days")
    print(f"{'='*80}\n")
    
    return df_regimes


def get_current_regime(current_date, df_regimes):
    """
    Get the regime classification for a given date.
    
    Returns:
    - -1: Bearish
    - 0: Neutral
    - 1: Bullish
    - None: No regime data available
    """
    if df_regimes is None or df_regimes.empty:
        return None
    
    # Find the regime for this date
    regime_row = df_regimes[df_regimes['Date'] == current_date]
    
    if regime_row.empty:
        # If exact date not found, use most recent prior date
        prior_regimes = df_regimes[df_regimes['Date'] <= current_date]
        if prior_regimes.empty:
            return None
        regime_row = prior_regimes.iloc[-1:]
    
    return regime_row['Regimes'].iloc[0]


def select_ecr_v2_regime_adaptive_normalized(df_universe, current_date, series='F', df_regimes=None):
    """
    ECR V2 with regime-adaptive weights (NORMALIZED).
    
    Switches composite weights quarterly based on S&P 500 regime:
    - Bullish regime: 15% Cap / 15% Cost / 70% Buffer (offense)
    - Neutral regime: 25% Cap / 25% Cost / 50% Buffer (balanced)
    - Bearish regime: 40% Cap / 40% Cost / 20% Buffer (defense)
    
    Components (normalized via MinMaxScaler):
    - Cap Utilization: Higher is better (1 - cap_utilization)
    - Enhanced Cost Ratio: Lower is better (inverse)
    - Buffer Integrity: Higher buffer protection is better
    """
    if df_universe.empty:
        return None
    
    df_series = df_universe[df_universe['Fund'].str.startswith(series)].copy()
    
    if df_series.empty:
        return None
    
    # Get current regime
    current_regime = get_current_regime(current_date, df_regimes)
    
    # Select weights based on regime
    if current_regime == 1:  # Bullish
        w_cap, w_cost, w_buffer = 0.15, 0.15, 0.70
        regime_label = "Bullish"
    elif current_regime == -1:  # Bearish
        w_cap, w_cost, w_buffer = 0.40, 0.40, 0.20
        regime_label = "Bearish"
    else:  # Neutral or no regime data
        w_cap, w_cost, w_buffer = 0.25, 0.25, 0.50
        regime_label = "Neutral" if current_regime == 0 else "Neutral (no data)"
    
    # Calculate components
    required_cols = ['Cap_Utilization', 'Enhanced_Cost_Ratio', 'Buffer_Level']
    
    if not all(col in df_series.columns for col in required_cols):
        return df_series.iloc[0]['Fund']
    
    # Component 1: Cap Utilization (higher available cap is better)
    df_series['Cap_Score_Raw'] = 1 - df_series['Cap_Utilization']
    
    # Component 2: Enhanced Cost Ratio (lower cost is better, so invert)
    max_cost = df_series['Enhanced_Cost_Ratio'].max()
    if max_cost > 0:
        df_series['Cost_Score_Raw'] = max_cost - df_series['Enhanced_Cost_Ratio']
    else:
        df_series['Cost_Score_Raw'] = 0
    
    # Component 3: Buffer Integrity (higher buffer is better)
    df_series['Buffer_Score_Raw'] = df_series['Buffer_Level']
    
    # NORMALIZE using MinMaxScaler
    scaler = MinMaxScaler()
    
    raw_scores = df_series[['Cap_Score_Raw', 'Cost_Score_Raw', 'Buffer_Score_Raw']].values
    normalized_scores = scaler.fit_transform(raw_scores)
    
    df_series['Cap_Score'] = normalized_scores[:, 0]
    df_series['Cost_Score'] = normalized_scores[:, 1]
    df_series['Buffer_Score'] = normalized_scores[:, 2]
    
    # Calculate weighted composite
    df_series['ECR_V2_Composite'] = (
        w_cap * df_series['Cap_Score'] +
        w_cost * df_series['Cost_Score'] +
        w_buffer * df_series['Buffer_Score']
    )
    
    # Select best fund
    best_fund_idx = df_series['ECR_V2_Composite'].idxmax()
    best_fund = df_series.loc[best_fund_idx, 'Fund']
    
    return best_fund


def select_ecr_v2_cap_moderate_normalized(df_universe, current_date, series='F', df_regimes=None):
    """
    ECR V2 Cap-Moderate (FIXED weights: 25/25/50) - NORMALIZED.
    
    This is the best fixed-weight variant from Batch 6.
    Uses same normalization as regime-adaptive but constant weights.
    """
    if df_universe.empty:
        return None
    
    df_series = df_universe[df_universe['Fund'].str.startswith(series)].copy()
    
    if df_series.empty:
        return None
    
    # Fixed weights (Cap-Moderate)
    w_cap, w_cost, w_buffer = 0.25, 0.25, 0.50
    
    # Calculate components
    required_cols = ['Cap_Utilization', 'Enhanced_Cost_Ratio', 'Buffer_Level']
    
    if not all(col in df_series.columns for col in required_cols):
        return df_series.iloc[0]['Fund']
    
    # Component 1: Cap Utilization
    df_series['Cap_Score_Raw'] = 1 - df_series['Cap_Utilization']
    
    # Component 2: Enhanced Cost Ratio
    max_cost = df_series['Enhanced_Cost_Ratio'].max()
    if max_cost > 0:
        df_series['Cost_Score_Raw'] = max_cost - df_series['Enhanced_Cost_Ratio']
    else:
        df_series['Cost_Score_Raw'] = 0
    
    # Component 3: Buffer Integrity
    df_series['Buffer_Score_Raw'] = df_series['Buffer_Level']
    
    # NORMALIZE
    scaler = MinMaxScaler()
    raw_scores = df_series[['Cap_Score_Raw', 'Cost_Score_Raw', 'Buffer_Score_Raw']].values
    normalized_scores = scaler.fit_transform(raw_scores)
    
    df_series['Cap_Score'] = normalized_scores[:, 0]
    df_series['Cost_Score'] = normalized_scores[:, 1]
    df_series['Buffer_Score'] = normalized_scores[:, 2]
    
    # Calculate composite
    df_series['ECR_V2_Composite'] = (
        w_cap * df_series['Cap_Score'] +
        w_cost * df_series['Cost_Score'] +
        w_buffer * df_series['Buffer_Score']
    )
    
    # Select best
    best_fund_idx = df_series['ECR_V2_Composite'].idxmax()
    best_fund = df_series.loc[best_fund_idx, 'Fund']
    
    return best_fund


def select_most_recent_launch(df_universe, current_date, series='F', df_regimes=None, **kwargs):
    """
    Select the most recently launched fund.

    This is the baseline strategy for comparison.

    Parameters
    ----------
    df_universe : DataFrame
        Available funds at current date
    current_date : datetime
        Current backtest date
    series : str
        Fund series (default 'F')
    df_regimes : DataFrame, optional
        Regime data (not used, for API compatibility)
    **kwargs : dict
        Additional arguments (ignored, for API compatibility)

    Returns
    -------
    str or None
        Fund identifier with most recent roll date
    """
    if df_universe.empty:
        return None

    df_series = df_universe[df_universe['Fund'].str.startswith(series)].copy()

    if df_series.empty:
        return None

    # Get most recent launch
    if 'Roll_Date' in df_series.columns:
        most_recent = df_series.loc[df_series['Roll_Date'].idxmax()]
    else:
        most_recent = df_series.iloc[0]

    return most_recent['Fund']


def get_batch_7_configs(regime_file_path='data/sp500_regimes.csv'):
    """
    BATCH 7: Regime-Adaptive ECR vs Fixed Weights
    
    Tests: 3 strategies × 12 months = 36 simulations
    
    Compares:
    1. Regime-Adaptive ECR (adjusts weights quarterly based on regime)
    2. Fixed Cap-Moderate ECR (constant 25/25/50 weights)
    3. Existing 90% threshold baseline
    """
    
    configs = []
    
    # All 12 launch months
    launch_months = [
        ['JAN'], ['FEB'], ['MAR'], ['APR'],
        ['MAY'], ['JUN'], ['JUL'], ['AUG'],
        ['SEP'], ['OCT'], ['NOV'], ['DEC']
    ]
    
    print(f"\n{'='*80}")
    print(f"BATCH 7: REGIME-ADAPTIVE ECR")
    print(f"{'='*80}")
    
    # Load regime data
    df_regimes = load_regime_data(regime_file_path)
    
    if df_regimes is None:
        print(f"❌ Cannot proceed without regime data")
        return []
    
    print(f"\n{'='*80}")
    print(f"CONFIGURATION SUMMARY")
    print(f"{'='*80}")
    
    # Create configs for each month
    for month_list in launch_months:
        month = month_list[0]
        
        # 1. Regime-Adaptive ECR
        configs.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': 'select_ecr_v2_regime_adaptive_normalized',
            'launch_months': month_list,
            'strategy_intent': 'regime_adaptive',
            'description': f'{month}: ECR Regime-Adaptive',
            'df_regimes': df_regimes
        })
        
        # 2. Fixed Cap-Moderate ECR
        configs.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': 'select_ecr_v2_cap_moderate_normalized',
            'launch_months': month_list,
            'strategy_intent': 'cost_optimized',
            'description': f'{month}: ECR Cap-Moderate (Fixed)'
        })
        
        # 3. Existing 90% Threshold
        configs.append({
            'trigger_type': 'cap_utilization_threshold',
            'trigger_params': {'threshold': 0.90},
            'selection_func_name': 'select_most_recent_launch',
            'launch_months': month_list,
            'strategy_intent': 'bullish',
            'description': f'{month}: Existing 90% Threshold'
        })
    
    print(f"\nTotal configurations: {len(configs)}")
    print(f"  Regime-Adaptive ECR:  {len([c for c in configs if 'regime_adaptive' in c.get('strategy_intent', '')])}")
    print(f"  Fixed Cap-Moderate:   {len([c for c in configs if 'cost_optimized' in c.get('strategy_intent', '')])}")
    print(f"  Existing 90%:         {len([c for c in configs if 'bullish' in c.get('strategy_intent', '')])}")
    print(f"\nLaunch months: {len(launch_months)}")
    print(f"Total simulations: {len(configs)}")
    print(f"Estimated time: ~{len(configs) * 0.3:.0f} minutes")
    print(f"{'='*80}\n")
    
    return configs, df_regimes


def get_batch_7b_configs():
    """
    BATCH 7B: Fixed-Weight ECR Performance by Regime
    Tests: 3 strategies × 12 months = 36 simulations

    Strategy behavior: Fixed weights (no regime switching)
    Performance analysis: Bucketed by regime periods
    """
    configs = []

    launch_months = [
        ['JAN'], ['FEB'], ['MAR'], ['APR'],
        ['MAY'], ['JUN'], ['JUL'], ['AUG'],
        ['SEP'], ['OCT'], ['NOV'], ['DEC']
    ]

    for month_list in launch_months:
        month = month_list[0]

        # 1. Fixed Cap-Moderate (constant 25/25/50)
        configs.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': 'select_ecr_v2_cap_moderate_normalized',
            'launch_months': month_list,
            'description': f'{month}: ECR Cap-Moderate (Fixed - Analyzed by Regime)'
        })

        # 2. Existing 90%
        configs.append({
            'trigger_type': 'cap_utilization_threshold',
            'trigger_params': {'threshold': 0.90},
            'selection_func_name': 'select_most_recent_launch',
            'launch_months': month_list,
            'description': f'{month}: Existing 90%'
        })

    return configs


"""
BATCH 10: Regime-Adaptive ECR with Updated Methodology
=======================================================

Add this function to backtesting/regime_adaptive.py

This batch tests the NEW ECR composite score methodology with regime-adaptive
weight switching, compared against the Existing 90% threshold baseline.

NEW ECR Formula Changes:
- DBB Score: EXP(X * 5) [exponential penalty vs old linear 1-ABS(X)]
- Buffer Integrity: MAX(0, (W + X)) / H [vs old W/H]
- Cap Integrity: MIN(U / G, 1) [capped vs old unbounded U/G]
- Time Scaling: 1 - LN(AA / 365) [unchanged]

Strategy:
- Switches weights quarterly based on regime (1/0/-1)
- Bullish (1): 15/15/70 (cap-focused)
- Neutral (0): 25/25/50 (balanced)
- Bearish (-1): 40/40/20 (protection-focused)
"""


def get_batch_10_configs():
    """Batch 10: New ECR Composite Scoring Weight Variations"""

    launch_months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    # Weight variations to test
    weight_configs = [
        ('111', 'Equal weights'),
        ('211', 'Emphasize DBB'),
        ('121', 'Emphasize Buffer Integrity'),
        ('112', 'Emphasize Cap Integrity'),
        ('101', 'DBB + Cap only'),
        ('221', 'Equal buffer emphasis'),
        ('110', 'Buffer-focused'),
    ]

    launch_months = ['MAR']#, 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    configs = []

    # 1. New ECR strategies with different weights
    for weight_code, description in weight_configs:
        configs.append({
            'trigger_type': 'rebalance_time_period',
            'trigger_params': {'frequency': 'quarterly'},
            'selection_func_name': f'select_highest_new_ecr_composite_{weight_code}',
            'selection_params': {},
            'launch_months': launch_months,
            'strategy_name': f'Time_Quarterly → New ECR (w={weight_code[0]},{weight_code[1]},{weight_code[2]})'
        })

    # 2. Existing 90% for comparison
    configs.append({
        'trigger_type': 'cap_utilization_threshold',
        'trigger_params': {'threshold': 0.90},
        'selection_func_name': 'select_most_recent_launch',
        'selection_params': {},
        'launch_months': launch_months,
        'strategy_name': 'Cap_Util_90% → Most Recent (Existing 90%)'
    })

    return configs


