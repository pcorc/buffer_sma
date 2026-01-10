"""
Regime-specific performance analysis.
"""

import pandas as pd
import numpy as np


def analyze_by_regime(results_list, df_regimes):
    """
    Calculate performance metrics separately for each market regime.

    For each backtest, splits the daily performance by regime (bull/bear/neutral)
    and calculates returns and trade counts for each regime period.

    Parameters:
      results_list: List of result dicts from run_single_ticker_backtest
      df_regimes: DataFrame with Date and Regime columns

    Returns:
      DataFrame with regime-specific performance metrics
    """

    if not results_list:
        print("ERROR: No results to analyze!")
        return pd.DataFrame()

    regime_records = []

    for result in results_list:
        daily_perf = result['daily_performance'].copy()

        # Merge with regime data
        daily_perf = daily_perf.merge(df_regimes[['Date', 'Regime']], on='Date', how='left')
        daily_perf['Regime'] = daily_perf['Regime'].fillna('neutral')

        # Analyze each regime separately
        for regime in ['bull', 'bear', 'neutral']:
            regime_data = daily_perf[daily_perf['Regime'] == regime].copy()

            if len(regime_data) < 2:
                continue

            # Calculate returns for this regime
            first_strat_nav = regime_data['Strategy_NAV'].iloc[0]
            last_strat_nav = regime_data['Strategy_NAV'].iloc[-1]
            strat_return = (last_strat_nav / first_strat_nav) - 1

            first_spy_nav = regime_data['SPY_NAV'].iloc[0]
            last_spy_nav = regime_data['SPY_NAV'].iloc[-1]
            spy_return = (last_spy_nav / first_spy_nav) - 1

            first_bufr_nav = regime_data['BUFR_NAV'].iloc[0]
            last_bufr_nav = regime_data['BUFR_NAV'].iloc[-1]
            bufr_return = (last_bufr_nav / first_bufr_nav) - 1

            first_hold_nav = regime_data['Hold_NAV'].iloc[0]
            last_hold_nav = regime_data['Hold_NAV'].iloc[-1]
            hold_return = (last_hold_nav / first_hold_nav) - 1

            # Count trades in this regime
            trade_history = result['trade_history']
            if not trade_history.empty:
                trades_in_regime = trade_history.merge(
                    df_regimes, on='Date', how='left'
                )
                num_trades_regime = (trades_in_regime['Regime'] == regime).sum()
            else:
                num_trades_regime = 0

            regime_record = {
                'launch_month': result['launch_month'],
                'trigger_type': result['trigger_type'],
                'trigger_params': str(result['trigger_params']),
                'selection_algo': result['selection_algo'],
                'regime': regime,
                'days_in_regime': len(regime_data),
                'strategy_return': strat_return,
                'spy_return': spy_return,
                'bufr_return': bufr_return,
                'hold_return': hold_return,
                'vs_spy_excess': strat_return - spy_return,
                'vs_bufr_excess': strat_return - bufr_return,
                'vs_hold_excess': strat_return - hold_return,
                'num_trades': num_trades_regime
            }

            regime_records.append(regime_record)

    regime_df = pd.DataFrame(regime_records)

    return regime_df


def compare_regime_performance(regime_df):
    """
    Compare average performance across different regimes.

    Parameters:
      regime_df: Regime-specific analysis DataFrame

    Returns:
      DataFrame with average metrics by regime
    """
    if regime_df.empty:
        return pd.DataFrame()

    regime_comparison = regime_df.groupby('regime').agg({
        'strategy_return': 'mean',
        'vs_spy_excess': 'mean',
        'vs_bufr_excess': 'mean',
        'vs_hold_excess': 'mean',
        'num_trades': 'sum',
        'days_in_regime': 'sum'
    }).round(4)

    regime_comparison = regime_comparison.reset_index()

    return regime_comparison


def find_best_by_regime(regime_df):
    """
    Find the best performing strategy for each regime.

    Parameters:
      regime_df: Regime-specific analysis DataFrame

    Returns:
      Dict with best strategy for each regime
    """
    if regime_df.empty:
        return {}

    best_strategies = {}

    for regime in ['bull', 'bear', 'neutral']:
        regime_data = regime_df[regime_df['regime'] == regime]
        if regime_data.empty:
            continue

        best_idx = regime_data['vs_bufr_excess'].idxmax()
        best_row = regime_data.loc[best_idx]

        best_strategies[regime] = {
            'launch_month': best_row['launch_month'],
            'trigger_type': best_row['trigger_type'],
            'selection_algo': best_row['selection_algo'],
            'vs_bufr_excess': best_row['vs_bufr_excess'],
            'strategy_return': best_row['strategy_return']
        }

    return best_strategies


def calculate_capture_ratios(regime_df):
    """
    Calculate upside and downside capture ratios.

    Upside capture = strategy return in bull / SPY return in bull
    Downside capture = strategy return in bear / SPY return in bear

    Parameters:
      regime_df: Regime-specific analysis DataFrame

    Returns:
      DataFrame with capture ratios by strategy
    """
    if regime_df.empty:
        return pd.DataFrame()

    # Calculate for each strategy (launch + trigger + selection combo)
    regime_df['strategy_id'] = (
            regime_df['launch_month'] + '_' +
            regime_df['trigger_type'] + '_' +
            regime_df['selection_algo']
    )

    capture_records = []

    for strategy_id in regime_df['strategy_id'].unique():
        strat_data = regime_df[regime_df['strategy_id'] == strategy_id]

        bull_data = strat_data[strat_data['regime'] == 'bull']
        bear_data = strat_data[strat_data['regime'] == 'bear']

        upside_capture = np.nan
        downside_capture = np.nan

        if not bull_data.empty and bull_data.iloc[0]['spy_return'] != 0:
            upside_capture = bull_data.iloc[0]['strategy_return'] / bull_data.iloc[0]['spy_return']

        if not bear_data.empty and bear_data.iloc[0]['spy_return'] != 0:
            downside_capture = bear_data.iloc[0]['strategy_return'] / bear_data.iloc[0]['spy_return']

        capture_records.append({
            'strategy_id': strategy_id,
            'launch_month': strat_data.iloc[0]['launch_month'],
            'trigger_type': strat_data.iloc[0]['trigger_type'],
            'selection_algo': strat_data.iloc[0]['selection_algo'],
            'upside_capture': upside_capture,
            'downside_capture': downside_capture
        })

    capture_df = pd.DataFrame(capture_records)

    return capture_df