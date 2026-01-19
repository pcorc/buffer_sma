"""
Debug export utilities for time series validation.
"""

import os
import pandas as pd
from datetime import datetime


def export_debug_time_series(results_list, output_dir, df_regimes=None):
    """
    Export daily time series for each backtest configuration for manual validation.

    Creates individual CSV files with complete NAV time series and trade information.

    Parameters:
        results_list: List of result dicts from backtests
        output_dir: Directory to save debug exports
        df_regimes: Optional DataFrame with regime classification
    """

    os.makedirs(output_dir, exist_ok=True)

    for idx, result in enumerate(results_list, 1):
        # Create filename from strategy parameters
        launch = result['launch_month']
        trigger = result['trigger_type']
        params = result['trigger_params']
        selection = result['selection_algo']

        # Shorten trigger name
        trigger_short = trigger.replace('_threshold', '').replace('rebalance_', '')

        # Format parameters
        if 'threshold' in params:
            param_str = f"{params['threshold']:.2f}".replace('.', 'p')
        elif 'frequency' in params:
            param_str = params['frequency'][:3]
        else:
            param_str = "default"

        # Create filename
        filename = f"{idx:03d}_{launch}_{trigger_short}_{param_str}_{selection}.csv"
        filepath = os.path.join(output_dir, filename)

        # Prepare time series data
        daily_df = result['daily_performance'].copy()

        # Add strategy metadata
        daily_df.insert(0, 'config_id', idx)
        daily_df.insert(1, 'launch_month', launch)
        daily_df.insert(2, 'trigger_type', trigger)
        daily_df.insert(3, 'trigger_params', str(params))
        daily_df.insert(4, 'selection_algo', selection)

        # Add regime if available
        if df_regimes is not None:
            daily_df = daily_df.merge(df_regimes[['Date', 'Regime']], on='Date', how='left')
            daily_df['Regime'] = daily_df['Regime'].fillna('unknown')

        # Add trade indicator
        daily_df['Trade_Occurred'] = False
        if not result['trade_history'].empty:
            trade_dates = set(result['trade_history']['Date'])
            daily_df['Trade_Occurred'] = daily_df['Date'].isin(trade_dates)

        # Calculate daily returns for validation
        daily_df['Strategy_Daily_Return'] = daily_df['Strategy_NAV'].pct_change()
        daily_df['SPY_Daily_Return'] = daily_df['SPY_NAV'].pct_change()
        daily_df['BUFR_Daily_Return'] = daily_df['BUFR_NAV'].pct_change()
        daily_df['Hold_Daily_Return'] = daily_df['Hold_NAV'].pct_change()

        # Add cumulative returns from start
        daily_df['Strategy_Cumulative_Return'] = (daily_df['Strategy_NAV'] / 100) - 1
        daily_df['SPY_Cumulative_Return'] = (daily_df['SPY_NAV'] / 100) - 1
        daily_df['BUFR_Cumulative_Return'] = (daily_df['BUFR_NAV'] / 100) - 1
        daily_df['Hold_Cumulative_Return'] = (daily_df['Hold_NAV'] / 100) - 1

        # Reorder columns for clarity
        column_order = [
            'Date',
            'config_id',
            'launch_month',
            'trigger_type',
            'trigger_params',
            'selection_algo',
            'Current_Fund',
            'Trade_Occurred',
        ]

        if 'Regime' in daily_df.columns:
            column_order.append('Regime')

        # NAV columns
        column_order.extend([
            'Strategy_NAV',
            'SPY_NAV',
            'BUFR_NAV',
            'Hold_NAV',
        ])

        # Daily returns
        column_order.extend([
            'Strategy_Daily_Return',
            'SPY_Daily_Return',
            'BUFR_Daily_Return',
            'Hold_Daily_Return',
        ])

        # Cumulative returns
        column_order.extend([
            'Strategy_Cumulative_Return',
            'SPY_Cumulative_Return',
            'BUFR_Cumulative_Return',
            'Hold_Cumulative_Return',
        ])

        # Roll date metrics
        if 'Roll_Date' in daily_df.columns:
            column_order.extend([
                'Roll_Date',
                'Original_Cap',
                'Starting_Fund_NAV',
                'Current_Fund_NAV',
                'Starting_Ref_Index',
                'Current_Ref_Index',
                'Fund_Return_From_Roll',
                'Ref_Index_Return_From_Roll',
                'Cap_Utilization',
                'Cap_Remaining_Pct',
            ])

        # Only include columns that exist
        final_columns = [col for col in column_order if col in daily_df.columns]
        daily_df = daily_df[final_columns]

        # Round numeric columns
        numeric_cols = daily_df.select_dtypes(include=['float64']).columns
        daily_df[numeric_cols] = daily_df[numeric_cols].round(8)

        # Export to CSV
        daily_df.to_csv(filepath, index=False)

    # Create index file with summary
    create_debug_index(results_list, output_dir)


def create_debug_index(results_list, output_dir):
    """
    Create an index file listing all exported configurations.

    Parameters:
        results_list: List of result dicts
        output_dir: Directory where exports are saved
    """

    index_records = []

    for idx, result in enumerate(results_list, 1):
        trigger = result['trigger_type']
        params = result['trigger_params']

        # Format trigger description
        if 'threshold' in params:
            param_str = f"{params['threshold']:.2f}"
            trigger_desc = f"{trigger} @ {param_str}"
        elif 'frequency' in params:
            trigger_desc = f"{trigger} ({params['frequency']})"
        else:
            trigger_desc = trigger

        index_records.append({
            'config_id': idx,
            'filename': f"{idx:03d}_*.csv",
            'launch_month': result['launch_month'],
            'trigger': trigger_desc,
            'selection': result['selection_algo'],
            'num_trades': result['num_trades'],
            'strategy_return': result['strategy_total_return'],
            'vs_bufr_excess': result['vs_bufr_excess'],
            'sharpe': result['strategy_sharpe'],
            'start_date': result['start_date'].date(),
            'end_date': result['end_date'].date()
        })

    index_df = pd.DataFrame(index_records)

    # Round numeric columns
    index_df['strategy_return'] = index_df['strategy_return'].round(4)
    index_df['vs_bufr_excess'] = index_df['vs_bufr_excess'].round(4)
    index_df['sharpe'] = index_df['sharpe'].round(2)

    # Export index
    index_path = os.path.join(output_dir, '_INDEX.csv')
    index_df.to_csv(index_path, index=False)

    print(f"   ✓ Created index file: _INDEX.csv")
