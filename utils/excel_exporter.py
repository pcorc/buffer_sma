"""
Excel export utilities for consolidated workbook output.
"""

import os
import pandas as pd
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows


def export_consolidated_workbook(results_list, summary_df, output_dir, run_name='backtest'):
    """
    Export all backtest results to a single Excel workbook with multiple tabs.

    Workbook Structure:
    - Tab 1: Summary (all iterations with key metrics)
    - Tab 2: Trade Log (all trades with iteration details)
    - Tab 3+: Daily Time Series for each iteration

    Parameters:
        results_list: List of result dicts from run_single_ticker_backtest
        summary_df: Consolidated summary DataFrame
        output_dir: Directory path for output file
        run_name: Name prefix for the workbook file

    Returns:
        Path to created workbook
    """

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{run_name}_consolidated_{timestamp}.xlsx'
    filepath = os.path.join(output_dir, filename)

    # Create Excel writer
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:

        # =====================================================================
        # TAB 1: SUMMARY
        # =====================================================================

        # Add iteration number for clarity
        summary_export = summary_df.copy()
        summary_export.insert(0, 'iteration', range(1, len(summary_export) + 1))

        # Format percentages and round numbers
        pct_cols = [col for col in summary_export.columns if 'return' in col or 'excess' in col or 'dd' in col or 'volatility' in col]
        for col in pct_cols:
            if col in summary_export.columns:
                summary_export[col] = summary_export[col].round(4)

        if 'strategy_sharpe' in summary_export.columns:
            summary_export['strategy_sharpe'] = summary_export['strategy_sharpe'].round(2)

        summary_export.to_excel(writer, sheet_name='Summary', index=False)

        # Format Summary sheet
        worksheet = writer.sheets['Summary']

        # Header formatting
        header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        header_font = Font(bold=True, color='FFFFFF')

        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

        # Freeze header row
        worksheet.freeze_panes = 'A2'

        # =====================================================================
        # TAB 2: TRADE LOG
        # =====================================================================

        all_trades = []

        for idx, result in enumerate(results_list, 1):
            if result['trade_history'].empty:
                continue

            trades = result['trade_history'].copy()

            # Add iteration details
            trades.insert(0, 'iteration', idx)
            trades['launch_month'] = result['launch_month']
            trades['trigger_type'] = result['trigger_type']
            trades['trigger_params'] = str(result['trigger_params'])
            trades['selection_algo'] = result['selection_algo']

            # Add selection reason by looking at the fund that was selected
            trades['Selection_Reason'] = trades.apply(
                lambda row: _get_selection_reason(result['selection_algo'], row['To_Fund']),
                axis=1
            )

            all_trades.append(trades)

        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)

            # Reorder columns for better readability
            col_order = [
                'iteration', 'Date', 'launch_month', 'trigger_type', 'trigger_params',
                'selection_algo', 'From_Fund', 'To_Fund', 'Trigger_Reason',
                'Selection_Reason', 'NAV_at_Switch'
            ]
            combined_trades = combined_trades[col_order]

            combined_trades.to_excel(writer, sheet_name='Trade Log', index=False)

            # Format Trade Log sheet
            worksheet = writer.sheets['Trade Log']

            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

            worksheet.freeze_panes = 'A2'

        else:
            print(f"    ⚠ No trades to export")

        # =====================================================================
        # TAB 3+: DAILY TIME SERIES FOR EACH ITERATION
        # =====================================================================

        for idx, result in enumerate(results_list, 1):
            # Create tab name
            trigger_short = result['trigger_type'].replace('_', '').replace('rebalance', 'reb')[:12]
            selection_short = result['selection_algo'].replace('_', '').replace('select', 'sel')[:12]
            tab_name = f"{idx}_{result['launch_month']}_{trigger_short}_{selection_short}"

            # Excel tab names max 31 characters
            if len(tab_name) > 31:
                tab_name = f"{idx}_{result['launch_month'][:3]}_{trigger_short[:8]}_{selection_short[:8]}"

            # Prepare daily data
            daily = result['daily_performance'].copy()

            # Add iteration info
            daily.insert(0, 'iteration', idx)
            daily.insert(1, 'launch_month', result['launch_month'])
            daily.insert(2, 'trigger_type', result['trigger_type'])
            daily.insert(3, 'selection_algo', result['selection_algo'])

            # Add trade indicator
            daily['Trade_Occurred'] = False
            if not result['trade_history'].empty:
                trade_dates = set(result['trade_history']['Date'])
                daily['Trade_Occurred'] = daily['Date'].isin(trade_dates)

            # Add returns
            daily['Strategy_Daily_Return'] = daily['Strategy_NAV'].pct_change()
            daily['SPY_Daily_Return'] = daily['SPY_NAV'].pct_change()
            daily['BUFR_Daily_Return'] = daily['BUFR_NAV'].pct_change()
            daily['Hold_Daily_Return'] = daily['Hold_NAV'].pct_change()

            # Add relative performance (indexed to 100)
            daily['Strategy_vs_SPY'] = (daily['Strategy_NAV'] / daily['SPY_NAV']) * 100
            daily['Strategy_vs_BUFR'] = (daily['Strategy_NAV'] / daily['BUFR_NAV']) * 100
            daily['Strategy_vs_Hold'] = (daily['Strategy_NAV'] / daily['Hold_NAV']) * 100

            # Round for readability
            numeric_cols = daily.select_dtypes(include=['float64']).columns
            daily[numeric_cols] = daily[numeric_cols].round(6)

            # Write to Excel
            daily.to_excel(writer, sheet_name=tab_name, index=False)

            # Format sheet
            worksheet = writer.sheets[tab_name]

            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

            # Auto-adjust columns (but faster than full iteration)
            for column_letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
                worksheet.column_dimensions[column_letter].width = 15

            worksheet.freeze_panes = 'A2'

    return filepath


def _get_selection_reason(selection_algo: str, selected_fund: str) -> str:
    """
    Generate human-readable selection reason based on algorithm.

    Parameters:
        selection_algo: Name of selection function
        selected_fund: The fund that was selected

    Returns:
        Human-readable reason string
    """
    reasons = {
        'select_most_recent_launch': f'Most recent roll date ({selected_fund})',
        'select_remaining_cap': f'Highest remaining cap ({selected_fund})',
        'select_cap_utilization': f'Lowest cap utilization ({selected_fund})',
        'select_highest_outcome_and_cap': f'Best outcome days + cap score ({selected_fund})',
        'select_cost_analysis': f'Lowest cost per day of protection ({selected_fund})'
    }

    return reasons.get(selection_algo, f'{selection_algo} selected {selected_fund}')


def export_consolidated_workbook_append(results_list, summary_df, output_dir, run_name='backtest'):
    """
    Export results and append to cumulative workbook if it exists.

    Parameters:
        results_list: List of result dicts from run_single_ticker_backtest
        summary_df: Consolidated summary DataFrame
        output_dir: Directory path for output file
        run_name: Name prefix for the workbook file

    Returns:
        Path to created/updated workbook
    """
    # First create the timestamped version
    timestamped_path = export_consolidated_workbook(results_list, summary_df, output_dir, run_name)

    # Then create/update cumulative version
    cumulative_filename = f'{run_name}_cumulative.xlsx'
    cumulative_path = os.path.join(output_dir, cumulative_filename)

    if os.path.exists(cumulative_path):

        existing_summary = pd.read_excel(cumulative_path, sheet_name='Summary')

        new_summary = summary_df.copy()
        new_summary.insert(0, 'iteration', range(len(existing_summary) + 1, len(existing_summary) + len(new_summary) + 1))
        new_summary['run_timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")

        combined_summary = pd.concat([existing_summary, new_summary], ignore_index=True)

        # Read existing trade log if it exists
        try:
            existing_trades = pd.read_excel(cumulative_path, sheet_name='Trade Log')
        except:
            existing_trades = pd.DataFrame()

        # Create combined trades
        all_trades = []
        start_iteration = len(existing_summary) + 1

        for idx, result in enumerate(results_list, start_iteration):
            if result['trade_history'].empty:
                continue

            trades = result['trade_history'].copy()
            trades.insert(0, 'iteration', idx)
            trades['launch_month'] = result['launch_month']
            trades['trigger_type'] = result['trigger_type']
            trades['selection_algo'] = result['selection_algo']
            trades['Selection_Reason'] = trades.apply(
                lambda row: _get_selection_reason(result['selection_algo'], row['To_Fund']),
                axis=1
            )
            all_trades.append(trades)

        if all_trades:
            new_trades = pd.concat(all_trades, ignore_index=True)
            if not existing_trades.empty:
                combined_trades = pd.concat([existing_trades, new_trades], ignore_index=True)
            else:
                combined_trades = new_trades
        else:
            combined_trades = existing_trades

        # Write updated cumulative workbook
        with pd.ExcelWriter(cumulative_path, engine='openpyxl') as writer:
            combined_summary.to_excel(writer, sheet_name='Summary', index=False)
            if not combined_trades.empty:
                combined_trades.to_excel(writer, sheet_name='Trade Log', index=False)

    else:
        # Just copy the timestamped version as the first cumulative
        import shutil
        shutil.copy(timestamped_path, cumulative_path)

    return cumulative_path