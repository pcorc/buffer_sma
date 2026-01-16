"""
Excel export utilities for consolidated workbook output.
"""

import os
import pandas as pd
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows


def export_consolidated_workbook(results_list, summary_df, output_dir, run_name='backtest',
                                df_regimes=None, regime_df=None, capture_ratios=None,
                                trigger_summary=None, selection_summary=None, month_summary=None):
    """
    Export all backtest results to a single Excel workbook with multiple tabs.

    Workbook Structure:
    - Tab 1: Summary (all iterations with key metrics)
    - Tab 2: Trade Log (all trades with iteration details)
    - Tab 3: Regime Analysis (regime-specific performance)
    - Tab 4: Capture Ratios (upside/downside capture)
    - Tab 5: Trigger Summary (aggregated by trigger type)
    - Tab 6: Selection Summary (aggregated by selection algo)
    - Tab 7+: Daily Time Series for each iteration (with regime data)

    Parameters:
        results_list: List of result dicts from run_single_ticker_backtest
        summary_df: Consolidated summary DataFrame
        output_dir: Directory path for output file
        run_name: Name prefix for the workbook file
        df_regimes: Optional DataFrame with Date and Regime columns
        regime_df: Optional DataFrame with regime-specific analysis
        capture_ratios: Optional DataFrame with capture ratios
        trigger_summary: Optional DataFrame with trigger aggregation
        selection_summary: Optional DataFrame with selection aggregation

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

            # Add selection reason
            trades['Selection_Reason'] = trades.apply(
                lambda row: _get_selection_reason(result['selection_algo'], row['To_Fund']),
                axis=1
            )

            # Add regime if available
            if df_regimes is not None:
                trades = trades.merge(df_regimes[['Date', 'Regime']], on='Date', how='left')
                trades['Regime'] = trades['Regime'].fillna('unknown')

            all_trades.append(trades)

        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)

            # Reorder columns for better readability
            base_cols = [
                'iteration', 'Date', 'launch_month', 'trigger_type', 'trigger_params',
                'selection_algo', 'From_Fund', 'To_Fund', 'Trigger_Reason',
                'Selection_Reason', 'NAV_at_Switch'
            ]

            if df_regimes is not None and 'Regime' in combined_trades.columns:
                base_cols.insert(2, 'Regime')

            combined_trades = combined_trades[base_cols]

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
        # TAB 3: ROLL DATE SNAPSHOTS
        # =====================================================================

        print("  Creating Roll Date Snapshots tab...")

        # Extract roll date snapshots from first result's daily performance
        # (all iterations use same funds, so roll dates are the same)
        if results_list:
            # Get all unique roll dates from the first iteration
            first_result = results_list[0]
            daily_df = first_result['daily_performance']

            # Group by Outcome_Period_ID and get first row of each period (the roll date)
            roll_snapshots = []

            for period_id in daily_df['Outcome_Period_ID'].dropna().unique():
                period_data = daily_df[daily_df['Outcome_Period_ID'] == period_id]
                if len(period_data) > 0:
                    snapshot = period_data.iloc[0].copy()

                    # Extract fund name from Current_Fund
                    fund_name = snapshot.get('Current_Fund', '')

                    roll_snapshots.append({
                        'Fund': fund_name,
                        'Outcome_Period_ID': period_id,
                        'Roll_Date': snapshot.get('Roll_Date'),
                        'Total_Outcome_Days': snapshot.get('Total_Outcome_Days'),
                        'Starting_Fund_NAV': snapshot.get('Starting_Fund_NAV'),
                        'Starting_Ref_Index': snapshot.get('Starting_Ref_Index'),
                        'Original_Cap': snapshot.get('Original_Cap'),
                        'Original_Buffer': snapshot.get('Original_Buffer'),
                        'Starting_Downside_Before_Buffer': snapshot.get('Starting_Downside_Before_Buffer'),
                        'Fund_Cap_Value': snapshot.get('Starting_Fund_NAV', 0) * (1 + snapshot.get('Original_Cap', 0)) if snapshot.get('Starting_Fund_NAV') and snapshot.get('Original_Cap') else None,
                        'Ref_Index_Cap_Value': snapshot.get('Starting_Ref_Index', 0) * (1 + snapshot.get('Original_Cap', 0)) if snapshot.get('Starting_Ref_Index') and snapshot.get('Original_Cap') else None
                    })

            if roll_snapshots:
                roll_snapshot_df = pd.DataFrame(roll_snapshots)

                # Sort by Roll_Date
                roll_snapshot_df = roll_snapshot_df.sort_values('Roll_Date')

                # Round numeric columns
                numeric_cols = roll_snapshot_df.select_dtypes(include=['float64']).columns
                roll_snapshot_df[numeric_cols] = roll_snapshot_df[numeric_cols].round(6)

                roll_snapshot_df.to_excel(writer, sheet_name='Roll Date Snapshots', index=False)

                # Format sheet
                worksheet = writer.sheets['Roll Date Snapshots']

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

                print(f"    ✓ Roll Date Snapshots: {len(roll_snapshot_df)} roll dates")


        # =====================================================================
        # TAB 4: REGIME ANALYSIS
        # =====================================================================

        if regime_df is not None and not regime_df.empty:
            regime_export = regime_df.copy()

            # Round numeric columns
            numeric_cols = regime_export.select_dtypes(include=['float64']).columns
            regime_export[numeric_cols] = regime_export[numeric_cols].round(4)

            regime_export.to_excel(writer, sheet_name='Regime Analysis', index=False)

            # Format sheet
            worksheet = writer.sheets['Regime Analysis']

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

        # =====================================================================
        # TAB 5: CAPTURE RATIOS
        # =====================================================================

        if capture_ratios is not None and not capture_ratios.empty:
            capture_export = capture_ratios.copy()

            # Round numeric columns
            numeric_cols = capture_export.select_dtypes(include=['float64']).columns
            capture_export[numeric_cols] = capture_export[numeric_cols].round(4)

            capture_export.to_excel(writer, sheet_name='Capture Ratios', index=False)

            # Format sheet
            worksheet = writer.sheets['Capture Ratios']

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

        # =====================================================================
        # TAB 6: TRIGGER SUMMARY
        # =====================================================================

        if trigger_summary is not None and not trigger_summary.empty:

            trigger_export = trigger_summary.copy()

            # Round numeric columns
            numeric_cols = trigger_export.select_dtypes(include=['float64']).columns
            trigger_export[numeric_cols] = trigger_export[numeric_cols].round(4)

            trigger_export.to_excel(writer, sheet_name='By Trigger', index=False)

            # Format sheet
            worksheet = writer.sheets['By Trigger']

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

        # =====================================================================
        # TAB 7: SELECTION SUMMARY
        # =====================================================================

        if selection_summary is not None and not selection_summary.empty:

            selection_export = selection_summary.copy()

            # Round numeric columns
            numeric_cols = selection_export.select_dtypes(include=['float64']).columns
            selection_export[numeric_cols] = selection_export[numeric_cols].round(4)

            selection_export.to_excel(writer, sheet_name='By Selection', index=False)

            # Format sheet
            worksheet = writer.sheets['By Selection']

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

        # =====================================================================
        # TAB 8+: DAILY TIME SERIES FOR EACH ITERATION
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

            # Add regime information if available
            if df_regimes is not None:
                daily = daily.merge(df_regimes[['Date', 'Regime']], on='Date', how='left')
                daily['Regime'] = daily['Regime'].fillna('unknown')
                # Move Regime column to position 4 (after selection_algo)
                cols = list(daily.columns)
                cols.insert(4, cols.pop(cols.index('Regime')))
                daily = daily[cols]

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

            # Reorder columns for better analysis flow
            base_cols = [
                'Date', 'iteration', 'launch_month', 'trigger_type', 'selection_algo'
            ]

            if 'Regime' in daily.columns:
                base_cols.append('Regime')

            # Add identification and snapshot columns
            snapshot_cols = [
                'Current_Fund', 'Outcome_Period_ID', 'Roll_Date',
                'Starting_Fund_NAV', 'Starting_Ref_Index', 'Original_Cap', 'Original_Buffer',
                'Total_Outcome_Days', 'Remaining_Outcome_Days'
            ]

            # Current values
            current_cols = [
                'Current_Fund_NAV', 'Current_Ref_Index', 'Current_Remaining_Cap_Pct'
            ]

            # Calculated returns
            return_cols = [
                'Fund_Return_From_Roll', 'Ref_Index_Return_From_Roll'
            ]

            # NAV performance
            nav_cols = [
                'Strategy_NAV', 'SPY_NAV', 'BUFR_NAV', 'Hold_NAV'
            ]

            # Daily returns
            daily_return_cols = [
                'Strategy_Daily_Return', 'SPY_Daily_Return', 'BUFR_Daily_Return', 'Hold_Daily_Return'
            ]

            # Relative performance
            relative_cols = [
                'Strategy_vs_SPY', 'Strategy_vs_BUFR', 'Strategy_vs_Hold'
            ]

            # Cap and buffer metrics
            metrics_cols = [
                'Cap_Utilization', 'Cap_Remaining_Pct',
                'Downside_Before_Buffer_Pct', 'Starting_Downside_Before_Buffer'
            ]

            # Trading
            trade_cols = ['Trade_Occurred']

            # Combine all columns in logical order
            all_cols = base_cols + snapshot_cols + current_cols + return_cols + nav_cols + daily_return_cols + relative_cols + metrics_cols + trade_cols

            # Only include columns that exist
            ordered_cols = [col for col in all_cols if col in daily.columns]
            daily = daily[ordered_cols]

            # Write to Excel
            daily.to_excel(writer, sheet_name=tab_name, index=False)

            # Format sheet
            worksheet = writer.sheets[tab_name]

            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

            # Auto-adjust columns (but faster than full iteration)
            for column_letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
                worksheet.column_dimensions[column_letter].width = 15

            worksheet.freeze_panes = 'A2'

    # Count total tabs
    total_tabs = 2  # Summary + Trade Log
    total_tabs += 1  # Roll Date Snapshots (always included if results exist)
    if regime_df is not None and not regime_df.empty:
        total_tabs += 1
    if capture_ratios is not None and not capture_ratios.empty:
        total_tabs += 1
    if trigger_summary is not None and not trigger_summary.empty:
        total_tabs += 1
    if selection_summary is not None and not selection_summary.empty:
        total_tabs += 1
    if month_summary is not None and not month_summary.empty:
        total_tabs += 1
    total_tabs += len(results_list)

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
    Export all backtest results to a single Excel workbook with multiple tabs.

    Workbook Structure:
    - Tab 1: Summary (all iterations with key metrics)
    - Tab 2: Trade Log (all trades with iteration details)
    - Tab 3+: Daily Time Series for each iteration (with regime data if provided)

    Parameters:
        results_list: List of result dicts from run_single_ticker_backtest
        summary_df: Consolidated summary DataFrame
        output_dir: Directory path for output file
        run_name: Name prefix for the workbook file
        df_regimes: Optional DataFrame with Date and Regime columns

    Returns:
        Path to created workbook
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


def export_main_consolidated_workbook(results_list, summary_df, output_dir, run_name='mainpy_consolidated',
                                      df_regimes=None, regime_df=None, capture_ratios=None,
                                      trigger_summary=None, selection_summary=None, month_summary=None):
    """
    Export all backtest results from main.py to a single Excel workbook with multiple tabs.

    Optimized for large-scale backtests (100+ iterations) with strategy intent classification.

    Workbook Structure:
    - Tab 1: Summary (all iterations with key metrics)
    - Tab 2: Trade Log (all trades with iteration details)
    - Tab 3: Regime Analysis (regime-specific performance)
    - Tab 4: Capture Ratios (upside/downside capture)
    - Tab 5: By Trigger (aggregated by trigger type)
    - Tab 6: By Selection (aggregated by selection algo)
    - Tab 7: By Launch Month (aggregated by month)
    - Tab 8: By Strategy Intent (aggregated by bullish/bearish/neutral)
    - Tab 9: By Intent & Regime (strategy intent vs actual regime performance)
    - Tab 10: By Month & Regime (launch month performance across regimes)
    - Tab 11: Daily Time Series (all iterations, filterable with slicers)

    Parameters:
        results_list: List of result dicts from run_single_ticker_backtest
        summary_df: Consolidated summary DataFrame
        output_dir: Directory path for output file
        run_name: Name prefix for the workbook file
        df_regimes: Optional DataFrame with Date and Regime columns
        regime_df: Optional DataFrame with regime-specific analysis
        capture_ratios: Optional DataFrame with capture ratios
        trigger_summary: Optional DataFrame with trigger aggregation
        selection_summary: Optional DataFrame with selection aggregation
        month_summary: Optional DataFrame with month aggregation

    Returns:
        Path to created workbook
    """
    from analysis.consolidator import (
        summarize_by_strategy_intent,
        summarize_by_intent_and_regime,
        summarize_by_month_and_regime
    )

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{run_name}_{timestamp}.xlsx'
    filepath = os.path.join(output_dir, filename)

    print(f"Creating consolidated workbook: {filename}")

    # Generate additional summaries
    intent_summary = summarize_by_strategy_intent(summary_df)
    intent_regime_summary = summarize_by_intent_and_regime(summary_df, regime_df) if regime_df is not None else pd.DataFrame()
    month_regime_summary = summarize_by_month_and_regime(summary_df, regime_df) if regime_df is not None else pd.DataFrame()

    # Create Excel writer
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:

        # Header formatting
        header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        header_font = Font(bold=True, color='FFFFFF')

        # =====================================================================
        # TAB 1: SUMMARY
        # =====================================================================

        summary_export = summary_df.copy()
        summary_export.insert(0, 'iteration', range(1, len(summary_export) + 1))

        # Format percentages
        pct_cols = [col for col in summary_export.columns if 'return' in col or 'excess' in col or 'dd' in col or 'volatility' in col]
        for col in pct_cols:
            if col in summary_export.columns:
                summary_export[col] = summary_export[col].round(4)

        if 'strategy_sharpe' in summary_export.columns:
            summary_export['strategy_sharpe'] = summary_export['strategy_sharpe'].round(2)

        summary_export.to_excel(writer, sheet_name='Summary', index=False)
        _format_sheet(writer.sheets['Summary'], header_fill, header_font)
        print(f"  ✓ Tab 1: Summary ({len(summary_export)} iterations)")

        # =====================================================================
        # TAB 2: TRADE LOG
        # =====================================================================

        all_trades = []
        for idx, result in enumerate(results_list, 1):
            if result['trade_history'].empty:
                continue

            trades = result['trade_history'].copy()
            trades.insert(0, 'iteration', idx)
            trades['launch_month'] = result['launch_month']
            trades['trigger_type'] = result['trigger_type']
            trades['trigger_params'] = str(result['trigger_params'])
            trades['selection_algo'] = result['selection_algo']
            trades['strategy_intent'] = result.get('strategy_intent', 'neutral')

            trades['Selection_Reason'] = trades.apply(
                lambda row: _get_selection_reason(result['selection_algo'], row['To_Fund']),
                axis=1
            )

            if df_regimes is not None:
                trades = trades.merge(df_regimes[['Date', 'Regime']], on='Date', how='left')
                trades['Regime'] = trades['Regime'].fillna('unknown')

            all_trades.append(trades)

        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)

            base_cols = [
                'iteration', 'Date', 'launch_month', 'trigger_type', 'trigger_params',
                'selection_algo', 'strategy_intent', 'From_Fund', 'To_Fund',
                'Trigger_Reason', 'Selection_Reason', 'NAV_at_Switch'
            ]

            if df_regimes is not None and 'Regime' in combined_trades.columns:
                base_cols.insert(2, 'Regime')

            combined_trades = combined_trades[base_cols]
            combined_trades.to_excel(writer, sheet_name='Trade Log', index=False)
            _format_sheet(writer.sheets['Trade Log'], header_fill, header_font)
            print(f"  ✓ Tab 2: Trade Log ({len(combined_trades)} trades)")
        else:
            print(f"  ⚠ Tab 2: Trade Log (no trades)")

        # =====================================================================
        # TAB 3: REGIME ANALYSIS
        # =====================================================================

        if regime_df is not None and not regime_df.empty:
            regime_export = regime_df.copy()
            numeric_cols = regime_export.select_dtypes(include=['float64']).columns
            regime_export[numeric_cols] = regime_export[numeric_cols].round(4)
            regime_export.to_excel(writer, sheet_name='Regime Analysis', index=False)
            _format_sheet(writer.sheets['Regime Analysis'], header_fill, header_font)
            print(f"  ✓ Tab 3: Regime Analysis ({len(regime_export)} records)")

        # =====================================================================
        # TAB 4: CAPTURE RATIOS
        # =====================================================================

        if capture_ratios is not None and not capture_ratios.empty:
            capture_export = capture_ratios.copy()
            numeric_cols = capture_export.select_dtypes(include=['float64']).columns
            capture_export[numeric_cols] = capture_export[numeric_cols].round(4)
            capture_export.to_excel(writer, sheet_name='Capture Ratios', index=False)
            _format_sheet(writer.sheets['Capture Ratios'], header_fill, header_font)
            print(f"  ✓ Tab 4: Capture Ratios ({len(capture_export)} strategies)")

        # =====================================================================
        # TAB 5: BY TRIGGER
        # =====================================================================

        if trigger_summary is not None and not trigger_summary.empty:
            trigger_export = trigger_summary.copy()
            numeric_cols = trigger_export.select_dtypes(include=['float64']).columns
            trigger_export[numeric_cols] = trigger_export[numeric_cols].round(4)
            trigger_export.to_excel(writer, sheet_name='By Trigger', index=False)
            _format_sheet(writer.sheets['By Trigger'], header_fill, header_font)
            print(f"  ✓ Tab 5: By Trigger ({len(trigger_export)} trigger types)")

        # =====================================================================
        # TAB 6: BY SELECTION
        # =====================================================================

        if selection_summary is not None and not selection_summary.empty:
            selection_export = selection_summary.copy()
            numeric_cols = selection_export.select_dtypes(include=['float64']).columns
            selection_export[numeric_cols] = selection_export[numeric_cols].round(4)
            selection_export.to_excel(writer, sheet_name='By Selection', index=False)
            _format_sheet(writer.sheets['By Selection'], header_fill, header_font)
            print(f"  ✓ Tab 6: By Selection ({len(selection_export)} selection algos)")

        # =====================================================================
        # TAB 7: BY LAUNCH MONTH
        # =====================================================================

        if month_summary is not None and not month_summary.empty:
            month_export = month_summary.copy()
            numeric_cols = month_export.select_dtypes(include=['float64']).columns
            month_export[numeric_cols] = month_export[numeric_cols].round(4)
            month_export.to_excel(writer, sheet_name='By Launch Month', index=False)
            _format_sheet(writer.sheets['By Launch Month'], header_fill, header_font)
            print(f"  ✓ Tab 7: By Launch Month ({len(month_export)} months)")

        # =====================================================================
        # TAB 8: BY STRATEGY INTENT
        # =====================================================================

        if not intent_summary.empty:
            intent_export = intent_summary.copy()
            numeric_cols = intent_export.select_dtypes(include=['float64']).columns
            intent_export[numeric_cols] = intent_export[numeric_cols].round(4)
            intent_export.to_excel(writer, sheet_name='By Strategy Intent', index=False)
            _format_sheet(writer.sheets['By Strategy Intent'], header_fill, header_font)
            print(f"  ✓ Tab 8: By Strategy Intent ({len(intent_export)} intents)")

        # =====================================================================
        # TAB 9: BY INTENT & REGIME
        # =====================================================================

        if not intent_regime_summary.empty:
            intent_regime_export = intent_regime_summary.copy()
            numeric_cols = intent_regime_export.select_dtypes(include=['float64']).columns
            intent_regime_export[numeric_cols] = intent_regime_export[numeric_cols].round(4)
            intent_regime_export.to_excel(writer, sheet_name='By Intent & Regime', index=False)
            _format_sheet(writer.sheets['By Intent & Regime'], header_fill, header_font)
            print(f"  ✓ Tab 9: By Intent & Regime ({len(intent_regime_export)} combinations)")

        # =====================================================================
        # TAB 10: BY MONTH & REGIME
        # =====================================================================

        if not month_regime_summary.empty:
            month_regime_export = month_regime_summary.copy()
            numeric_cols = month_regime_export.select_dtypes(include=['float64']).columns
            month_regime_export[numeric_cols] = month_regime_export[numeric_cols].round(4)
            month_regime_export.to_excel(writer, sheet_name='By Month & Regime', index=False)
            _format_sheet(writer.sheets['By Month & Regime'], header_fill, header_font)
            print(f"  ✓ Tab 10: By Month & Regime ({len(month_regime_export)} combinations)")

        # =====================================================================
        # TAB 11: DAILY TIME SERIES (ALL) - WITH SLICERS
        # =====================================================================

        print(f"  ⏳ Tab 11: Compiling daily time series...")

        all_daily = []
        for idx, result in enumerate(results_list, 1):
            daily = result['daily_performance'].copy()

            # Add iteration metadata
            daily.insert(0, 'iteration', idx)
            daily.insert(1, 'launch_month', result['launch_month'])
            daily.insert(2, 'trigger_type', result['trigger_type'])
            daily.insert(3, 'trigger_params', str(result['trigger_params']))
            daily.insert(4, 'selection_algo', result['selection_algo'])
            daily.insert(5, 'strategy_intent', result.get('strategy_intent', 'neutral'))

            # Add regime if available
            if df_regimes is not None:
                daily = daily.merge(df_regimes[['Date', 'Regime']], on='Date', how='left')
                daily['Regime'] = daily['Regime'].fillna('unknown')
                # Move Regime column
                cols = list(daily.columns)
                cols.insert(6, cols.pop(cols.index('Regime')))
                daily = daily[cols]

            # Add trade indicator
            daily['Trade_Occurred'] = False
            if not result['trade_history'].empty:
                trade_dates = set(result['trade_history']['Date'])
                daily['Trade_Occurred'] = daily['Date'].isin(trade_dates)

            # Calculate daily returns
            daily['Strategy_Daily_Return'] = daily['Strategy_NAV'].pct_change()
            daily['SPY_Daily_Return'] = daily['SPY_NAV'].pct_change()
            daily['BUFR_Daily_Return'] = daily['BUFR_NAV'].pct_change()
            daily['Hold_Daily_Return'] = daily['Hold_NAV'].pct_change()

            # Calculate relative performance
            daily['Strategy_vs_SPY'] = (daily['Strategy_NAV'] / daily['SPY_NAV']) * 100
            daily['Strategy_vs_BUFR'] = (daily['Strategy_NAV'] / daily['BUFR_NAV']) * 100
            daily['Strategy_vs_Hold'] = (daily['Strategy_NAV'] / daily['Hold_NAV']) * 100

            all_daily.append(daily)

        combined_daily = pd.concat(all_daily, ignore_index=True)

        # Round numeric columns
        numeric_cols = combined_daily.select_dtypes(include=['float64']).columns
        combined_daily[numeric_cols] = combined_daily[numeric_cols].round(6)

        # Write to Excel
        combined_daily.to_excel(writer, sheet_name='Daily Time Series', index=False, startrow=0)

        # Format as Excel Table with filters
        worksheet = writer.sheets['Daily Time Series']
        _format_sheet(worksheet, header_fill, header_font)

        # Convert to Excel Table for filtering
        from openpyxl.worksheet.table import Table, TableStyleInfo

        tab = Table(displayName="TimeSeriesTable", ref=f"A1:{chr(65 + len(combined_daily.columns) - 1)}{len(combined_daily) + 1}")
        style = TableStyleInfo(
            name="TableStyleMedium9",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False
        )
        tab.tableStyleInfo = style
        worksheet.add_table(tab)

        print(f"  ✓ Tab 11: Daily Time Series ({len(combined_daily)} rows)")
        print(f"    Filter dropdowns enabled on all columns")

    print(f"\n✅ Workbook created: {filepath}")
    print(f"   Total tabs: 11")

    return filepath


def _format_sheet(worksheet, header_fill, header_font):
    """
    Apply consistent formatting to a worksheet.
    """
    # Format header row
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