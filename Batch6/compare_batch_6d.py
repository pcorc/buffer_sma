"""
BATCH 6D COMPARISON VISUALIZATION (FIXED)
==========================================

This script loads Batch 6D results and creates:
1. Time series plot with 4 lines (SPY, BUFR, ECR, Existing)
2. Trade logs for ECR and Existing strategies
3. Summary statistics comparison table

Run after Batch 6D completes:
    python compare_batch_6d.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np


def inspect_excel_file(excel_path):
    """
    Inspect Excel file and print all sheet names for debugging.
    """
    print(f"\n{'='*80}")
    print(f"INSPECTING EXCEL FILE")
    print(f"{'='*80}")
    print(f"File: {excel_path}")

    xl_file = pd.ExcelFile(excel_path)
    print(f"\nFound {len(xl_file.sheet_names)} sheets:")
    for i, sheet in enumerate(xl_file.sheet_names, 1):
        print(f"  {i}. {sheet}")
    print(f"{'='*80}\n")

    return xl_file.sheet_names


def load_batch_6d_results(batch_number=6):
    """
    Load Batch 6D Excel results file.

    Returns:
        tuple: (excel_file_path, strategy_results_df, sheet_names)
    """
    results_dir = Path('output/backtest_results')
    batch_dir = results_dir / f'batch_{batch_number}'

    # Find the Excel file
    excel_files = list(batch_dir.glob('*.xlsx'))

    if not excel_files:
        raise FileNotFoundError(f"No Excel results found in {batch_dir}")

    excel_path = excel_files[0]
    print(f"Loading results from: {excel_path}")

    # Inspect file first
    sheet_names = inspect_excel_file(excel_path)

    # Load summary sheet if it exists
    summary_df = None
    if 'Summary' in sheet_names:
        summary_df = pd.read_excel(excel_path, sheet_name='Summary')

    return excel_path, summary_df, sheet_names


def find_strategy_sheets(sheet_names):
    """
    Find strategy sheets using multiple patterns.

    Returns:
        tuple: (ecr_sheet, existing_sheet, benchmark_sheet)
    """
    ecr_sheet = None
    existing_sheet = None
    benchmark_sheet = None

    # Try multiple patterns
    for sheet in sheet_names:
        sheet_lower = sheet.lower()

        # Skip non-strategy sheets
        if sheet in ['Summary', 'Regime_Analysis', 'Trade_Log']:
            continue

        # Look for ECR sheets
        if 'enhanced_cost_ratio' in sheet_lower or 'ecr' in sheet_lower:
            if 'v2' in sheet_lower or 'neutral' in sheet_lower:
                ecr_sheet = sheet
                print(f"  ✅ Found ECR sheet: {sheet}")

        # Look for Existing/Threshold sheets
        elif 'cap_or_buffer' in sheet_lower or 'threshold' in sheet_lower or 'utilization' in sheet_lower:
            existing_sheet = sheet
            print(f"  ✅ Found Existing sheet: {sheet}")

        # Use any strategy sheet for benchmarks
        if benchmark_sheet is None and sheet not in ['Summary', 'Regime_Analysis', 'Trade_Log']:
            benchmark_sheet = sheet

    # If still not found, try broader patterns
    if ecr_sheet is None or existing_sheet is None:
        print("\n⚠️  Trying broader search patterns...")

        for sheet in sheet_names:
            if sheet in ['Summary', 'Regime_Analysis', 'Trade_Log']:
                continue

            # First non-summary sheet = ECR
            if ecr_sheet is None:
                ecr_sheet = sheet
                print(f"  → Using as ECR: {sheet}")
            # Second non-summary sheet = Existing
            elif existing_sheet is None:
                existing_sheet = sheet
                print(f"  → Using as Existing: {sheet}")

            if ecr_sheet and existing_sheet:
                break

    return ecr_sheet, existing_sheet, benchmark_sheet


def load_daily_nav_data(excel_path, strategy_name):
    """
    Load daily NAV time series for a specific strategy.

    Parameters:
        excel_path: Path to Excel file
        strategy_name: Sheet name

    Returns:
        DataFrame with Date and NAV columns
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=strategy_name)

        # Find Date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break

        if date_col is None:
            print(f"  ⚠️  No Date column found in {strategy_name}")
            return None

        # Find NAV column
        nav_col = None
        for col in df.columns:
            if 'nav' in col.lower() and 'spy' not in col.lower() and 'bufr' not in col.lower():
                nav_col = col
                break

        if nav_col is None:
            print(f"  ⚠️  No NAV column found in {strategy_name}")
            return None

        df[date_col] = pd.to_datetime(df[date_col])
        df = df[[date_col, nav_col]].copy()
        df.columns = ['Date', 'NAV']

        # Normalize to 1.0 at start
        if df['NAV'].iloc[0] != 0:
            df['NAV'] = df['NAV'] / df['NAV'].iloc[0]

        return df

    except Exception as e:
        print(f"  ❌ Error loading {strategy_name}: {e}")
        return None


def load_benchmark_data(excel_path, sheet_names):
    """
    Load SPY and BUFR benchmark data from any strategy sheet.

    Returns:
        tuple: (spy_df, bufr_df)
    """
    # Find a strategy sheet
    strategy_sheet = None
    for sheet in sheet_names:
        if sheet not in ['Summary', 'Regime_Analysis', 'Trade_Log']:
            strategy_sheet = sheet
            break

    if strategy_sheet is None:
        print("  ⚠️  No strategy sheets found for benchmarks")
        return None, None

    print(f"  Loading benchmarks from: {strategy_sheet}")

    df = pd.read_excel(excel_path, sheet_name=strategy_sheet)

    # Find Date column
    date_col = None
    for col in df.columns:
        if 'date' in col.lower():
            date_col = col
            break

    if date_col is None:
        print("  ⚠️  No Date column found")
        return None, None

    df[date_col] = pd.to_datetime(df[date_col])

    # Extract SPY
    spy_col = None
    for col in df.columns:
        if 'spy' in col.lower() and 'nav' in col.lower():
            spy_col = col
            break

    spy_df = None
    if spy_col:
        spy_df = df[[date_col, spy_col]].copy()
        spy_df.columns = ['Date', 'NAV']
        if spy_df['NAV'].iloc[0] != 0:
            spy_df['NAV'] = spy_df['NAV'] / spy_df['NAV'].iloc[0]
        print(f"  ✅ SPY loaded: {len(spy_df)} days")
    else:
        print("  ⚠️  SPY column not found")

    # Extract BUFR
    bufr_col = None
    for col in df.columns:
        if 'bufr' in col.lower() and 'nav' in col.lower():
            bufr_col = col
            break

    bufr_df = None
    if bufr_col:
        bufr_df = df[[date_col, bufr_col]].copy()
        bufr_df.columns = ['Date', 'NAV']
        if bufr_df['NAV'].iloc[0] != 0:
            bufr_df['NAV'] = bufr_df['NAV'] / bufr_df['NAV'].iloc[0]
        print(f"  ✅ BUFR loaded: {len(bufr_df)} days")
    else:
        print("  ⚠️  BUFR column not found")

    return spy_df, bufr_df


def load_trade_logs(excel_path, sheet_names):
    """
    Load trade logs for both strategies.

    Returns:
        dict: {strategy_name: trade_log_df}
    """
    trade_log_sheets = [s for s in sheet_names if 'trade' in s.lower() and 'log' in s.lower()]

    trade_logs = {}

    for sheet in trade_log_sheets:
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet)
            strategy_name = sheet.replace('_Trade_Log', '').replace('_Trades', '')
            trade_logs[strategy_name] = df
            print(f"  ✅ Trade log loaded: {sheet}")
        except Exception as e:
            print(f"  ⚠️  Error loading trade log {sheet}: {e}")

    return trade_logs


def create_comparison_plot(spy_df, bufr_df, ecr_df, existing_df, output_path):
    """
    Create 4-line comparison plot.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot all 4 strategies
    if spy_df is not None and not spy_df.empty:
        ax.plot(spy_df['Date'], spy_df['NAV'], label='SPY (Buy & Hold)',
                linewidth=2, color='black', alpha=0.7)

    if bufr_df is not None and not bufr_df.empty:
        ax.plot(bufr_df['Date'], bufr_df['NAV'], label='BUFR (Benchmark)',
                linewidth=2, color='gray', alpha=0.7, linestyle='--')

    if ecr_df is not None and not ecr_df.empty:
        ax.plot(ecr_df['Date'], ecr_df['NAV'], label='ECR (Quarterly, Equal Weights)',
                linewidth=2.5, color='blue')

    if existing_df is not None and not existing_df.empty:
        ax.plot(existing_df['Date'], existing_df['NAV'], label='Existing (90% Threshold)',
                linewidth=2.5, color='red')

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized NAV (Starting = 1.0)', fontsize=12)
    ax.set_title('Batch 6D: ECR vs Existing Strategy Comparison\n(SEP Launch Month)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")

    return fig


def create_summary_table(summary_df):
    """
    Create formatted summary statistics table.
    """
    if summary_df is None or summary_df.empty:
        return "⚠️ No summary data available"

    table = []
    table.append("="*100)
    table.append("BATCH 6D COMPARISON - SUMMARY STATISTICS")
    table.append("="*100)
    table.append("")

    # Headers
    headers = f"{'Strategy':<50} {'Sharpe':<10} {'Return':<12} {'vs BUFR':<12} {'MaxDD':<12} {'Trades':<8}"
    table.append(headers)
    table.append("-"*100)

    for idx, row in summary_df.iterrows():
        strategy_name = str(row.get('strategy', 'Unknown'))

        # Simplify name
        if len(strategy_name) > 50:
            if 'enhanced_cost_ratio' in strategy_name.lower():
                display_name = "ECR (Quarterly, Equal Weights V2)"
            elif 'cap_or_buffer' in strategy_name.lower():
                display_name = "Existing (90% Cap OR Buffer)"
            else:
                display_name = strategy_name[:47] + "..."
        else:
            display_name = strategy_name

        sharpe = row.get('strategy_sharpe', 0)
        total_return = row.get('strategy_return', 0)
        vs_bufr = row.get('vs_bufr_excess', 0)
        max_dd = row.get('strategy_max_dd', 0)
        trades = row.get('num_trades', 0)

        line = (f"{display_name:<50} {sharpe:>6.2f}    "
                f"{total_return*100:>7.2f}%     "
                f"{vs_bufr*100:>+7.2f}%     "
                f"{max_dd*100:>7.2f}%     "
                f"{int(trades):<8}")
        table.append(line)

    table.append("="*100)

    return "\n".join(table)


def format_trade_log(trade_df, strategy_name):
    """
    Format trade log for display.
    """
    lines = []
    lines.append("="*100)
    lines.append(f"TRADE LOG: {strategy_name}")
    lines.append("="*100)
    lines.append("")

    if trade_df is None or trade_df.empty:
        lines.append("No trades executed (buy and hold)")
        lines.append("="*100)
        return "\n".join(lines)

    # Format header
    header = f"{'Date':<12} {'Action':<8} {'Fund':<10} {'NAV':<12} {'Reason':<50}"
    lines.append(header)
    lines.append("-"*100)

    # Format each trade
    for idx, row in trade_df.iterrows():
        date = pd.to_datetime(row['Date']).strftime('%Y-%m-%d') if 'Date' in row else 'N/A'
        action = row.get('Action', 'N/A')
        fund = row.get('Fund', 'N/A')
        nav = row.get('NAV', 0)
        reason = row.get('Reason', 'N/A')[:50]  # Truncate long reasons

        line = f"{date:<12} {action:<8} {fund:<10} {nav:>10.4f}  {reason:<50}"
        lines.append(line)

    lines.append("="*100)
    lines.append(f"Total Trades: {len(trade_df)}")
    lines.append("="*100)

    return "\n".join(lines)


def main():
    """
    Main comparison script.
    """
    print("="*80)
    print("BATCH 6D COMPARISON - ECR VS EXISTING")
    print("="*80)
    print()

    batch_number = 6

    # Load results
    try:
        excel_path, summary_df, sheet_names = load_batch_6d_results(batch_number)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you've run Batch 6D first:")
        print("  python run_batch_tests.py")
        return

    # Find strategy sheets
    print("\nFinding strategy sheets...")
    ecr_sheet, existing_sheet, benchmark_sheet = find_strategy_sheets(sheet_names)

    if not ecr_sheet:
        print("❌ Could not find ECR strategy sheet")
        return
    if not existing_sheet:
        print("❌ Could not find Existing strategy sheet")
        return

    # Load benchmark data
    print("\nLoading benchmark data...")
    spy_df, bufr_df = load_benchmark_data(excel_path, sheet_names)

    # Load strategy data
    print("\nLoading strategy data...")
    ecr_df = load_daily_nav_data(excel_path, ecr_sheet)
    if ecr_df is not None:
        print(f"  ✅ ECR loaded: {len(ecr_df)} days")

    existing_df = load_daily_nav_data(excel_path, existing_sheet)
    if existing_df is not None:
        print(f"  ✅ Existing loaded: {len(existing_df)} days")

    # Create plot
    print("\nCreating comparison plot...")
    output_dir = Path('output/backtest_results') / f'batch_{batch_number}'
    plot_path = output_dir / 'batch_6d_comparison_plot.png'

    create_comparison_plot(spy_df, bufr_df, ecr_df, existing_df, plot_path)

    # Print summary table
    if summary_df is not None:
        print("\n")
        print(create_summary_table(summary_df))
        print()

    # Load and print trade logs
    print("\nLoading trade logs...")
    trade_logs = load_trade_logs(excel_path, sheet_names)

    if trade_logs:
        for strategy_name, trade_df in trade_logs.items():
            print("\n")
            print(format_trade_log(trade_df, strategy_name))
            print()
    else:
        print("  ⚠️  No trade logs found")
    
    # Final summary
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\n📊 Plot saved to: {plot_path}")
    print(f"📁 Full results: {excel_path}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
