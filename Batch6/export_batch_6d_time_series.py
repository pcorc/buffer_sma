"""
STANDALONE DAILY TIME SERIES EXPORTER
======================================

This script extracts daily NAV and returns for all strategies in a batch
and exports to CSV, even if comparison hasn't been run.

Usage:
    python export_batch_6d_time_series.py
    
Output:
    output/backtest_results/batch_6/batch_6d_daily_time_series.csv
"""

import pandas as pd
from pathlib import Path
import sys


def find_batch_excel(batch_number=6):
    """Find the batch results Excel file."""
    results_dir = Path('output/backtest_results')
    batch_dir = results_dir / f'batch_{batch_number}'
    
    if not batch_dir.exists():
        print(f"❌ Batch directory not found: {batch_dir}")
        return None
    
    excel_files = list(batch_dir.glob('*.xlsx'))
    
    if not excel_files:
        print(f"❌ No Excel results found in {batch_dir}")
        return None
    
    excel_path = excel_files[0]
    print(f"✅ Found batch results: {excel_path.name}")
    return excel_path


def load_all_nav_data(excel_path):
    """
    Load NAV data from all sheets in the Excel file.
    
    Returns:
        dict: {sheet_name: DataFrame with Date and NAV columns}
    """
    print(f"\n{'='*80}")
    print("LOADING NAV DATA FROM ALL SHEETS")
    print(f"{'='*80}")
    
    xl_file = pd.ExcelFile(excel_path)
    
    nav_data = {}
    
    for sheet_name in xl_file.sheet_names:
        # Skip non-strategy sheets
        if sheet_name in ['Summary', 'Regime_Analysis']:
            continue
        
        print(f"\nProcessing sheet: {sheet_name}")
        
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # Find Date column
            date_col = None
            for col in df.columns:
                if 'date' in col.lower():
                    date_col = col
                    break
            
            if date_col is None:
                print(f"  ⚠️  No Date column found, skipping")
                continue
            
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Find all NAV columns
            nav_cols = {}
            for col in df.columns:
                if 'nav' in col.lower():
                    col_lower = col.lower()
                    
                    if 'spy' in col_lower:
                        nav_cols['SPY'] = col
                    elif 'bufr' in col_lower:
                        nav_cols['BUFR'] = col
                    elif 'hold' not in col_lower and 'benchmark' not in col_lower:
                        # This is the strategy NAV
                        nav_cols['Strategy'] = col
            
            if not nav_cols:
                print(f"  ⚠️  No NAV columns found, skipping")
                continue
            
            # Extract relevant columns
            result_df = df[[date_col]].copy()
            result_df.rename(columns={date_col: 'Date'}, inplace=True)
            
            for label, col in nav_cols.items():
                result_df[label] = df[col]
            
            nav_data[sheet_name] = result_df
            print(f"  ✅ Loaded {len(result_df)} rows with columns: {list(nav_cols.keys())}")
            
        except Exception as e:
            print(f"  ❌ Error loading sheet: {e}")
            continue
    
    return nav_data


def merge_and_export(nav_data, output_path):
    """
    Merge all NAV data and export to CSV.
    
    Output columns:
    - Date
    - SPY_NAV, BUFR_NAV
    - Strategy1_NAV, Strategy2_NAV, ...
    - SPY_Return, BUFR_Return, Strategy1_Return, Strategy2_Return, ...
    """
    print(f"\n{'='*80}")
    print("MERGING AND EXPORTING TIME SERIES")
    print(f"{'='*80}\n")
    
    if not nav_data:
        print("❌ No data to export")
        return
    
    # Get all unique dates
    all_dates = set()
    for sheet_name, df in nav_data.items():
        all_dates.update(df['Date'].tolist())
    
    all_dates = sorted(list(all_dates))
    print(f"Date range: {min(all_dates)} to {max(all_dates)}")
    print(f"Total days: {len(all_dates)}")
    
    # Start with date column
    result_df = pd.DataFrame({'Date': all_dates})
    
    # Track benchmark columns (SPY/BUFR) - only include once
    benchmarks_added = {'SPY': False, 'BUFR': False}
    
    # Add each strategy
    strategy_count = 0
    
    for sheet_name, df in nav_data.items():
        print(f"\nProcessing: {sheet_name}")
        
        # Determine strategy name
        if 'enhanced_cost_ratio' in sheet_name.lower() or 'ecr' in sheet_name.lower():
            strategy_label = 'ECR_V2'
        elif 'cap_utilization' in sheet_name.lower() and '0.9' in sheet_name:
            strategy_label = 'Existing_90'
        else:
            strategy_count += 1
            strategy_label = f'Strategy{strategy_count}'
        
        # Merge benchmarks (once)
        if 'SPY' in df.columns and not benchmarks_added['SPY']:
            result_df = result_df.merge(
                df[['Date', 'SPY']].rename(columns={'SPY': 'SPY_NAV'}),
                on='Date', how='left'
            )
            benchmarks_added['SPY'] = True
            print(f"  ✅ Added SPY benchmark")
        
        if 'BUFR' in df.columns and not benchmarks_added['BUFR']:
            result_df = result_df.merge(
                df[['Date', 'BUFR']].rename(columns={'BUFR': 'BUFR_NAV'}),
                on='Date', how='left'
            )
            benchmarks_added['BUFR'] = True
            print(f"  ✅ Added BUFR benchmark")
        
        # Merge strategy
        if 'Strategy' in df.columns:
            result_df = result_df.merge(
                df[['Date', 'Strategy']].rename(columns={'Strategy': f'{strategy_label}_NAV'}),
                on='Date', how='left'
            )
            print(f"  ✅ Added {strategy_label} NAV")
    
    # Calculate daily returns
    print(f"\nCalculating daily returns...")
    for col in result_df.columns:
        if col.endswith('_NAV'):
            return_col = col.replace('_NAV', '_Return')
            result_df[return_col] = result_df[col].pct_change()
            print(f"  ✅ Calculated {return_col}")
    
    # Reorder columns: Date, NAVs, then Returns
    nav_cols = [c for c in result_df.columns if c.endswith('_NAV')]
    return_cols = [c for c in result_df.columns if c.endswith('_Return')]
    ordered_cols = ['Date'] + sorted(nav_cols) + sorted(return_cols)
    result_df = result_df[ordered_cols]
    
    # Export
    result_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print("EXPORT COMPLETE")
    print(f"{'='*80}")
    print(f"\n✅ CSV exported to: {output_path}")
    print(f"   Rows: {len(result_df):,}")
    print(f"   Columns: {len(result_df.columns)}")
    print(f"\nColumns included:")
    for i, col in enumerate(result_df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Show sample
    print(f"\n{'='*80}")
    print("SAMPLE DATA (First 5 rows)")
    print(f"{'='*80}")
    print(result_df.head().to_string(index=False))
    
    return result_df


def main():
    """Main export script."""
    print("="*80)
    print("BATCH 6D DAILY TIME SERIES EXPORTER")
    print("="*80)
    
    # Find batch results
    excel_path = find_batch_excel(batch_number=6)
    if excel_path is None:
        return
    
    # Load all NAV data
    nav_data = load_all_nav_data(excel_path)
    
    if not nav_data:
        print("\n❌ No strategy data found to export")
        return
    
    # Export
    output_path = excel_path.parent / 'batch_6d_daily_time_series.csv'
    result_df = merge_and_export(nav_data, output_path)
    
    print(f"\n{'='*80}")
    print("DONE! 🎉")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
