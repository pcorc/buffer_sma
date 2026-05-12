"""
SIMPLE BATCH 6D PLOTTER - Uses CSV Directly
============================================

This script loads the already-created CSV file and generates plots.
No need to parse Excel file.

Run after batch completes:
    python plot_batch_6d_from_csv.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def load_time_series_csv(csv_path):
    """Load the time series CSV."""
    print(f"Loading CSV: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"  ✅ Loaded {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")
    
    return df


def create_comparison_plot(df, output_path):
    """Create 4-line comparison plot."""
    print(f"\nCreating comparison plot...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Normalize all NAV columns to start at 1.0
    nav_cols = [c for c in df.columns if c.endswith('_NAV')]
    
    for col in nav_cols:
        if df[col].notna().any():
            first_valid = df[col].dropna().iloc[0]
            df[f'{col}_normalized'] = df[col] / first_valid
    
    # Plot each strategy
    colors = {
        'SPY': ('black', 'SPY (Buy & Hold)', '-', 2.0, 0.7),
        'BUFR': ('gray', 'BUFR (Benchmark)', '--', 2.0, 0.7),
        'ECR_V2': ('blue', 'ECR V2 (Quarterly, Equal Weights)', '-', 2.5, 1.0),
        'Existing_90': ('red', 'Existing (90% Cap Threshold)', '-', 2.5, 1.0)
    }
    
    for prefix, (color, label, style, width, alpha) in colors.items():
        col_name = f'{prefix}_NAV_normalized'
        if col_name in df.columns:
            ax.plot(df['Date'], df[col_name], 
                   label=label, color=color, linestyle=style, 
                   linewidth=width, alpha=alpha)
            print(f"  ✅ Plotted: {label}")
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized NAV (Starting = 1.0)', fontsize=12)
    ax.set_title('Batch 6D: ECR V2 vs Existing (90% Threshold) Comparison\nSEP Launch Month', 
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
    
    plt.close()


def calculate_summary_stats(df):
    """Calculate summary statistics from the CSV."""
    print(f"\n{'='*100}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*100}\n")
    
    # Find all strategy NAV columns (exclude benchmarks for separate display)
    strategy_cols = [c for c in df.columns if c.endswith('_NAV') and c not in ['SPY_NAV', 'BUFR_NAV']]
    benchmark_cols = [c for c in df.columns if c.endswith('_NAV') and c in ['SPY_NAV', 'BUFR_NAV']]
    
    stats = []
    
    # Calculate for each column
    for col in strategy_cols + benchmark_cols:
        if col not in df.columns or df[col].isna().all():
            continue
        
        return_col = col.replace('_NAV', '_Return')
        
        if return_col not in df.columns:
            continue
        
        returns = df[return_col].dropna()
        nav = df[col].dropna()
        
        if len(returns) == 0 or len(nav) == 0:
            continue
        
        # Calculate metrics
        total_return = (nav.iloc[-1] / nav.iloc[0]) - 1
        ann_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * (252 ** 0.5)
        sharpe = ann_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        strategy_name = col.replace('_NAV', '')
        
        stats.append({
            'Strategy': strategy_name,
            'Total Return': total_return,
            'Ann Return': ann_return,
            'Volatility': volatility,
            'Sharpe': sharpe,
            'Max DD': max_dd
        })
    
    # Print table
    print(f"{'Strategy':<20} {'Total Return':<15} {'Ann Return':<15} {'Sharpe':<10} {'Max DD':<12}")
    print("-"*100)
    
    for stat in stats:
        print(f"{stat['Strategy']:<20} "
              f"{stat['Total Return']*100:>10.2f}%    "
              f"{stat['Ann Return']*100:>10.2f}%    "
              f"{stat['Sharpe']:>6.2f}    "
              f"{stat['Max DD']*100:>7.2f}%")
    
    print("="*100)
    
    # Calculate vs BUFR
    if any(s['Strategy'] == 'BUFR' for s in stats):
        bufr_return = next(s['Total Return'] for s in stats if s['Strategy'] == 'BUFR')
        
        print(f"\nVS BUFR EXCESS:")
        print("-"*100)
        for stat in stats:
            if stat['Strategy'] not in ['SPY', 'BUFR']:
                excess = stat['Total Return'] - bufr_return
                print(f"  {stat['Strategy']:<20} {excess*100:>+7.2f}%")
    
    print("="*100)
    
    return stats


def show_sample_data(df):
    """Show sample of the data."""
    print(f"\n{'='*100}")
    print(f"SAMPLE DATA (first 10 rows)")
    print(f"{'='*100}\n")
    
    display_cols = ['Date'] + [c for c in df.columns if c.endswith('_NAV')][:4]
    print(df[display_cols].head(10).to_string(index=False))


def main():
    """Main plotting script."""
    print("="*80)
    print("BATCH 6D PLOTTER - FROM CSV")
    print("="*80)
    print()
    
    # Find CSV file
    csv_path = Path('output/backtest_results/batch_6/batch_6_daily_time_series.csv')
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        print("\nMake sure you've run the batch first:")
        print("  python run_batch_tests.py")
        return
    
    # Load data
    df = load_time_series_csv(csv_path)
    
    # Show sample
    show_sample_data(df)
    
    # Calculate stats
    stats = calculate_summary_stats(df)
    
    # Create plot
    plot_path = csv_path.parent / 'batch_6d_comparison_plot.png'
    create_comparison_plot(df, plot_path)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📊 Plot: {plot_path}")
    print(f"📈 Data: {csv_path}")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
