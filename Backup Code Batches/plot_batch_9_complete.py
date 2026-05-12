"""
BATCH 9 COMPLETE PLOTTER - FIXED-WEIGHT ECR BY REGIME
======================================================

Creates:
1. Time series with regime shading (Fixed Cap-Moderate + Existing 90% + benchmarks)
2. Bar chart: Performance by regime (Bullish/Neutral/Bearish)
3. Bar chart: Fixed ECR outperformance by regime
4. Comprehensive regime performance table (CSV + console)

Run: python plot_batch_9_complete.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np


def load_regime_data(regime_file_path='data/sp500_regimes.csv'):
    """Load S&P 500 regime classifications."""
    df_regimes = pd.read_csv(regime_file_path)
    df_regimes.columns = df_regimes.columns.str.strip()
    df_regimes['Date'] = pd.to_datetime(df_regimes['Date'])
    df_regimes = df_regimes[['Date', 'Regimes']].copy()
    df_regimes = df_regimes[df_regimes['Regimes'].notna()].copy()
    df_regimes['Regimes'] = df_regimes['Regimes'].astype(int)
    return df_regimes


def load_csv(csv_path):
    """Load performance CSV and detect available strategies."""
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Find months
    months = set()
    for col in df.columns:
        if col.startswith((
                'JAN_', 'FEB_', 'MAR_', 'APR_', 'MAY_', 'JUN_',
                'JUL_', 'AUG_', 'SEP_', 'OCT_', 'NOV_', 'DEC_'
        )):
            month = col.split('_')[0]
            months.add(month)
    
    # Detect which strategies are actually available
    print(f"\n{'='*80}")
    print(f"DETECTING AVAILABLE STRATEGIES")
    print(f"{'='*80}")
    
    available_strategies = {
        'cap_moderate': False,  # ← Looking for Fixed Cap-Moderate
        'Existing_90': False
    }
    
    # Check for each strategy type
    for col in df.columns:
        if 'cap_moderate' in col or 'cap_mod' in col or 'ECR_V2_cap_moderate' in col:
            available_strategies['cap_moderate'] = True
        if 'Existing_90' in col or ('Existing' in col and '90' in col):
            available_strategies['Existing_90'] = True
    
    print(f"\nStrategies found in CSV:")
    print(f"  {'Fixed Cap-Moderate ECR:':<30} {'✅ YES' if available_strategies['cap_moderate'] else '❌ NO'}")
    print(f"  {'Existing 90% Threshold:':<30} {'✅ YES' if available_strategies['Existing_90'] else '❌ NO'}")
    
    if not available_strategies['cap_moderate'] or not available_strategies['Existing_90']:
        print(f"\n⚠️  WARNING: Some strategies are missing.")
        print(f"   Run complete Batch 9 for all strategies.")
    
    print(f"{'='*80}\n")
    
    return df, sorted(months), available_strategies


def find_strategy_column(df, month, strategy_keywords):
    """Find strategy column by searching for keywords."""
    for keyword in strategy_keywords:
        for col in df.columns:
            if col.startswith(f'{month}_') and keyword in col and col.endswith('_NAV'):
                return col
    return None


def create_regime_shaded_plot(df, df_regimes, month, output_dir, available_strategies):
    """Create time series with regime shading."""
    print(f"Creating plot for {month}...", end=' ')
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Merge regime data
    df_merged = df.merge(df_regimes, on='Date', how='left')
    df_merged['Regimes'] = df_merged['Regimes'].ffill()
    
    # Add regime shading
    regime_colors = {-1: ('lightcoral', 'Bearish'), 0: ('lightgray', 'Neutral'), 1: ('lightgreen', 'Bullish')}
    
    current_regime = None
    start_date = None
    
    for idx, row in df_merged.iterrows():
        regime = row['Regimes']
        date = row['Date']
        
        if pd.isna(regime):
            continue
        
        if regime != current_regime:
            if current_regime is not None and start_date is not None:
                color, label = regime_colors.get(current_regime, ('white', 'Unknown'))
                ax.axvspan(start_date, date, alpha=0.15, color=color, zorder=0)
            current_regime = regime
            start_date = date
    
    if current_regime is not None and start_date is not None:
        color, label = regime_colors.get(current_regime, ('white', 'Unknown'))
        ax.axvspan(start_date, df_merged['Date'].max(), alpha=0.15, color=color, zorder=0)
    
    # Plot only available strategies
    strategies_to_plot = []
    
    if available_strategies['cap_moderate']:
        strategies_to_plot.append((
            ['cap_moderate', 'cap_mod', 'ECR_V2_cap_moderate'], 
            'darkblue', '-', 3.5, 'Fixed Cap-Moderate ECR'
        ))
    
    if available_strategies['Existing_90']:
        strategies_to_plot.append((
            ['Existing_90', 'Existing', '90'], 
            'red', '--', 2.5, 'Existing 90%'
        ))
    
    # Plot strategies
    for keywords, color, style, width, label in strategies_to_plot:
        col = find_strategy_column(df_merged, month, keywords)
        
        if col and df_merged[col].notna().any():
            first_valid = df_merged[col].dropna().iloc[0]
            normalized = df_merged[col] / first_valid
            ax.plot(df_merged['Date'], normalized, label=label, color=color,
                   linestyle=style, linewidth=width, alpha=1.0, zorder=2)
    
    # Benchmarks
    for bench_col, color, style, label in [
        ('SPY_NAV', 'black', '-', 'SPY'),
        ('BUFR_NAV', 'gray', '--', 'BUFR')
    ]:
        if bench_col in df_merged.columns and df_merged[bench_col].notna().any():
            first_valid = df_merged[bench_col].dropna().iloc[0]
            normalized = df_merged[bench_col] / first_valid
            ax.plot(df_merged['Date'], normalized, label=label, color=color,
                   linestyle=style, linewidth=2.0, alpha=0.7, zorder=2)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized NAV (Starting = 1.0)', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Batch 9: Fixed-Weight ECR Performance by Regime\n{month} Launch Month\n'
        f'Green=Bullish | Gray=Neutral | Red=Bearish', 
        fontsize=16, fontweight='bold'
    )
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    regime_patches = [
        mpatches.Patch(color='lightgreen', alpha=0.3, label='Bullish Regime'),
        mpatches.Patch(color='lightgray', alpha=0.3, label='Neutral Regime'),
        mpatches.Patch(color='lightcoral', alpha=0.3, label='Bearish Regime')
    ]
    handles = handles + regime_patches
    ax.legend(handles=handles, loc='upper left', fontsize=11, framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, zorder=1)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = output_dir / f'batch_9_regime_shaded_{month}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved")


def calculate_regime_stats(df, df_regimes, months):
    """Calculate performance stats by regime."""
    print(f"\nCalculating regime statistics...")
    
    df_merged = df.merge(df_regimes, on='Date', how='left')
    df_merged['Regimes'] = df_merged['Regimes'].ffill()
    
    results = {
        'cap_moderate': {-1: [], 0: [], 1: []},  # ← Changed from 'Regime_Adaptive'
        'Existing_90': {-1: [], 0: [], 1: []},
        'BUFR': {-1: [], 0: [], 1: []}
    }
    
    for month in months:
        # Find return columns
        ecr_col = find_strategy_column(df_merged, month, ['cap_moderate', 'cap_mod', 'ECR_V2_cap_moderate'])
        ecr_col = ecr_col.replace('_NAV', '_Return') if ecr_col else None
        
        existing_col = find_strategy_column(df_merged, month, ['Existing_90', 'Existing', '90'])
        existing_col = existing_col.replace('_NAV', '_Return') if existing_col else None
        
        bufr_col = 'BUFR_Return' if 'BUFR_Return' in df_merged.columns else None
        
        # Calculate by regime
        for regime in [-1, 0, 1]:
            regime_mask = df_merged['Regimes'] == regime
            
            if ecr_col and ecr_col in df_merged.columns:
                returns = df_merged.loc[regime_mask, ecr_col].dropna()
                if len(returns) > 0:
                    results['cap_moderate'][regime].append({
                        'month': month,
                        'total_return': (1 + returns).prod() - 1,
                        'days': len(returns)
                    })
            
            if existing_col and existing_col in df_merged.columns:
                returns = df_merged.loc[regime_mask, existing_col].dropna()
                if len(returns) > 0:
                    results['Existing_90'][regime].append({
                        'month': month,
                        'total_return': (1 + returns).prod() - 1,
                        'days': len(returns)
                    })
            
            if bufr_col:
                returns = df_merged.loc[regime_mask, bufr_col].dropna()
                if len(returns) > 0:
                    results['BUFR'][regime].append({
                        'month': month,
                        'total_return': (1 + returns).prod() - 1,
                        'days': len(returns)
                    })
    
    return results


def create_bar_charts_and_table(regime_stats, output_dir):
    """Create bar charts and print table."""
    print(f"\nCreating bar charts and table...")
    
    regime_labels = {-1: 'Bearish', 0: 'Neutral', 1: 'Bullish'}
    
    # Prepare data
    data_by_regime = []
    
    for regime in [1, 0, -1]:
        ecr_stats = regime_stats['cap_moderate'].get(regime, [])  # ← Changed
        existing_stats = regime_stats['Existing_90'].get(regime, [])
        bufr_stats = regime_stats['BUFR'].get(regime, [])
        
        if not ecr_stats:
            continue
        
        ecr_avg = np.mean([s['total_return'] for s in ecr_stats]) * 100
        existing_avg = np.mean([s['total_return'] for s in existing_stats]) * 100 if existing_stats else 0
        bufr_avg = np.mean([s['total_return'] for s in bufr_stats]) * 100 if bufr_stats else 0
        
        # Win counts
        ecr_wins_existing = sum(1 for es in ecr_stats 
                               if any(ex['month'] == es['month'] and es['total_return'] > ex['total_return'] 
                                      for ex in existing_stats))
        ecr_wins_bufr = sum(1 for es in ecr_stats 
                           if any(b['month'] == es['month'] and es['total_return'] > b['total_return'] 
                                  for b in bufr_stats))
        
        data_by_regime.append({
            'regime': regime_labels[regime],
            'regime_code': regime,
            'ecr_avg': ecr_avg,
            'existing_avg': existing_avg,
            'bufr_avg': bufr_avg,
            'ecr_vs_existing': ecr_avg - existing_avg,
            'ecr_vs_bufr': ecr_avg - bufr_avg,
            'ecr_wins_existing': ecr_wins_existing,
            'total_months': len(ecr_stats),
            'ecr_wins_bufr': ecr_wins_bufr,
            'days': sum([s['days'] for s in ecr_stats])
        })
    
    # PRINT TABLE
    print(f"\n{'='*130}")
    print(f"REGIME PERFORMANCE ANALYSIS TABLE - BATCH 9 (FIXED WEIGHTS)")
    print(f"{'='*130}\n")
    
    print(f"{'Regime':<12} | {'ECR Avg':<10} | {'Exist90 Avg':<12} | {'BUFR Avg':<10} | "
          f"{'ECR vs E90':<12} | {'ECR vs BUFR':<12} | {'ECR Wins E90':<13} | {'ECR Wins BUFR':<14} | {'Days':<6}")
    print("-" * 130)
    
    for d in data_by_regime:
        print(f"{d['regime']:<12} | "
              f"{d['ecr_avg']:>8.2f}%  | "
              f"{d['existing_avg']:>10.2f}%  | "
              f"{d['bufr_avg']:>8.2f}%  | "
              f"{d['ecr_vs_existing']:>+9.2f}%  | "
              f"{d['ecr_vs_bufr']:>+9.2f}%  | "
              f"{d['ecr_wins_existing']}/{d['total_months']:<10} | "
              f"{d['ecr_wins_bufr']}/{d['total_months']:<11} | "
              f"{d['days']:<6}")
    
    print("=" * 130)
    
    # Summary stats
    if data_by_regime:
        overall_ecr = np.mean([d['ecr_avg'] for d in data_by_regime])
        overall_existing = np.mean([d['existing_avg'] for d in data_by_regime])
        overall_bufr = np.mean([d['bufr_avg'] for d in data_by_regime])
        
        total_wins_e90 = sum([d['ecr_wins_existing'] for d in data_by_regime])
        total_opps_e90 = sum([d['total_months'] for d in data_by_regime])
        total_wins_bufr = sum([d['ecr_wins_bufr'] for d in data_by_regime])
        
        print(f"\nOVERALL SUMMARY:")
        print(f"  Average Performance:    ECR: {overall_ecr:>7.2f}% | Existing 90%: {overall_existing:>7.2f}% | BUFR: {overall_bufr:>7.2f}%")
        print(f"  Performance Diff:       ECR vs E90: {overall_ecr - overall_existing:>+7.2f}% | ECR vs BUFR: {overall_ecr - overall_bufr:>+7.2f}%")
        if total_opps_e90 > 0:
            print(f"  Win Rates (all regimes): vs E90: {total_wins_e90}/{total_opps_e90} ({total_wins_e90/total_opps_e90*100:.1f}%) | vs BUFR: {total_wins_bufr}/{total_opps_e90} ({total_wins_bufr/total_opps_e90*100:.1f}%)")
        print(f"\n{'='*130}\n")
        
        # Save CSV
        df_table = pd.DataFrame(data_by_regime)
        csv_path = output_dir / 'batch_9_regime_performance_table.csv'
        df_table.to_csv(csv_path, index=False)
        print(f"✅ Table saved: {csv_path.name}\n")
        
        # CREATE BAR CHARTS
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # LEFT: Performance by regime
        regimes = [d['regime'] for d in data_by_regime]
        x = np.arange(len(regimes))
        width = 0.25
        
        ecr_values = [d['ecr_avg'] for d in data_by_regime]
        existing_values = [d['existing_avg'] for d in data_by_regime]
        bufr_values = [d['bufr_avg'] for d in data_by_regime]
        
        ax1.bar(x - width, ecr_values, width, label='Fixed Cap-Moderate ECR', color='darkblue', alpha=0.8)
        ax1.bar(x, existing_values, width, label='Existing 90%', color='red', alpha=0.8)
        ax1.bar(x + width, bufr_values, width, label='BUFR', color='gray', alpha=0.8)
        
        ax1.set_ylabel('Average Return (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Fixed-Weight ECR Performance During Different Market Regimes', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(regimes, fontsize=11)
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # RIGHT: Outperformance
        vs_existing = [d['ecr_vs_existing'] for d in data_by_regime]
        vs_bufr = [d['ecr_vs_bufr'] for d in data_by_regime]
        
        ax2.bar(x - width/2, vs_existing, width, label='ECR vs Existing 90%', color='orange', alpha=0.8)
        ax2.bar(x + width/2, vs_bufr, width, label='ECR vs BUFR', color='green', alpha=0.8)
        
        ax2.set_ylabel('Outperformance (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Fixed-Weight ECR Outperformance by Regime', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(regimes, fontsize=11)
        ax2.legend(loc='upper left', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        
        plot_path = output_dir / 'batch_9_regime_bar_charts.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Bar charts saved: {plot_path.name}")


def main():
    """Main execution."""
    print("="*80)
    print("BATCH 9 COMPLETE PLOTTER")
    print("="*80)
    
    # Load data
    regime_file = Path('data/sp500_regimes.csv')
    df_regimes = load_regime_data(regime_file)
    
    csv_path = Path('output/backtest_results/batch_9/batch_9_daily_time_series.csv')
    df, months, available_strategies = load_csv(csv_path)
    
    output_dir = csv_path.parent
    
    # Time series plots
    print(f"Creating {len(months)} time series plots...")
    for month in months:
        create_regime_shaded_plot(df, df_regimes, month, output_dir, available_strategies)
    
    # Regime stats
    regime_stats = calculate_regime_stats(df, df_regimes, months)
    
    # Bar charts and table
    create_bar_charts_and_table(regime_stats, output_dir)
    
    print(f"\n{'='*80}")
    print(f"COMPLETE!")
    print(f"{'='*80}")
    print(f"\nGenerated:")
    print(f"  - {len(months)} time series plots")
    print(f"  - 1 bar chart comparison")
    print(f"  - 1 CSV table")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
