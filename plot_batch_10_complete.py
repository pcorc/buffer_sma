
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
import os


def main():
    print("=" * 80)
    print("BATCH 10 PLOTTER - COMBINED CANVAS")
    print("=" * 80)

    # Show current directory
    print(f"\nCurrent directory: {os.getcwd()}")

    # Use relative paths
    regime_file = Path('data/sp500_regimes.csv')
    csv_file = Path('output/backtest_results/batch_10/batch_10_daily_time_series.csv')
    output_dir = Path('output/backtest_results/batch_10')

    # Check files exist
    print(f"\nChecking files:")
    print(f"  Regime file exists: {regime_file.exists()}")
    print(f"  CSV file exists: {csv_file.exists()}")

    if not regime_file.exists():
        print(f"\n❌ ERROR: Regime file not found at: {regime_file.absolute()}")
        return

    if not csv_file.exists():
        print(f"\n❌ ERROR: CSV file not found at: {csv_file.absolute()}")
        print(f"\n   Searching for the file...")
        base = Path('output')
        if base.exists():
            for file in base.rglob('batch_10_daily_time_series.csv'):
                print(f"   Found: {file}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load regime data
    print(f"\nLoading regime data...")
    df_regimes = pd.read_csv(regime_file)
    df_regimes.columns = df_regimes.columns.str.strip()
    df_regimes['Date'] = pd.to_datetime(df_regimes['Date'])
    df_regimes = df_regimes[['Date', 'Regimes']].copy()
    df_regimes = df_regimes[df_regimes['Regimes'].notna()].copy()
    df_regimes['Regimes'] = df_regimes['Regimes'].astype(int)
    print(f"✅ Loaded {len(df_regimes)} regime rows")

    # Load CSV
    print(f"\nLoading CSV data...")
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Find months
    print(f"\nDetecting months...")
    months = []
    for col in df.columns:
        if col.endswith('_New_ECR_Composite_NAV'):
            month = col.split('_')[0]
            if month not in months:
                months.append(month)

    months = sorted(months)
    has_existing_90 = any('Existing_90' in col for col in df.columns)

    print(f"✅ Found {len(months)} months: {', '.join(months)}")
    print(f"✅ Existing 90% available: {has_existing_90}")

    if len(months) == 0:
        print(f"\n❌ ERROR: No months detected in CSV!")
        return

    # Create BOTH combined plots AND individual plots
    print(f"\n{'=' * 80}")
    print(f"CREATING COMBINED CANVAS PLOTS")
    print(f"{'=' * 80}")

    # Strategy: Split into groups of 6 for optimal viewing
    if len(months) <= 6:
        # Single plot with all months
        create_combined_canvas(df, df_regimes, months, output_dir, has_existing_90,
                               plot_name='batch_10_combined_all')
    else:
        # Split into two plots (first 6, rest)
        mid = 6
        create_combined_canvas(df, df_regimes, months[:mid], output_dir, has_existing_90,
                               plot_name='batch_10_combined_part1')
        create_combined_canvas(df, df_regimes, months[mid:], output_dir, has_existing_90,
                               plot_name='batch_10_combined_part2')

    # Create INDIVIDUAL plots for each month
    print(f"\n{'=' * 80}")
    print(f"CREATING INDIVIDUAL PLOTS")
    print(f"{'=' * 80}")

    for month in months:
        create_individual_plot(df, df_regimes, month, output_dir, has_existing_90)

    print(f"\n{'=' * 80}")
    print(f"✅ COMPLETE!")
    print(f"{'=' * 80}")
    print(f"\nOutputs in: {output_dir.absolute()}")

    print(f"\nCombined plots:")
    if len(months) <= 6:
        print(f"  - batch_10_combined_all.png ({len(months)} months)")
    else:
        print(f"  - batch_10_combined_part1.png (first 6 months)")
        print(f"  - batch_10_combined_part2.png (remaining {len(months) - 6} months)")

    print(f"\nIndividual plots:")
    for month in months:
        print(f"  - batch_10_regime_shaded_{month}.png")


def create_combined_canvas(df, df_regimes, months, output_dir, has_existing_90, plot_name):
    """Create one combined canvas with multiple months."""
    n_months = len(months)
    n_cols = 3
    n_rows = 2  # Always use 2 rows for consistency

    print(f"\n  Creating {plot_name} with {n_months} months ({n_rows}x{n_cols} grid)...")

    # Create figure - larger for better readability
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 16))
    axes = axes.flatten()

    for idx, month in enumerate(months):
        ax = axes[idx]
        plot_month_subplot(ax, df, df_regimes, month, has_existing_90)

    # Hide unused subplots
    for idx in range(n_months, len(axes)):
        axes[idx].axis('off')

    # Title
    month_list = ', '.join(months)
    fig.suptitle(f'Batch 10: New ECR vs Existing 90%\n{month_list}\n'
                 'Green=Bullish | Gray=Neutral | Red=Bearish',
                 fontsize=18, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    plot_path = output_dir / f'{plot_name}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Saved: {plot_path.name}")


def create_individual_plot(df, df_regimes, month, output_dir, has_existing_90):
    """Create individual full-size plot for one month."""
    print(f"  Creating {month}...", end=' ')

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_month_subplot(ax, df, df_regimes, month, has_existing_90, full_size=True)

    plt.tight_layout()

    plot_path = output_dir / f'batch_10_regime_shaded_{month}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅")


def plot_month_subplot(ax, df, df_regimes, month, has_existing_90, full_size=False):
    """Plot one month on an axis with performance metrics."""
    ecr_col = f'{month}_New_ECR_Composite_NAV'
    month_start = df[df[ecr_col].notna()]['Date'].min()

    if pd.isna(month_start):
        ax.text(0.5, 0.5, f'No data for {month}',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{month} - No Data', fontweight='bold', fontsize=12)
        return

    # Filter to this month's data
    df_month = df[df['Date'] >= month_start].copy()
    df_merged = df_month.merge(df_regimes, on='Date', how='left')
    df_merged['Regimes'] = df_merged['Regimes'].ffill()

    # Calculate performance metrics
    ecr_return_col = f'{month}_New_ECR_Composite_Return'
    existing_return_col = f'{month}_Existing_90_Return'

    ecr_returns = df_merged[ecr_return_col].dropna() if ecr_return_col in df_merged.columns else pd.Series()
    ecr_total_return = ((1 + ecr_returns).prod() - 1) * 100 if len(ecr_returns) > 0 else 0
    ecr_sharpe = (ecr_returns.mean() / ecr_returns.std()) * np.sqrt(252) if len(ecr_returns) > 0 and ecr_returns.std() > 0 else 0

    existing_total_return = 0
    if has_existing_90 and existing_return_col in df_merged.columns:
        existing_returns = df_merged[existing_return_col].dropna()
        existing_total_return = ((1 + existing_returns).prod() - 1) * 100 if len(existing_returns) > 0 else 0

    bufr_navs = df_merged['BUFR_NAV'].dropna()
    bufr_return = 0
    if len(bufr_navs) > 1:
        bufr_return = ((bufr_navs.iloc[-1] - bufr_navs.iloc[0]) / bufr_navs.iloc[0]) * 100

    # Add regime shading
    regime_colors = {-1: ('lightcoral', 'Bearish'),
                     0: ('lightgray', 'Neutral'),
                     1: ('lightgreen', 'Bullish')}

    current_regime = None
    start_date = None

    for _, row in df_merged.iterrows():
        regime = row['Regimes']
        date = row['Date']

        if pd.isna(regime):
            continue

        if regime != current_regime:
            if current_regime is not None and start_date is not None:
                color, _ = regime_colors.get(current_regime, ('white', 'Unknown'))
                ax.axvspan(start_date, date, alpha=0.15, color=color, zorder=0)
            current_regime = regime
            start_date = date

    if current_regime is not None and start_date is not None:
        color, _ = regime_colors.get(current_regime, ('white', 'Unknown'))
        ax.axvspan(start_date, df_merged['Date'].max(), alpha=0.15, color=color, zorder=0)

    # Plot strategies
    if ecr_col in df_merged.columns and df_merged[ecr_col].notna().any():
        ax.plot(df_merged['Date'], df_merged[ecr_col],
                label='ECR', color='darkblue', linestyle='-', linewidth=2.5, alpha=1.0, zorder=2)

    if has_existing_90:
        existing_col = f'{month}_Existing_90_NAV'
        if existing_col in df_merged.columns and df_merged[existing_col].notna().any():
            ax.plot(df_merged['Date'], df_merged[existing_col],
                    label='E90', color='red', linestyle='--', linewidth=2.0, alpha=0.9, zorder=2)

    # Plot benchmarks (normalized to 100 at month start)
    for bench_col, color, style, label in [('SPY_NAV', 'black', '-', 'SPY'),
                                           ('BUFR_NAV', 'gray', '--', 'BUFR')]:
        if bench_col in df_merged.columns and df_merged[bench_col].notna().any():
            first = df_merged[bench_col].dropna().iloc[0]
            normalized = (df_merged[bench_col] / first) * 100
            ax.plot(df_merged['Date'], normalized, label=label, color=color,
                    linestyle=style, linewidth=1.5, alpha=0.6, zorder=2)

    # Formatting
    fontsize = 14 if full_size else 12
    ax.set_title(f'{month} (start: {month_start.strftime("%Y-%m-%d")})',
                 fontsize=fontsize, fontweight='bold')
    ax.set_xlabel('Date', fontsize=fontsize - 2)
    ax.set_ylabel('NAV (100 = Start)', fontsize=fontsize - 2)
    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend(loc='upper left', fontsize=fontsize - 3, framealpha=0.9)

    # Performance metrics box (upper right)
    perf_text = f'PERFORMANCE\n━━━━━━━━━━━━━━\nECR: {ecr_total_return:+.2f}%\n'
    if has_existing_90:
        diff = ecr_total_return - existing_total_return
        perf_text += f'E90: {existing_total_return:+.2f}%\nΔ:    {diff:+.2f}%\n'
    perf_text += f'BUFR: {bufr_return:+.2f}%\n━━━━━━━━━━━━━━\nSharpe: {ecr_sharpe:.2f}'

    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9,
                 edgecolor='black', linewidth=1.5)
    ax.text(0.98, 0.98, perf_text, transform=ax.transAxes, fontsize=fontsize - 3,
            verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace', weight='bold')

    # Format axes
    ax.tick_params(axis='x', rotation=45, labelsize=fontsize - 4)
    ax.tick_params(axis='y', labelsize=9)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))


if __name__ == "__main__":
    main()