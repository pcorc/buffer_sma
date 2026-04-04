import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
import os

# Color scheme for different weight variations
WEIGHT_COLORS = {
    '111': ('darkblue', 'solid', 'Equal (1,1,1)'),
    '211': ('blue', 'dashed', 'DBB Heavy (2,1,1)'),
    '121': ('green', 'dashed', 'Buffer Heavy (1,2,1)'),
    '112': ('purple', 'dashed', 'Cap Heavy (1,1,2)'),
    '101': ('orange', 'dotted', 'DBB+Cap (1,0,1)'),
    '221': ('cyan', 'dashdot', 'Buffer Equal (2,2,1)'),
    '110': ('magenta', 'dotted', 'Buffer Only (1,1,0)'),
}


def main():
    print("=" * 80)
    print("BATCH 10 PLOTTER - MULTI-WEIGHT VARIATIONS")
    print("=" * 80)

    regime_file = Path('data/sp500_regimes.csv')
    csv_file = Path('output/backtest_results/batch_10/batch_10_daily_time_series.csv')
    output_dir = Path('output/backtest_results/batch_10')


    if not regime_file.exists() or not csv_file.exists():
        print(f"\n❌ ERROR: Files not found!")
        if not csv_file.exists():
            print(f"\n   Searching for batch_10_daily_time_series.csv...")
            base = Path('output')
            if base.exists():
                for file in base.rglob('batch_10_daily_time_series.csv'):
                    print(f"   Found: {file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading data...")
    df_regimes = load_regime_data(regime_file)
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Detect months and weight variations
    months, weight_variations = detect_strategies(df)

    print(f"\n✅ Found {len(months)} months: {', '.join(months)}")
    print(f"✅ Found {len(weight_variations)} weight variations: {', '.join(weight_variations)}")

    has_existing_90 = any('Existing_90' in col for col in df.columns)
    print(f"✅ Existing 90% available: {has_existing_90}")

    if len(months) == 0:
        print(f"\n❌ ERROR: No months detected!")
        return

    # Create BOTH combined and individual plots
    print(f"\n{'=' * 80}")
    print(f"CREATING COMBINED CANVAS PLOTS")
    print(f"{'=' * 80}")

    if len(months) <= 6:
        create_combined_canvas(df, df_regimes, months, weight_variations, output_dir, has_existing_90,
                               plot_name='batch_10_combined_all')
    else:
        mid = 6
        create_combined_canvas(df, df_regimes, months[:mid], weight_variations, output_dir, has_existing_90,
                               plot_name='batch_10_combined_part1')
        create_combined_canvas(df, df_regimes, months[mid:], weight_variations, output_dir, has_existing_90,
                               plot_name='batch_10_combined_part2')

    print(f"\n{'=' * 80}")
    print(f"CREATING INDIVIDUAL PLOTS")
    print(f"{'=' * 80}")

    for month in months:
        create_individual_plot(df, df_regimes, month, weight_variations, output_dir, has_existing_90)

    print(f"\n{'=' * 80}")
    print(f"✅ COMPLETE!")

    if len(months) <= 6:
        print(f"  - batch_10_combined_all.png ({len(months)} months)")
    else:
        print(f"  - batch_10_combined_part1.png (first 6 months)")
        print(f"  - batch_10_combined_part2.png (remaining {len(months) - 6} months)")
    print(f"\nIndividual plots:")
    for month in months:
        print(f"  - batch_10_all_weights_{month}.png")


def load_regime_data(regime_file_path):
    """Load regime data."""
    df_regimes = pd.read_csv(regime_file_path)
    df_regimes.columns = df_regimes.columns.str.strip()
    df_regimes['Date'] = pd.to_datetime(df_regimes['Date'])
    df_regimes = df_regimes[['Date', 'Regimes']].copy()
    df_regimes = df_regimes[df_regimes['Regimes'].notna()].copy()
    df_regimes['Regimes'] = df_regimes['Regimes'].astype(int)
    print(f"✅ Loaded {len(df_regimes)} regime rows")
    return df_regimes


def detect_strategies(df):
    """Detect all months and weight variations in the CSV."""
    months = set()
    weight_variations = set()

    # Pattern: {MONTH}_New_ECR_{WEIGHTS}_NAV
    for col in df.columns:
        if '_New_ECR_' in col and col.endswith('_NAV'):
            parts = col.split('_')
            month = parts[0]
            # Find weight pattern (3 digits)
            for i, part in enumerate(parts):
                if part.isdigit() and len(part) == 3:
                    weight = part
                    months.add(month)
                    weight_variations.add(weight)
                    break

    return sorted(months), sorted(weight_variations)


def create_combined_canvas(df, df_regimes, months, weight_variations, output_dir, has_existing_90, plot_name):
    """Create combined canvas with all months."""
    n_months = len(months)
    n_cols = 3
    n_rows = 2

    print(f"\n  Creating {plot_name} with {n_months} months...")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 16))
    axes = axes.flatten()

    for idx, month in enumerate(months):
        ax = axes[idx]
        plot_month_subplot(ax, df, df_regimes, month, weight_variations, has_existing_90)

    for idx in range(n_months, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'Batch 10: ECR Weight Variations - {", ".join(months)}\n'
                 f'Weight variations: {", ".join([f"({w[0]},{w[1]},{w[2]})" for w in weight_variations])}\n'
                 'Green=Bullish | Gray=Neutral | Red=Bearish',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    plot_path = output_dir / f'{plot_name}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Saved: {plot_path.name}")


def create_individual_plot(df, df_regimes, month, weight_variations, output_dir, has_existing_90):
    """Create individual full-size plot for one month with all weight variations."""
    print(f"  Creating {month}...", end=' ')

    fig, ax = plt.subplots(figsize=(22, 12))
    plot_month_subplot(ax, df, df_regimes, month, weight_variations, has_existing_90, full_size=True)

    plt.tight_layout()

    plot_path = output_dir / f'batch_10_all_weights_{month}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅")


def plot_month_subplot(ax, df, df_regimes, month, weight_variations, has_existing_90, full_size=False):
    """Plot one month with all weight variations."""

    # Find start date from first available weight variation
    month_start = None
    for weight in weight_variations:
        col = f'{month}_New_ECR_{weight}_NAV'
        if col in df.columns:
            month_start = df[df[col].notna()]['Date'].min()
            if not pd.isna(month_start):
                break

    if pd.isna(month_start):
        ax.text(0.5, 0.5, f'No data for {month}',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{month} - No Data', fontweight='bold')
        return

    df_month = df[df['Date'] >= month_start].copy()
    df_merged = df_month.merge(df_regimes, on='Date', how='left')
    df_merged['Regimes'] = df_merged['Regimes'].ffill()

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

    # Plot all weight variations
    performance_data = []

    for weight in weight_variations:
        nav_col = f'{month}_New_ECR_{weight}_NAV'
        return_col = f'{month}_New_ECR_{weight}_Return'

        if nav_col in df_merged.columns and df_merged[nav_col].notna().any():
            color, linestyle, label = WEIGHT_COLORS.get(weight, ('gray', 'solid', f'w={weight}'))

            linewidth = 3.0 if weight == '111' else 2.0
            alpha = 1.0 if weight == '111' else 0.8

            ax.plot(df_merged['Date'], df_merged[nav_col],
                    label=label, color=color, linestyle=linestyle,
                    linewidth=linewidth, alpha=alpha, zorder=2)

            # Calculate performance
            if return_col in df_merged.columns:
                returns = df_merged[return_col].dropna()
                if len(returns) > 0:
                    total_return = ((1 + returns).prod() - 1) * 100
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                    performance_data.append({
                        'weight': weight,
                        'return': total_return,
                        'sharpe': sharpe
                    })

    # Plot Existing 90% if available
    if has_existing_90:
        existing_col = f'{month}_Existing_90_NAV'
        if existing_col in df_merged.columns and df_merged[existing_col].notna().any():
            ax.plot(df_merged['Date'], df_merged[existing_col],
                    label='Existing 90%', color='red', linestyle='--',
                    linewidth=2.5, alpha=0.9, zorder=2)

    # Plot benchmarks (resampled)
    for bench_col, color, style, label in [('SPY_NAV', 'black', '-', 'SPY'), ('BUFR_NAV', 'gray', '--', 'BUFR')]:
        if bench_col in df_merged.columns and df_merged[bench_col].notna().any():
            first = df_merged[bench_col].dropna().iloc[0]
            normalized = (df_merged[bench_col] / first) * 100
            ax.plot(df_merged['Date'], normalized, label=label, color=color,
                    linestyle=style, linewidth=1.5, alpha=0.6, zorder=1)

    # Formatting
    fontsize = 14 if full_size else 11
    ax.set_title(f'{month} - All Weight Variations (start: {month_start.strftime("%Y-%m-%d")})',
                 fontsize=fontsize, fontweight='bold')
    ax.set_xlabel('Date', fontsize=fontsize - 2)
    ax.set_ylabel('NAV (100 = Start)', fontsize=fontsize - 2)
    ax.grid(True, alpha=0.3, zorder=1)

    # Legend - split into two columns for readability
    ncol = 2 if len(weight_variations) > 4 else 1
    ax.legend(loc='upper left', fontsize=fontsize - 4, framealpha=0.95, ncol=ncol)

    # Performance summary box (upper right)
    if performance_data:
        perf_text = 'PERFORMANCE\n' + '━' * 20 + '\n'
        for perf in sorted(performance_data, key=lambda x: x['return'], reverse=True)[:5]:  # Top 5
            weight_label = f"({perf['weight'][0]},{perf['weight'][1]},{perf['weight'][2]})"
            perf_text += f"{weight_label}: {perf['return']:+.1f}%\n"

        props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9,
                     edgecolor='black', linewidth=1.5)
        ax.text(0.98, 0.98, perf_text, transform=ax.transAxes, fontsize=fontsize - 4,
                verticalalignment='top', horizontalalignment='right',
                bbox=props, family='monospace', weight='bold')

    ax.tick_params(axis='x', rotation=45, labelsize=fontsize - 4)
    ax.tick_params(axis='y', labelsize=fontsize - 4)


if __name__ == "__main__":
    main()