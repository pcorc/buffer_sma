"""
Batch 10b Visualization: ECR Percentile Trigger vs Existing 90% Baseline
=========================================================================

Plots NAV performance of each percentile threshold (p70, p75, p80, p90)
against the Existing 90% cap utilization baseline, with:
  - Regime shading (bull/neutral/bear)
  - Performance summary box (total return + trade count per strategy)
  - One chart per launch month
  - Combined canvas when multiple months are present

Usage:
    python plot_batch_10b.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path


# ── Color / style config ──────────────────────────────────────────────────────
PERCENTILE_STYLES = {
    'p70':  {'color': '#2166ac', 'linestyle': '-',   'linewidth': 2.5, 'label': 'ECR p70'},
    'p75':  {'color': '#4dac26', 'linestyle': '-',   'linewidth': 2.5, 'label': 'ECR p75'},
    'p80':  {'color': '#d6604d', 'linestyle': '-',   'linewidth': 2.5, 'label': 'ECR p80'},
    'p90':  {'color': '#7b2d8b', 'linestyle': '-',   'linewidth': 2.5, 'label': 'ECR p90'},
}

BASELINE_STYLE = {'color': '#b2182b', 'linestyle': '--', 'linewidth': 2.0, 'label': 'Existing 90%'}
SPY_STYLE      = {'color': '#000000', 'linestyle': '-',  'linewidth': 1.2, 'alpha': 0.45, 'label': 'SPY'}
BUFR_STYLE     = {'color': '#636363', 'linestyle': '--', 'linewidth': 1.2, 'alpha': 0.45, 'label': 'BUFR'}

REGIME_COLORS = {
     1: ('#d4edda', 'Bullish'),
     0: ('#e9e9e9', 'Neutral'),
    -1: ('#fde8e8', 'Bearish'),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(csv_path: Path, regime_path: Path):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])

    df_reg = pd.read_csv(regime_path)
    df_reg.columns = df_reg.columns.str.strip()
    df_reg['Date'] = pd.to_datetime(df_reg['Date'])
    df_reg = df_reg[['Date', 'Regimes']].dropna()
    df_reg['Regimes'] = df_reg['Regimes'].astype(int)

    return df, df_reg


def detect_strategies(df):
    """
    Returns dict: {month: {pct_key: nav_col_name}}
    e.g. {'JAN': {'p70': 'JAN_ECR_Pct_111_p70_h63_NAV', ...}}
    Also returns dict of baseline cols: {month: col_name}
    """
    months = {}
    baselines = {}

    for col in df.columns:
        if not col.endswith('_NAV'):
            continue

        if 'ECR_Pct_' in col:
            # Pattern: {MONTH}_ECR_Pct_{WEIGHT}_p{PCT}_h{HOLD}_NAV
            parts = col.replace('_NAV', '').split('_')
            month = parts[0]
            # Find pXX part
            pct_key = next((p for p in parts if p.startswith('p') and p[1:].isdigit()), None)
            if pct_key:
                months.setdefault(month, {})[pct_key] = col

        elif 'Existing_90' in col:
            month = col.split('_')[0]
            baselines[month] = col

    return months, baselines


def add_regime_shading(ax, df_merged):
    """Shade background by regime."""
    current_regime = None
    start_date = None

    for _, row in df_merged.iterrows():
        regime = row.get('Regimes')
        date = row['Date']
        if pd.isna(regime):
            continue
        regime = int(regime)
        if regime != current_regime:
            if current_regime is not None and start_date is not None:
                color, _ = REGIME_COLORS.get(current_regime, ('#ffffff', ''))
                ax.axvspan(start_date, date, alpha=0.18, color=color, zorder=0)
            current_regime = regime
            start_date = date

    if current_regime is not None and start_date is not None:
        color, _ = REGIME_COLORS.get(current_regime, ('#ffffff', ''))
        ax.axvspan(start_date, df_merged['Date'].max(), alpha=0.18, color=color, zorder=0)


def compute_stats(df, nav_col, return_col):
    """Return (total_return_pct, num_trades_proxy) from NAV series."""
    series = df[nav_col].dropna()
    if len(series) < 2:
        return np.nan, np.nan
    total_return = (series.iloc[-1] / series.iloc[0] - 1) * 100

    # Trade count proxy: count sign changes in daily return that are large
    if return_col in df.columns:
        rets = df[return_col].dropna()
        # Count days where return differs significantly from prior day
        # (rough proxy since trade history not in CSV)
        trades = int((rets.diff().abs() > 0.001).sum())
    else:
        trades = 0

    return total_return, trades


def plot_month(ax, df, df_reg, month, pct_cols, baseline_col, full_size=False):
    """Plot one month's strategies on ax."""

    # Determine start date from first available strategy
    all_nav_cols = list(pct_cols.values()) + ([baseline_col] if baseline_col else [])
    month_start = None
    for col in all_nav_cols:
        if col in df.columns:
            first = df[df[col].notna()]['Date'].min()
            if not pd.isna(first):
                month_start = first
                break

    if month_start is None:
        ax.text(0.5, 0.5, f'No data for {month}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{month} — No Data')
        return

    df_m = df[df['Date'] >= month_start].copy()
    df_merged = df_m.merge(df_reg, on='Date', how='left')
    df_merged['Regimes'] = df_merged['Regimes'].ffill()

    # Regime shading
    add_regime_shading(ax, df_merged)

    # Plot benchmarks (normalized to 100 at month start)
    for bench_col, style in [('SPY_NAV', SPY_STYLE), ('BUFR_NAV', BUFR_STYLE)]:
        if bench_col in df_merged.columns and df_merged[bench_col].notna().any():
            first = df_merged[bench_col].dropna().iloc[0]
            norm = (df_merged[bench_col] / first) * 100
            ax.plot(df_merged['Date'], norm, zorder=1, **style)

    # Collect performance for summary box
    perf_lines = []

    # Plot each percentile threshold
    sorted_pcts = sorted(pct_cols.keys(), key=lambda x: int(x[1:]))
    for pct_key in sorted_pcts:
        nav_col = pct_cols[pct_key]
        ret_col = nav_col.replace('_NAV', '_Return')

        if nav_col not in df_merged.columns or not df_merged[nav_col].notna().any():
            continue

        style = PERCENTILE_STYLES.get(pct_key, {'color': 'blue', 'linestyle': '-',
                                                  'linewidth': 2.0, 'label': pct_key})
        ax.plot(df_merged['Date'], df_merged[nav_col],
                color=style['color'], linestyle=style['linestyle'],
                linewidth=style['linewidth'], label=style['label'], zorder=3)

        total_ret, _ = compute_stats(df_merged, nav_col, ret_col)
        perf_lines.append((style['label'], style['color'], total_ret))

    # Plot baseline
    if baseline_col and baseline_col in df_merged.columns and df_merged[baseline_col].notna().any():
        ret_col = baseline_col.replace('_NAV', '_Return')
        ax.plot(df_merged['Date'], df_merged[baseline_col],
                color=BASELINE_STYLE['color'], linestyle=BASELINE_STYLE['linestyle'],
                linewidth=BASELINE_STYLE['linewidth'], label=BASELINE_STYLE['label'], zorder=3)
        total_ret, _ = compute_stats(df_merged, baseline_col, ret_col)
        perf_lines.append(('Existing 90%', BASELINE_STYLE['color'], total_ret))

    # Formatting
    fs = 13 if full_size else 10
    ax.set_title(f'{month}  —  ECR Percentile Thresholds vs Existing 90%\n'
                 f'Start: {month_start.strftime("%Y-%m-%d")}',
                 fontsize=fs, fontweight='bold')
    ax.set_ylabel('NAV (100 = Start)', fontsize=fs - 1)
    ax.set_xlabel('Date', fontsize=fs - 1)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    ax.tick_params(axis='x', rotation=35, labelsize=fs - 2)
    ax.tick_params(axis='y', labelsize=fs - 2)

    # Legend
    ax.legend(loc='upper left', fontsize=fs - 3, framealpha=0.92,
              ncol=2 if len(perf_lines) > 4 else 1)

    # Performance summary box (upper right)
    if perf_lines:
        box_lines = ['PERFORMANCE SUMMARY', '─' * 22]
        for label, color, ret in sorted(perf_lines, key=lambda x: x[2] if not np.isnan(x[2]) else -999, reverse=True):
            ret_str = f'{ret:+.1f}%' if not np.isnan(ret) else 'n/a'
            box_lines.append(f'{label:<14} {ret_str:>7}')

        box_text = '\n'.join(box_lines)
        props = dict(boxstyle='round,pad=0.5', facecolor='#fffde7',
                     edgecolor='#9e9d24', linewidth=1.5, alpha=0.95)
        ax.text(0.98, 0.98, box_text, transform=ax.transAxes,
                fontsize=fs - 3, verticalalignment='top', horizontalalignment='right',
                bbox=props, family='monospace', fontweight='bold')

    # Regime legend patches
    import matplotlib.patches as mpatches
    regime_patches = [
        mpatches.Patch(color='#d4edda', alpha=0.6, label='Bullish'),
        mpatches.Patch(color='#e9e9e9', alpha=0.6, label='Neutral'),
        mpatches.Patch(color='#fde8e8', alpha=0.6, label='Bearish'),
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + regime_patches,
              labels=ax.get_legend_handles_labels()[1] + ['Bullish', 'Neutral', 'Bearish'],
              loc='upper left', fontsize=fs - 3, framealpha=0.92,
              ncol=2 if len(perf_lines) > 3 else 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    batcher = "batch_10b"
    csv_path    = Path('output/backtest_results/' + batcher + '/' + batcher + '_daily_time_series.csv')
    regime_path = Path('../input_data/sp500_regimes.csv')
    output_dir  = Path('output/backtest_results/' + batcher)

    if not csv_path.exists():
        print(f'\n❌ CSV not found: {csv_path}')
        print('   Searching...')
        for f in Path('../output').rglob(batcher + '_daily_time_series.csv'):
            print(f'   Found: {f}')
        return

    if not regime_path.exists():
        print(f'❌ Regime file not found: {regime_path}')
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading data...')
    df, df_reg = load_data(csv_path, regime_path)
    pct_by_month, baseline_by_month = detect_strategies(df)

    months = sorted(pct_by_month.keys())
    print(f'✅ Found months: {", ".join(months)}')
    print(f'✅ Columns: {[c for c in df.columns if c.endswith("_NAV")]}')

    if not months:
        print('❌ No ECR percentile strategies detected in CSV')
        return

    # ── Individual plots ──────────────────────────────────────────────────────
    for month in months:
        fig, ax = plt.subplots(figsize=(20, 11))
        plot_month(
            ax=ax,
            df=df,
            df_reg=df_reg,
            month=month,
            pct_cols=pct_by_month.get(month, {}),
            baseline_col=baseline_by_month.get(month),
            full_size=True
        )
        plt.tight_layout()
        out_path = output_dir / f'batch_{month}.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'✅ Saved: {out_path.name}')

    # ── Combined canvas (if multiple months) ──────────────────────────────────
    if len(months) > 1:
        n_cols = 3
        n_rows = int(np.ceil(len(months) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 9 * n_rows))
        axes = axes.flatten()

        for i, month in enumerate(months):
            plot_month(
                ax=axes[i],
                df=df,
                df_reg=df_reg,
                month=month,
                pct_cols=pct_by_month.get(month, {}),
                baseline_col=baseline_by_month.get(month),
                full_size=False
            )

        for j in range(len(months), len(axes)):
            axes[j].axis('off')

        fig.suptitle(batcher + ' ECR Percentile Trigger — All Months\n'
                     'p70 / p75 / p80 / p90 vs Existing 90% Baseline  |  '
                     'Green=Bullish  Gray=Neutral  Red=Bearish',
                     fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        out_path = output_dir / 'batch_combined.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'✅ Saved: {out_path.name}')

    print(f'\n{"=" * 60}')
    print(f'✅ COMPLETE — outputs in {output_dir}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
