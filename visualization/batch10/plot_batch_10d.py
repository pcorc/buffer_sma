"""
Batch 10d Visualization: ECR Percentile Rank Trigger
=====================================================

Plots produced:
  1. batch_10d_score_analysis.png       — Score distribution + trade counts + perf table
  2. batch_10d_score_evolution.png      — Score at each trade over time per month (max 4 per fig)
  3. batch_10d_combined_time_series.png — All strategies + benchmarks per month (max 4 per fig)
  4. batch_10d_vs_baseline_table.png    — Month x weight x percentile vs Existing 90% matrix

Update the three path constants below to match your output folder.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from pathlib import Path
import glob

# ============================================================================
# CONFIGURATION — update these three paths
# ============================================================================

batch_10d_dir = Path('../../output/backtest_results/batch_10d')
matches = sorted(glob.glob(str(batch_10d_dir / 'batch10d_*.xlsx')))
if not matches:
    raise FileNotFoundError(f'No batch 10d summary workbook found in {batch_10d_dir}')
SUMMARY_PATH = Path(matches[-1])

CSV_PATH     = Path('../../output/backtest_results/batch_10d/batch_10d_daily_time_series.csv')
SCORING_PATH = Path('../../output/backtest_results/batch_10d/batch_10d_rebalance_scoring.xlsx')
OUTPUT_DIR   = Path('../../output/backtest_results/batch_10d')

WEIGHT_COLORS = {
    '1,1,1': '#2166ac',
    '1,2,1': '#4dac26',
    '2,1,1': '#d6604d',
}
WEIGHT_LABELS = {
    '1,1,1': 'Equal (1,1,1)',
    '1,2,1': 'Buffer Heavy (1,2,1)',
    '2,1,1': 'DBB Heavy (2,1,1)',
}

# One color per percentile threshold
PCT_COLORS = {
    25:  '#1f78b4',
    33:  '#33a02c',
    50:  '#e31a1c',
    67:  '#ff7f00',
    75:  '#6a3d9a',
}

ECR_PALETTE = [
    '#2166ac', '#4dac26', '#d6604d', '#7b2d8b',
    '#f46d43', '#74add1', '#a6d96a', '#fdae61',
    '#1a9850', '#d73027',
]

CHUNK_SIZE = 4   # max months per figure


# ============================================================================
# HELPERS
# ============================================================================

def parse_tp(tp):
    if isinstance(tp, str):
        try:
            return ast.literal_eval(tp)
        except Exception:
            return {}
    return tp if isinstance(tp, dict) else {}


def load_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Summary workbook ──────────────────────────────────────────────────────
    df_summary = pd.read_excel(SUMMARY_PATH, sheet_name='Summary')
    df_summary['_tp']                 = df_summary['trigger_params'].apply(parse_tp)
    df_summary['weight_code']         = df_summary['_tp'].apply(lambda x: x.get('weight_code'))
    df_summary['percentile_threshold']= df_summary['_tp'].apply(lambda x: x.get('percentile_threshold'))
    df_summary['start_date']          = pd.to_datetime(df_summary['start_date'])
    df_summary['end_date']            = pd.to_datetime(df_summary['end_date'])

    df_ecr  = df_summary[df_summary['trigger_type'] == 'ecr_percentile_rank'].copy()
    df_base = df_summary[df_summary['trigger_type'] != 'ecr_percentile_rank'].copy()

    # ── Daily NAV CSV ─────────────────────────────────────────────────────────
    df_nav = None
    if CSV_PATH.exists():
        df_nav = pd.read_csv(CSV_PATH)
        df_nav['Date'] = pd.to_datetime(df_nav['Date'])
    else:
        print(f'  ⚠️  CSV not found: {CSV_PATH}')

    # ── Scoring file ──────────────────────────────────────────────────────────
    df_sel = pd.read_excel(SCORING_PATH, sheet_name='Selected Funds')
    df_all = pd.read_excel(SCORING_PATH, sheet_name='All Rebalances')
    df_sel['Date'] = pd.to_datetime(df_sel['Date'])
    df_all['Date'] = pd.to_datetime(df_all['Date'])

    months      = sorted(df_sel['Month'].unique())
    weights     = sorted(df_ecr['weight_code'].dropna().unique()) if not df_ecr.empty else sorted(df_sel['Weight_Config'].unique())
    percentiles = sorted(df_ecr['percentile_threshold'].dropna().unique()) if not df_ecr.empty else []

    print(f'  Months:          {months}')
    print(f'  Weight configs:  {weights}')
    print(f'  Percentiles:     {percentiles}')
    print(f'  ECR strategies:  {len(df_ecr)}')
    print(f'  Baselines:       {len(df_base)}')
    print(f'  Selected trades: {len(df_sel)}')

    return df_summary, df_ecr, df_base, df_nav, df_sel, df_all, months, weights, percentiles


def save(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path.name}')


def chunk_months(months):
    return [months[i:i + CHUNK_SIZE] for i in range(0, len(months), CHUNK_SIZE)]


# ============================================================================
# PLOT 1 — Score Analysis: distribution + trade counts + perf table
# ============================================================================

def plot_score_analysis(df_sel, df_ecr, df_base, months, weights, percentiles):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('Batch 10d: ECR Percentile Rank — Score Analysis at Trade Time',
                 fontsize=13, fontweight='bold')

    # ── Left: composite score boxplot by weight config ────────────────────────
    ax = axes[0]
    data = [df_sel[df_sel['Weight_Config'] == w]['Composite_Score'].values
            for w in weights]
    if any(len(d) > 0 for d in data):
        bp = ax.boxplot(data,
                        labels=[WEIGHT_LABELS.get(w, w) for w in weights],
                        patch_artist=True)
        for patch, w in zip(bp['boxes'], weights):
            patch.set_facecolor(WEIGHT_COLORS.get(w, 'steelblue'))
            patch.set_alpha(0.7)
    ax.set_title('Composite Score at Trade Date\nby Weight Config',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('ECR Composite Score')
    ax.grid(True, alpha=0.3, axis='y')

    # ── Middle: trade counts per month per weight ─────────────────────────────
    ax = axes[1]
    tc = df_sel.groupby(['Month', 'Weight_Config'])['Date'].count().reset_index()
    tc.columns = ['Month', 'Weight_Config', 'Trades']
    x  = np.arange(len(months))
    bw = 0.25
    for i, w in enumerate(weights):
        vals = []
        for m in months:
            v = tc[(tc['Month'] == m) & (tc['Weight_Config'] == w)]['Trades'].values
            vals.append(v[0] if len(v) > 0 else 0)
        bars = ax.bar(x + i * bw, vals, bw,
                      label=WEIGHT_LABELS.get(w, w),
                      color=WEIGHT_COLORS.get(w, 'steelblue'), alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2, str(val),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title('Trades Fired\nby Month + Weight Config',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Trades')
    ax.set_xticks(x + bw)
    ax.set_xticklabels(months, fontsize=8, rotation=30)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Right: performance summary table ─────────────────────────────────────
    ax = axes[2]
    ax.axis('off')
    headers = ['Month', 'w', 'Pct', 'Return', 'Sharpe', 'vs BUFR', 'Trades']
    rows = []
    for m in months:
        for _, row in df_ecr[df_ecr['launch_month'] == m].iterrows():
            rows.append([
                m,
                row['weight_code'] or '?',
                f"p{int(row['percentile_threshold'])}",
                f"{row['strategy_return']*100:+.1f}%",
                f"{row['strategy_sharpe']:.2f}",
                f"{row['vs_bufr_excess']*100:+.1f}%",
                str(int(row['num_trades'])),
            ])
        for _, row in df_base[df_base['launch_month'] == m].iterrows():
            tp  = parse_tp(row.get('trigger_params', '{}'))
            thr = tp.get('threshold', '?')
            rows.append([
                m, 'E90%',
                f'{thr*100:.0f}%' if isinstance(thr, float) else str(thr),
                f"{row['strategy_return']*100:+.1f}%",
                f"{row['strategy_sharpe']:.2f}",
                f"{row['vs_bufr_excess']*100:+.1f}%",
                str(int(row['num_trades'])),
            ])

    if rows:
        tbl = ax.table(cellText=rows, colLabels=headers,
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.6)
        for j in range(len(headers)):
            tbl[(0, j)].set_facecolor('#2166ac')
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')
        for i in range(1, len(rows) + 1):
            bg = '#f0f4fa' if i % 2 == 0 else 'white'
            for j in range(len(headers)):
                tbl[(i, j)].set_facecolor(bg)
    ax.set_title('Strategy Performance Summary', fontsize=11, fontweight='bold', pad=20)

    plt.tight_layout()
    save(fig, 'batch_10d_score_analysis.png')


# ============================================================================
# PLOT 2 — Score Evolution per month (max 4 per figure)
# ============================================================================

def plot_score_evolution(df_sel, months, weights):
    for chunk_idx, month_chunk in enumerate(chunk_months(months)):
        fig, axes = plt.subplots(len(month_chunk), 1,
                                 figsize=(20, 6 * len(month_chunk)), sharex=False)
        if len(month_chunk) == 1:
            axes = [axes]

        n_total = len(months)
        start_n = chunk_idx * CHUNK_SIZE + 1
        end_n   = chunk_idx * CHUNK_SIZE + len(month_chunk)
        fig.suptitle(
            f'Batch 10d: ECR Score at Each Trade Event '
            f'({start_n}–{end_n} of {n_total})',
            fontsize=13, fontweight='bold', y=1.01
        )

        for ax, month in zip(axes, month_chunk):
            month_data = df_sel[df_sel['Month'] == month].copy()

            for w in weights:
                wd = month_data[month_data['Weight_Config'] == w].sort_values('Date')
                if wd.empty:
                    continue
                ax.scatter(wd['Date'], wd['Composite_Score'],
                           color=WEIGHT_COLORS.get(w, 'steelblue'),
                           label=WEIGHT_LABELS.get(w, w),
                           s=80, zorder=3, alpha=0.85)
                ax.plot(wd['Date'], wd['Composite_Score'],
                        color=WEIGHT_COLORS.get(w, 'steelblue'),
                        linewidth=1.2, alpha=0.4, zorder=2)
                for _, row in wd.iterrows():
                    ax.annotate(row['Fund'][-3:],
                                (row['Date'], row['Composite_Score']),
                                textcoords='offset points', xytext=(0, 6),
                                fontsize=6.5, ha='center',
                                color=WEIGHT_COLORS.get(w, 'black'))

            ax.set_title(f'{month} — Selected Fund Score at Each Trade',
                         fontsize=11, fontweight='bold')
            ax.set_ylabel('ECR Composite Score')
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=9, loc='upper right')
            ax.tick_params(axis='x', rotation=30, labelsize=8)

        plt.tight_layout()
        suffix = f'_part{chunk_idx+1}' if len(chunk_months(months)) > 1 else ''
        save(fig, f'batch_10d_score_evolution{suffix}.png')


# ============================================================================
# PLOT 3 — Combined Time Series (max 4 per figure)
# ============================================================================

def plot_combined_time_series(df_nav, df_ecr, df_base, months):
    for chunk_idx, month_chunk in enumerate(chunk_months(months)):
        n = len(month_chunk)
        fig, axes = plt.subplots(n, 1, figsize=(22, 9 * n))
        if n == 1:
            axes = [axes]

        n_total = len(months)
        start_n = chunk_idx * CHUNK_SIZE + 1
        end_n   = chunk_idx * CHUNK_SIZE + n
        fig.suptitle(
            f'Batch 10d: Full Time Series — All Strategies vs Benchmarks '
            f'({start_n}–{end_n} of {n_total})',
            fontsize=15, fontweight='bold', y=1.01
        )

        for ax, month in zip(axes, month_chunk):
            if df_nav is None:
                ax.set_title(f'{month} — No NAV data available', fontsize=11)
                continue

            # Find this month's start date
            month_cols = [c for c in df_nav.columns
                          if c.startswith(f'{month}_') and c.endswith('_NAV')
                          and '_SPY_NAV' not in c and '_BUFR_NAV' not in c]
            if not month_cols:
                ax.set_title(f'{month} — No strategy NAV columns found', fontsize=11)
                continue

            month_start = df_nav.loc[df_nav[month_cols[0]].notna(), 'Date'].min()
            df_month    = df_nav[df_nav['Date'] >= month_start].copy()

            # ── Per-month benchmarks ──────────────────────────────────────────
            for bench_suffix, color, ls, lw, alpha, lbl in [
                ('SPY_NAV',  'black',   '-',  1.0, 0.35, 'SPY'),
                ('BUFR_NAV', '#636363', '--', 1.2, 0.45, 'BUFR'),
            ]:
                bench_col = f'{month}_{bench_suffix}'
                if bench_col in df_month.columns:
                    ax.plot(df_month['Date'], df_month[bench_col],
                            color=color, linestyle=ls, linewidth=lw,
                            alpha=alpha, label=lbl, zorder=1)

            # ── Existing 90% ──────────────────────────────────────────────────
            base_col = f'{month}_Existing_90_NAV'
            if base_col in df_month.columns:
                ax.plot(df_month['Date'], df_month[base_col],
                        color='#b2182b', linestyle='--', linewidth=2.2,
                        label='Existing 90%', zorder=3)

            # ── ECR PctRank NAV columns for this month ────────────────────────
            ecr_nav_cols = sorted([
                c for c in df_month.columns
                if c.startswith(f'{month}_ECR_PctRank_') and c.endswith('_NAV')
            ])
            for i, col in enumerate(ecr_nav_cols):
                color = ECR_PALETTE[i % len(ECR_PALETTE)]
                # e.g. JAN_ECR_PctRank_w111_p50_NAV → w=111 p=50
                lbl = (col
                       .replace(f'{month}_ECR_PctRank_', '')
                       .replace('_NAV', '')
                       .replace('w', 'w=')
                       .replace('_p', ' p=')
                       .replace('_', '.'))
                ax.plot(df_month['Date'], df_month[col],
                        color=color, linewidth=2.0, label=lbl, zorder=4)

            # ── Performance summary box ───────────────────────────────────────
            lines = []
            for _, row in df_ecr[df_ecr['launch_month'] == month].iterrows():
                pct = row['percentile_threshold']
                lines.append(
                    f"w={row['weight_code']} p={int(pct) if pd.notna(pct) else '?':<3}  "
                    f"{row['strategy_return']*100:+.1f}%  "
                    f"sh={row['strategy_sharpe']:.2f}  "
                    f"vs_B={row['vs_bufr_excess']*100:+.1f}%  "
                    f"T={int(row['num_trades'])}"
                )
            for _, row in df_base[df_base['launch_month'] == month].iterrows():
                tp      = parse_tp(row.get('trigger_params', '{}'))
                thr     = tp.get('threshold', '?')
                thr_str = f'{thr*100:.0f}%' if isinstance(thr, float) else str(thr)
                lines.append(
                    f"Exist {thr_str:<9}  "
                    f"{row['strategy_return']*100:+.1f}%  "
                    f"sh={row['strategy_sharpe']:.2f}  "
                    f"vs_B={row['vs_bufr_excess']*100:+.1f}%  "
                    f"T={int(row['num_trades'])}"
                )

            if lines:
                ax.text(0.01, 0.97,
                        f'{month} PERFORMANCE\n' + '─' * 45 + '\n' + '\n'.join(lines),
                        transform=ax.transAxes, fontsize=9, va='top', ha='left',
                        family='monospace',
                        bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                                  edgecolor='#cccccc', alpha=0.95))

            ax.set_title(f'{month} — All Strategies', fontsize=13, fontweight='bold')
            ax.set_ylabel('NAV (100 = Start)', fontsize=11)
            ax.set_xlabel('Date', fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.legend(loc='lower right', fontsize=9, framealpha=0.92,
                      ncol=2 if len(ax.lines) > 5 else 1)
            ax.tick_params(axis='x', rotation=30, labelsize=9)

        plt.tight_layout()
        suffix = f'_part{chunk_idx+1}' if len(chunk_months(months)) > 1 else ''
        save(fig, f'batch_10d_combined_time_series{suffix}.png')


# ============================================================================
# PLOT 4 — Month x Weight x Percentile vs Existing 90% matrix table
# ============================================================================

def plot_vs_baseline_table(df_ecr, df_base, months, weights, percentiles):
    if not percentiles:
        print('  ⚠️  No ECR percentiles found — skipping vs baseline table')
        return

    row_keys = [(m, w) for m in months for w in weights]
    col_keys = [int(p) for p in percentiles] + ['Existing 90%']

    def fmt(ret, vsb, trades):
        return f"{ret:+.1f}%\nvs B: {vsb:+.1f}%\nT={int(trades)}"

    cells = {}
    for m in months:
        month_base = df_base[df_base['launch_month'] == m]
        base_ret = base_vsb = base_trades = None
        for _, row in month_base.iterrows():
            base_ret    = row['strategy_return'] * 100
            base_vsb    = row['vs_bufr_excess'] * 100
            base_trades = row['num_trades']
            break

        for w in weights:
            for p in percentiles:
                subset = df_ecr[
                    (df_ecr['launch_month'] == m) &
                    (df_ecr['weight_code'] == w) &
                    (df_ecr['percentile_threshold'] == p)
                ]
                if not subset.empty:
                    r = subset.iloc[0]
                    cells[(m, w, int(p))] = (
                        r['strategy_return'] * 100,
                        r['vs_bufr_excess'] * 100,
                        r['num_trades']
                    )
                else:
                    cells[(m, w, int(p))] = None

            cells[(m, w, 'Existing 90%')] = (
                base_ret, base_vsb, base_trades
            ) if base_ret is not None else None

    n_rows  = len(row_keys)
    n_cols  = len(col_keys)
    fig_h   = max(6, n_rows * 1.2 + 2)
    fig, ax = plt.subplots(figsize=(4 * n_cols + 2, fig_h))
    ax.axis('off')

    col_labels = [f'p={p}th' if p != 'Existing 90%' else 'Existing\n90%'
                  for p in col_keys]
    row_labels = [f'{m}\n{WEIGHT_LABELS.get(w, w)}' for m, w in row_keys]

    table_data  = []
    cell_colors = []

    for m, w in row_keys:
        row_data   = []
        row_colors = []
        base_cell  = cells.get((m, w, 'Existing 90%'))
        base_ret   = base_cell[0] if base_cell else None

        for t in col_keys:
            c = cells.get((m, w, t))
            if c is None or c[0] is None:
                row_data.append('—')
                row_colors.append('#f5f5f5')
            else:
                ret, vsb, trades = c
                row_data.append(fmt(ret, vsb, trades))
                if t == 'Existing 90%':
                    row_colors.append('#fff3cd')
                elif base_ret is not None and ret > base_ret:
                    row_colors.append('#d4edda')
                elif base_ret is not None and ret < base_ret:
                    row_colors.append('#fce4e4')
                else:
                    row_colors.append('#ffffff')

        table_data.append(row_data)
        cell_colors.append(row_colors)

    tbl = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 3.0)

    for j in range(n_cols):
        tbl[(0, j)].set_facecolor('#d6604d')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    fig.suptitle(
        'Batch 10d: ECR Percentile Rank vs Existing 90% Baseline\n'
        'Green = beats baseline return | Yellow = baseline | Red = underperforms',
        fontsize=12, fontweight='bold', y=0.98
    )
    plt.tight_layout()
    save(fig, 'batch_10d_vs_baseline_table.png')


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('=' * 60)
    print('Batch 10d — Generating plots')
    print('=' * 60)
    print()
    print('Loading data...')
    df_summary, df_ecr, df_base, df_nav, df_sel, df_all, months, weights, percentiles = load_data()

    print('\nPlot 1/4: Score Analysis...')
    plot_score_analysis(df_sel, df_ecr, df_base, months, weights, percentiles)

    print('Plot 2/4: Score Evolution...')
    plot_score_evolution(df_sel, months, weights)

    print('Plot 3/4: Combined Time Series...')
    plot_combined_time_series(df_nav, df_ecr, df_base, months)

    print('Plot 4/4: vs Baseline Table...')
    plot_vs_baseline_table(df_ecr, df_base, months, weights, percentiles)

    print()
    print('=' * 60)
    print(f'All plots saved to: {OUTPUT_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
