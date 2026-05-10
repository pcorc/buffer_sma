"""
Batch 10c Visualization: ECR Score Threshold Trigger
=====================================================

Plots produced:
  1. batch_10c_score_analysis.png       — Score distribution + trade counts + perf table
  2. batch_10c_score_evolution.png      — Score at each trade over time per month
  3. batch_10c_trade_comparison.png     — Outgoing vs incoming fund component scores
  4. batch_10c_remaining_days.png       — When in the outcome period does trigger fire?
  5. batch_10c_combined_time_series.png — All strategies + benchmarks per month
  6. batch_10c_vs_baseline_table.png    — Month × weight config vs Existing 90% summary table

Note: batch_10c_nav_performance.png is intentionally commented out.

Update the three path constants below to match your output folder.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import ast
from pathlib import Path
import glob

# ============================================================================
# CONFIGURATION — update these three paths
# ============================================================================

batch_10d_dir = Path('../output/backtest_results/batch_10d')
matches = sorted(glob.glob(str(batch_10d_dir / 'batch10d_*.xlsx')))
if not matches:
    raise FileNotFoundError(f'No batch 10d summary workbook found in {batch_10d_dir}')
SUMMARY_PATH = Path(matches[-1])

CSV_PATH     = Path('output/backtest_results/batch_10c/batch_10c_daily_time_series.csv')
SCORING_PATH = Path('output/backtest_results/batch_10c/batch_10c_rebalance_scoring.xlsx')
OUTPUT_DIR   = Path('../output/backtest_results/batch_10c')

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
ECR_PALETTE = [
    '#2166ac', '#4dac26', '#d6604d', '#7b2d8b',
    '#f46d43', '#74add1', '#a6d96a', '#fdae61',
    '#1a9850', '#d73027',
]


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
    df_summary['_tp']            = df_summary['trigger_params'].apply(parse_tp)
    df_summary['weight_code']    = df_summary['_tp'].apply(lambda x: x.get('weight_code'))
    df_summary['score_threshold']= df_summary['_tp'].apply(lambda x: x.get('score_threshold'))
    df_summary['start_date']     = pd.to_datetime(df_summary['start_date'])
    df_summary['end_date']       = pd.to_datetime(df_summary['end_date'])

    df_ecr  = df_summary[df_summary['trigger_type'] == 'ecr_score_threshold'].copy()
    df_base = df_summary[df_summary['trigger_type'] != 'ecr_score_threshold'].copy()

    # ── Daily NAV CSV ─────────────────────────────────────────────────────────
    df_nav = None
    if CSV_PATH.exists():
        df_nav = pd.read_csv(CSV_PATH)
        df_nav['Date'] = pd.to_datetime(df_nav['Date'])

    # ── Scoring file ──────────────────────────────────────────────────────────
    df_sel = pd.read_excel(SCORING_PATH, sheet_name='Selected Funds')
    df_all = pd.read_excel(SCORING_PATH, sheet_name='All Rebalances')
    df_sel['Date'] = pd.to_datetime(df_sel['Date'])
    df_all['Date'] = pd.to_datetime(df_all['Date'])

    months     = sorted(df_sel['Month'].unique())
    weights    = sorted(df_sel['Weight_Config'].unique())
    thresholds = sorted(df_ecr['score_threshold'].dropna().unique()) if not df_ecr.empty else []

    print(f'  Months:          {months}')
    print(f'  Weight configs:  {weights}')
    print(f'  Thresholds:      {thresholds}')
    print(f'  ECR strategies:  {len(df_ecr)}')
    print(f'  Baselines:       {len(df_base)}')
    print(f'  Selected trades: {len(df_sel)}')

    return df_summary, df_ecr, df_base, df_nav, df_sel, df_all, months, weights, thresholds


def save(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path.name}')


# ============================================================================
# PLOT 1 — Score Analysis (commented-out nav_performance replacement)
# ============================================================================

def plot_score_analysis(df_sel, df_ecr, df_base, months, weights):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('Batch 10c: ECR Score Analysis at Trade Time',
                 fontsize=13, fontweight='bold')

    # ── Left: score distribution boxplot by weight config ────────────────────
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

    # ── Middle: trade counts bar chart per month per weight ───────────────────
    ax = axes[1]
    tc = df_sel.groupby(['Month', 'Weight_Config'])['Date'].count().reset_index()
    tc.columns = ['Month', 'Weight_Config', 'Trades']
    x  = np.arange(len(months))
    bw = 0.25
    for i, w in enumerate(weights):
        vals = []
        for m in months:
            # Filter for this specific month
            month_w = tc[(tc['Month'] == m) & (tc['Weight_Config'] == w)]['Trades'].values
            vals.append(month_w[0] if len(month_w) > 0 else 0)
        bars = ax.bar(x + i * bw, vals, bw,
                      label=WEIGHT_LABELS.get(w, w),
                      color=WEIGHT_COLORS.get(w, 'steelblue'), alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    str(val), ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
    ax.set_title('Trades Fired\nby Month + Weight Config',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Trades')
    ax.set_xticks(x + bw)
    ax.set_xticklabels(months, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Right: performance summary table from workbook ────────────────────────
    ax = axes[2]
    ax.axis('off')
    headers = ['Month', 'w', 'Threshold', 'Return', 'Sharpe', 'vs BUFR', 'Trades']
    rows = []
    for m in months:
        # Filter ECR strategies for this month
        month_ecr = df_ecr[df_ecr['launch_month'] == m]
        for _, row in month_ecr.iterrows():
            rows.append([
                m,
                row['weight_code'] or '?',
                row['score_threshold'],
                f"{row['strategy_return']*100:+.1f}%",
                f"{row['strategy_sharpe']:.2f}",
                f"{row['vs_bufr_excess']*100:+.1f}%",
                str(int(row['num_trades'])),
            ])
        # Filter baseline for this month
        month_base = df_base[df_base['launch_month'] == m]
        for _, row in month_base.iterrows():
            tp  = parse_tp(row.get('trigger_params', '{}'))
            thr = tp.get('threshold', '?')
            rows.append([
                m, 'E90%', f'{thr*100:.0f}%' if isinstance(thr, float) else thr,
                f"{row['strategy_return']*100:+.1f}%",
                f"{row['strategy_sharpe']:.2f}",
                f"{row['vs_bufr_excess']*100:+.1f}%",
                str(int(row['num_trades'])),
            ])

    if rows:
        tbl = ax.table(cellText=rows, colLabels=headers,
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.7)
        for j in range(len(headers)):
            tbl[(0, j)].set_facecolor('#2166ac')
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')
        for i in range(1, len(rows) + 1):
            bg = '#f0f4fa' if i % 2 == 0 else 'white'
            for j in range(len(headers)):
                tbl[(i, j)].set_facecolor(bg)
    ax.set_title('Strategy Performance Summary', fontsize=11, fontweight='bold', pad=20)

    plt.tight_layout()
    save(fig, 'batch_10c_score_analysis.png')


# ============================================================================
# PLOT 2 — Score Evolution per month
# ============================================================================

def plot_score_evolution(df_sel, months, weights):
    fig, axes = plt.subplots(len(months), 1,
                             figsize=(20, 6 * len(months)), sharex=False)
    if len(months) == 1:
        axes = [axes]

    fig.suptitle('Batch 10c: ECR Score at Each Trade Event\n'
                 'Fund label shown at each point',
                 fontsize=13, fontweight='bold', y=1.01)

    for ax, month in zip(axes, months):
        # Filter data for this month only
        month_data = df_sel[df_sel['Month'] == month].copy()

        for w in weights:
            # Filter for this weight config within this month
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
    save(fig, 'batch_10c_score_evolution.png')


# ============================================================================
# PLOT 3 — Trade Comparison: outgoing vs incoming
# ============================================================================

def plot_trade_comparison(df_sel, df_all, months, weights):
    nw = len(weights)
    nm = len(months)
    fig, axes = plt.subplots(nm, nw, figsize=(7 * nw, 5 * nm))

    if nm == 1 and nw == 1:
        axes = [[axes]]
    elif nm == 1:
        axes = [list(axes)]
    elif nw == 1:
        axes = [[ax] for ax in axes]
    else:
        axes = [list(row) for row in axes]

    fig.suptitle('Batch 10c: Outgoing vs Incoming Fund — Component Scores at Each Trade\n'
                 'Red = fund leaving | Green = fund entering',
                 fontsize=13, fontweight='bold')

    components  = ['DBB_Score', 'Buffer_Integrity', 'Cap_Integrity']
    comp_labels = ['DBB', 'Buf Int', 'Cap Int']

    for ri, month in enumerate(months):
        # Filter selected funds for this month
        month_sel = df_sel[df_sel['Month'] == month].copy()
        # Filter all rebalances for this month
        month_all = df_all[df_all['Month'] == month].copy()

        for ci, w in enumerate(weights):
            ax = axes[ri][ci]

            # Filter incoming trades for this month + weight
            incoming = month_sel[month_sel['Weight_Config'] == w].sort_values('Date').reset_index(drop=True)

            if incoming.empty:
                ax.axis('off')
                continue

            x  = np.arange(len(components))
            bw = 0.35
            plotted = 0

            for _, inc_row in incoming.iterrows():
                # Filter all rebalances for this specific trade date + weight
                day_scores = month_all[
                    (month_all['Date'] == inc_row['Date']) &
                    (month_all['Weight_Config'] == w)
                ].sort_values('Composite_Score')

                if day_scores.empty:
                    continue

                out_row  = day_scores.iloc[0]
                out_vals = [out_row[c] for c in components]
                in_vals  = [inc_row[c] for c in components]

                alpha = 0.7 if plotted == 0 else 0.25
                if plotted == 0:
                    ax.bar(x - bw/2, out_vals, bw, color='#d6604d', alpha=alpha,
                           label=f'Leaving ({out_row["Fund"]})')
                    ax.bar(x + bw/2, in_vals,  bw, color='#4dac26', alpha=alpha,
                           label=f'Entering ({inc_row["Fund"]})')
                else:
                    ax.bar(x - bw/2, out_vals, bw, color='#d6604d', alpha=alpha)
                    ax.bar(x + bw/2, in_vals,  bw, color='#4dac26', alpha=alpha)
                plotted += 1

            ax.set_title(f'{month} | {WEIGHT_LABELS.get(w, w)}',
                         fontsize=10, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(comp_labels, fontsize=9)
            ax.set_ylabel('Score', fontsize=9)
            ax.set_ylim(0, 1.2)
            ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8)
            ax.legend(fontsize=7, framealpha=0.9)
            ax.grid(True, alpha=0.2, axis='y')
            ax.text(0.98, 0.95, f'{plotted} trades',
                    transform=ax.transAxes, fontsize=8, ha='right', va='top',
                    color='#555555')

    plt.tight_layout()
    save(fig, 'batch_10c_trade_comparison.png')


# ============================================================================
# PLOT 4 — Remaining Days violin
# ============================================================================

def plot_remaining_days(df_sel, weights):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('Batch 10c: Remaining Outcome Days When Trigger Fires\n'
                 'High = fires early | Low = fires late in outcome period',
                 fontsize=13, fontweight='bold')

    data = [df_sel[df_sel['Weight_Config'] == w]['Remaining_Days'].dropna().values
            for w in weights]

    if any(len(d) > 0 for d in data):
        vp = ax.violinplot(data, positions=range(len(weights)),
                           showmedians=True, showextrema=True)
        for pc, w in zip(vp['bodies'], weights):
            pc.set_facecolor(WEIGHT_COLORS.get(w, 'steelblue'))
            pc.set_alpha(0.7)

    ax.axhline(y=182, color='orange', linestyle='--', linewidth=1.5, label='6 months (~182d)')
    ax.axhline(y=91,  color='red',    linestyle='--', linewidth=1.5, label='3 months (~91d)')
    ax.axhline(y=365, color='green',  linestyle=':',  linewidth=1.5, label='Full period (365d)')
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels([WEIGHT_LABELS.get(w, w) for w in weights], fontsize=10)
    ax.set_ylabel('Remaining Outcome Days at Trade')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (d, w) in enumerate(zip(data, weights)):
        if len(d) > 0:
            ax.text(i, float(np.mean(d)) + 3, f'avg={np.mean(d):.0f}d',
                    ha='center', fontsize=9, fontweight='bold',
                    color=WEIGHT_COLORS.get(w, 'black'))

    plt.tight_layout()
    save(fig, 'batch_10c_remaining_days.png')


# ============================================================================
# PLOT 5 — Combined Time Series per month
# ============================================================================

def plot_combined_time_series(df_nav, df_ecr, df_base, months):
    n = len(months)
    fig, axes = plt.subplots(n, 1, figsize=(22, 9 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle('Batch 10c: Full Time Series — All Strategies vs Benchmarks',
                 fontsize=15, fontweight='bold', y=1.01)

    for ax, month in zip(axes, months):
        if df_nav is not None:

            # ── Find this month's start date from its first non-null strategy column ──
            month_cols = [c for c in df_nav.columns
                          if c.startswith(f'{month}_') and c.endswith('_NAV')]

            if not month_cols:
                continue

            # Get the start date for this month specifically
            first_col = month_cols[0]
            month_start = df_nav.loc[df_nav[first_col].notna(), 'Date'].min()

            # Slice the full df_nav to this month's window
            df_month = df_nav[df_nav['Date'] >= month_start].copy()

            # ── Benchmarks — use per-month columns ───────────────────────────────
            for bench_suffix, color, ls, lw, alpha, lbl in [
                ('SPY_NAV', 'black', '-', 1.0, 0.35, 'SPY'),
                ('BUFR_NAV', '#636363', '--', 1.2, 0.45, 'BUFR'),
            ]:
                bench_col = f'{month}_{bench_suffix}'
                if bench_col in df_month.columns:
                    ax.plot(df_month['Date'], df_month[bench_col],
                            color=color, linestyle=ls, linewidth=lw,
                            alpha=alpha, label=lbl, zorder=1)
            # ── Existing 90% for this month ───────────────────────────────────────
            base_col = f'{month}_Existing_90_NAV'
            if base_col in df_month.columns:
                ax.plot(df_month['Date'], df_month[base_col],
                        color='#b2182b', linestyle='--', linewidth=2.2,
                        label='Existing 90%', zorder=3)

            # ── ECR NAV columns for this month ────────────────────────────────────
            ecr_nav_cols = sorted([c for c in df_month.columns
                                   if c.startswith(f'{month}_ECR_') and
                                   c.endswith('_NAV')])
            for i, col in enumerate(ecr_nav_cols):
                color = ECR_PALETTE[i % len(ECR_PALETTE)]
                lbl = (col
                       .replace(f'{month}_ECR_', '')
                       .replace('_NAV', '')
                       .replace('w', 'w=')
                       .replace('_t', ' t=')
                       .replace('_s', ' s=')
                       .replace('_', '.'))
                ax.plot(df_month['Date'], df_month[col],
                        color=color, linewidth=2.0, label=lbl, zorder=4)

            # Fallback if label builder hasn't been fixed yet
            if not ecr_nav_cols:
                ecr_col = f'{month}_ecr_score__select_highest__NAV'
                if ecr_col in df_month.columns:
                    ax.plot(df_month['Date'], df_month[ecr_col],
                            color='#2166ac', linewidth=2.5,
                            label='ECR Score Threshold', zorder=4)

        # ── Performance summary box — filtered per month ──────────────────────
        lines = []
        for _, row in df_ecr[df_ecr['launch_month'] == month].iterrows():
            lines.append(
                f"w={row['weight_code']} t={row['score_threshold']:<4}  "
                f"{row['strategy_return']*100:+.1f}%  "
                f"sh={row['strategy_sharpe']:.2f}  "
                f"vs_B={row['vs_bufr_excess']*100:+.1f}%  "
                f"T={int(row['num_trades'])}"
            )
        for _, row in df_base[df_base['launch_month'] == month].iterrows():
            tp  = parse_tp(row.get('trigger_params', '{}'))
            thr = tp.get('threshold', '?')
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
    save(fig, 'batch_10c_combined_time_series.png')


# ============================================================================
# PLOT 6 — Month × Weight Config vs Existing 90% summary table
# ============================================================================

def plot_vs_baseline_table(df_ecr, df_base, months, weights):
    """
    Matrix table: rows = month × weight config, columns = threshold values.
    Each cell shows: Total Return | vs BUFR | Trades
    Existing 90% shown as a reference row per month.
    Green cells beat the baseline, red cells underperform.
    """
    thresholds = sorted(df_ecr['score_threshold'].dropna().unique())

    if not thresholds:
        print('  ⚠️  No ECR thresholds found — skipping vs baseline table')
        return

    # Build data matrix
    # Rows: (month, weight) combos
    # Cols: thresholds + baseline

    row_keys = [(m, w) for m in months for w in weights]
    col_keys  = thresholds + ['Existing 90%']

    # cell format: "ret% | vsB% | T"
    def fmt(ret, vsb, trades):
        return f"{ret:+.1f}%\nvs B: {vsb:+.1f}%\nT={int(trades)}"

    cells = {}
    for m in months:
        month_base = df_base[df_base['launch_month'] == m]

        # Baseline values for this month
        base_ret = base_vsb = base_trades = None
        for _, row in month_base.iterrows():
            base_ret    = row['strategy_return'] * 100
            base_vsb    = row['vs_bufr_excess'] * 100
            base_trades = row['num_trades']
            break

        for w in weights:
            for t in thresholds:
                # Filter for this specific month + weight + threshold
                subset = df_ecr[
                    (df_ecr['launch_month'] == m) &
                    (df_ecr['weight_code'] == w) &
                    (df_ecr['score_threshold'] == t)
                ]
                if not subset.empty:
                    r = subset.iloc[0]
                    cells[(m, w, t)] = (
                        r['strategy_return'] * 100,
                        r['vs_bufr_excess'] * 100,
                        r['num_trades']
                    )
                else:
                    cells[(m, w, t)] = None

            cells[(m, w, 'Existing 90%')] = (
                base_ret, base_vsb, base_trades
            ) if base_ret is not None else None

    # ── Draw table ────────────────────────────────────────────────────────────
    n_rows = len(row_keys)
    n_cols = len(col_keys)

    fig_h = max(6, n_rows * 1.2 + 2)
    fig, ax = plt.subplots(figsize=(4 * n_cols + 2, fig_h))
    ax.axis('off')

    # Header row
    col_labels = [f't={t}' if t != 'Existing 90%' else 'Existing\n90%'
                  for t in col_keys]
    row_labels = [f'{m}\n{WEIGHT_LABELS.get(w, w)}' for m, w in row_keys]

    table_data = []
    cell_colors = []

    for m, w in row_keys:
        row_data   = []
        row_colors = []

        # Get baseline return for this month to color-code
        base_cell = cells.get((m, w, 'Existing 90%'))
        base_ret  = base_cell[0] if base_cell else None

        for t in col_keys:
            c = cells.get((m, w, t))
            if c is None or c[0] is None:
                row_data.append('—')
                row_colors.append('#f5f5f5')
            else:
                ret, vsb, trades = c
                row_data.append(fmt(ret, vsb, trades))
                # Color: green if beats baseline, red if not, gray for baseline col
                if t == 'Existing 90%':
                    row_colors.append('#fff3cd')   # yellow for baseline
                elif base_ret is not None and ret > base_ret:
                    row_colors.append('#d4edda')   # green
                elif base_ret is not None and ret < base_ret:
                    row_colors.append('#fce4e4')   # red
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

    # Header formatting
    for j in range(n_cols):
        tbl[(0, j)].set_facecolor('#2166ac')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    fig.suptitle(
        'Batch 10c: Strategy Performance vs Existing 90% Baseline\n'
        'Green = beats baseline return | Yellow = baseline | Red = underperforms',
        fontsize=12, fontweight='bold', y=0.98
    )
    plt.tight_layout()
    save(fig, 'batch_10c_vs_baseline_table.png')


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('=' * 60)
    print('Batch 10c — Generating plots')
    print('=' * 60)
    print()
    print('Loading data...')
    df_summary, df_ecr, df_base, df_nav, df_sel, df_all, months, weights, thresholds = load_data()

    print('\nPlot 1/5: Score Analysis...')
    plot_score_analysis(df_sel, df_ecr, df_base, months, weights)

    print('Plot 2/5: Score Evolution...')
    plot_score_evolution(df_sel, months, weights)

    # print('Plot 3/5: Trade Comparison...')
    # plot_trade_comparison(df_sel, df_all, months, weights)
    #
    # print('Plot 4/5: Remaining Days...')
    # plot_remaining_days(df_sel, weights)

    print('Plot 5/5: Combined Time Series...')
    plot_combined_time_series(df_nav, df_ecr, df_base, months)
    #
    # print('Plot 6/6: vs Baseline Table...')
    # plot_vs_baseline_table(df_ecr, df_base, months, weights)

    print()
    print('=' * 60)
    print(f'All plots saved to: {OUTPUT_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
