"""
Batch 11: Unified Plotting Suite — Par Proximity Strategy Variants
====================================================================

ONE script that produces all Batch 11 plots, with dynamic filters for:
  - Launch months (subset of FEB, MAR, ..., DEC)
  - w_par values (subset of 0.5, 1.0, 1.5, 2.0)
  - time_shape values (subset of linear, sqrt, squared)
  - Baseline inclusion

Plots produced (configurable via PLOTS_TO_RUN):
  1. performance_matrix       — Heatmap of vs BUFR / return / Sharpe across the (w_par × shape) grid
  2. per_month_rankings       — Ranked bar chart per launch month
  3. par_bonus_decomposition  — Scatter + stacked bar showing par contribution
  4. score_summary            — Boxplot + trade counts + summary table
  5. top_bottom_strategies    — Dual-axis ECR score + NAV for top/bottom N
  6. overlay_per_month        — NAV overlay + par bonus contribution per month

USAGE:
    Edit the CONFIGURATION block below, then run: python plot_batch_11_unified.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import glob
from pathlib import Path

# ============================================================================
# CONFIGURATION — EDIT HERE
# ============================================================================

# Output folder (where Batch 11 result files live)

BATCH_11_DIR = Path(r'C:\Users\PatrickCorcoran\PyCharmProjects\Buffer SMA\MachineLearning_April2025\output\backtest_results\batch_11_all')
# ── Filters ──────────────────────────────────────────────────────────────────
# Set any of these to None to disable that filter (show all).
MONTHS = None                          # e.g. ['FEB', 'JUN'] or None for all
W_PAR_FILTER = ['w1p5', 'w2', 'w2p5']   # ← was None
SHAPE_FILTER = ['lin', 'sqrt']          # ← was None
INCLUDE_BASELINE = True

# ── Plot selection ───────────────────────────────────────────────────────────
# Comment out any you don't want to run.
PLOTS_TO_RUN = [
    'performance_matrix',
    'per_month_rankings',
    'par_bonus_decomposition',
    'score_summary',
    'top_bottom_strategies',
    'overlay_per_month',
]

# ── Plot-specific config ─────────────────────────────────────────────────────
N_TOP = 3       # Number of top strategies (for top_bottom_strategies plot)
N_BOTTOM = 3    # Number of bottom strategies

# ============================================================================
# CONSTANTS — DO NOT EDIT
# ============================================================================

#itera1
W_PAR_ALL = ['w0p5', 'w1', 'w1p5', 'w2', 'w2p5']
W_PAR_LABELS = {'w0p5': '0.5', 'w1': '1.0', 'w1p5': '1.5', 'w2': '2.0', 'w2p5': '2.5'}
W_PAR_INTENSITY = {'w0p5': 0.30, 'w1': 0.48, 'w1p5': 0.65, 'w2': 0.82, 'w2p5': 1.0}
SHAPES_ALL = ['lin', 'sqrt', 'sq']
SHAPE_LABELS = {'lin': 'Linear', 'sqrt': 'Sqrt', 'sq': 'Squared'}

# Color palette
SHAPE_HUE = {'lin': '#2166ac', 'sqrt': '#4dac26', 'sq': '#d6604d'}

COLOR_BASELINE = '#000000'
COLOR_SPY = '#888888'
COLOR_BUFR = '#555555'
COLOR_STRATEGY = '#2166ac'
COLOR_SPY_TB = '#d6604d'      # for top/bottom plot (different scheme)
COLOR_BUFR_TB = '#4dac26'
COLOR_ECR = '#7b2d8b'


# Runtime paths
SUMMARY_PATH = None
SCORING_PATH = None
DAILY_NAV_PATH = None
OUTPUT_DIR = None


# ============================================================================
# SETUP
# ============================================================================

def resolve_paths():
    global SUMMARY_PATH, SCORING_PATH, DAILY_NAV_PATH, OUTPUT_DIR

    matches = sorted(glob.glob(str(BATCH_11_DIR / 'batch11_*.xlsx')))
    if not matches:
        raise FileNotFoundError(f'No batch 11 summary workbook found in {BATCH_11_DIR}')

    SUMMARY_PATH = Path(matches[-1])
    SCORING_PATH = BATCH_11_DIR / 'batch_11_rebalance_scoring.xlsx'
    DAILY_NAV_PATH = BATCH_11_DIR / 'batch_11_daily_time_series.csv'
    OUTPUT_DIR = BATCH_11_DIR


def get_active_w_pars():
    """Return list of w_par values to plot, after filter applied."""
    if W_PAR_FILTER is None:
        return W_PAR_ALL[:]
    return [w for w in W_PAR_ALL if w in W_PAR_FILTER]


def get_active_shapes():
    """Return list of shapes to plot, after filter applied."""
    if SHAPE_FILTER is None:
        return SHAPES_ALL[:]
    return [s for s in SHAPES_ALL if s in SHAPE_FILTER]


def filter_suffix():
    """Generate a suffix for output filenames based on active filters."""
    parts = []
    if MONTHS is not None:
        parts.append('mo-' + '-'.join(MONTHS))
    if W_PAR_FILTER is not None:
        parts.append('w-' + '-'.join(W_PAR_FILTER))
    if SHAPE_FILTER is not None:
        parts.append('sh-' + '-'.join(SHAPE_FILTER))
    if not INCLUDE_BASELINE:
        parts.append('nobase')
    if not parts:
        return ''
    return '_' + '_'.join(parts)


def variant_color(w_par, shape):
    base = mcolors.to_rgb(SHAPE_HUE[shape])
    intensity = W_PAR_INTENSITY[w_par]
    return tuple(c * intensity + 1.0 * (1 - intensity) for c in base)


# ============================================================================
# HELPERS
# ============================================================================

def parse_config(algo):
    """Parse selection algo → (w_par, shape) tuple. 'baseline' for v1."""
    if algo == 'select_highest_new_ecr_composite_111':
        return ('baseline', 'baseline')
    m = re.match(r'select_ecr_par_prox_(w[\dp]+)_(\w+)', algo)
    if m:
        return (m.group(1), m.group(2))
    return ('?', '?')


def matching_weight_desc(selection_algo):
    """Return Weight_Description string the tracker writes for this selection."""
    if selection_algo == 'select_highest_new_ecr_composite_111':
        return 'Weights (1,1,1)'
    w, s = parse_config(selection_algo)
    w_val = float(w.replace('p', '.').replace('w', ''))
    shape_full = {'lin': 'linear', 'sqrt': 'sqrt', 'sq': 'sq'}.get(s, s)
    return f'Weights (1,1,1) par={w_val} shape={shape_full}'


def is_kept(w_par, shape):
    """Apply filters to a (w_par, shape) tuple — does it survive?"""
    if w_par == 'baseline':
        return INCLUDE_BASELINE
    if W_PAR_FILTER is not None and w_par not in W_PAR_FILTER:
        return False
    if SHAPE_FILTER is not None and shape not in SHAPE_FILTER:
        return False
    return True


def apply_filters(df_summary):
    """Filter the summary df by MONTHS / W_PAR_FILTER / SHAPE_FILTER / INCLUDE_BASELINE."""
    df = df_summary.copy()
    df[['w_par', 'shape']] = df['selection_algo'].apply(
        lambda x: pd.Series(parse_config(x))
    )

    # Month filter
    if MONTHS is not None:
        df = df[df['launch_month'].isin(MONTHS)]

    # Apply w_par / shape / baseline filters
    df = df[df.apply(lambda r: is_kept(r['w_par'], r['shape']), axis=1)]

    return df.reset_index(drop=True)


def filter_summary_msg(df_summary_filtered):
    """Print what got included after filtering."""
    months = sorted(df_summary_filtered['launch_month'].unique())
    n_configs = df_summary_filtered['selection_algo'].nunique()
    n_rows = len(df_summary_filtered)
    print(f'  After filters:')
    print(f'    Months ({len(months)}): {months}')
    print(f'    Unique configs: {n_configs}')
    print(f'    Total rows: {n_rows}')


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_summary = pd.read_excel(SUMMARY_PATH, sheet_name='Summary')

    try:
        df_rebal = pd.read_excel(SCORING_PATH, sheet_name='All Rebalances')
    except Exception:
        try:
            df_rebal = pd.read_excel(SCORING_PATH, sheet_name='Selected Funds')
        except Exception:
            df_rebal = pd.read_excel(SCORING_PATH, sheet_name='Selected Funds (2)')
    df_rebal['Date'] = pd.to_datetime(df_rebal['Date'])

    df_nav = None
    if DAILY_NAV_PATH.exists():
        df_nav = pd.read_csv(DAILY_NAV_PATH)
        df_nav['Date'] = pd.to_datetime(df_nav['Date'])
        print(f'  ✅ Loaded daily NAV CSV ({len(df_nav)} rows)')
    else:
        print(f'  ⚠️  Daily NAV CSV not found at {DAILY_NAV_PATH}')

    return df_summary, df_rebal, df_nav


def save(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path.name}')


# ============================================================================
# NAV column helpers (for daily NAV CSV)
# ============================================================================

def find_nav_columns_for_month(df_nav, launch_month):
    """Map label → column name for NAV columns of one month."""
    if df_nav is None:
        return {}

    out = {}
    for c in df_nav.columns:
        if c == f'{launch_month}_SPY_NAV':
            out['SPY'] = c
        elif c == f'{launch_month}_BUFR_NAV':
            out['BUFR'] = c

    for c in df_nav.columns:
        if c.startswith(f'{launch_month}_ECR_Score_') and c.endswith('_NAV'):
            out['Baseline'] = c
            break

    for c in df_nav.columns:
        m = re.match(rf'{launch_month}_ParProx_(w[\dp]+)_(\w+?)_.*_NAV$', c)
        if m:
            w_part = m.group(1)
            shape = m.group(2)
            if w_part in W_PAR_LABELS and shape in SHAPE_LABELS:
                key = f'{W_PAR_LABELS[w_part]}/{SHAPE_LABELS[shape]}'
                out[key] = c

    return out


def find_strategy_nav_column(df_nav, launch_month, selection_algo):
    """Locate strategy NAV column in daily CSV."""
    if df_nav is None:
        return None

    w, s = parse_config(selection_algo)
    if w == 'baseline':
        candidates = [c for c in df_nav.columns
                      if c.startswith(f'{launch_month}_ECR_Score_')
                      and c.endswith('_NAV')]
        return candidates[0] if candidates else None

    candidates = [c for c in df_nav.columns
                  if c.startswith(f'{launch_month}_ParProx_{w}_{s}_')
                  and c.endswith('_NAV')]
    return candidates[0] if candidates else None


# ============================================================================
# Selected-fund helpers
# ============================================================================

def get_strategy_trades(df_rebal, launch_month, selection_algo):
    """Selected-fund rows for one (month, selection_algo) strategy."""
    weight_desc = matching_weight_desc(selection_algo)
    sub = df_rebal[
        (df_rebal['Month'] == launch_month)
        & (df_rebal['Weight_Description'] == weight_desc)
        & (df_rebal['Selected'] == True)
    ].copy()
    return sub.sort_values('Date').reset_index(drop=True)


def get_strategy_par_bonuses(df_rebal, launch_month):
    """
    Dict keyed by (w_par_label, shape) → DataFrame of selected rebalances.
    Respects filters.
    """
    out = {}
    sub = df_rebal[
        (df_rebal['Month'] == launch_month)
        & (df_rebal['Selected'] == True)
        & (df_rebal['Par_Bonus'].notna())
    ].copy()

    if sub.empty:
        return out

    for desc in sub['Weight_Description'].unique():
        m = re.search(r'par=(\S+)\s+shape=(\w+)', str(desc))
        if not m:
            continue
        par_val = m.group(1)
        shape_full = m.group(2)
        shape_short = {'linear': 'lin', 'sqrt': 'sqrt', 'sq': 'sq'}.get(shape_full, shape_full)
        par_f = float(par_val)
        if par_f == int(par_f):
            w_label = f'w{int(par_f)}'
        else:
            w_label = f'w{par_val.replace(".", "p")}'

        if not is_kept(w_label, shape_short):
            continue

        strat = sub[sub['Weight_Description'] == desc].copy()
        strat = strat.sort_values('Date').reset_index(drop=True)
        out[(w_label, shape_short)] = strat[['Date', 'Composite_Score', 'Par_Bonus', 'Fund']]

    return out


# ============================================================================
# PLOT 1 — Performance Matrix
# ============================================================================

def plot_performance_matrix(df_summary, df_rebal, df_nav, months_list):
    """
    Heatmap matrix (active_w_pars × active_shapes) showing avg vs BUFR, return, Sharpe.
    Baseline shown in side panel if INCLUDE_BASELINE.
    """
    w_pars = get_active_w_pars()
    shapes = get_active_shapes()

    if not w_pars or not shapes:
        print('  ⚠️  No w_par or shape values in filter — skipping performance_matrix')
        return

    agg = df_summary.groupby(['w_par', 'shape']).agg(
        avg_return=('strategy_return', 'mean'),
        avg_vs_bufr=('vs_bufr_excess', 'mean'),
        avg_sharpe=('strategy_sharpe', 'mean'),
        avg_trades=('num_trades', 'mean'),
        avg_max_dd=('strategy_max_dd', 'mean'),
    ).reset_index()

    baseline_row = agg[agg['w_par'] == 'baseline']
    if baseline_row.empty:
        base_vs_bufr = 0.0
        baseline = None
    else:
        baseline = baseline_row.iloc[0]
        base_vs_bufr = baseline['avg_vs_bufr']

    # Build matrices
    matrix_return = np.full((len(w_pars), len(shapes)), np.nan)
    matrix_sharpe = np.full((len(w_pars), len(shapes)), np.nan)
    matrix_vs_bufr = np.full((len(w_pars), len(shapes)), np.nan)

    for i, w in enumerate(w_pars):
        for j, s in enumerate(shapes):
            row = agg[(agg['w_par'] == w) & (agg['shape'] == s)]
            if not row.empty:
                matrix_return[i, j] = row['avg_return'].iloc[0] * 100
                matrix_sharpe[i, j] = row['avg_sharpe'].iloc[0]
                matrix_vs_bufr[i, j] = row['avg_vs_bufr'].iloc[0] * 100

    n_panels = 3 + (1 if (baseline is not None and INCLUDE_BASELINE) else 0)
    width_ratios = [1.3, 1.3, 1.3] + ([0.5] if n_panels == 4 else [])

    fig = plt.figure(figsize=(6 * n_panels, 7))
    gs = fig.add_gridspec(1, n_panels, width_ratios=width_ratios)

    def draw_matrix(ax, matrix, title, fmt, cmap='RdYlGn', center=None):
        if center is not None:
            vmax = np.nanmax(np.abs(matrix - center))
            if vmax == 0 or np.isnan(vmax):
                vmax = 1.0
            im = ax.imshow(matrix, cmap=cmap, vmin=center - vmax, vmax=center + vmax, aspect='auto')
        else:
            im = ax.imshow(matrix, cmap=cmap, aspect='auto')

        ax.set_xticks(range(len(shapes)))
        ax.set_xticklabels([SHAPE_LABELS[s] for s in shapes], fontsize=10)
        ax.set_yticks(range(len(w_pars)))
        ax.set_yticklabels([f'w_par = {W_PAR_LABELS[w]}' for w in w_pars], fontsize=10)
        ax.set_xlabel('Time Shape', fontsize=11, fontweight='bold')
        ax.set_ylabel('Par Bonus Weight', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, fmt.format(matrix[i, j]),
                            ha='center', va='center', fontsize=10,
                            fontweight='bold', color='black')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax1 = fig.add_subplot(gs[0, 0])
    title1 = f'vs BUFR Excess (%)'
    if baseline is not None:
        title1 += f'\nBaseline: {base_vs_bufr*100:+.2f}%'
    draw_matrix(ax1, matrix_vs_bufr, title1, '{:+.2f}%',
                cmap='RdYlGn', center=base_vs_bufr * 100 if baseline is not None else None)

    ax2 = fig.add_subplot(gs[0, 1])
    draw_matrix(ax2, matrix_return, 'Avg Total Return (%)', '{:.1f}%', cmap='Blues')

    ax3 = fig.add_subplot(gs[0, 2])
    draw_matrix(ax3, matrix_sharpe, 'Avg Sharpe', '{:.2f}', cmap='Greens')

    if baseline is not None and INCLUDE_BASELINE:
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        base_text = (
            f"BASELINE\n"
            f"(ECR v1, w=1,1,1)\n"
            f"{'─' * 18}\n\n"
            f"Return:    {baseline['avg_return']*100:.1f}%\n"
            f"Sharpe:    {baseline['avg_sharpe']:.2f}\n"
            f"vs BUFR:   {baseline['avg_vs_bufr']*100:+.2f}%\n"
            f"Max DD:    {baseline['avg_max_dd']*100:.1f}%\n"
            f"Trades:    {baseline['avg_trades']:.1f}\n"
            f"\n{'─' * 18}\n"
            f"Months:    {len(months_list)}\n"
            f"({', '.join(months_list)})"
        )
        ax4.text(0.05, 0.5, base_text, transform=ax4.transAxes,
                 fontsize=10, family='monospace', va='center', ha='left',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#fff3cd',
                           edgecolor='#856404', linewidth=1.5))

    fig.suptitle(
        f'Batch 11: Performance Matrix — averaged across launch months',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    save(fig, f'batch_11_performance_matrix{filter_suffix()}.png')


# ============================================================================
# PLOT 2 — Per-Month Rankings
# ============================================================================

def plot_per_month_rankings(df_summary, df_rebal, df_nav, months_list):
    n = len(months_list)
    if n == 0:
        print('  ⚠️  No months in filter — skipping per_month_rankings')
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 8), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        'Batch 11: Strategy Rankings by Launch Month (vs BUFR)',
        fontsize=14, fontweight='bold', y=1.02
    )

    for ax, month in zip(axes, months_list):
        sub = df_summary[df_summary['launch_month'] == month].copy()
        sub['vs_bufr_pct'] = sub['vs_bufr_excess'] * 100
        sub = sub.sort_values('vs_bufr_pct')

        def lbl(row):
            if row['w_par'] == 'baseline':
                return 'BASELINE'
            return f'{W_PAR_LABELS[row["w_par"]]} / {SHAPE_LABELS[row["shape"]]}'

        sub['label'] = sub.apply(lbl, axis=1)

        def color(row):
            if row['w_par'] == 'baseline':
                return '#fff3cd'
            return variant_color(row['w_par'], row['shape'])

        colors = sub.apply(color, axis=1).tolist()

        bars = ax.barh(range(len(sub)), sub['vs_bufr_pct'],
                       color=colors, edgecolor='black', linewidth=0.6)

        for bar, vs_bufr, trades in zip(bars, sub['vs_bufr_pct'], sub['num_trades']):
            ax.text(bar.get_width() + (0.05 if vs_bufr >= 0 else -0.05),
                    bar.get_y() + bar.get_height() / 2,
                    f'{vs_bufr:+.2f}%  (T={int(trades)})',
                    va='center', ha='left' if vs_bufr >= 0 else 'right',
                    fontsize=8)

        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub['label'], fontsize=9)
        ax.set_xlabel('vs BUFR Excess (%)', fontsize=10)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_title(f'{month} launch', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    save(fig, f'batch_11_per_month_rankings{filter_suffix()}.png')


# ============================================================================
# PLOT 3 — Par Bonus Decomposition
# ============================================================================

def plot_par_bonus_decomposition(df_summary, df_rebal, df_nav, months_list):
    if df_rebal is None or 'Par_Bonus' not in df_rebal.columns:
        print('  ⚠️  No par_bonus data — skipping decomposition plot')
        return

    par_data = df_rebal[df_rebal['Par_Bonus'].notna()].copy()
    if MONTHS is not None:
        par_data = par_data[par_data['Month'].isin(MONTHS)]

    if par_data.empty:
        print('  ⚠️  No par_bonus rows after filter — skipping decomposition')
        return

    def parse_desc(desc):
        m = re.search(r'par=(\S+)\s+shape=(\w+)', str(desc))
        if not m:
            return '?', '?'
        par_val = m.group(1)
        shape_full = m.group(2)
        par_f = float(par_val)
        w_label = f'w{int(par_f)}' if par_f == int(par_f) else f'w{par_val.replace(".", "p")}'
        shape_short = {'linear': 'lin', 'sqrt': 'sqrt', 'sq': 'sq'}.get(shape_full, shape_full)
        return w_label, shape_short

    par_data[['w_par_val', 'shape_name']] = par_data['Weight_Description'].apply(
        lambda x: pd.Series(parse_desc(x))
    )

    # Apply filter
    par_data = par_data[par_data.apply(
        lambda r: is_kept(r['w_par_val'], r['shape_name']), axis=1
    )]

    if par_data.empty:
        print('  ⚠️  No par_bonus rows after w_par/shape filter — skipping decomposition')
        return

    par_data['structural_base'] = (
        par_data['DBB_Score'] + par_data['Buffer_Integrity'] + par_data['Cap_Integrity']
    )
    par_data['par_pct_of_composite'] = par_data['Par_Bonus'] / par_data['Composite_Score'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        'Batch 11: Par Bonus Decomposition — How decisive is the par signal?',
        fontsize=14, fontweight='bold', y=1.02
    )

    # ── Left: scatter ─────────────────────────────────────────────────────────
    ax = axes[0]
    for shape_name in get_active_shapes():
        sub = par_data[par_data['shape_name'] == shape_name]
        if sub.empty:
            continue
        c = SHAPE_HUE[shape_name]
        label = SHAPE_LABELS[shape_name]
        ax.scatter(sub['structural_base'], sub['Par_Bonus'],
                   color=c, alpha=0.5, s=40, label=label, edgecolor='black', linewidth=0.4)

    ax.set_xlabel('Structural Base (DBB + BI + CI)', fontsize=11)
    ax.set_ylabel('Par Bonus', fontsize=11)
    ax.set_title('Par Bonus vs Structural Score at Trade Time',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, title='Time Shape', loc='best')
    ax.grid(True, alpha=0.3)

    if not par_data.empty:
        xmax = par_data['structural_base'].max() * 1.05
        ymax = par_data['Par_Bonus'].max() * 1.1
        for total in [2.0, 2.5, 3.0, 3.5]:
            xs = np.linspace(0, xmax, 50)
            ys = total - xs
            mask = ys >= 0
            ax.plot(xs[mask], ys[mask], '--', color='gray', alpha=0.3, linewidth=0.8)
            ax.text(xmax * 0.85, total - xmax * 0.85, f'composite={total}',
                    color='gray', fontsize=8, alpha=0.7, ha='right')
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)

    # ── Right: stacked bar ───────────────────────────────────────────────────
    ax = axes[1]
    agg = par_data.groupby('Weight_Description').agg(
        avg_struct=('structural_base', 'mean'),
        avg_par_bonus=('Par_Bonus', 'mean'),
        avg_par_pct=('par_pct_of_composite', 'mean'),
    ).reset_index()
    agg = agg.sort_values('avg_par_bonus', ascending=True)

    short_names = agg['Weight_Description'].str.replace(
        r'Weights \(1,1,1\)\s*', '', regex=True
    ).str.replace('par=', 'p=', regex=False).str.replace('shape=', '', regex=False)

    y = np.arange(len(agg))
    ax.barh(y, agg['avg_struct'], color='#a6cee3', label='Structural (DBB+BI+CI)',
            edgecolor='black', linewidth=0.5)
    ax.barh(y, agg['avg_par_bonus'], left=agg['avg_struct'],
            color='#e31a1c', label='Par Bonus', edgecolor='black', linewidth=0.5)

    for i, (struct, par_b, par_pct) in enumerate(zip(agg['avg_struct'], agg['avg_par_bonus'], agg['avg_par_pct'])):
        ax.text(struct + par_b + 0.05, i, f'{par_pct:.1f}% par',
                va='center', ha='left', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel('Composite Score', fontsize=11)
    ax.set_title('Avg Composite Decomposition at Selected Trade Events',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    save(fig, f'batch_11_par_bonus_decomposition{filter_suffix()}.png')


# ============================================================================
# PLOT 4 — Score Distribution + Trade Counts + Summary
# ============================================================================

def plot_score_summary(df_summary, df_rebal, df_nav, months_list):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(
        'Batch 11: Score Distribution + Trade Activity + Performance Summary',
        fontsize=14, fontweight='bold', y=1.02
    )

    shapes = get_active_shapes()

    # ── Left: boxplot ─────────────────────────────────────────────────────────
    ax = axes[0]
    if df_rebal is not None and 'Par_Bonus' in df_rebal.columns:
        par_data = df_rebal[df_rebal['Par_Bonus'].notna()].copy()
        if MONTHS is not None:
            par_data = par_data[par_data['Month'].isin(MONTHS)]

        def get_shape(desc):
            m = re.search(r'shape=(\w+)', str(desc))
            if m:
                s = m.group(1)
                return {'linear': 'lin', 'sqrt': 'sqrt', 'sq': 'sq'}.get(s, '?')
            return '?'

        par_data['shape'] = par_data['Weight_Description'].apply(get_shape)

        # Optionally filter par_data by w_par via re-check
        def get_w(desc):
            m = re.search(r'par=(\S+)\s+shape', str(desc))
            if not m:
                return '?'
            par_f = float(m.group(1))
            return f'w{int(par_f)}' if par_f == int(par_f) else f'w{m.group(1).replace(".", "p")}'

        par_data['w_par'] = par_data['Weight_Description'].apply(get_w)
        par_data = par_data[par_data.apply(
            lambda r: is_kept(r['w_par'], r['shape']), axis=1
        )]

        data = []
        labels = []
        colors = []
        for s in shapes:
            sub = par_data[par_data['shape'] == s]['Composite_Score'].values
            if len(sub) > 0:
                data.append(sub)
                labels.append(SHAPE_LABELS[s])
                colors.append(SHAPE_HUE[s])

        if INCLUDE_BASELINE:
            base_data = df_rebal[df_rebal['Par_Bonus'].isna()]
            if MONTHS is not None:
                base_data = base_data[base_data['Month'].isin(MONTHS)]
            base_vals = base_data['Composite_Score'].values
            if len(base_vals) > 0:
                data.append(base_vals)
                labels.append('Baseline')
                colors.append(COLOR_BUFR)

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)

    ax.set_title('Composite Score at Trade Time\nby Time Shape',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('ECR Composite Score')
    ax.grid(True, alpha=0.3, axis='y')

    # ── Middle: trade count per config ────────────────────────────────────────
    ax = axes[1]
    df_summary = df_summary.copy()
    df_summary['config_label'] = df_summary.apply(
        lambda r: 'Baseline' if r['w_par'] == 'baseline'
                  else f'{W_PAR_LABELS[r["w_par"]]}/{SHAPE_LABELS[r["shape"]][:3]}',
        axis=1
    )

    pivot = df_summary.pivot_table(
        index='config_label', columns='launch_month', values='num_trades',
        aggfunc='mean'
    )

    config_order = []
    for w in get_active_w_pars():
        for s in shapes:
            config_order.append(f'{W_PAR_LABELS[w]}/{SHAPE_LABELS[s][:3]}')
    if INCLUDE_BASELINE:
        config_order.append('Baseline')
    pivot = pivot.reindex([c for c in config_order if c in pivot.index])

    if not pivot.empty:
        pivot.plot(kind='barh', ax=ax, width=0.8, edgecolor='black', linewidth=0.4)
    ax.set_title('Number of Trades\nby Launch Month', fontsize=11, fontweight='bold')
    ax.set_xlabel('Trades')
    ax.set_ylabel('')
    ax.legend(title='Month', fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    # ── Right: table ──────────────────────────────────────────────────────────
    ax = axes[2]
    ax.axis('off')

    agg = df_summary.groupby(['w_par', 'shape']).agg(
        ret=('strategy_return', 'mean'),
        sharpe=('strategy_sharpe', 'mean'),
        vs_bufr=('vs_bufr_excess', 'mean'),
        dd=('strategy_max_dd', 'mean'),
        trades=('num_trades', 'mean'),
    ).reset_index()
    agg = agg.sort_values('vs_bufr', ascending=False)

    headers = ['Config', 'Return', 'Sharpe', 'vs BUFR', 'Max DD', 'Trades']
    rows = []
    for _, r in agg.iterrows():
        if r['w_par'] == 'baseline':
            cfg = 'BASELINE'
        else:
            cfg = f'w={W_PAR_LABELS[r["w_par"]]}, {SHAPE_LABELS[r["shape"]]}'
        rows.append([
            cfg,
            f'{r["ret"]*100:.1f}%',
            f'{r["sharpe"]:.2f}',
            f'{r["vs_bufr"]*100:+.2f}%',
            f'{r["dd"]*100:.1f}%',
            f'{r["trades"]:.1f}',
        ])

    if not rows:
        ax.text(0.5, 0.5, 'No configs to summarize (filter too restrictive)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='gray', style='italic')
    else:
        tbl = ax.table(cellText=rows, colLabels=headers,
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.5)

        for j in range(len(headers)):
            tbl[(0, j)].set_facecolor('#2166ac')
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')

        for i in range(1, len(rows) + 1):
            cfg_str = rows[i - 1][0]
            if cfg_str == 'BASELINE':
                bg = '#fff3cd'
            elif i == 1:
                bg = '#d4edda'
            else:
                bg = '#f0f4fa' if i % 2 == 0 else 'white'
            for j in range(len(headers)):
                tbl[(i, j)].set_facecolor(bg)

    ax.set_title('Performance Ranked by vs BUFR\n(Best at top)',
                 fontsize=11, fontweight='bold', pad=20)

    plt.tight_layout()
    save(fig, f'batch_11_score_summary{filter_suffix()}.png')


# ============================================================================
# PLOT 5 — Top/Bottom N strategies (dual axis)
# ============================================================================

def short_label_for_strategy(launch_month, selection_algo):
    w, s = parse_config(selection_algo)
    if w == 'baseline':
        return f'{launch_month} | BASELINE'
    w_clean = w.replace('p', '.').replace('w', 'w=')
    shape_clean = SHAPE_LABELS.get(s, s)
    return f'{launch_month} | {w_clean}, {shape_clean}'


def plot_one_strategy(ax_left, ax_right, df_nav, df_rebal,
                      launch_month, selection_algo, summary_row, panel_color):
    """Plot one strategy on a pair of dual axes."""
    title = short_label_for_strategy(launch_month, selection_algo)
    vs_bufr = summary_row['vs_bufr_excess'] * 100
    total_ret = summary_row['strategy_return'] * 100
    sharpe = summary_row['strategy_sharpe']
    trades = int(summary_row['num_trades'])

    ax_left.set_facecolor('#fafafa')

    trades_df = get_strategy_trades(df_rebal, launch_month, selection_algo)

    if not trades_df.empty:
        ax_left.step(trades_df['Date'], trades_df['Composite_Score'],
                     where='post', color=COLOR_ECR, linewidth=2.0,
                     label='Held-Fund ECR Score', zorder=3)
        ax_left.scatter(trades_df['Date'], trades_df['Composite_Score'],
                        color=COLOR_ECR, s=50, zorder=4, edgecolor='white',
                        linewidth=1.0)
        for _, r in trades_df.iterrows():
            ax_left.annotate(r['Fund'][-3:],
                             (r['Date'], r['Composite_Score']),
                             textcoords='offset points', xytext=(0, 9),
                             ha='center', fontsize=7,
                             color=COLOR_ECR, fontweight='bold')

    ax_left.set_ylabel('ECR Composite Score', color=COLOR_ECR, fontsize=10,
                       fontweight='bold')
    ax_left.tick_params(axis='y', labelcolor=COLOR_ECR)
    ax_left.grid(True, alpha=0.25)
    ax_left.set_ylim(bottom=0)

    if df_nav is not None:
        spy_col = next((c for c in df_nav.columns
                        if c == f'{launch_month}_SPY_NAV'), None)
        bufr_col = next((c for c in df_nav.columns
                         if c == f'{launch_month}_BUFR_NAV'), None)
        strat_col = find_strategy_nav_column(df_nav, launch_month, selection_algo)

        if spy_col:
            d = df_nav[['Date', spy_col]].dropna()
            ax_right.plot(d['Date'], d[spy_col], color=COLOR_SPY_TB,
                          linewidth=1.4, alpha=0.85, label='SPY', zorder=2)
        if bufr_col:
            d = df_nav[['Date', bufr_col]].dropna()
            ax_right.plot(d['Date'], d[bufr_col], color=COLOR_BUFR_TB,
                          linewidth=1.4, linestyle='--', alpha=0.85,
                          label='BUFR', zorder=2)
        if strat_col:
            d = df_nav[['Date', strat_col]].dropna()
            ax_right.plot(d['Date'], d[strat_col], color=COLOR_STRATEGY,
                          linewidth=2.2, label='Strategy', zorder=3)

    ax_right.set_ylabel('NAV (rebased to 100)', color='black', fontsize=10,
                        fontweight='bold')
    ax_right.tick_params(axis='y', labelcolor='black')

    title_full = (
        f'{title}\n'
        f'Return: {total_ret:+.1f}%  |  '
        f'vs BUFR: {vs_bufr:+.2f}%  |  '
        f'Sharpe: {sharpe:.2f}  |  '
        f'Trades: {trades}'
    )
    ax_left.set_title(title_full, fontsize=10, fontweight='bold',
                      color=panel_color, pad=10)

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_right.legend(lines_left + lines_right, labels_left + labels_right,
                    loc='upper left', fontsize=8, framealpha=0.9, ncol=2)
    ax_left.tick_params(axis='x', rotation=30, labelsize=8)


def plot_top_bottom_strategies(df_summary, df_rebal, df_nav, months_list):
    n_total = len(df_summary)
    if n_total < 2:
        print(f'  ⚠️  Only {n_total} strategies after filter — skipping top/bottom')
        return

    # Clamp N if too small
    n_top = min(N_TOP, max(1, n_total // 2))
    n_bot = min(N_BOTTOM, max(1, n_total // 2))

    if n_top < N_TOP or n_bot < N_BOTTOM:
        print(f'  ⚠️  Universe is {n_total}; using top {n_top} / bottom {n_bot}')

    ranked = df_summary.sort_values('vs_bufr_excess', ascending=False).reset_index(drop=True)
    top = ranked.head(n_top).copy()
    bottom = ranked.tail(n_bot).copy().sort_values('vs_bufr_excess', ascending=False)

    n_rows = n_top + n_bot
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(
        f'Batch 11: Top {n_top} & Bottom {n_bot} Strategies\n'
        f'ECR Score (left) vs. NAV Performance (right) over time',
        fontsize=14, fontweight='bold', y=1.0
    )

    for i, (_, row) in enumerate(top.iterrows()):
        ax_left = axes[i]
        ax_right = ax_left.twinx()
        plot_one_strategy(ax_left, ax_right, df_nav, df_rebal,
                          row['launch_month'], row['selection_algo'], row,
                          panel_color='#1a9850')
        ax_left.text(-0.06, 0.5, f'TOP #{i+1}', transform=ax_left.transAxes,
                     fontsize=12, fontweight='bold', color='#1a9850',
                     rotation=90, va='center', ha='center')

    for i, (_, row) in enumerate(bottom.iterrows()):
        ax_left = axes[n_top + i]
        ax_right = ax_left.twinx()
        plot_one_strategy(ax_left, ax_right, df_nav, df_rebal,
                          row['launch_month'], row['selection_algo'], row,
                          panel_color='#d73027')
        rank_from_bottom = n_bot - i
        ax_left.text(-0.06, 0.5, f'BOTTOM #{rank_from_bottom}',
                     transform=ax_left.transAxes,
                     fontsize=12, fontweight='bold', color='#d73027',
                     rotation=90, va='center', ha='center')

    plt.tight_layout()
    save(fig, f'batch_11_top_bottom_strategies{filter_suffix()}.png')


# ============================================================================
# PLOT 6 — Overlay per launch month (NAV + par bonus)
# ============================================================================

def plot_overlay_per_month(df_summary, df_rebal, df_nav, months_list):
    for month in months_list:
        plot_overlay_month(month, df_summary, df_rebal, df_nav)


def plot_overlay_month(launch_month, df_summary, df_rebal, df_nav):
    nav_cols = find_nav_columns_for_month(df_nav, launch_month) if df_nav is not None else {}
    have_nav = df_nav is not None and bool(nav_cols)

    w_pars = get_active_w_pars()
    shapes = get_active_shapes()

    if have_nav:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 11),
                                             gridspec_kw={'height_ratios': [2, 1]},
                                             sharex=True)
    else:
        fig, ax_bot = plt.subplots(1, 1, figsize=(16, 7))
        ax_top = None

    # ── Top: NAV curves ───────────────────────────────────────────────────────
    if have_nav:
        if 'SPY' in nav_cols:
            ax_top.plot(df_nav['Date'], df_nav[nav_cols['SPY']],
                        color=COLOR_SPY, linewidth=1.2, alpha=0.7,
                        label='SPY', zorder=1)
        if 'BUFR' in nav_cols:
            ax_top.plot(df_nav['Date'], df_nav[nav_cols['BUFR']],
                        color=COLOR_BUFR, linewidth=1.2, linestyle='--',
                        alpha=0.7, label='BUFR', zorder=1)

        for w in w_pars:
            for s in shapes:
                key = f'{W_PAR_LABELS[w]}/{SHAPE_LABELS[s]}'
                if key in nav_cols:
                    color = variant_color(w, s)
                    ax_top.plot(df_nav['Date'], df_nav[nav_cols[key]],
                                color=color, linewidth=1.3, alpha=0.85,
                                label=f'w={W_PAR_LABELS[w]}, {SHAPE_LABELS[s]}',
                                zorder=2)

        if INCLUDE_BASELINE and 'Baseline' in nav_cols:
            ax_top.plot(df_nav['Date'], df_nav[nav_cols['Baseline']],
                        color=COLOR_BASELINE, linewidth=2.5,
                        label='BASELINE (ECR v1)', zorder=3)

        ax_top.set_ylabel('NAV (rebased to 100)', fontsize=11, fontweight='bold')
        ax_top.set_title(
            f'{launch_month} Launch — All Strategies vs Benchmarks',
            fontsize=12, fontweight='bold'
        )
        ax_top.legend(loc='upper left', fontsize=8, ncol=3, framealpha=0.92)
        ax_top.grid(True, alpha=0.3)

    # ── Bottom: par_bonus contribution ────────────────────────────────────────
    par_bonuses = get_strategy_par_bonuses(df_rebal, launch_month)

    if not par_bonuses:
        ax_bot.text(0.5, 0.5, 'No par_bonus data for this filter',
                    transform=ax_bot.transAxes, ha='center', va='center',
                    fontsize=12, color='gray', style='italic')
    else:
        for w in w_pars:
            for s in shapes:
                key = (w, s)
                if key not in par_bonuses:
                    continue
                df_b = par_bonuses[key]
                color = variant_color(w, s)
                label = f'w={W_PAR_LABELS[w]}, {SHAPE_LABELS[s]}'
                ax_bot.plot(df_b['Date'], df_b['Par_Bonus'],
                            color=color, linewidth=1.3, alpha=0.7,
                            label=label, zorder=2)
                ax_bot.scatter(df_b['Date'], df_b['Par_Bonus'],
                               color=color, s=45, alpha=0.85, edgecolor='white',
                               linewidth=0.5, zorder=3)

        ax_bot.set_ylabel('Par Bonus\n(w_par × par_proximity × time_elapsed)',
                          fontsize=10, fontweight='bold')
        ax_bot.set_xlabel('Date', fontsize=11)
        title = (f'Par Bonus Contribution — "new time scaling" ({launch_month} Launch)'
                 if not have_nav else
                 'Par Bonus Contribution at Each Trade Event — "new time scaling"')
        ax_bot.set_title(title, fontsize=11, fontweight='bold')
        ax_bot.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.tick_params(axis='x', rotation=30, labelsize=9)
        ax_bot.legend(loc='upper left', fontsize=8, ncol=3, framealpha=0.92)

    plt.tight_layout()
    save(fig, f'batch_11_overlay_{launch_month}{filter_suffix()}.png')


# ============================================================================
# DISPATCH
# ============================================================================

PLOT_FUNCS = {
    'performance_matrix':      plot_performance_matrix,
    'per_month_rankings':      plot_per_month_rankings,
    'par_bonus_decomposition': plot_par_bonus_decomposition,
    'score_summary':           plot_score_summary,
    'top_bottom_strategies':   plot_top_bottom_strategies,
    'overlay_per_month':       plot_overlay_per_month,
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('=' * 60)
    print('Batch 11 — Unified plotting suite')
    print('=' * 60)
    print()
    print('Active filters:')
    print(f'  MONTHS:           {MONTHS}')
    print(f'  W_PAR_FILTER:     {W_PAR_FILTER}')
    print(f'  SHAPE_FILTER:     {SHAPE_FILTER}')
    print(f'  INCLUDE_BASELINE: {INCLUDE_BASELINE}')
    print(f'  PLOTS_TO_RUN:     {PLOTS_TO_RUN}')
    print()

    resolve_paths()
    print('Loading data...')
    df_summary_raw, df_rebal, df_nav = load_data()
    df_summary = apply_filters(df_summary_raw)
    filter_summary_msg(df_summary)

    months_list = sorted(df_summary['launch_month'].unique())

    if df_summary.empty:
        print('\n  ❌ No data to plot after filters. Check your filter settings.')
        return

    for plot_name in PLOTS_TO_RUN:
        if plot_name not in PLOT_FUNCS:
            print(f'\n  ⚠️  Unknown plot: {plot_name}')
            continue
        print(f'\nRunning: {plot_name}...')
        try:
            PLOT_FUNCS[plot_name](df_summary, df_rebal, df_nav, months_list)
        except Exception as e:
            print(f'  ❌ Error in {plot_name}: {e}')
            import traceback
            traceback.print_exc()

    print()
    print('=' * 60)
    print(f'Done. Output saved to: {OUTPUT_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
