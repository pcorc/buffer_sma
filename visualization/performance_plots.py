"""Headless ECR NAV and component reports; no hardcoded machine paths."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_results(result, output):
    cohorts = list(result['daily'].Cohort.unique())
    fig, axes = plt.subplots((len(cohorts) + 2) // 3, 3, figsize=(15, 3.5 * ((len(cohorts) + 2) // 3)),
                             squeeze=False)
    for ax, cohort in zip(axes.flat, cohorts):
        daily = result['daily'].loc[result['daily'].Cohort == cohort]
        for column in ('Strategy NAV', 'SPY NAV', 'BUFR NAV', 'Hold NAV'):
            ax.plot(daily.Date, daily[column], label=column)
        ax.set_title(cohort)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(cohorts):]:
        ax.set_visible(False)
    fig.suptitle('ECR — NAV by launch cohort (each starts at 100)')
    fig.tight_layout()
    fig.savefig(output / 'nav_by_cohort.png', dpi=120)
    plt.close(fig)
    diagnostics = result['diagnostics']
    fig, ax = plt.subplots(figsize=(10, 4))
    for cohort in cohorts:
        rows = diagnostics.loc[diagnostics.Cohort == cohort]
        ax.scatter(rows.Date, rows['ECR'], label=cohort, s=12)
    ax.set(ylabel='ECR at entry/switch', title='Selected-fund ECR diagnostics')
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'selected_scores.png', dpi=120)
    plt.close(fig)


def plot_studio_results(result, output):
    """Portable PNGs of performance, risk, triggers and actual holdings per launch."""
    config = result['parameters']['backtest']
    colors = ['#147D73', '#9AA7B4', '#5789B1', '#C99D5B', '#8578AD', '#CE8475']
    for cohort in result['summary'].Cohort:
        curve = result['daily'].loc[result['daily'].Cohort == cohort]
        positions = result['holdings'].loc[result['holdings'].Cohort == cohort]
        with plt.rc_context({'axes.spines.top': False, 'axes.spines.right': False,
                             'axes.titleweight': 'bold', 'font.size': 9}):
            fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
            for column, color in zip(('Strategy NAV', 'SPY NAV', 'BUFR NAV', 'Hold NAV'), colors):
                axes[0].plot(curve.Date, curve[column], label=column.replace(' NAV', ''), color=color)
            axes[0].set_title('Growth of 100', loc='left')
            for column, color in [('Strategy NAV', colors[0]), ('BUFR NAV', colors[1])]:
                drawdown = (curve[column] / curve[column].cummax() - 1) * 100
                axes[1].plot(curve.Date, drawdown, label=column.replace(' NAV', ''), color=color)
            axes[1].set(title='Drawdown', ylabel='%')
            is_ecr = config['trigger'] == 'ecr'
            for slot, frame in positions.groupby('Position'):
                axes[2].plot(frame.Date, frame['Held ECR Score' if is_ecr else 'Held Buffer Score'], label=f'Position {slot}')
            floor = config['threshold'] if is_ecr else config['metric_floor']
            axes[2].axhline(floor, color='#CB7A43', linestyle=':', label=f'Trigger < {floor:g}')
            axes[2].set_title(('ECR' if is_ecr else config['buffer_metric'].replace('_', ' ').title())+' per holding')
            weights = positions.pivot(index='Date', columns='Fund', values='Weight').fillna(0)
            axes[3].stackplot(weights.index, weights.to_numpy().T * 100, labels=weights.columns, alpha=.85)
            axes[3].set(title='Actual portfolio weights', ylabel='%', ylim=(0, 100))
            for ax in axes:
                ax.grid(axis='y', alpha=.12)
                ax.legend(loc='upper left', ncol=4, fontsize=7)
            fig.suptitle(f'{config["client_name"]} | {cohort} launch | {config["portfolio_size"]} fund(s)', fontsize=15)
            fig.tight_layout(rect=(0, 0, 1, .97))
            fig.savefig(output / f'{cohort.lower()}-report.png', dpi=150)
            plt.close(fig)
