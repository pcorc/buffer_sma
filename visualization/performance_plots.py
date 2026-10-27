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
