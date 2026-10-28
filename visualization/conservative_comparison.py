"""Readable batch reports: cohort-aligned curves, downside rankings and run links."""
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative

from backtesting.research import comparison_table
from visualization.research_plots import style_figure


def export_batch_comparison(results, cases, folders, output):
    """Results contain summary/daily/parameters only, keeping batch memory bounded."""
    output = Path(output)
    charts = output / 'charts'
    charts.mkdir()
    table = comparison_table(results)
    table['Mean Annualized Return'] = [r['summary']['Annualized Return'].mean() for r in results.values()]
    table['Drawdown Rank'] = table['Worst Drawdown'].rank(ascending=False, method='min').astype(int)
    table['Sortino Rank'] = table['Mean Sortino'].rank(ascending=False, method='min', na_option='bottom').astype(int)
    table['Turnover Rank'] = table['Mean Annualized Turnover'].rank(ascending=True, method='min', na_option='bottom').astype(int)
    configs = []
    for case in cases:
        e = case.experiment
        cfg = e.backtest
        config = {'Experiment': e.name, 'Universe': ', '.join(cfg.universe), 'N': cfg.portfolio_size,
                  'Review': cfg.review, 'Trigger': 'ecr' if cfg.trigger == 'ecr' else cfg.buffer_metric,
                  'Threshold': cfg.threshold if cfg.trigger == 'ecr' else cfg.metric_floor,
                  'Score improvement': cfg.min_selection_improvement, 'Max replacements': cfg.max_replacements_per_review,
                  'Turnover budget': cfg.max_turnover_per_review, 'Weight band': cfg.weight_band,
                  'Report folder': folders[e.name].relative_to(output).as_posix()}
        configs.append(config)
    config_table = pd.DataFrame(configs).set_index('Experiment')
    table = table.join(config_table.drop(columns=['N']))
    table = table.sort_values(['Drawdown Rank', 'Sortino Rank', 'Turnover Rank'], kind='stable')
    cohorts = pd.concat([r['summary'].assign(Experiment=name) for name,r in results.items()], ignore_index=True)
    table.to_csv(output / 'comparison.csv')
    cohorts.to_csv(output / 'cohort_comparison.csv', index=False)
    config_table.to_csv(output / 'configurations.csv')
    launches = next(iter(results.values()))['summary'].Cohort.tolist()
    curves = {}
    with pd.ExcelWriter(output / 'comparison.xlsx', engine='openpyxl') as writer:
        table.to_excel(writer, sheet_name='Comparison')
        cohorts.to_excel(writer, sheet_name='Launch Cohorts', index=False)
        config_table.to_excel(writer, sheet_name='Configurations')
        pd.DataFrame({'Read me': [
            'Sorted by least severe worst-cohort drawdown, then Sortino and turnover. This is a research ordering, not a validated optimum.',
            'Each launch month is a separate backtest. Aggregate columns are arithmetic means of those tests, not returns of one investable portfolio.',
            'Compare curves within the same launch month. Windows must match exactly; missing sessions are not filled.',
            'All scenarios select highest multiplicative ECR, use next-close execution, and share the specified transaction cost.',
            'Inspect the individual report folders for decisions, holdings, candidates, and diagnostics.',
            'Thresholds, portfolio size and turnover rules are in Configurations; full replay settings are in each run/config.json.',
        ]}).to_excel(writer, sheet_name='Read Me', index=False)
        for cohort in launches:
            series = {name: result['daily'].loc[result['daily'].Cohort == cohort].set_index('Date')['Strategy NAV']
                      for name,result in results.items()}
            frame = pd.concat(series, axis=1)
            # Never compare a backfilled or silently shortened price series.
            if frame.isna().any().any():
                raise ValueError(f'{cohort}: scenario NAV calendars differ or contain missing values.')
            curves[cohort] = frame
            frame.to_excel(writer, sheet_name=f'NAV {cohort}')
        from openpyxl.styles import Font, PatternFill
        for sheet in writer.book:
            sheet.freeze_panes = 'B2'
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(color='FFFFFF', bold=True)
                cell.fill = PatternFill('solid', fgColor='1B3442')
            for column in sheet.columns:
                letter = column[0].column_letter
                sheet.column_dimensions[letter].width = min(60, max(16, len(str(column[0].value or ''))+2))
            # Return, drawdown and turnover columns remain numeric and easy to compare.
            for cell in sheet[1]:
                label = str(cell.value or '')
                if any(word in label for word in ('Return', 'Drawdown', 'Deviation', 'Turnover', 'Cumulative Excess', 'vs BUFR', 'Weight band')) and 'Rank' not in label:
                    for row in sheet.iter_rows(min_row=2, min_col=cell.column, max_col=cell.column):
                        row[0].number_format = '0.00%'

    top = table.index[:min(6, len(table))].tolist()
    plot_links = []
    for cohort, frame in curves.items():
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=['Growth of 100', 'Drawdown'], vertical_spacing=.12)
        for i, name in enumerate(frame):
            visible = True if name in top else 'legendonly'
            dd = frame[name] / frame[name].cummax() - 1
            color = qualitative.Dark24[i % len(qualitative.Dark24)]
            for row, values in [(1, frame[name]), (2, dd)]:
                fig.add_trace(go.Scatter(x=frame.index, y=values, name=name, legendgroup=name,
                                        showlegend=row==1, visible=visible, line=dict(color=color)), row=row, col=1)
        ref = next(iter(results.values()))['daily'].query('Cohort == @cohort').set_index('Date')
        for benchmark, color in [('BUFR NAV', '#888888'), ('SPY NAV', '#1B3442')]:
            for row, values in [(1,ref[benchmark]), (2,ref[benchmark]/ref[benchmark].cummax()-1)]:
                fig.add_trace(go.Scatter(x=ref.index, y=values, name=benchmark, legendgroup=benchmark,
                                        showlegend=row==1, line=dict(dash='dot',color=color)), row=row,col=1)
        buttons = []
        for label, selected in [('Downside shortlist',set(top)), ('All scenarios',set(frame)),
                                 *[(f'{n}-fund portfolios', set(table.index[table['Portfolio Size']==n])) for n in sorted(table['Portfolio Size'].unique())]]:
            visibility = [True if name in selected else 'legendonly' for name in frame for _ in range(2)] + [True]*4
            buttons.append(dict(label=label,method='update',args=[{'visible':visibility}]))
        fig.update_layout(title=f'{cohort} launch · all iterations (legend toggles individual lines)',height=950,
                          updatemenus=[dict(buttons=buttons,x=1,y=1.16,xanchor='right')], legend=dict(groupclick='togglegroup'))
        fig.update_yaxes(tickformat='.0%',row=2,col=1)
        style_figure(fig).update_layout(legend_maxheight=.25).write_html(charts / f'{cohort.lower()}-comparison.html',include_plotlyjs='directory')
        plot_links.append((f'{cohort} · interactive performance & drawdown', f'charts/{cohort.lower()}-comparison.html'))
        # Small static panels by portfolio size avoid an unreadable 85-line image.
        sizes = sorted(table['Portfolio Size'].unique())
        figure, axes = plt.subplots(len(sizes),2,figsize=(16,4.5*len(sizes)),squeeze=False)
        for i,n in enumerate(sizes):
            selected = table.index[table['Portfolio Size']==n][:6]
            for name in selected:
                short = name.split(' / ',1)[-1]
                axes[i,0].plot(frame.index,frame[name],label=short,linewidth=1.6)
                axes[i,1].plot(frame.index,(frame[name]/frame[name].cummax()-1)*100,linewidth=1.6)
            for benchmark in ['BUFR NAV', 'SPY NAV']:
                axes[i,0].plot(ref.index,ref[benchmark],label=benchmark,linestyle=':',linewidth=1)
                axes[i,1].plot(ref.index,(ref[benchmark]/ref[benchmark].cummax()-1)*100,linestyle=':',linewidth=1)
            axes[i,0].set_title(f'N={n} · up to six configurations with least severe worst-cohort drawdown')
            axes[i,1].set_title('Drawdown (%)')
            axes[i,0].legend(fontsize=7,loc='upper left')
            for ax in axes[i]:
                ax.grid(alpha=.15)
                ax.tick_params(axis='x',rotation=20)
        figure.suptitle(f'{cohort} launch · downside shortlist (full set in interactive chart)')
        figure.tight_layout()
        figure.savefig(charts / f'{cohort.lower()}-shortlist.png',dpi=140)
        plt.close(figure)

    risk = go.Figure()
    for n, rows in table.groupby('Portfolio Size'):
        risk.add_trace(go.Scatter(x=-rows['Worst Drawdown'],y=rows['Mean Annualized Return'],mode='markers',
                                  name=f'{n} fund(s)',text=rows.index,
                                  customdata=rows[['Mean Annualized Turnover','Mean Sortino']].to_numpy(),
                                  hovertemplate='%{text}<br>Worst drawdown: %{x:.2%}<br>Mean annualized return: %{y:.2%}<br>Turnover/year: %{customdata[0]:.2f}×<br>Sortino: %{customdata[1]:.2f}<extra></extra>'))
    risk.update_layout(title='All iterations · return and downside',xaxis_title='Worst cohort drawdown magnitude (less is better)',
                       yaxis_title='Mean annualized return',xaxis_tickformat='.0%',yaxis_tickformat='.0%',height=600)
    style_figure(risk).write_html(charts/'risk-return.html',include_plotlyjs='directory')
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    for n,rows in table.groupby('Portfolio Size'):
        axes[0].scatter(-rows['Worst Drawdown']*100,rows['Mean Annualized Return']*100,label=f'N={n}',alpha=.7)
        axes[1].scatter(rows['Mean Annualized Turnover'], -rows['Worst Drawdown']*100,label=f'N={n}',alpha=.7)
    axes[0].set(xlabel='Worst drawdown magnitude (%)',ylabel='Mean annualized return (%)',title='Return / downside')
    axes[1].set(xlabel='Mean annualized one-way turnover (×)',ylabel='Worst drawdown magnitude (%)',title='Turnover / downside')
    for ax in axes:
        ax.grid(alpha=.15);ax.legend()
    fig.tight_layout();fig.savefig(charts/'risk-return.png',dpi=150);plt.close(fig)
    links = ''.join(f'<li><a href="{url}">{escape(label)}</a></li>' for label,url in plot_links)
    display = table[['Portfolio Size','Mean Annualized Return','Worst Drawdown','Mean Sortino','Mean Annualized Turnover']].head(12)
    display = display.to_html(float_format=lambda v:f'{v:.4f}',escape=True)
    runs = ''.join(f'<tr><td>{escape(name)}</td><td><a href="{folder.relative_to(output).as_posix()}/ecr_results.xlsx">Excel</a></td><td><a href="{folder.relative_to(output).as_posix()}/charts/">Charts</a></td></tr>' for name,folder in folders.items())
    (output/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Conservative ECR comparison</title>
<style>body{{font:16px Arial;color:#1b3442;max-width:1400px;margin:40px auto;padding:0 24px}}h1{{background:#f9eed7;padding:24px;border-left:4px solid #eaaa43}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}a{{color:#1672c4}}img{{width:100%}}</style>
<h1>Conservative ECR · {len(results)} configurations</h1><p>Highest ECR selects every replacement. Inspect each launch separately; cohort averages are research summaries, not a combined portfolio.</p>
<p><a href="comparison.xlsx">Comparison workbook</a> · <a href="comparison.csv">Comparison CSV</a> · <a href="charts/risk-return.html">Interactive risk / return</a></p>
<img src="charts/risk-return.png" alt="Return, downside and turnover trade-offs"><h2>Performance by launch</h2><ul>{links}</ul>
<h2>Downside shortlist</h2><p>Sorted by least severe worst-cohort drawdown, then Sortino and turnover. Returns and drawdowns below are decimals (0.10 = 10%). This ordering does not establish an out-of-sample winner.</p>{display}
<h2>Individual reports</h2><table><tr><th>Configuration</th><th>Tables</th><th>Pictures & interactive charts</th></tr>{runs}</table></html>''',encoding='utf-8')
    return table
