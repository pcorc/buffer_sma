"""Interactive research views, shared by the notebook and exported HTML."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def dashboard(result, cohort='Jan'):
    daily = result['daily'].query('Cohort == @cohort')
    events = result['diagnostics'].query('Cohort == @cohort')
    if daily.empty:
        raise ValueError(f'No cohort {cohort}')
    config = result['parameters']['backtest']
    is_ecr = config['trigger'] == 'ecr'
    metric_label = 'ECR' if is_ecr else config['buffer_metric'].replace('_', ' ').title()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.09,
                        subplot_titles=('Growth of 100', 'Drawdown', f'{metric_label} · per holding'))
    for column in ['Strategy NAV', 'SPY NAV', 'BUFR NAV', 'Hold NAV']:
        fig.add_trace(go.Scatter(x=daily.Date, y=daily[column], name=column.replace(' NAV', ''),
                                line=dict(width=3 if column == 'Strategy NAV' else 1.5)), row=1, col=1)
    marked = events.merge(daily[['Date', 'Strategy NAV']], on='Date', validate='many_to_one')
    fig.add_trace(go.Scatter(x=marked.Date, y=marked['Strategy NAV'], mode='markers',
                            marker=dict(symbol='diamond', size=7, color='#147D73'), text=marked.Fund,
                            name='Entry / switch', hovertemplate='%{x}<br>%{text}<extra></extra>'), row=1, col=1)
    for column, color in [('Strategy NAV', '#147D73'), ('BUFR NAV', '#8F9BA8')]:
        drawdown = daily[column] / daily[column].cummax() - 1
        fig.add_trace(go.Scatter(x=daily.Date, y=drawdown, name=column.replace(' NAV', '')+' drawdown',
                                line=dict(color=color), fill='tozeroy' if column=='Strategy NAV' else None), row=2, col=1)
    positions = result['holdings'].query('Cohort == @cohort')
    for slot, frame in positions.groupby('Position'):
        fig.add_trace(go.Scatter(x=frame.Date, y=frame['Held ECR Score' if is_ecr else 'Held Buffer Score'],
                                text=frame.Fund, name=f'Position {slot} / {metric_label}',
                                hovertemplate='%{x}<br>%{text}: %{y:.3f}<extra></extra>'), row=3, col=1)
        if config['trigger'] not in ('ecr', 'metric_floor'):
            fig.add_trace(go.Scatter(x=frame.Date, y=frame['Entry Buffer Score']*(1-config['buffer_drop']),
                                    name=f'Position {slot} entry-relative floor', line=dict(dash='dot')), row=3, col=1)
    if config['trigger'] in ('ecr', 'metric_floor'):
        threshold = config['threshold'] if is_ecr else config['metric_floor']
        fig.add_hline(y=threshold, line_dash='dot', line_color='#CB7A43', row=3, col=1,
                      annotation_text=f'Trigger <{ "=" if config["comparator"] == "<=" else ""} {threshold:g}')
    fig.update_layout(height=900, title=f'{config["client_name"]} — {cohort} — N={config["portfolio_size"]}', hovermode='x unified')
    fig.update_yaxes(tickformat='.0%', row=2, col=1)
    return style_figure(fig)


def holdings_figure(result, cohort='Jan'):
    positions = result['holdings'].query('Cohort == @cohort')
    weights = positions.pivot(index='Date', columns='Fund', values='Weight').fillna(0)
    fig = go.Figure()
    for fund in weights:
        fig.add_trace(go.Scatter(x=weights.index, y=weights[fund], name=fund, stackgroup='weights',
                                hovertemplate='%{x}<br>%{y:.1%}<extra>%{fullData.name}</extra>'))
    fig.update_layout(title=f'Actual portfolio weights — {cohort}', yaxis_tickformat='.0%',
                      yaxis_range=[0, 1], height=450, hovermode='x unified')
    return style_figure(fig)


def comparison_figures(results, table):
    names = list(results)
    cohorts = results[names[0]]['summary'].Cohort.tolist()
    z = [results[n]['summary'].set_index('Cohort').loc[cohorts, 'vs BUFR Cumulative Excess'].to_list() for n in names]
    heatmap = go.Figure(go.Heatmap(z=z, x=cohorts, y=names, colorscale='RdBu', zmid=0,
                                  colorbar=dict(tickformat='.0%'), hovertemplate='%{y}<br>%{x}: %{z:.2%}<extra></extra>'))
    heatmap.update_layout(title='Cumulative excess vs BUFR by cohort', height=550)
    tradeoff = go.Figure(go.Scatter(x=-table['Worst Drawdown'], y=table['Mean vs BUFR'],
                                   mode='markers+text', text=table.index, textposition='top center',
                                   marker=dict(size=12, color=table['Mean Switches per Year'],
                                               colorscale='Viridis', colorbar=dict(title='Switches/year'))))
    tradeoff.update_layout(title='Return / drawdown / turnover trade-off', height=500,
                          xaxis_title='Worst cohort drawdown magnitude', yaxis_title='Mean cumulative excess vs BUFR')
    tradeoff.update_xaxes(tickformat='.0%'); tradeoff.update_yaxes(tickformat='.0%')
    return heatmap, tradeoff


def style_figure(fig):
    fig.update_layout(template='plotly_white', paper_bgcolor='white', plot_bgcolor='white',
                      font=dict(family='Arial, sans-serif', color='#34495A'),
                      colorway=['#147D73', '#9AA7B4', '#5789B1', '#C99D5B', '#8578AD', '#CE8475'],
                      margin=dict(l=30, r=25, t=65, b=40),
                      legend=dict(orientation='h', y=-.09, x=0), hovermode='x unified')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#EEF1F4', zerolinecolor='#E2E8ED')
    return fig
