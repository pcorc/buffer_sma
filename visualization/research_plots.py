"""Interactive research views, shared by the notebook and exported HTML."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def dashboard(result, cohort='Jan'):
    daily = result['daily'].query('Cohort == @cohort')
    events = result['diagnostics'].query('Cohort == @cohort')
    if daily.empty:
        raise ValueError(f'No cohort {cohort}')
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=('NAV and executed switches', 'Drawdown', 'Held buffer score versus entry anchor'))
    for column in ['Strategy NAV', 'SPY NAV', 'BUFR NAV', 'Hold NAV']:
        fig.add_trace(go.Scatter(x=daily.Date, y=daily[column], name=column), row=1, col=1)
    marked = events.merge(daily[['Date', 'Strategy NAV']], on='Date', validate='many_to_one')
    fig.add_trace(go.Scatter(x=marked.Date, y=marked['Strategy NAV'], mode='markers',
                            marker=dict(symbol='diamond', size=9), text=marked.Fund,
                            name='Entry / switch', hovertemplate='%{x}<br>%{text}<extra></extra>'), row=1, col=1)
    drawdown = daily['Strategy NAV'] / daily['Strategy NAV'].cummax() - 1
    fig.add_trace(go.Scatter(x=daily.Date, y=drawdown, name='Strategy drawdown', fill='tozeroy'), row=2, col=1)
    drop = result['parameters']['backtest']['buffer_drop']
    config = result['parameters']['backtest']
    floor = (daily['Entry Buffer Score'] * 0 + config['metric_floor'] if config['trigger'] == 'metric_floor'
             else daily['Entry Buffer Score'] * (1-drop))
    if config['portfolio_size'] > 1:
        positions = result['holdings'].query('Cohort == @cohort')
        for slot, frame in positions.groupby('Position'):
            fig.add_trace(go.Scatter(x=frame.Date, y=frame['Held Buffer Score'],
                                    text=frame.Fund, name=f'Position {slot} / {config["buffer_metric"]}',
                                    hovertemplate='%{x}<br>%{text}: %{y:.3f}<extra></extra>'), row=3, col=1)
            if config['trigger'] != 'metric_floor':
                fig.add_trace(go.Scatter(x=frame.Date, y=frame['Entry Buffer Score']*(1-drop),
                                        name=f'Position {slot} entry-relative floor', line=dict(dash='dot')), row=3, col=1)
        if config['trigger'] == 'metric_floor':
            fig.add_trace(go.Scatter(x=daily.Date, y=floor, name='Buffer integrity floor', line=dict(dash='dot')), row=3, col=1)
    else:
        for name, values in [(config['buffer_metric'], daily['Held Buffer Score']), ('Trigger floor', floor)]:
            fig.add_trace(go.Scatter(x=daily.Date, y=values, name=name), row=3, col=1)
    fig.update_layout(height=900, title=f'{config["client_name"]} — {cohort} — N={config["portfolio_size"]}', hovermode='x unified')
    fig.update_yaxes(tickformat='.0%', row=2, col=1)
    return fig


def holdings_figure(result, cohort='Jan'):
    positions = result['holdings'].query('Cohort == @cohort')
    weights = positions.pivot(index='Date', columns='Fund', values='Weight').fillna(0)
    fig = go.Figure()
    for fund in weights:
        fig.add_trace(go.Scatter(x=weights.index, y=weights[fund], name=fund, stackgroup='weights',
                                hovertemplate='%{x}<br>%{y:.1%}<extra>%{fullData.name}</extra>'))
    fig.update_layout(title=f'Actual portfolio weights — {cohort}', yaxis_tickformat='.0%',
                      yaxis_range=[0, 1], height=450, hovermode='x unified')
    return fig


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
