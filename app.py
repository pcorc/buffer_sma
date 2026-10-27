"""Launch with: python -m streamlit run app.py"""
from dataclasses import asdict
from html import escape
import time

import pandas as pd
import streamlit as st

from backtesting.engine import simulate
from backtesting.scenarios import conservative_suite
from config.settings import BacktestConfig, MONTHS, ScoreConfig
from studio.service import (CADENCES, METRICS, SELECTIONS, SCHEDULE_NOTES, build_downloads,
                            config_json, load_dataset, load_preset, source_signature, validate_window)
from visualization.research_plots import dashboard, holdings_figure

st.set_page_config(page_title='ECR · Backtest Studio', page_icon='◈', layout='wide')
st.markdown('''<style>
.block-container {max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem;}
h1 {font-size: 2.3rem !important; letter-spacing: -.07rem; font-weight: 650 !important;}
h2 {font-size: 1.4rem !important; letter-spacing: -.025rem;}
h3 {font-size: 1.08rem !important; font-weight: 600 !important;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-radius: 14px;}
[data-testid="stMetric"] {background:white; border:1px solid #E3E9ED; border-radius:12px; padding:18px;}
[data-testid="stMetricLabel"] {color:#607383; font-size:.8rem;}
[data-testid="stMetricValue"] {font-size:1.8rem;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color:#596F7D !important;}
.eyebrow {color:#147D73; letter-spacing:.16em; font-size:.73rem; font-weight:700; margin-bottom:12px;}
.step {font-size:.72rem; color:#147D73; letter-spacing:.12em; font-weight:700; margin-bottom:6px;}
.summary-item {padding:13px 0; border-bottom:1px solid #E8EDF1;}
.summary-item span {display:block; font-size:.74rem; color:#718291; margin-bottom:3px;}
.summary-item strong {font-size:.95rem; font-weight:600;}
.pill {display:inline-block; color:#147D73; background:#E7F3F0; border-radius:30px; padding:5px 11px; font-size:.73rem;}
.stButton > button, .stDownloadButton > button {border-radius:9px; min-height:2.8rem;}
</style>''', unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=8)
def prepared_dataset(signature, score):
    # The signature invalidates cached data when source files change.
    return load_dataset(score)


RESEARCH_PRESETS = {e.name: e for e in conservative_suite() if e.name.startswith('UCITS')}


def apply_preset():
    if st.session_state.preset in RESEARCH_PRESETS:
        experiment = RESEARCH_PRESETS[st.session_state.preset]
        score, config = experiment.score, experiment.backtest
    else:
        filename = {'Super Buffer': 'super-buffer.json', 'UCITS Conservative': 'ucit-conservative.json'}[st.session_state.preset]
        score, config = load_preset(filename)
    values = {'client': st.session_state.preset if st.session_state.preset in RESEARCH_PRESETS else config.client_name, 'structure': 'Single ticker' if config.portfolio_size == 1 else 'Fund of funds',
              'fund_count': max(2, config.portfolio_size), 'cadence': config.review,
              'selection': config.selection, 'trigger_metric': 'ecr' if config.trigger == 'ecr' else config.buffer_metric,
              'ecr_threshold': config.threshold, 'component_threshold': config.metric_floor,
              'lag': config.execution_lag, 'cost': float(config.cost_bps), 'holding_days': config.min_holding_days,
              'remaining_days': config.min_remaining_days, 'audit': config.candidate_audit,
              'confirmations': config.confirmation_reviews, 'score_gap': float(config.min_selection_improvement),
              'entry_margin': float(config.entry_metric_margin), 'cooldown': config.reentry_cooldown_days,
              'replacement_limit': config.max_replacements_per_review, 'turnover_budget': float(config.max_turnover_per_review*100),
              'use_bands': config.weight_band is not None, 'band_pp': float((config.weight_band if config.weight_band is not None else .05)*100)}
    values.update({'score_'+k: float(v) for k,v in asdict(score).items()})
    st.session_state.update(values)


if 'preset' not in st.session_state:
    st.session_state.preset = 'Super Buffer'
    apply_preset()

# Preserve neutral defaults when an already-open app reloads this version.
for key, value in {'confirmations': 1, 'score_gap': 0., 'entry_margin': 0., 'cooldown': 0,
                   'replacement_limit': 0, 'turnover_budget': 100., 'use_bands': False, 'band_pp': 5.}.items():
    st.session_state.setdefault(key, value)

st.markdown('<div class="eyebrow">ECR / RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
head, badge = st.columns([5, 1])
with head:
    st.title('Backtest Studio')
    st.write('Build a strategy. Understand its behavior. Export the evidence.')
with badge:
    st.markdown('<div class="pill">F-series · 12 funds</div>', unsafe_allow_html=True)
if 'view' not in st.session_state:
    st.session_state.view = 'Configure'
if '_navigate' in st.session_state:
    st.session_state.view = st.session_state.pop('_navigate')
view = st.segmented_control('Workspace', ['Configure', 'Results'], key='view', label_visibility='collapsed')

# Preserve configuration widget values when navigating to the results page.
CONFIG_KEYS = ['preset', 'client', 'structure', 'fund_count', 'cadence', 'selection', 'trigger_metric',
               'ecr_threshold', 'component_threshold', 'lag', 'cost', 'holding_days', 'remaining_days',
               'audit', 'start', 'end', 'cohorts', 'confirmations', 'score_gap', 'entry_margin', 'cooldown',
               'replacement_limit', 'turnover_budget', 'use_bands', 'band_pp'] + ['score_'+k for k in asdict(ScoreConfig())]
for key in CONFIG_KEYS:
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]

if view == 'Results':
    if 'result' not in st.session_state:
        st.info('Your results will appear here after you run a backtest. Start in Configure.')
        st.stop()
    result = st.session_state.result
    saved = result['parameters']['backtest']
    st.subheader(saved['client_name'])
    label = METRICS['ecr' if saved['trigger'] == 'ecr' else saved['buffer_metric']]
    floor = saved['threshold'] if saved['trigger'] == 'ecr' else saved['metric_floor']
    st.caption(f"Saved run · {saved['portfolio_size']} fund(s) · {CADENCES[saved['review']]} · {label} < {floor:g} → {SELECTIONS[saved['selection']]} · {result['strategy_id']}")
    st.caption('These results reflect the settings used at execution. Configuration edits take effect on the next run.')
    cohorts = result['summary'].Cohort.tolist()
    cohort = st.selectbox('Launch cohort to display', cohorts, key='result_cohort',
                          help='Each launch month is a separate backtest with its own start date, not an allocation.')
    row = result['summary'].set_index('Cohort').loc[cohort]
    metrics = st.columns(4)
    for col, title, value in zip(metrics, ['Annualized return', 'Maximum drawdown', 'Sortino ratio', 'Replacements'],
                                 [f"{row['Annualized Return']:.1%}", f"{row['Max Drawdown']:.1%}",
                                  f"{row['Sortino']:.2f}", str(int(row['Switches']))]):
        col.metric(title, value)
    st.caption(f"{row['Start']:%d %b %Y} – {row['End']:%d %b %Y} · {saved['cost_bps']:g} bps per replacement · "
               f"{'Next-close execution' if saved['execution_lag'] else 'Theoretical same-close execution'}")
    st.caption(f"One-way turnover: {row['One-way Turnover']:.2f}× · Weight resets: {int(row.get('Weight Rebalances', 0))}. Detailed trades and any execution-budget deferrals are included in Excel.")
    if row['Missing Benchmark Sessions']:
        st.info(f"{int(row['Missing Benchmark Sessions'])} benchmark sessions are missing inside this window. Charts preserve those gaps; endpoint returns remain available.")
    chart_tab, holdings_tab, decisions_tab = st.tabs(['Performance & triggers', 'Holdings', 'Review log'])
    with chart_tab:
        st.plotly_chart(dashboard(result, cohort), width='stretch', config={'displaylogo': False})
        st.caption('The trigger line is checked only on scheduled review dates. A breach can result in no trade if the incumbent still ranks highest.')
    with holdings_tab:
        st.plotly_chart(holdings_figure(result, cohort), width='stretch', config={'displaylogo': False})
        latest = result['holdings'].query('Cohort == @cohort')
        latest = latest.loc[latest.Date == latest.Date.max(), ['Position', 'Fund', 'Weight', 'Held ECR Score']]
        st.dataframe(latest, hide_index=True, width='stretch', column_config={'Weight': st.column_config.NumberColumn(format='percent')})
    with decisions_tab:
        decisions = result['decisions']
        if decisions.empty:
            st.info('No scheduled review occurred in this window.')
        else:
            st.dataframe(decisions.loc[decisions.Cohort == cohort], hide_index=True, width='stretch')
    if 'trades' in result:
        with st.expander('Executed trades & weight reviews'):
            st.dataframe(result['trades'], hide_index=True, width='stretch')
            st.dataframe(result['weight_reviews'], hide_index=True, width='stretch')
    with st.expander('All launch cohorts · performance summary'):
        st.dataframe(result['summary'], hide_index=True, width='stretch')
    st.subheader('Take the results with you')
    st.caption('Excel includes diagnostics at each entry and switch. The complete package adds PNGs, interactive charts, CSVs and the exact configuration.')
    downloads = st.session_state.downloads
    a, b, c = st.columns(3)
    stem = 'ecr-'+result['strategy_id']
    a.download_button('↓ Excel workbook', downloads['excel'], file_name=stem+'.xlsx',
                      mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', width='stretch')
    b.download_button('↓ All results & graphs', downloads['zip'], file_name=stem+'.zip', mime='application/zip', width='stretch', type='primary')
    c.download_button('↓ Configuration', downloads['config'], file_name=stem+'.json', mime='application/json', width='stretch')
    st.stop()

st.caption('Follow the numbered cards. All key choices stay visible as you build your strategy.')
preset_col, name_col = st.columns([1, 2])
with preset_col:
    st.selectbox('Start from a client preset', ['Super Buffer', 'UCITS Conservative', *RESEARCH_PRESETS], key='preset', on_change=apply_preset)
with name_col:
    st.text_input('Run name', key='client', max_chars=100)

main, summary = st.columns([2.7, 1], gap='large')
with main:
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown('<div class="step">01 / PORTFOLIO</div>', unsafe_allow_html=True)
        st.subheader('Choose the structure')
        structure = st.selectbox('Portfolio type', ['Single ticker', 'Fund of funds'], key='structure')
        if structure == 'Single ticker':
            n = 1
            st.text_input('Number of funds', value='1', disabled=True)
        else:
            n = st.number_input('Number of funds', min_value=2, max_value=12, step=1, key='fund_count')
        st.caption('Start equally weighted. Replace only breached holdings; allow weights to drift.')
    with right, st.container(border=True):
        st.markdown('<div class="step">02 / SCHEDULE</div>', unsafe_allow_html=True)
        st.subheader('Set the review rhythm')
        cadence = st.selectbox('Rebalance review cadence', list(CADENCES), format_func=CADENCES.get, key='cadence')
        st.info(SCHEDULE_NOTES[cadence])
        st.caption('Check the trigger on these dates. Trade only when the rule calls for it.')
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown('<div class="step">03 / SELECTION</div>', unsafe_allow_html=True)
        st.subheader('Decide what to buy')
        selection = st.selectbox('Rank eligible funds by', list(SELECTIONS), format_func=SELECTIONS.get, key='selection')
        st.caption('Use this ranking at inception and for replacements. Exact ties favor existing holdings.')
    with right, st.container(border=True):
        st.markdown('<div class="step">04 / TRIGGER</div>', unsafe_allow_html=True)
        st.subheader('Decide when to review a holding')
        metric = st.selectbox('Sell trigger metric', list(METRICS), format_func=METRICS.get, key='trigger_metric')
        if metric == 'ecr':
            threshold = st.selectbox('Review holding when ECR is below', [1.0, 1.5, 2.0], key='ecr_threshold', format_func=lambda v: f'{v:g}')
            st.caption('Strictly below the threshold. The standard ECR model has an intended 0–3 scale.')
        else:
            threshold = st.number_input(f'Review holding when {METRICS[metric]} is below', min_value=0.0, max_value=1.0,
                                        step=.05, format='%.2f', key='component_threshold')
            st.caption(f'{threshold:.0%} of the ideal fresh score of 1.00. This is not a decline from the purchase score.')
    with st.container(border=True):
        st.markdown('<div class="step">05 / TEST WINDOW</div>', unsafe_allow_html=True)
        st.subheader('Choose the period and launch months')
        # Load the default panel only once; alternate scoring is prepared when running.
        try:
            default_data, sources = prepared_dataset(source_signature(), ScoreConfig())
        except (ValueError, OSError) as error:
            st.error(f'Unable to load the repository data: {error}')
            st.stop()
        earliest, latest = default_data.dates[0].date(), default_data.dates[-1].date()
        a, b = st.columns(2)
        if 'start' not in st.session_state:
            st.session_state.start = max(earliest, pd.Timestamp('2021-01-19').date())
        if 'end' not in st.session_state:
            st.session_state.end = latest
        if 'cohorts' not in st.session_state:
            st.session_state.cohorts = ['JAN']
        start = a.date_input('Start no earlier than', value=None,
                              min_value=earliest, max_value=latest, key='start')
        end = b.date_input('End date', value=None, min_value=earliest, max_value=latest, key='end')
        cohorts = st.multiselect('Launch months', MONTHS, key='cohorts',
                                help='Each month is an independent launch test. For JAN, entry is the first trading day after the January roll date on or after your start limit.')
        st.caption(f'Data available {earliest:%d %b %Y} – {latest:%d %b %Y}. Launch dates follow the supplied roll calendar.')
    with st.expander('Execution & eligibility', expanded=False):
        a, b = st.columns(2)
        lag = a.selectbox('Trade execution', [1, 0], format_func=lambda v: 'Next trading close' if v else 'Same close (theoretical)', key='lag')
        cost = b.number_input('Cost per replacement (bps)', min_value=0.0, max_value=9999.0, step=1.0, key='cost')
        hold = a.number_input('Minimum holding sessions', min_value=0, max_value=2520, step=1, key='holding_days')
        remaining = b.number_input('Minimum outcome days remaining', min_value=0, max_value=366, step=1, key='remaining_days')
        audit = st.checkbox('Include full candidate rankings in Excel', key='audit')
        st.caption('Costs apply once to replaced capital. Initial entry is uncharged. The universe is the 12 F-series funds in the repository.')
    with st.expander('Turnover & weight controls', expanded=False):
        st.caption('Optional controls slow replacements. Defaults preserve the original strategy. Budgets and confirmation can delay exits from deteriorating holdings.')
        a, b = st.columns(2)
        confirmations = a.number_input('Consecutive breached reviews', min_value=1, max_value=12, step=1, key='confirmations',
                                        help='At monthly cadence, 2 requires breaches on two consecutive monthly reviews. Recovery resets the count.')
        score_gap = b.number_input('Minimum selection-score improvement', min_value=0., step=.05, key='score_gap',
                                   help='Require the replacement to improve on the held score by this many points. Measured in the chosen selection metric.')
        entry_margin = a.number_input('Replacement entry margin', min_value=0., max_value=1., step=.05, key='entry_margin', disabled=metric=='ecr',
                                       help='For component triggers only: floor 0.80 plus margin 0.10 requires a replacement component score of at least 0.90. Zero disables this filter.')
        cooldown = b.number_input('Repurchase cooldown (sessions)', min_value=0, max_value=2520, step=1, key='cooldown')
        replacement_limit = a.number_input('Maximum replacements per review', min_value=0, max_value=12, step=1, key='replacement_limit',
                                           help='Zero means unlimited. Process the worst triggered metric first.')
        budget = b.number_input('Turnover budget per review (%)', min_value=.1, max_value=100., step=5., key='turnover_budget',
                                 help='One-way turnover: replaced capital / portfolio capital; weight resets add half the absolute weight change. Applies at execution too.')
        use_bands = st.checkbox('Reset equal weights only outside a drift band', key='use_bands', disabled=n==1)
        band = st.number_input('Weight band around 1/N (percentage points)', min_value=0., max_value=100., step=1., key='band_pp', disabled=n==1 or not use_bands)
        if n>1 and use_bands:
            st.caption(f'Target {1/n:.1%}; review band {max(0,1/n-band/100):.1%}–{min(1,1/n+band/100):.1%}. A breach requests a full equal-weight reset at the configured execution time, subject to remaining turnover budget.')
        st.caption('Minimum holding sessions are set under Execution & eligibility. Replacement controls do not suppress the separate weight-band policy. A band is a review trigger, not a continuously enforced weight cap.')
        if budget/100 < 1/n:
            st.warning('This budget is below an initially equal-sized holding. Whole-holding replacements can be deferred until their weight fits the budget.')
    with st.expander('ECR formula parameters', expanded=False):
        st.caption('The presets use the workbook multiplicative formula. Changing these values creates a custom experiment; it can change the score scale.')
        st.code('(w_DBB × DBB + w_Reset × Reset) × max(0, Buffer)^γB × max(0, Cap)^γC + bonus × Period', language=None)
        labels = {'dbb_weight':'DBB coefficient', 'reset_weight':'Reset coefficient', 'buffer_exponent':'Buffer gate exponent',
                  'cap_exponent':'Cap gate exponent', 'period_bonus':'Period bonus', 'dbb_decay':'DBB decay',
                  'sigma_down':'Reset sigma · downside', 'sigma_up':'Reset sigma · upside',
                  'period_center':'Period center (days)', 'period_width':'Period width (days)'}
        a, b = st.columns(2)
        score_options = {}
        for index, (key, label) in enumerate(labels.items()):
            score_options[key] = (a if index % 2 == 0 else b).number_input(
                label, min_value=.001 if key in ('sigma_down','sigma_up','period_width') else 0.0,
                step=.01 if 'sigma' in key else .1, format='%.3f', key='score_'+key)

error = None
try:
    if start is None or end is None:
        raise ValueError('Choose both a start and an end date.')
    score = ScoreConfig(**score_options)
    config = BacktestConfig(client_name=st.session_state.client, portfolio_size=int(n), selection=selection,
                            review=cadence, initial_selection='highest_score', trigger='ecr' if metric == 'ecr' else 'metric_floor',
                            buffer_metric='buffer_integrity' if metric == 'ecr' else metric,
                            threshold=threshold if metric == 'ecr' else 1.5,
                            metric_floor=threshold if metric != 'ecr' else .8, comparator='<',
                            execution_lag=lag, cost_bps=cost, min_holding_days=hold, min_remaining_days=remaining,
                            candidate_audit=audit, start_date=start.isoformat(), end_date=end.isoformat(), cohorts=tuple(cohorts),
                            confirmation_reviews=confirmations, min_selection_improvement=score_gap,
                            entry_metric_margin=entry_margin if metric != 'ecr' else 0., reentry_cooldown_days=cooldown,
                            max_replacements_per_review=replacement_limit, max_turnover_per_review=budget/100,
                            weight_band=band/100 if n>1 and use_bands else None)
    validate_window(default_data, config)
except ValueError as exc:
    error = str(exc)

with summary, st.container(border=True):
    st.markdown('<div class="step">YOUR STRATEGY</div>', unsafe_allow_html=True)
    st.subheader('Ready to test')
    for label, value in [('Portfolio', f'{n} fund'+('s' if n != 1 else '')+' · '+structure),
                         ('Review', CADENCES[cadence]), ('Selection', SELECTIONS[selection]),
                         ('Trigger', f'{METRICS[metric]} < {threshold:g}'),
                         ('Execution', 'Next trading close' if lag else 'Same close · theoretical'),
                         ('Turnover budget', f'{budget:g}% per review'),
                         ('Weight policy', f'1/N ± {band:g}pp' if n>1 and use_bands else 'Allow weights to drift'),
                         ('Launch tests', ', '.join(cohorts) or 'Choose a launch month')]:
        st.markdown(f'<div class="summary-item"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>', unsafe_allow_html=True)
    st.write('')
    st.caption('Excel workbook + interactive charts + PNG reports. Every run includes its exact settings.')
    if error:
        st.error(error)
    if n == 12:
        st.info('All 12 funds are held. There are no outside replacements; a breach may leave the portfolio unchanged.')
    if st.button('Run backtest →', type='primary', width='stretch', disabled=bool(error)):
        try:
            with st.status('Running your strategy…', expanded=True) as status:
                began = time.perf_counter()
                st.write('Preparing the selected scoring model')
                data, sources = prepared_dataset(source_signature(), score)
                validate_window(data, config)
                st.write('Simulating holdings, triggers and trading costs')
                result = simulate(data, config)
                result['sources'] = sources
                result['timings'] = {'run_seconds': time.perf_counter() - began}
                st.write('Building Excel and chart downloads')
                downloads = build_downloads(result)
                status.update(label='Backtest complete', state='complete')
            st.session_state.result = result
            st.session_state.downloads = downloads
            st.session_state.pop('result_cohort', None)
            st.session_state['_navigate'] = 'Results'
            st.rerun()
        except (ValueError, OSError, KeyError) as exc:
            st.error(f'Unable to complete this run: {exc}. Adjust the settings and try again.')
    if not error:
        st.download_button('Save configuration', config_json(score, config), file_name='ecr-config.json',
                           mime='application/json', width='stretch')
    st.caption('Changing the starting preset restores its portfolio, trigger, execution and formula defaults. Your test window stays in place.')
