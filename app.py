"""Launch with: python -m streamlit run app.py"""
from dataclasses import asdict
from html import escape
import time

import pandas as pd
import streamlit as st

from backtesting.engine import simulate
from core.universe import GROUPS, STRATEGIES, BUFFER_BANDS, filter_catalog, validate_universe
from studio.archive import DEFAULT_ARCHIVE_ROOT, save_archive
from config.settings import BacktestConfig, MONTHS, ScoreConfig
from studio.service import (CADENCES, METRICS, SELECTIONS, SCHEDULE_NOTES, build_downloads,
                            config_json, load_dataset, load_preset, source_signature, validate_window, available_catalogue)
from visualization.research_plots import dashboard, holdings_figure

st.set_page_config(page_title='ECR · Backtest Studio', page_icon='◈', layout='wide')
st.markdown('''<style>
.block-container {max-width:1440px; padding-top:1.5rem; padding-bottom:3rem;}
h1,h2,h3 {color:#1B3442; letter-spacing:.01em;}
h1 {font-size:2.5rem !important;} h2 {font-size:1.45rem !important;} h3 {font-size:1.08rem !important;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stCaptionContainer"] p {color:#536C7A !important;}
[data-testid="stMetric"] {background:#F9EED7; border-top:3px solid #EAAA43; padding:18px;}
[data-testid="stMetricValue"] {font-size:1.8rem;}
[data-testid="stExpander"] {background:#F5F4EF;}
.eyebrow {color:#1B3442; border-bottom:2px solid #EAAA43; padding-bottom:10px; font-size:.8rem; font-weight:700; letter-spacing:.13em;}
.step {font-size:.75rem; color:#86561F; letter-spacing:.12em; font-weight:700; margin-bottom:6px;}
.summary-item {padding:12px 0; border-bottom:1px solid #DFE3E5;}
.summary-item span {display:block; font-size:.75rem; color:#536C7A; margin-bottom:3px;}
.summary-item strong {font-size:.94rem; font-weight:600;}
.pill {display:inline-block; color:#1B3442; background:#F9EED7; padding:6px 12px; font-size:.8rem;}
.hero {background:#F9EED7; padding:24px 30px; margin:18px 0 20px; border-left:4px solid #EAAA43;}
.hero h1 {margin:0 0 8px;} .hero p {margin:0; color:#1B3442;}
.stButton > button, .stDownloadButton > button {border-radius:3px; min-height:2.7rem;}
button[kind="primary"] {background:#EAAA43 !important; color:#1B3442 !important; border-color:#EAAA43 !important; font-weight:700;}
</style>''', unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=8)
def prepared_dataset(signature, score, universe=None):
    # The signature invalidates cached data when source files change.
    return load_dataset(score, universe)


@st.cache_data(show_spinner=False)
def catalogue_for_ui(signature):
    return available_catalogue()


def apply_outcome_profile():
    quarterly = st.session_state.universe_outcome == 'Quarterly'
    st.session_state.score_period_center = 45. if quarterly else 180.
    st.session_state.score_period_width = 18.75 if quarterly else 75.


def apply_universe_group():
    group = st.session_state.universe_group
    if group == 'Quarterly structures':
        st.session_state.universe_outcome = 'Quarterly'
    elif group.endswith(' series'):
        st.session_state.universe_outcome = 'Annual'
    apply_outcome_profile()


CLIENT_PRESETS = ['Super Buffer', 'UCITS Conservative']


def apply_preset():
    filename = {'Super Buffer': 'super-buffer.json', 'UCITS Conservative': 'ucit-conservative.json'}[st.session_state.preset]
    score, config = load_preset(filename)
    values = {'client': config.client_name, 'structure': 'Single ticker' if config.portfolio_size == 1 else 'Fund of funds',
              'fund_count': max(2, config.portfolio_size), 'cadence': config.review,
              'selection': config.selection, 'trigger_metric': 'ecr' if config.trigger == 'ecr' else config.buffer_metric,
              'ecr_threshold': config.threshold, 'component_threshold': config.metric_floor,
              'lag': config.execution_lag, 'cost': float(config.cost_bps),
              'remaining_days': config.min_remaining_days, 'audit': config.candidate_audit,
              'score_gap': float(config.min_selection_improvement),
              'replacement_limit': config.max_replacements_per_review, 'turnover_budget': float(config.max_turnover_per_review*100),
              'use_bands': config.weight_band is not None, 'band_pp': float((config.weight_band if config.weight_band is not None else .05)*100)}
    values.update({'score_'+k: float(v) for k,v in asdict(score).items()})
    st.session_state.update(values)
    if st.session_state.get('universe_outcome') == 'Quarterly':
        apply_outcome_profile()


if st.session_state.get('preset') not in CLIENT_PRESETS:
    st.session_state.preset = 'Super Buffer'
    apply_preset()

# Preserve neutral defaults when an already-open app reloads this version.
for key, value in {'score_gap': 0.,
                   'replacement_limit': 0, 'turnover_budget': 100., 'use_bands': False, 'band_pp': 5.}.items():
    st.session_state.setdefault(key, value)

st.session_state.setdefault('universe_group', 'F series')
st.session_state.setdefault('universe_outcome', 'Annual')
st.session_state.setdefault('archive_root', DEFAULT_ARCHIVE_ROOT)
st.session_state.setdefault('archive_enabled', True)
st.markdown('<div class="eyebrow">ECR / BACKTEST STUDIO</div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Build an outcome portfolio.</h1><p>Choose your universe, define the rules, and evaluate the results.</p></div>', unsafe_allow_html=True)
if 'view' not in st.session_state:
    st.session_state.view = 'Configure'
if '_navigate' in st.session_state:
    st.session_state.view = st.session_state.pop('_navigate')
view = st.segmented_control('Workspace', ['Configure', 'Results'], key='view', label_visibility='collapsed')

# Preserve configuration widget values when navigating to the results page.
CONFIG_KEYS = ['preset', 'client', 'structure', 'fund_count', 'cadence', 'selection', 'trigger_metric',
               'ecr_threshold', 'component_threshold', 'lag', 'cost', 'remaining_days',
               'audit', 'start', 'end', 'cohorts', 'score_gap',
               'replacement_limit', 'turnover_budget', 'use_bands', 'band_pp', 'universe_group', 'universe_outcome',
               'fund_series', 'fund_strategies', 'fund_references', 'fund_months', 'buffer_band', 'selected_tickers',
               'archive_root', 'archive_enabled'] + ['score_'+k for k in asdict(ScoreConfig())]
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
    if st.session_state.get('archive_path'):
        st.success('Results archived successfully.')
        st.code(st.session_state.archive_path, language=None)
    elif st.session_state.get('archive_error'):
        st.warning('Archive was not saved: ' + st.session_state.archive_error + ' Your results and ZIP download remain available below.')
    else:
        st.caption('Automatic archiving was off for this run. Download the ZIP to keep a copy.')
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
    if 'universe' in result:
        with st.expander('Fund universe used in this run'):
            st.dataframe(result['universe'], hide_index=True, width='stretch')
    chart_tab, holdings_tab, decisions_tab = st.tabs(['Performance & triggers', 'Holdings', 'Review log'])
    with chart_tab:
        st.plotly_chart(dashboard(result, cohort), width='stretch', config={'displaylogo': False})
        st.caption('Hold NAV shows the initially selected fund or basket held without trading.')
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
            st.dataframe(decisions.loc[decisions.Cohort == cohort].drop(columns=['Confirmation Reviews'], errors='ignore'), hide_index=True, width='stretch')
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

st.caption('Start with the fund universe, then follow the numbered strategy cards.')

with st.container(border=True):
    st.subheader('Fund universe')
    st.caption('1 · Choose a fund group and outcome period.  2 · Narrow the filters.  3 · Confirm the funds included.')
    a, b = st.columns(2)
    a.selectbox('Fund group', GROUPS, key='universe_group', on_change=apply_universe_group)
    outcome = b.selectbox('Outcome period', ['Annual','Quarterly'], key='universe_outcome', on_change=apply_outcome_profile)
    try:
        catalogue = catalogue_for_ui(source_signature())
    except (ValueError, OSError) as error:
        st.error(f'Unable to load the fund catalogue: {error}')
        st.stop()
    with st.expander('Filter funds · series, strategy, reference asset and buffer'):
        a,b,c = st.columns(3)
        series = a.multiselect('Fund series', sorted(set(['F','D','G','S','Y','Q','B']) | set(catalogue.series)), key='fund_series', placeholder='All series')
        strategies = b.multiselect('Strategy', sorted(set(STRATEGIES) | set(catalogue.strategy)), key='fund_strategies', placeholder='All strategies')
        references = c.multiselect('Reference asset', sorted(set(['EEM','EFA','GLD','IWM','QQQ','RSP','SPY']) | set(catalogue.reference_asset)), key='fund_references', placeholder='All reference assets')
        months = a.multiselect('Fund outcome month', [*MONTHS,'Quarterly'], key='fund_months', placeholder='All outcome months', help='Filters fund products. This is separate from the backtest launch month below.')
        buffer_band = b.selectbox('Buffer category', BUFFER_BANDS, key='buffer_band')
        c.caption('One broad buffer category keeps the list short. Exact ranges stay in the fund catalogue.')
    matches = filter_catalog(catalogue, st.session_state.universe_group, outcome, series, strategies, references, months, buffer_band)
    ready = matches.loc[matches.history_available]
    matching_tickers = tuple(sorted(ready.ticker))
    filter_identity = (st.session_state.universe_group, outcome, tuple(series), tuple(strategies), tuple(references), tuple(months), buffer_band, matching_tickers)
    if st.session_state.get('_universe_filter') != filter_identity:
        st.session_state.selected_tickers = list(matching_tickers)
        st.session_state['_universe_filter'] = filter_identity
    selected_tickers = st.multiselect('Funds included in this backtest', matching_tickers, key='selected_tickers', placeholder='No available funds match these filters')
    if not matching_tickers:
        st.info('No historical data is loaded for this selection yet. F-series annual funds are available now; other groups will become runnable when their catalogue records and history are added.')
    else:
        st.caption(f'{len(selected_tickers)} funds selected. Initial purchases and replacements are both restricted to this list.')
    with st.expander('View matching fund details'):
        details = matches.rename(columns={'history_available': 'History available'}).copy()
        details = details[['ticker', 'History available', *[c for c in details if c not in ('ticker', 'History available')]]]
        details['History available'] = details['History available'].map({True: '✓ Available', False: '— Not loaded'})
        styled = details.style.map(lambda v: 'color: #1672C4; font-weight: 700' if v == '✓ Available' else 'color: #687783', subset=['History available'])
        st.dataframe(styled, hide_index=True, width='stretch')

with st.container(border=True):
    st.subheader('Client configuration')
    preset_col, name_col = st.columns([1, 2])
    preset_col.selectbox('Start from a client preset', CLIENT_PRESETS, key='preset', on_change=apply_preset)
    name_col.text_input('Run name', key='client', max_chars=100)
    st.caption('Two starting points. Customize the rules below; use the research scripts for batch comparisons.')

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
            default_data, sources = prepared_dataset(source_signature(), ScoreConfig(), tuple(selected_tickers) if selected_tickers else None)
        except (ValueError, OSError) as error:
            st.error(f'Unable to load the repository data: {error}')
            st.stop()
        earliest, latest = default_data.dates[0].date(), default_data.dates[-1].date()
        a, b = st.columns(2)
        if 'start' not in st.session_state:
            st.session_state.start = max(earliest, pd.Timestamp('2021-01-19').date())
        if 'end' not in st.session_state:
            st.session_state.end = latest
        for date_key in ('start', 'end'):
            value = st.session_state[date_key]
            if value is not None:
                st.session_state[date_key] = min(latest, max(earliest, value))
        if 'cohorts' not in st.session_state:
            st.session_state.cohorts = ['JAN']
        start = a.date_input('Start no earlier than', value=None,
                              min_value=earliest, max_value=latest, key='start')
        end = b.date_input('End date', value=None, min_value=earliest, max_value=latest, key='end')
        cohorts = st.multiselect('Launch months', MONTHS, key='cohorts',
                                help='Each month is an independent launch test. For JAN, entry is the first trading day after the January roll date on or after your start limit.')
        st.write('Each launch month is a separate backtest. A January launch enters on the first observed trading session after the qualifying January roll date, on or after your start limit. Performance starts at 100 on that entry date.')
        st.write(f'At entry, buy the top {n} eligible fund(s) under your selection rule, equally weighted for a portfolio. With next-close trading, the ranking comes from the preceding session. A January launch does not require holding FJAN.')
        st.info('The review calendar stays fixed: quarterly reviews are in March, June, September and December—even when the backtest launches in January. At those reviews, only triggered holdings are reconsidered; optional weight bands are checked separately.')
        st.caption(f'Data available {earliest:%d %b %Y} – {latest:%d %b %Y}. Multiple launch months create independent comparisons, not one combined portfolio.')
    with st.expander('Trading assumptions & fund eligibility', expanded=False):
        st.write('By default, calculate the ranking and trigger at a trading close, then execute any resulting trade at the following trading session’s close. This avoids assuming the final closing signal could already have been traded at that same price.')
        a, b = st.columns(2)
        lag = a.selectbox('When to execute a signaled trade', [1, 0], format_func=lambda v: 'Following trading day’s close (default)' if v else 'Signal day’s close (theoretical comparison)', key='lag')
        cost = b.number_input('Trading cost (basis points)', min_value=0.0, max_value=9999.0, step=1.0, key='cost', help='0 means no transaction cost. 5 bps = 0.05%, or $5 per $10,000 of one-way traded capital. Charged on replacements and weight adjustments; initial entry is uncharged.')
        remaining = b.number_input('Minimum outcome days remaining', min_value=0, max_value=366, step=1, key='remaining_days')
        audit = st.checkbox('Include full candidate rankings in Excel', key='audit')
    with st.expander('Turnover & weight controls', expanded=False):
        st.caption('Optional controls slow replacements. Defaults preserve the original strategy. Replacement budgets can defer trades.')
        a, b = st.columns(2)
        score_gap = a.number_input('Minimum selection-score improvement', min_value=0., step=.05, key='score_gap',
                                   help='Require the replacement to improve on the held score by this many points. Measured in the chosen selection metric.')
        replacement_limit = b.number_input('Maximum replacements per review', min_value=0, max_value=12, step=1, key='replacement_limit',
                                           help='Zero means unlimited. Process the worst triggered metric first.')
        budget = a.number_input('Turnover budget per review (%)', min_value=.1, max_value=100., step=5., key='turnover_budget',
                                 help='One-way turnover: replaced capital / portfolio capital; weight resets add half the absolute weight change. Applies at execution too.')
        use_bands = st.checkbox('Reset equal weights only outside a drift band', key='use_bands', disabled=n==1)
        band = st.number_input('Weight band around 1/N (percentage points)', min_value=0., max_value=100., step=1., key='band_pp', disabled=n==1 or not use_bands)
        if n>1 and use_bands:
            st.caption(f'Target {1/n:.1%}; review band {max(0,1/n-band/100):.1%}–{min(1,1/n+band/100):.1%}. A breach requests a full equal-weight reset at the configured execution time, subject to remaining turnover budget.')
        if budget/100 < 1/n:
            st.warning('This budget is below an initially equal-sized holding. Whole-holding replacements can be deferred until their weight fits the budget.')
    with st.expander('ECR formula · components & outcome period', expanded=False):
        st.subheader('Components of the ECR score')
        st.caption('Economics drive the score; buffer and cap integrity act as multiplicative gates. Period position adds a small timing bonus. The default coefficients have an intended maximum of 3.0.')
        st.latex(r'\mathrm{ECR} = (w_D \cdot D + w_R \cdot R) \cdot \max(0,B)^{\gamma_B} \cdot \max(0,C)^{\gamma_C} + b_P \cdot P')
        economics, integrity, timing = st.tabs(['1 · Economic signals', '2 · Integrity gates', '3 · Period position'])
        score_options = {}

        def score_field(column, key, label, step=.1):
            score_options[key] = column.number_input(label, min_value=.001 if key in ('sigma_down','sigma_up','period_width') else 0.,
                                                      step=step, format='%.3f', key='score_'+key)

        with economics:
            a, b = st.columns(2)
            with a, st.container(border=True):
                st.subheader('D · Downside Before Buffer')
                st.caption('Scores the loss exposure before protection begins. The coefficient sets its contribution to ECR.')
                score_field(st, 'dbb_weight', 'DBB coefficient')
                score_field(st, 'dbb_decay', 'DBB sensitivity')
            with b, st.container(border=True):
                st.subheader('R · Reset proximity')
                st.caption('Rewards proximity to the reset level. Larger tolerance gives credit to funds further from par.')
                score_field(st, 'reset_weight', 'Reset coefficient')
                left, right = st.columns(2)
                score_field(left, 'sigma_down', 'Downside tolerance', .01)
                score_field(right, 'sigma_up', 'Upside tolerance', .01)
        with integrity:
            a, b = st.columns(2)
            for column, symbol, title, key in [(a, 'B', 'Buffer Integrity', 'buffer_exponent'), (b, 'C', 'Cap Integrity', 'cap_exponent')]:
                with column, st.container(border=True):
                    st.subheader(f'{symbol} · {title}')
                    st.caption('Fraction of the original protection remaining.' if symbol == 'B' else 'Fraction of the original upside cap remaining.')
                    score_field(st, key, f'{title} exponent')
            st.caption('Both gates multiply the economic signals. A larger exponent penalizes deterioration more strongly.')
        with timing:
            st.subheader('P · Outcome-period position')
            st.caption('A small additive bonus centered within the outcome period. These values describe the product, independently of the review cadence.')
            st.table(pd.DataFrame({'Outcome profile': ['Annual', 'Quarterly'], 'Peak days remaining': [180.,45.], 'Timing width (days)': [75.,18.75], 'Reset tolerance (down / up)': ['0.07 / 0.05','0.07 / 0.05']}))
            st.caption('Quarterly timing is a proportional research setting; it still needs quarterly fund history for evaluation.')
            st.info(f'Active profile: {outcome}. Changing the universe outcome period restores its timing defaults.')
            a, b, c = st.columns(3)
            score_field(a, 'period_bonus', 'Period-position bonus')
            score_field(b, 'period_center', 'Peak days remaining')
            score_field(c, 'period_width', 'Timing width (days)')

error = None
try:
    if start is None or end is None:
        raise ValueError('Choose both a start and an end date.')
    selected_catalogue = validate_universe(catalogue, selected_tickers, catalogue.loc[catalogue.history_available].ticker)
    if n > len(selected_tickers):
        raise ValueError('Number of portfolio funds exceeds the selected universe.')
    score = ScoreConfig(**score_options)
    config = BacktestConfig(client_name=st.session_state.client, portfolio_size=int(n), selection=selection, universe=tuple(selected_tickers),
                            review=cadence, initial_selection='highest_score', trigger='ecr' if metric == 'ecr' else 'metric_floor',
                            buffer_metric='buffer_integrity' if metric == 'ecr' else metric,
                            threshold=threshold if metric == 'ecr' else 1.5,
                            metric_floor=threshold if metric != 'ecr' else .8, comparator='<',
                            execution_lag=lag, cost_bps=cost, min_remaining_days=remaining,
                            candidate_audit=audit, start_date=start.isoformat(), end_date=end.isoformat(), cohorts=tuple(cohorts),
                            min_selection_improvement=score_gap,
                            max_replacements_per_review=replacement_limit, max_turnover_per_review=budget/100,
                            weight_band=band/100 if n>1 and use_bands else None)
    validate_window(default_data, config)
except ValueError as exc:
    error = str(exc)

with summary, st.container(border=True):
    st.markdown('<div class="step">YOUR STRATEGY</div>', unsafe_allow_html=True)
    st.subheader('Ready to test')
    for label, value in [('Portfolio', f'{n} fund'+('s' if n != 1 else '')+' · '+structure),
                         ('Universe', f'{len(selected_tickers)} funds · {outcome}'), ('Review', CADENCES[cadence]), ('Selection', SELECTIONS[selection]),
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
    with st.expander('Archive location', expanded=False):
        st.checkbox('Save each completed run automatically', key='archive_enabled')
        st.text_input('Results folder on this computer', key='archive_root')
        st.caption('Each run gets its own dated subfolder with Excel, PNG and HTML charts, settings, and a ZIP. This path belongs to the computer running the app.')
    st.caption('Automatic archive: ' + st.session_state.archive_root if st.session_state.archive_enabled else 'Automatic archiving is off. ZIP downloads remain available.')
    if n == len(selected_tickers) and n>1:
        st.info('All selected funds are held. There are no outside replacements; a breach may leave the portfolio unchanged.')
    if st.button('Run backtest →', type='primary', width='stretch', disabled=bool(error)):
        try:
            with st.status('Running your strategy…', expanded=True) as status:
                began = time.perf_counter()
                st.write('Preparing the selected scoring model')
                data, sources = prepared_dataset(source_signature(), score, config.universe)
                validate_window(data, config)
                st.write('Simulating holdings, triggers and trading costs')
                result = simulate(data, config)
                result['sources'] = sources
                result['universe'] = selected_catalogue
                result['requested_archive_root'] = st.session_state.archive_root if st.session_state.archive_enabled else None
                result['timings'] = {'run_seconds': time.perf_counter() - began}
                st.write('Building Excel and chart downloads')
                downloads = build_downloads(result)
                archive_path, archive_error = None, None
                if st.session_state.archive_enabled:
                    st.write('Saving the completed report to the archive')
                    try:
                        archive_path = str(save_archive(downloads, st.session_state.archive_root, config.client_name, result['strategy_id']))
                    except OSError as exc:
                        archive_error = str(exc)
                status.update(label='Backtest complete', state='complete')
            st.session_state.result = result
            st.session_state.downloads = downloads
            st.session_state.archive_path = archive_path
            st.session_state.archive_error = archive_error
            st.session_state.pop('result_cohort', None)
            st.session_state['_navigate'] = 'Results'
            st.rerun()
        except (ValueError, OSError, KeyError) as exc:
            st.error(f'Unable to complete this run: {exc}. Adjust the settings and try again.')
    if not error:
        st.download_button('Save configuration', config_json(score, config), file_name='ecr-config.json',
                           mime='application/json', width='stretch')
    st.caption('Changing the starting preset restores its portfolio, trigger, execution and formula defaults. Your test window stays in place.')
