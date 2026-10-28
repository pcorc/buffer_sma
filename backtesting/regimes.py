"""Matched stress windows for exported NAVs; evaluation does not reset holdings."""
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Regime:
    name: str
    start: str
    end: str
    kind: str = 'Stress'
    fresh_start: str | None = None
    fresh_cohort: str | None = None
    simulation_end: str | None = None


REGIMES = (
    Regime('2020 COVID selloff', '2020-02-19', '2020-03-23'),
    Regime('2022 bear market', '2021-12-31', '2022-10-12', fresh_start='2021-12-01', fresh_cohort='DEC', simulation_end='2022-12-30'),
    Regime('2022 calendar year', '2021-12-31', '2022-12-30', 'Context'),
    Regime('2023 correction', '2023-07-31', '2023-10-27', fresh_start='2023-07-01', fresh_cohort='JUL', simulation_end='2023-12-29'),
    Regime('2024 summer selloff', '2024-07-16', '2024-08-05', fresh_start='2024-05-01', fresh_cohort='MAY', simulation_end='2024-08-30'),
    Regime('2025 spring selloff', '2025-02-19', '2025-04-08', fresh_start='2025-01-01', fresh_cohort='JAN', simulation_end='2025-06-30'),
    Regime('2025 rebound', '2025-04-08', '2025-06-30', 'Context'),
    Regime('2023-24 expansion', '2022-12-30', '2024-12-31', 'Context'),
)


def simple_configurations(configs, review='monthly'):
    required = {'Experiment','N','Review','Trigger','Threshold','Universe','Score improvement','Max replacements','Turnover budget','Weight band'}
    if required - set(configs):
        raise ValueError(f'Configuration table is missing: {sorted(required-set(configs))}')
    if configs.Experiment.duplicated().any():
        raise ValueError('Experiment names must be unique.')
    mask = (configs['Score improvement'].eq(0) & configs['Max replacements'].eq(0)
            & configs['Turnover budget'].eq(1) & configs['Weight band'].isna())
    if review != 'all':
        mask &= configs.Review.eq(review)
    selected = configs.loc[mask].copy()
    if selected.empty:
        raise ValueError('No simple configurations match the requested cadence.')
    return selected


def nav_metrics(nav):
    if len(nav) < 2 or not np.isfinite(nav).all() or (nav <= 0).any():
        raise ValueError('NAV requires at least two finite positive observations.')
    returns = nav.pct_change(fill_method=None).iloc[1:]
    vol = returns.std()
    return {'Total Return':nav.iloc[-1]/nav.iloc[0]-1,
            'Max Drawdown':(nav/nav.cummax()-1).min(),
            'Sharpe':returns.mean()/vol*np.sqrt(252) if vol > 0 else np.nan}


def load_export(path, review='monthly'):
    """Reconcile all exported summaries, then retain the simple NAV columns."""
    curves, audits = {}, []
    with pd.ExcelFile(path) as book:
        configs = pd.read_excel(book,sheet_name='Configurations')
        selected = simple_configurations(configs,review)
        summary = pd.read_excel(book,sheet_name='Launch Cohorts')
        comparison = pd.read_excel(book,sheet_name='Comparison')
        sheets = [s for s in book.sheet_names if s.startswith('NAV ')]
        if not sheets:
            raise ValueError('Workbook contains no NAV launch sheets.')
        if summary.duplicated(['Experiment','Cohort']).any():
            raise ValueError('Duplicate experiment/cohort summary rows.')
        for sheet in sheets:
            cohort = sheet[4:]
            frame = pd.read_excel(book,sheet_name=sheet,index_col=0)
            frame.index = pd.to_datetime(frame.index,errors='raise')
            if frame.index.hasnans or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
                raise ValueError(f'{sheet}: NAV dates must be nonmissing, unique and sorted.')
            reported = summary.loc[summary.Cohort.eq(cohort)].set_index('Experiment')
            if set(frame) != set(configs.Experiment) or set(frame) != set(reported.index):
                raise ValueError(f'{sheet}: configurations, summary rows and NAV columns must match.')
            for name in frame:
                metrics = nav_metrics(frame[name])
                row = reported.loc[name]
                if pd.Timestamp(row.Start)!=frame.index[0] or pd.Timestamp(row.End)!=frame.index[-1]:
                    raise ValueError(f'{sheet}/{name}: reported dates do not match NAV dates.')
                error = {}
                for key,computed in metrics.items():
                    cached = row[key]
                    if pd.isna(cached) and pd.isna(computed):
                        error[key+' Error'] = 0.
                    elif not np.isclose(cached,computed,atol=1e-9,rtol=1e-9):
                        raise ValueError(f'{sheet}/{name}: {key} does not reconcile to the NAV history.')
                    else:
                        error[key+' Error'] = abs(cached-computed)
                audits.append({'Experiment':name,'Cohort':cohort,'Start':frame.index[0],'End':frame.index[-1],**error})
            curves[cohort] = frame.loc[:,selected.Experiment]
        if set(summary.Cohort) != set(curves):
            raise ValueError('Some launch summaries have no corresponding NAV sheet.')
    return selected, curves, pd.DataFrame(audits), comparison


def evaluate_windows(curves, regimes=REGIMES):
    """Exact closing anchors; no backfill or partial-window comparisons."""
    rows, coverage = [], []
    for cohort,frame in curves.items():
        for regime in regimes:
            start,end = pd.Timestamp(regime.start),pd.Timestamp(regime.end)
            covered = start in frame.index and end in frame.index
            coverage.append({'Regime':regime.name,'Kind':regime.kind,'Cohort':cohort,'Requested Start':start,
                             'Requested End':end,'Available Start':frame.index[0],'Available End':frame.index[-1],
                             'Status':'Covered' if covered else 'Unavailable: exact window boundary missing'})
            if not covered:
                continue
            window = frame.loc[start:end]
            for name in window:
                rows.append({'Regime':regime.name,'Kind':regime.kind,'Experiment':name,'Cohort':cohort,
                             'Start':start,'End':end,'Observations':len(window),**nav_metrics(window[name])})
    if not rows:
        raise ValueError('No complete regime windows are covered by the workbook.')
    details = pd.DataFrame(rows)
    summary = details.groupby(['Regime','Kind','Experiment'],sort=False).agg(
        **{'Launch Tests':('Cohort','size'),'Mean Return':('Total Return','mean'),
           'Worst Return':('Total Return','min'),'Best Return':('Total Return','max'),
           'Worst Drawdown':('Max Drawdown','min'),'Mean Drawdown':('Max Drawdown','mean')}).reset_index()
    return details,summary,pd.DataFrame(coverage)


def fresh_checks(configs, output, regimes=REGIMES):
    """Reuse the engine with declared new inceptions before each stress period."""
    from backtesting.conservative_runner import make_case, research_session, preflight_cases
    from core.universe import load_catalog
    from config.settings import BacktestConfig
    from studio.service import config_json
    from utils.exporters import export_results
    session = research_session()
    catalogue = load_catalog()
    rows, plans = [], []
    for ri,regime in enumerate(regimes,1):
        if not regime.fresh_start:
            continue
        print(f'Fresh start: {regime.name} · {len(configs)} simple configurations',flush=True)
        actual_entry = None
        for ci,(_,config) in enumerate(configs.iterrows(),1):
            base = BacktestConfig(initial_selection='highest_score',start_date=regime.fresh_start,
                                  end_date=regime.simulation_end,cohorts=(regime.fresh_cohort,),execution_lag=1,
                                  comparator='<',candidate_audit=False,cost_bps=0)
            case = make_case(config.Experiment,{'group':'Custom universe','outcome':'Annual',
                             'tickers':[t.strip() for t in config.Universe.split(',')]},catalogue,
                             session.metrics.ticker.unique(),base,n=int(config.N),metric=config.Trigger,
                             floor=float(config.Threshold),cadence=config.Review)
            preflight_cases(session,[case])
            e = case.experiment
            result = session.run(e.score,e.backtest)
            entry = result['summary'].Start.iloc[0]
            if actual_entry is not None and entry != actual_entry:
                raise ValueError('Fresh-start configurations must have identical inception dates.')
            actual_entry = entry
            if entry > pd.Timestamp(regime.start):
                raise ValueError('Fresh inception must occur before the stress window.')
            nav = result['daily'].set_index('Date')['Strategy NAV']
            detail,_,_ = evaluate_windows({regime.fresh_cohort:nav.to_frame(config.Experiment)},[regime])
            events = result['diagnostics']
            initial = events.loc[events['Event Type'].eq('initial'),'Fund'].tolist()
            at_start = result['holdings'].loc[result['holdings'].Date.eq(pd.Timestamp(regime.start)),'Fund'].tolist()
            switches = events.loc[events['Event Type'].eq('switch') & events.Date.gt(pd.Timestamp(regime.start)) & events.Date.le(pd.Timestamp(regime.end))]
            rows.append({**detail.iloc[0].to_dict(),'Fresh Entry':entry,'Initial Funds':', '.join(initial),
                         'Funds at Window Start':', '.join(at_start),'Window Replacements':len(switches)})
            folder = Path(output)/'fresh_runs'/f'{ri:02d}'/f'{ci:03d}'
            export_results(result,folder,{**result['parameters'],'sources':session.sources,'regime':asdict(regime)},excel=False,plots=False)
            (folder/'config.json').write_text(config_json(e.score,e.backtest))
            plans.append({'Regime':regime.name,'Experiment':config.Experiment,'Requested Entry':regime.fresh_start,
                          'Launch Month':regime.fresh_cohort,'Actual Entry':entry,'Simulation End':result['summary'].End.iloc[0],
                          'Report folder':folder.relative_to(output).as_posix()})
    return pd.DataFrame(rows),pd.DataFrame(plans),session.sources


def drawdown_heatmap(summary, configs, output, filename='stress-drawdowns.png', value='Worst Drawdown'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stress = [r.name for r in REGIMES if r.kind=='Stress' and r.name in set(summary.Regime)]
    if not stress:
        return
    matrix = summary.pivot(index='Experiment',columns='Regime',values=value).reindex(index=configs.Experiment,columns=stress)
    labels = []
    for _,c in configs.iterrows():
        trigger = {'ecr':'ECR','buffer_integrity':'Buffer','dbb':'DBB','cap_integrity':'Cap'}[c.Trigger]
        labels.append(f'N={int(c.N)} | {trigger} < {c.Threshold:g} | {c.Review}')
    fig,ax=plt.subplots(figsize=(11,max(5,len(matrix)*.37)))
    ax.imshow(-matrix.to_numpy()*100,cmap='YlOrRd',vmin=0,vmax=max(15.,-matrix.min().min()*100),aspect='auto')
    ax.set_xticks(range(len(stress)),labels=stress,rotation=20,ha='right')
    ax.set_yticks(range(len(matrix)),labels=labels)
    for i in range(len(matrix)):
        for j in range(len(stress)):
            if pd.notna(matrix.iloc[i,j]):
                ax.text(j,i,f'{matrix.iloc[i,j]:.2%}',ha='center',va='center',fontsize=9)
    ax.set_title('Stress-window drawdown · fresh starts' if value=='Max Drawdown' else 'Stress-window drawdown · worst of the existing launch portfolios')
    fig.tight_layout();fig.savefig(Path(output)/filename,dpi=140);plt.close(fig)
