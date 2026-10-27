"""Run the conservative scenario library with cached data and export comparisons."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtesting.research import ResearchSession, export_comparison
from backtesting.scenarios import conservative_suite
from config.settings import BacktestConfig, MONTHS
from studio.service import config_json


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohorts',default=','.join(MONTHS))
    parser.add_argument('--start-date',default='2021-01-19')
    parser.add_argument('--end-date')
    parser.add_argument('--cost-bps',type=float,default=0.)
    parser.add_argument('--filter',default='',help='Case-insensitive substring, e.g. combined or N=3')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    config=BacktestConfig(start_date=args.start_date,end_date=args.end_date,execution_lag=1,
                          cost_bps=args.cost_bps,cohorts=tuple(x.strip().upper() for x in args.cohorts.split(',')))
    tests=[e for e in conservative_suite(config) if args.filter.lower() in e.name.lower()]
    if not tests:
        parser.error('Filter matched no scenarios')
    session=ResearchSession(root/'input_data/website_metrics.csv',root/'input_data/benchmark_ts.csv',root/'input_data/roll_dates.csv')
    results={}
    for i, experiment in enumerate(tests,1):
        print(f'[{i}/{len(tests)}] {experiment.name}',flush=True)
        results[experiment.name]=session.run(experiment.score,experiment.backtest)
    output=args.output or root/'output'/('conservative-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    export_comparison(results,output)
    (output/'configs').mkdir()
    import pandas as pd
    from backtesting.research import comparison_table
    table=comparison_table(results)
    with pd.ExcelWriter(output/'comparison.xlsx',engine='openpyxl') as writer:
        table.to_excel(writer,sheet_name='Scenarios')
        pd.concat([r['summary'].assign(Experiment=name) for name,r in results.items()]).to_excel(writer,sheet_name='Cohorts',index=False)
    for i, experiment in enumerate(tests,1):
        (output/'configs'/f'{i:02d}.json').write_text(config_json(experiment.score,experiment.backtest))
    try:
        from visualization.research_plots import comparison_figures
        heatmap,tradeoff=comparison_figures(results,table)
        heatmap.write_html(output/'cohort-heatmap.html',include_plotlyjs='directory')
        tradeoff.write_html(output/'risk-return.html',include_plotlyjs='directory')
    except ImportError:
        print('Install requirements-app.txt to include interactive comparison charts.')
    print(f'Results: {output}\nPrepared score panels: {session.score_builds}',flush=True)


if __name__=='__main__':
    main()
