"""Run the filtered conservative grid and archive individual and aggregate reports."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtesting.conservative_runner import (DEFAULT_GRID, add_common_arguments, base_config,
    ensure_archive_root, grid_cases, preflight_cases, research_session, run_and_archive)
from config.settings import MONTHS
from core.universe import load_catalog
from studio.archive import new_archive_path
from visualization.conservative_comparison import export_batch_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, cohorts=MONTHS)
    parser.add_argument('--grid', type=Path, default=DEFAULT_GRID, help='JSON containing universe filters and the scenario grid.')
    parser.add_argument('--filter', default='', help='Case-insensitive substring in scenario names, e.g. N=3 or DBB.')
    parser.add_argument('--list', action='store_true', help='Validate and print the selected configurations without running or writing results.')
    parser.add_argument('--limit', type=int, help='Run only the first N matching cases, for a small inspection batch.')
    parser.add_argument('--plots', choices=['top5', 'none'], default='top5',
                        help='Default: three top-five comparison charts only. Use none to skip all plotting.')
    parser.add_argument('--chart-cohort', type=str.upper, choices=MONTHS,
                        help='Launch shown in static charts; defaults to the first requested cohort. HTML can switch launches.')
    args = parser.parse_args(argv)
    output = None
    status = None
    try:
        if args.limit is not None and args.limit < 1:
            raise ValueError('--limit must be at least 1.')
        grid = json.loads(args.grid.read_text())
        catalogue = load_catalog()
        session = research_session()
        base = replace(base_config(args), candidate_audit=False)
        if args.chart_cohort and args.chart_cohort not in base.cohorts:
            raise ValueError('--chart-cohort must be one of the requested --cohorts.')
        cases = grid_cases(grid, catalogue, session.metrics.ticker.unique(), base)
        cases = [c for c in cases if args.filter.lower() in c.experiment.name.lower()]
        if args.limit:
            cases = cases[:args.limit]
        if not cases:
            raise ValueError('Filter matched no configurations.')
        preflight_cases(session, cases)
        print(f'{len(cases)} configurations × {len(base.cohorts)} launch month(s); next-close execution; {base.cost_bps:g} bps.', flush=True)
        if args.list:
            for i,case in enumerate(cases,1):
                print(f'{i:03d}  {case.experiment.name}')
            return
        print('Individual charts: off. Comparison charts: ' + ('top five by return / drawdown / Sharpe.' if args.plots == 'top5' else 'off.'), flush=True)
        destination = ensure_archive_root(args.archive_root)
        output = new_archive_path(destination,'Conservative-comparison','batch')
        output.mkdir(parents=True,exist_ok=False)
        (output/'grid.json').write_text(json.dumps(grid,indent=2)+'\n')
        plan = [{'name':c.experiment.name,'filters':c.filters,'score':asdict(c.experiment.score),
                 'backtest':asdict(c.experiment.backtest)} for c in cases]
        (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        status = {'state':'running','created_utc':datetime.now(timezone.utc).isoformat(),
                  'planned':len(cases),'completed':[],'sources':session.sources,
                  'plots':args.plots,'individual_plots':False,'chart_cohort':args.chart_cohort or base.cohorts[0]}

        def write_status():
            temp = output/'batch-status.tmp'
            temp.write_text(json.dumps(status,indent=2)+'\n')
            temp.replace(output/'batch-status.json')

        write_status()
        results, folders = {}, {}
        print(f'Batch folder: {output}',flush=True)
        for i,case in enumerate(cases,1):
            name = case.experiment.name
            print(f'[{i}/{len(cases)}] {name}',flush=True)
            # Short child names keep the user's deep Windows archive below path limits.
            result, folder = run_and_archive(session,case,output/'runs',catalogue,folder_name=f'{i:03d}',plots=False)
            # Detailed ledgers are already on disk; keep only comparison inputs in RAM.
            results[name] = {k:result[k] for k in ('summary','daily','parameters','timings','strategy_id','sources')}
            folders[name] = folder
            status['completed'].append({'name':name,'folder':folder.relative_to(output).as_posix(),
                                        'strategy_id':result['strategy_id']})
            write_status()
            del result
        export_batch_comparison(results,cases,folders,output,plots=args.plots,chart_cohort=args.chart_cohort)
        status.update(state='complete',prepared_panels=session.score_builds,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        write_status()
        (output/'ARCHIVE_COMPLETE.txt').write_text('All individual reports and aggregate comparisons completed. Start with index.html or comparison.xlsx.\n')
        print(f'Comparison ready: {output / "index.html"}\nWorkbook: {output / "comparison.xlsx"}\nPrepared score panels: {session.score_builds}',flush=True)
        return output
    except (ValueError, OSError, KeyError, TypeError, KeyboardInterrupt) as exc:
        if output is not None and status is not None:
            status.update(state='incomplete',error=str(exc) or 'Interrupted')
            try:
                (output/'batch-status.json').write_text(json.dumps(status,indent=2)+'\n')
            except OSError:
                # A disconnected/full archive may also prevent the failure update.
                # The missing completion marker still identifies a partial batch.
                pass
        parser.exit(1,f'Batch not completed: {exc}\n'+(f'Completed individual reports remain in {output}\n' if output else ''))


if __name__ == '__main__':
    main()
