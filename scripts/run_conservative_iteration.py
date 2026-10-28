"""Run one auditable conservative iteration and archive Excel, charts, and settings."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtesting.conservative_runner import (
    add_common_arguments, add_universe_arguments, base_config, ensure_archive_root,
    filters_from_args, make_case, preflight_cases, research_session, run_and_archive)
from core.universe import load_catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    add_universe_arguments(parser)
    parser.add_argument('--name', default='UCITS Conservative inspection')
    parser.add_argument('--funds', type=int, default=3)
    parser.add_argument('--trigger', choices=['ecr', 'buffer_integrity', 'dbb', 'cap_integrity'], default='buffer_integrity')
    parser.add_argument('--threshold', type=float, help='Default: 0.80 for a component, 1.50 for ECR.')
    parser.add_argument('--cadence', choices=['monthly', 'quarterly', 'semiannual', 'annual'], default='monthly')
    parser.add_argument('--score-improvement', type=float, default=0.)
    parser.add_argument('--max-replacements', type=int, default=0, help='0 means unlimited.')
    parser.add_argument('--turnover-budget', type=float, default=1., help='Fraction of capital per review, e.g. 0.4 for 40%%.')
    parser.add_argument('--weight-band', type=float, help='Absolute drift from 1/N, e.g. 0.05 for 5 percentage points.')
    args = parser.parse_args(argv)
    try:
        destination = ensure_archive_root(args.archive_root)
        catalogue = load_catalog()
        session = research_session()
        threshold = args.threshold if args.threshold is not None else (1.5 if args.trigger == 'ecr' else .8)
        controls = {'min_selection_improvement': args.score_improvement,
                    'max_replacements_per_review': args.max_replacements,
                    'max_turnover_per_review': args.turnover_budget, 'weight_band': args.weight_band}
        case = make_case(args.name, filters_from_args(args), catalogue, session.metrics.ticker.unique(),
                         base_config(args), args.funds, args.trigger, threshold, args.cadence, controls)
        preflight_cases(session, [case])
        print(f'{args.name}: N={args.funds}, {args.trigger} < {threshold:g}, {args.cadence}; highest ECR selects.', flush=True)
        result, folder = run_and_archive(session, case, destination, catalogue)
        print(result['summary'][['Cohort', 'Annualized Return', 'Max Drawdown', 'Switches']].to_string(index=False))
        print(f'Open {folder / "ecr_results.xlsx"}\nCharts: {folder / "charts"}', flush=True)
        return folder
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Unable to complete the iteration: {exc}\n')


if __name__ == '__main__':
    main()
