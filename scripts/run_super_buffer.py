"""Run the single-fund Super Buffer strategy: highest ECR selection, ECR < 1.5 trigger."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtesting.conservative_runner import (
    add_common_arguments, base_config, ensure_archive_root, make_case,
    preflight_cases, research_session, run_and_archive)
from core.universe import load_catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    try:
        destination = ensure_archive_root(args.archive_root)
        catalogue = load_catalog()
        session = research_session()
        case = make_case(
            'Super Buffer', {'group': 'F series', 'outcome': 'Annual'},
            catalogue, session.metrics.ticker.unique(), base_config(args),
            n=1, metric='ecr', floor=1.5, cadence='monthly')
        preflight_cases(session, [case])
        print('Super Buffer: one fund; highest ECR selects; replace when ECR < 1.5 '
              'at monthly reviews; next-close execution.', flush=True)
        result, folder = run_and_archive(session, case, destination, catalogue)
        print(result['summary'][['Cohort', 'Annualized Return', 'Max Drawdown', 'Switches']].to_string(index=False))
        print(f'Open {folder / "ecr_results.xlsx"}\nCharts: {folder / "charts"}', flush=True)
        return folder
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Unable to complete the Super Buffer run: {exc}\n')


if __name__ == '__main__':
    main()
