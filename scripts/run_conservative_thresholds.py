"""Compare lower Buffer Integrity/DBB floors with the simple conservative cases.

Run without arguments for 27 configurations across all 12 launch months.
All suite flags are supported, including --list, --cohorts and --plots none.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_conservative_suite import main as run_suite


def main(argv=None):
    return run_suite(argv, default_grid=ROOT / 'config/conservative-thresholds.json')


if __name__ == '__main__':
    main()
