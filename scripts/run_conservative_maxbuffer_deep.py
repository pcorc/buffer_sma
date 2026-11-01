"""Run the conservative grid on Max Buffer and D-series (Deep Buffer) annual funds only.

Run without arguments for 85 configurations across all 12 launch months.
All suite flags are supported, including --list, --cohorts, --end-date and --plots none.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_conservative_suite import main as run_suite


def main(argv=None):
    return run_suite(argv, default_grid=ROOT / 'config/conservative-maxbuffer-deep.json')


if __name__ == '__main__':
    main()
