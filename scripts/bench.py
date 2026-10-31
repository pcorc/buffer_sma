"""Time the paths the Studio and suites depend on; best of N repeats, written as JSON.

    python scripts/bench.py --repeat 3 --output bench.json
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtesting.engine import simulate
from config.settings import MONTHS, ScoreConfig
from studio.service import build_downloads, load_dataset, load_preset

UNIVERSE = tuple('F' + month for month in MONTHS)


def best(function, repeat):
    times = []
    for _ in range(repeat):
        began = time.perf_counter()
        value = function()
        times.append(time.perf_counter() - began)
    return min(times), value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', type=Path, help='Optional JSON destination; printed either way.')
    parser.add_argument('--reports', action='store_true', help='Also time Excel/ZIP report building (slow).')
    args = parser.parse_args(argv)
    timings = {}
    timings['prepare_data'], data = best(lambda: load_dataset(ScoreConfig(), UNIVERSE)[0], args.repeat)
    for preset in ('super-buffer.json', 'ucit-conservative.json'):
        config = replace(load_preset(preset)[1], universe=UNIVERSE, candidate_audit=False)
        name = preset.removesuffix('.json')
        timings[f'simulate {name} 1 cohort'], _ = best(lambda: simulate(data, replace(config, cohorts=('JAN',))), args.repeat)
        timings[f'simulate {name} 12 cohorts'], result = best(lambda: simulate(data, config), args.repeat)
        if args.reports:
            timings[f'build_downloads {name} 12 cohorts, no plots'], _ = best(lambda: build_downloads(result, plots=False), 1)
    report = {key: round(value, 4) for key, value in timings.items()}
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + '\n')
    return report


if __name__ == '__main__':
    main()
