"""Golden economic fingerprints on the repository data; regenerate only for a named, intended change.

    python tests/golden_cases.py --write

Hashes exclude identity columns (Strategy ID, Event/Review IDs, Client) so renames and id migrations do
not move them; any change to NAVs, holdings, decisions or trades does.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtesting.engine import simulate
from config.settings import BacktestConfig, MONTHS, ScoreConfig
from studio.service import load_dataset, load_preset

FIXTURE = ROOT / 'tests/fixtures/golden_hashes.json'
FRAMES = ('summary', 'daily', 'holdings', 'diagnostics', 'decisions', 'trades', 'weight_reviews')
IDENTITY_COLUMNS = ['Strategy ID', 'Event ID', 'Review ID', 'Client']
F_SERIES = tuple('F' + month for month in MONTHS)


def cases():
    super_buffer, ucits = load_preset('super-buffer.json')[1], load_preset('ucit-conservative.json')[1]
    research = BacktestConfig(initial_selection='highest_score', universe=F_SERIES, execution_lag=1, comparator='<')
    controls = dict(min_selection_improvement=.1, max_replacements_per_review=1, max_turnover_per_review=.4, weight_band=.05)
    configs = {
        'workbook-default': BacktestConfig(),
        'super-buffer-preset': super_buffer,
        'super-buffer-explicit-universe': replace(super_buffer, universe=F_SERIES),
        'ucits-preset': ucits,
        'ucits-explicit-universe': replace(ucits, universe=F_SERIES),
        'n1-cohort-daily-hold22-5bp': BacktestConfig(review='daily', min_holding_days=22, execution_lag=1, cost_bps=5),
        'n3-buffer-drop': replace(research, portfolio_size=3, trigger='buffer_drop', buffer_drop=.2),
        'n5-dbb-0.9': replace(research, portfolio_size=5, trigger='metric_floor', buffer_metric='dbb', metric_floor=.9),
        'n3-controls-band': replace(research, portfolio_size=3, trigger='metric_floor', metric_floor=.8, **controls),
        'n3-noop-score-gap': replace(research, portfolio_size=3, trigger='metric_floor', metric_floor=.9,
                                     min_selection_improvement=1e-300),
        'n5-quarterly-5bp': replace(research, portfolio_size=5, trigger='metric_floor', metric_floor=.8,
                                    review='quarterly', cost_bps=5),
        'n8-lag0': replace(research, portfolio_size=8, trigger='metric_floor', metric_floor=.8, execution_lag=0),
        'n6-lag0-ecr': replace(research, portfolio_size=6, trigger='ecr', threshold=1.5, execution_lag=0),
    }
    return {name: replace(config, candidate_audit=False, end_date='2026-03-27') for name, config in configs.items()}


def frame_hash(frame):
    frame = frame.drop(columns=[c for c in IDENTITY_COLUMNS if c in frame], errors='ignore')
    digest = hashlib.sha256(','.join(map(str, frame.columns)).encode())
    if len(frame):
        digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return digest.hexdigest()


def fingerprint(result):
    summary = result['summary'].set_index('Cohort')
    return {'frames': {key: frame_hash(result[key]) for key in FRAMES if key in result},
            'final_nav': summary['Total Return'].add(1).mul(100).round(10).to_dict(),
            'max_drawdown': summary['Max Drawdown'].round(10).to_dict(),
            'switches': summary['Switches'].astype(int).to_dict()}


def compute(names=None):
    prepared = {}
    output = {}
    for name, config in cases().items():
        if names and name not in names:
            continue
        key = tuple(sorted(config.universe)) if config.universe else None
        if key not in prepared:
            prepared[key] = load_dataset(ScoreConfig(), key)[0]
        output[name] = fingerprint(simulate(prepared[key], config))
    return output


if __name__ == '__main__':
    if '--write' not in sys.argv:
        sys.exit('Pass --write to regenerate the golden fixture after an intended economic change.')
    FIXTURE.write_text(json.dumps(compute(), indent=1, sort_keys=True) + '\n')
    print(f'Wrote {FIXTURE}')
