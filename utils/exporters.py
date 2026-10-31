"""Portable reports with immutable run directories and complete configuration."""
import json
from pathlib import Path

import pandas as pd

SHEETS = {'summary': 'Summary', 'daily': 'Daily NAV', 'diagnostics': 'ECR Diagnostics',
          'decisions': 'Review Decisions', 'candidates': 'Candidate Audit', 'holdings': 'Portfolio Holdings',
          'trades': 'Executed Trades', 'weight_reviews': 'Weight Reviews', 'universe': 'Fund Universe'}
# Per-session ledgers dominate Excel write time; compact workbooks leave them to the CSV files.
LEDGERS = ('daily', 'decisions', 'candidates', 'holdings')


def export_results(result, output, manifest, excel=True, plots=True):
    """Write CSVs for every table, an Excel workbook and optional PNGs into a new directory.

    excel: True/'full' (every table), 'compact' (summary, NAV by launch, events and trades;
    ledgers stay in CSV) or False/None.
    """
    mode = {True: 'full', False: None}.get(excel, excel)
    if mode not in ('full', 'compact', None):
        raise ValueError('excel must be full, compact or None')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    names = {key: sheet for key, sheet in SHEETS.items() if key in result}
    for key in names:
        if len(result[key].columns):
            result[key].to_csv(output / f'{key}.csv', index=False)
    if mode:
        sheets = names if mode == 'full' else {k: v for k, v in names.items() if k not in LEDGERS}
        with pd.ExcelWriter(output / 'ecr_results.xlsx', engine='openpyxl') as writer:
            for key, sheet in sheets.items():
                result[key].to_excel(writer, sheet_name=sheet, index=False)
                if key == 'summary' and mode == 'compact':
                    # One column per launch: small enough for Excel, the shape analysts chart.
                    result['daily'].pivot(index='Date', columns='Cohort', values='Strategy NAV').to_excel(
                        writer, sheet_name='NAV by Launch')
            pd.DataFrame([{'Key': k, 'Value': json.dumps(v, default=str)}
                          for k, v in manifest.items()]).to_excel(writer, sheet_name='Run Manifest', index=False)
            if mode == 'compact':
                pd.DataFrame({'Read me': [f'{SHEETS[key]}: {key}.csv' for key in LEDGERS if key in names]
                              + ['This compact workbook omits per-session ledgers; the CSV files beside it are complete.']}
                             ).to_excel(writer, sheet_name='Read Me', index=False)
    if plots:
        from visualization.performance_plots import plot_results
        plot_results(result, output)
    return output
