"""Portable reports with immutable run directories and complete configuration."""
import json
from pathlib import Path

import pandas as pd


def export_results(result, output, manifest, excel=True, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    names = {'summary': 'Summary', 'daily': 'Daily NAV', 'diagnostics': 'ECR Diagnostics',
             'decisions': 'Review Decisions', 'candidates': 'Candidate Audit'}
    if 'holdings' in result:
        names['holdings'] = 'Portfolio Holdings'
    for key, sheet in [('trades', 'Executed Trades'), ('weight_reviews', 'Weight Reviews'), ('universe', 'Fund Universe')]:
        if key in result:
            names[key] = sheet
    for key in names:
        if len(result[key].columns):
            result[key].to_csv(output / f'{key}.csv', index=False)
    if excel:
        with pd.ExcelWriter(output / 'ecr_results.xlsx', engine='openpyxl') as writer:
            for key, sheet in names.items():
                result[key].to_excel(writer, sheet_name=sheet, index=False)
            pd.DataFrame([{'Key': k, 'Value': json.dumps(v, default=str)}
                          for k, v in manifest.items()]).to_excel(writer, sheet_name='Run Manifest', index=False)
    if plots:
        from visualization.performance_plots import plot_results
        plot_results(result, output)
    return output
