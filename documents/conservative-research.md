# Conservative ECR research

The studio has two client presets: **Super Buffer** and **UCITS Conservative**.
Batch experiments live in scripts, keeping the interface simple. Every new
research configuration selects the highest **multiplicative ECR** at inception
and when a trigger calls for replacement. The scoring coefficients stay fixed.

## First: inspect one run

From the repository in PowerShell, with `requirements-app.txt` installed:

```powershell
.\.venv\Scripts\python.exe scripts/run_conservative_iteration.py
```

Default: F-series annual universe, three funds, Buffer Integrity **< 0.80** of
ideal fresh score 1.00, monthly reviews, January launch, highest ECR selection,
next-close execution and zero transaction cost. Start is no earlier than
2021-01-19; the end is the last available observation. N stays fixed per run.
The initial basket is equally weighted; only breached positions are reconsidered.

The script prints its unique output folder under:

```text
G:\Shared drives\Portfolio Management\Funds\Archive\Model Portfolio Allocation\ECR Backtest Results
```

Open `ecr_results.xlsx` first. It includes daily NAV, positions and weights,
review decisions, candidate rankings, entry/replacement component snapshots,
executed trades, weight reviews and summary statistics. `charts/` contains PNG
reports and interactive HTML performance/trigger/weight charts. `config.json`,
source hashes, `manifest.json`, CSVs and `results.zip` are saved alongside them.
`ARCHIVE_COMPLETE.txt` appears only after a successful save.

For a different single iteration:

```powershell
.\.venv\Scripts\python.exe scripts/run_conservative_iteration.py --funds 5 --trigger dbb --threshold 0.90 --cadence quarterly --cohorts JAN
.\.venv\Scripts\python.exe scripts/run_conservative_iteration.py --funds 1 --trigger ecr --threshold 1.5
.\.venv\Scripts\python.exe scripts/run_conservative_iteration.py --funds 3 --score-improvement 0.1 --max-replacements 1 --turnover-budget 0.4 --weight-band 0.05
```

`--weight-band 0.05` means ±5 percentage points around 1/N, not 5% of the target.
`--turnover-budget 0.4` means 40% of portfolio capital at a review. A zero maximum
replacement count means unlimited. ECR thresholds are score points; component
thresholds are fractions of an ideal fresh score of 1.00. DBB is a score, not the
raw percentage loss before protection.

## Then: compare the full grid

Inspect the plan without writing results:

```powershell
.\.venv\Scripts\python.exe scripts/run_conservative_suite.py --list
```

Run all default cases once the inspection report looks right:

```powershell
.\.venv\Scripts\python.exe scripts/run_conservative_suite.py
```

The default [grid](../config/conservative-grid.json) has **85 configurations**:

| Dimension | Values |
| --- | --- |
| Universe | Available annual F-series funds |
| Portfolio size | 3, 5 |
| Review cadence | Monthly, quarterly |
| ECR trigger reference | Below 1.50 |
| Buffer Integrity trigger | Below 0.70, 0.80, 0.90 |
| DBB trigger | Below 0.80, 0.90, 0.95 |
| Turnover profiles | Standard; ECR improvement ≥0.15; limited turnover |
| Single-ticker reference | ECR <1.50, monthly review |

The limited-turnover profile combines ECR improvement ≥0.10, at most one
replacement per review, a 40% turnover budget for N=3 or 25% for N=5, and a ±5pp
weight band. The matrix is 2 sizes × 2 cadences × 7 triggers × 3 profiles = 84,
plus one single-ticker reference. All twelve launch months are separate tests
by default. The full run writes **1,020 cohort reports** and can take several
minutes or longer depending on Excel/chart generation and shared-drive speed.
The script prints progress and saves each completed iteration immediately.
Scored data are cached across cases; full ledgers are released from memory after
saving. Candidate rankings are included in the single inspection run; the batch
omits the full candidate audit to reduce output size, retaining decisions,
positions, diagnostics and trades.

For a small comparison or an evaluation period:

```powershell
.\.venv\Scripts\python.exe scripts/run_conservative_suite.py --filter "Buffer < 0.8" --cohorts JAN --limit 3
.\.venv\Scripts\python.exe scripts/run_conservative_suite.py --start-date 2024-01-01 --end-date 2026-03-27
```

Both scripts accept `--archive-root "C:\ECR Results"` for a different destination,
`--start-date`, `--end-date`, `--cohorts JAN,APR` and `--cost-bps 5`. An unavailable
shared drive is an error, never a silently substituted location. Each invocation
creates a new folder; it never overwrites an existing run.

## Reading the batch output

Start with **index.html**, which links all charts and individual reports.

- **comparison.xlsx:** sortable scenario metrics, launch-cohort results, exact
  configuration dimensions, and a daily NAV sheet per launch month with every
  iteration side by side. Returns/drawdowns are numeric percentage cells.
- **charts/*-comparison.html:** performance and drawdown for every iteration in
  that launch. The dropdown shows the downside shortlist, all scenarios or a
  portfolio size; the legend toggles individual configurations.
- **charts/*-shortlist.png:** readable static panels by portfolio size, showing
  up to six configurations selected by worst-cohort drawdown, with SPY/BUFR.
- **charts/risk-return.html / .png:** all scenarios' return, downside and turnover
  trade-offs. Hover in the HTML for the configuration and statistics.
- **runs/:** full individual Excel/CSV/PNG/HTML/ZIP packages and replay configs.
- **grid.json / plan.json:** the requested grid and exact expanded configurations.
- **batch-status.json:** planned/completed runs, source hashes and completion status.
  Interrupted/failed batches retain completed individual reports; no complete
  marker is written and the process exits with an error. A rerun starts a fresh batch.

The table sorts by least severe **worst-cohort drawdown**, then mean Sortino and
annualized turnover. Separate ranks make these trade-offs visible; this is not
an automatic recommendation. Aggregate columns are arithmetic means across
launch tests, **not** the return of an investable combined portfolio. Each
launch's curves use identical dates; mismatched windows are rejected before
running. Validate shortlisted parameters on a separate period before choosing
one for a client. Budget constraints can retain a deteriorated position, and
weight bands can increase turnover versus letting weights drift indefinitely.

## Customize filtered iterations

The single-run script supports the same filters as the UI: `--group`, `--outcome`,
`--series`, `--strategies`, `--reference-assets`, `--outcome-months`,
`--buffer-category`, and `--tickers`. Comma-separated filters are intersected;
explicit tickers must match those filters. For example:

```powershell
.\.venv\Scripts\python.exe scripts/run_conservative_iteration.py --group "F series" --reference-assets SPY --buffer-category "Up to 10%" --tickers FJAN,FFEB,FMAR,FAPR,FMAY
```

For batches, copy `config/conservative-grid.json` and pass `--grid path/to/copy.json`.
Each item in `universes` accepts these keys (lists for multi-value filters):

```json
{
  "name": "Selected F funds",
  "group": "F series",
  "outcome": "Annual",
  "series": ["F"],
  "strategies": ["Buffer"],
  "references": ["SPY"],
  "months": ["JAN", "FEB", "MAR", "APR", "MAY"],
  "buffer_band": "Up to 10%"
}
```

An optional `tickers` list narrows this further. Multiple universe definitions
are crossed with the entire scenario matrix. Duplicate scenarios are rejected.
New portfolio sizes require matching `turnover_budget_by_size` entries where
that profile is used. Cadences also support `semiannual` and `annual`; triggers
also support `cap_integrity`. ECR thresholds can include 1.0 and 2.0. The grid
keeps selection fixed to highest ECR so the trigger comparisons stay focused.

Only real loaded histories are used. D/G, UCITS and quarterly universes become
available after their catalogue entries and normalized CSVs are added to
`input_data/fund_metrics/`. See [universes and archives](universes-and-archives.md).
Metadata alone does not create runnable history. The current F-series backtests
are strategy research for the conservative client, not tests of UCITS fund data.

## Quarterly outcome products

Annual products use period center/width 180/75 days; quarterly products use
45/18.75. Both retain reset sigmas 0.07 downside and 0.05 upside and the same
multiplicative coefficients. This proportional quarterly timing remains a
hypothesis requiring actual quarterly data. Quarterly **reviews** of annual
funds retain the annual formula; annual/quarterly products are never mixed in
one universe. Current currency and benchmark checks remain enforced.
