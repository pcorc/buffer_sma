# Simple conservative comparisons across regimes

The 85-case comparison starts every launch in 2021 and ends in March 2026. Cases
are aligned within each launch, but twelve overlapping entry months are not
independent tests of twelve different market regimes. An existing portfolio's
performance during a selloff and a newly purchased portfolio's performance are
different questions. The regime report keeps those questions separate.

Use the simpler monthly grid (15 configurations, no extra turnover controls):

```powershell
.\.venv\Scripts\python.exe scripts/run_conservative_suite.py --grid config/conservative-simple.json --plots none
```

The grid varies only portfolio size (3/5) and trigger (ECR 1.5, Buffer Integrity
0.70/0.80/0.90, DBB 0.80/0.90/0.95), plus a one-fund ECR reference. Selection is
always highest multiplicative ECR. Initial weights are equal and then drift;
only breached positions are reconsidered. There are no extra replacement limits.

## Lower threshold comparison

To add 30%, 50% and 60% score floors while keeping all the simple cases:

```powershell
.\.venv\Scripts\python.exe scripts/run_conservative_thresholds.py
```

This runs **27 configurations**, each across the same twelve 2021 launch months:

| Portfolio size | Trigger floors (strictly below) | Selection / review |
| --- | --- | --- |
| 3 and 5 funds | Buffer Integrity: 0.30, 0.50, 0.60, 0.70, 0.80, 0.90 | Highest ECR / monthly |
| 3 and 5 funds | DBB score: 0.30, 0.50, 0.60, 0.80, 0.90, 0.95 | Highest ECR / monthly |
| 3 and 5 funds | ECR: 1.50 | Highest ECR / monthly |
| 1 fund (Super Buffer reference) | ECR: 1.50 | Highest ECR / monthly |

All use the original multiplicative ECR, next-close execution and zero costs.
Extra turnover controls are disabled. A lower floor permits **more** deterioration
before a review triggers a replacement search. A 0.30 floor means below 30% of
the ideal 1.00 score (70% deterioration), not a 30% decline from purchase.
Buffer Integrity measures the remaining fraction of the original buffer; DBB is
a nonlinear score, not a percentage of protection. With the current DBB formula
`exp(5 * downside_before_buffer_net)`, scores 0.30/0.50/0.60 correspond to inputs
of approximately -24.1%/-13.9%/-10.2%. These are permissive exits and may rarely
trigger. They are turnover-versus-drawdown experiments, not inherently safer rules.

Each run creates a new folder in the configured G-drive archive with
`comparison.xlsx`, CSV tables, individual run workbooks/configurations and three
top-five chart pairs (return, drawdown, Sharpe). Each chart identifies its parameter
sets. No individual-run charts are generated. Use `--plots none` to skip all
plotting, `--list` to preview the cases, or `--cohorts JAN` for a smaller first run.
`--start-date`, `--end-date` and `--archive-root` work as on the regular suite.
The grid is saved in `config/conservative-thresholds.json`.

Compare matched launch dates and stress windows before choosing a floor. Pass the
newly exported `comparison.xlsx` to the regime analysis command below; it supports
all 27 cases, including fresh-start checks, without adding trading-rule controls.

## Stress-window reports

To study a saved comparison without rerunning its backtests:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_conservative_regimes.py --workbook "C:\path\to\comparison.xlsx"
```

Use the actual path to your exported workbook. The report goes to a new folder
in the same G-drive archive. `--archive-root "C:\ECR Results"` changes that location.
The script checks **all** NAV/summary rows for consistent total return, maximum
drawdown, Sharpe and start/end dates, then selects only the simple monthly cases.
`--review all` also includes simple quarterly cases from the source workbook.

For new portfolios started before each stress event, add:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_conservative_regimes.py --workbook "C:\path\to\comparison.xlsx" --fresh-starts
```

| Stress test | Exact evaluation closes | Fresh portfolio entry under the supplied roll calendar |
| --- | --- | --- |
| 2022 bear market | Dec 31, 2021–Oct 12, 2022 | Dec 20, 2021 |
| 2023 correction | Jul 31–Oct 27, 2023 | Jul 24, 2023 |
| 2024 summer selloff | Jul 16–Aug 5, 2024 | May 20, 2024 |
| 2025 spring selloff | Feb 19–Apr 8, 2025 | Jan 21, 2025 |

The report also shows the 2022 calendar year, 2023–24 expansion and 2025 rebound
as context. COVID 2020 appears explicitly as unavailable for the uploaded 2021
launches. A valid COVID test would need the then-investable universe and suitable
benchmark history; it cannot assume all twelve current funds already existed.

May 2024 is intentional: the current source file has missing SPY and BUFR values
at the June 24 inception, which the engine correctly rejects. No prices are
filled. Fresh simulations run through the declared later end date and evaluate
only the stated stress interval; this does not change their chronological trading
rules. The full replay plan, input hashes and CSV ledgers are saved in `fresh_runs/`.

## Read the results

- **Regime Summary:** mean period return and worst period drawdown across existing
  launch cohorts. Compare rows *within one regime*, not one pooled performance rank.
- **Carry-in Detail:** one row per regime/configuration/original launch, all using
  the same exact closing dates. No partial windows or boundary forward-fills.
- **Fresh Start Results:** newly initiated portfolios, with actual entry, initial
  funds, funds at the stress boundary and number of replacements inside the window.
- **Fresh Start Plan:** requested/actual entry dates and simulation end dates.
- **Coverage / NAV Audit:** unavailable history and numerical reconciliation.
- **Drawdown heatmaps:** one for existing portfolios, one for fresh portfolios if
  requested. `--no-plots` suppresses both; all tables remain.

Period drawdown resets the high-water mark at the beginning of each evaluation
window. It therefore differs from full-history drawdown carried over from an
earlier peak. Entry-month cohorts overlap; averaging them does not add twelve
independent stress events. The chosen dates are retrospective diagnostics, not a
trading signal. Short-window Sharpe is not used to pick a winner. The original
engine Sharpe assumes a zero risk-free rate.

Fresh checks use the repository's current data, original multiplicative ECR,
strict thresholds, next-close execution and zero costs. No new trading rules or
engine changes are introduced. For other entry dates, the existing suite already
supports them:

```powershell
.\.venv\Scripts\python.exe scripts/run_conservative_suite.py --grid config/conservative-simple.json --start-date 2022-01-01 --end-date 2022-12-30 --cohorts JAN --plots none
```

This means the first qualifying January roll entry on/after the date bound,
**not** an assumed January 1 purchase. The workbook records the actual entry date.
