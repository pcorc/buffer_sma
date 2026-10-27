# ECR buffer ETF backtesting

ECR is the only strategy model in this repository. It implements **Website
Metrics AR**, including the AM: AQ components, from the supplied methodology
workbook. The legacy ECR/additive-par strategies, batch scripts, tracked bytecode,
backup code and historical result files have been removed. Raw reference inputs
remain available; all new results are ignored by Git.

## Install and run

Python 3.12 is the validated runtime. No service or API credentials are needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py --config config/ecr.json
```

Outputs go into a new timestamped directory under `output/`. A supplied
`--output` directory must not already exist. Results include CSVs, an Excel
workbook, headless NAV/score plots and a configuration/data provenance manifest.
For fast exploratory runs, omit Excel/plots with `--no-excel --no-plots`.
`--no-audit` omits candidate tables but retains all entry/switch snapshots and
review decisions.

```bash
python main.py --review daily --threshold 1.5 --min-holding-days 22 --cohorts FEB,SEP --no-excel --no-plots
python main.py --sigma-down 0.09 --sigma-up 0.06 --period-center 160 --period-width 80
python main.py --execution-lag 1 --cost-bps 5
python main.py --metrics "/path/to/methodology.xlsx"
```

`--help` lists every model lever. JSON config accepts `score` and `backtest`
objects; command-line arguments override JSON values. Models/configurations are
never inferred from encoded function names. Selection and trigger are independent:
ECR is the default ranking, with DBB, Buffer Integrity and Cap Integrity available
for component-ranking experiments.

## Backtest Studio — browser interface

```bash
python -m pip install -r requirements-app.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).
In the prepared cloud workspace, use `/workspace/buffer_sma-env/bin/python`.
The app is local; no account or API keys are required.

Start with **Super Buffer** or **UCITS Conservative**, then follow the numbered
cards: portfolio → cadence → selection → trigger → test window. All core choices
stay visible alongside a live strategy summary. The presets remain unchanged;
editing a card creates a custom run. **Results** provides return, drawdown,
per-holding trigger and allocation charts, a review log, and Excel/ZIP downloads.
The ZIP includes PNGs, offline interactive HTML, CSVs, source hashes and replayable
JSON. Extract the whole ZIP before opening its HTML files.

Selection supports `highest_ecr`, `highest_dbb`, `highest_buffer_integrity` and
`highest_cap_integrity`, for both initial entry and replacements. Review supports
`monthly`, `quarterly` (Mar/Jun/Sep/Dec), `semiannual` (Jun/Dec) and `annual` (Dec),
using the supplied holiday-adjusted monthly roll dates. These are trigger checks,
not forced rotations or portfolio-wide weight resets. The CLI also retains daily
review and the workbook's named-cohort initial-entry option.

ECR trigger choices are **1, 1.5, 2**. Component floors are fractions of ideal
fresh score **1.00**, e.g. **0.80**; thresholds above 1 would not describe component
deterioration. The UI uses strict `<`. The multiplicative ECR formula and both
client presets are retained. Advanced sections expose execution/costs, holding
and eligibility limits, candidate auditing and formula parameters.

See [the app guide](documents/backtest-studio.md) for assumptions, downloads,
validation, and suggested next improvements.

## Exact scoring model

All returns and net cap/buffer inputs are **decimals**. For reference return `r`
and remaining outcome days `d`:

```
DBB    = exp(5 * downside_before_buffer_net)
Buffer = min(1, max(0, remaining_buffer_net / buffer_net))
Cap    = min(remaining_cap_net / fund_cap_net, 1)
Reset  = exp(-r² / (2 * sigma²)), sigma = 0.07 if r <= 0 else 0.05
Period = exp(-(d - 180)² / (2 * 75²))
ECR   = (1.5 * DBB + 1.2 * Reset)
         * max(0, Buffer)^0.5 * max(0, Cap)^0.3 + 0.3 * Period
```

The weights, gate exponents, sigmas, DBB decay and period parameters are all
configurable. AQ is the period-position component; AR is the composite. There
is no additive elapsed-time par bonus or legacy time-scaling divisor.
Zero remaining days is valid for formula parity; `min_remaining_days` separately
controls selection eligibility. Zero/negative original denominators, missing
inputs and nonfinite scores are invalid, never replaced by a zero score. Missing
held-fund prices or invalid held scores on review dates fail the run.

## Rebalance and execution rules

- Default: monthly review on the supplied holiday-adjusted option-cycle calendar,
  held ECR **<= 1.5**, with no additional minimum holding period.
- Available alternatives: daily, quarterly, semiannual or annual review, strict `<`, arbitrary nonnegative
  threshold, minimum trading sessions held, and minimum remaining outcome days.
- Each cohort enters its named F-series fund on the first available trading
  session after its first qualifying monthly cycle. Default earliest start is
  2021-01-19. This reproduces all twelve workbook cohort starts.
- On a triggered review, pick the highest eligible selection score (ECR by default). Exact ties keep the current
  holding; other ties use alphabetical ticker order. Picking the current fund
  does not create a switch or restart the holding clock.
- Daily returns first accrue to the fund held coming into that close. The
  default `execution_lag=0` assumes theoretical same-close execution using that
  close's metrics, matching the workbook events. This is **not evidence that
  close-derived data can be traded at that same close in practice**.
- `execution_lag=1` selects using today's data and executes at the next observed
  trading close, keeping the outgoing fund's return through execution. A signal
  at the end of the sample is recorded as unexecuted. Diagnostic fields reflect
  the execution date; signal date and selection score are preserved separately.
- Optional `cost_bps` is one proportional cost charged per executed switch (the
  whole rotation), not per leg. Initial entry and no-op decisions are uncharged.

Scores are computed once for the complete input panel. Simulation uses indexed
date/fund arrays and shares prepared scores across cohorts; no daily full-frame
filtering or per-fund recomputation is needed. The stateful holdings loop remains
sequential to preserve trading logic.

## Inputs and provenance

The default `input_data/website_metrics.csv` is the supplied workbook's raw A:T
observations, exported as decimals. `website_metrics_provenance.json` records the
source workbook hash and sheet. The XLSX adapter reads **only Website Metrics**.
The original `input_data/data.csv` is also supported via an explicit column/unit
adapter, but its rollover-day observations differ; it is not the parity source.
Reported reference return G is used directly. No missing roll-date proxy is used.

Workbook data ends on 2026-03-27. The supplied benchmark CSV lacks twelve
interior sessions in 2024. Those prices remain NaN in daily reports and are
counted as `Missing Benchmark Sessions`; strategy sessions are retained. Endpoint
benchmark prices are required, so cumulative benchmark comparisons use observed
start/end prices. No interpolation or forward-fill is applied. Complete
benchmark data is still needed for continuous daily benchmark analysis.

The manifest records parameters, input hashes, strategy/data IDs, timings,
execution assumptions, Git commit and dirty status. Every selected snapshot and
candidate review identifies its strategy. No pivot combines distinct configurations.

## Diagnostics and validation

`ECR Diagnostics` contains the worksheet's fifteen columns in their original
order, then event identity/provenance. It captures **initial entries and actual
switches**. Separate `Review Decisions` and `Candidate Audit` tables show holds,
no-op triggers, delayed signals, ranks, eligibility, raw inputs and component
scores. Snapshot data are persisted values, not formulas linked to later inputs.

Independent fixtures contain Excel's cached component values and all 166
Diagnostics events. Regression tests reproduce the actual date/fund sequence
and every snapshot field, rather than merely looking up expected selections.
To compare all 16,978 formula rows directly against the original workbook:

```bash
python scripts/validate_workbook.py "/path/to/ECR vs ECR + Par Proximity Methodology (5).xlsx"
```

That optional verifier reads only Website Metrics and ECR Diagnostics.
The ordinary test suite needs no external workbook. It also tests threshold
boundaries, minimum hold, calendar reviews, ties, costs, execution timing,
missing data, no look-ahead, export round trips and configuration identity.

Returns in reports are decimal fractions. Excess vs BUFR means the difference
in **cumulative total returns**, not annual alpha. Sharpe is mean daily return /
sample daily standard deviation times sqrt(252), with zero risk-free return.
Annualized return uses elapsed calendar days; trades/year is separate from total
switches. A one-observation run has no annualized return/Sharpe estimate.

The first implementation focuses on correct AR scoring, event parity, configurable
backtesting and snapshots. Parameter-grid orchestration, parallel workers,
forward-trade attribution and expanded research charts are follow-up work.

## Jupyter research interface

Open `notebooks/ecr_backtest.ipynb` and run the cells in order. The configuration
cell uses the same tested Python classes as the CLI; no strategy logic lives in
notebook cells. Edit parameters, rerun the backtest cell, then inspect charts and
review decisions. Input data and scored panels are reused within a session.

```bash
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --prefix "$VIRTUAL_ENV" --name buffer_sma --display-name 'Python (buffer_sma)'
jupyter lab notebooks/ecr_backtest.ipynb
```

In the prepared cloud environment, use `/workspace/buffer_sma-env/bin/jupyter lab
notebooks/ecr_backtest.ipynb` from the repository. The `buffer_sma` kernel is
already installed in that virtual environment. The tracked notebook contains no
stored outputs; use its explicit export cell to save results and self-contained
interactive HTML. Exported directories are ignored by Git.

The formula remains the workbook's multiplicative ECR. There is no additive
33/33/33 model. Adjust `dbb_weight`, `reset_weight`, `buffer_exponent`,
`cap_exponent`, and `period_bonus` directly. Increasing a gate exponent applies a
stronger penalty when integrity is below 1. Increasing DBB alone changes the score
scale; the example DBB experiment changes DBB/reset to 1.8/0.9 to retain the intended
maximum of 3.0. This also reduces reset emphasis, so interpret it as a trade-off.

New controls (also accepted by CLI and JSON):

- `initial_selection`: `cohort` or `highest_score`. Cohort start dates remain the
  same. With next-close execution, highest-score initialization uses the previous
  observed session's scores; it fails if no prior session exists.
- `trigger`: `ecr`, `buffer_drop`, `ecr_or_buffer`, or `ecr_and_buffer`.
- `buffer_metric`: `buffer_integrity` (default), `dbb` or `cap_integrity`.
- `buffer_drop`: relative deterioration from the execution-date entry metric;
  default 0.20 means 20%, not 20 percentage points. The anchor resets on an actual
  switch, not on a no-op review. A zero entry metric has no relative-drop signal.

All trigger combinations respect the review calendar and minimum holding period.
Decision records include both component signals, the entry anchor and the measured
decline, even when the holding period prevents trading. The daily buffer chart
shows the holding at close; inspect the decision table for the outgoing fund.

The notebook compares four trigger experiments with identical cohort windows and then
repeats them with next-close execution and 5 bp per-switch costs. These overlapping
cohorts are not independent samples, and this comparison is not an out-of-sample
optimization. The notebook includes a visualization roadmap for holdings timelines,
rank heatmaps, gated component contributions and parameter-sensitivity surfaces.

## Conservative client research

Use [the conservative methods guide](documents/ucit-conservative.md) and
`notebooks/ecr_backtest.ipynb`. Standard multiplicative ECR selects the fund;
`--trigger metric_floor --buffer-metric buffer_integrity --metric-floor 0.8`
checks whether buffer integrity falls below 80% of ideal fresh score 1.00.
The preset `config/ucit-conservative.json` uses this trigger. The notebook compares
70%, 80% and 90% floors against the regular ECR-threshold baseline with identical
selection scoring and other assumptions. Three-component allocation controls
have been removed; this is not a weight-sweep experiment.

## Saved client presets and portfolio size

- **Super Buffer:** `python main.py --config config/super-buffer.json` — one fund,
  strict ECR < 1.5 trigger, highest-ECR selection.
- **UCITS Conservative:** `python main.py --config config/ucit-conservative.json
  --portfolio-size 3 --metric-floor 0.8` — N distinct funds, per-position buffer
  integrity trigger, highest-ECR replacement selection.

N is configurable per run/client, not a time-varying allocation rule. UCITS starts
with equal-weight top-N ECR holdings, replaces only breached positions, lets other
weights drift, and charges costs only on the positions traded. The preset's N=3
is an editable example. Daily position capital/weights are exported in a separate
holdings table. See the conservative methods guide for simultaneous triggers,
no-op selections, timing, insufficient-universe behavior and risk metrics.

The notebook loads a named client preset and compares N/floor combinations.
Running without `--config`, or using `config/ecr.json`, uses workbook-parity
research defaults (including `<=`), rather than either named client preset.

## Conservative scenario library and turnover controls

The app's preset selector includes 34 UCITS research variants for 3- and 5-fund
portfolios, plus the existing clients. Use **Turnover & weight controls** to set
review confirmation, a minimum score improvement, a replacement entry margin,
repurchase cooldown, replacement count, turnover budget and optional weight bands.
All default to preserving the original behavior. Weight bands trigger a reset
around 1/N at scheduled reviews, subject to the shared turnover budget; they are
not hard daily weight caps. Excel includes the executed-trade and weight-review
ledgers, including deferred actions.

Run all 35 comparisons (including the single-ticker reference) with:

```bash
python scripts/run_conservative_suite.py
```

See [conservative research](documents/conservative-research.md) for every preset,
exact accounting, CLI examples, quarterly-outcome scoring proposals and tradeoffs.
The supplied dataset still represents annual F-series outcome structures; a
quarterly review cadence does not change their outcome horizon.
