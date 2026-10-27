# ECR Backtest Studio

Run `python -m pip install -r requirements-app.txt`, then
`python -m streamlit run app.py` from the repository. The browser application
uses the same engine, inputs and report exporters as the CLI and notebook.

## Configure a run

1. Choose a client preset and give the run a name. Super Buffer starts with one
   fund and ECR < 1.5; UCITS Conservative starts with three funds and Buffer
   Integrity < 0.80. Both use highest ECR selection, monthly reviews, next-close
   execution and a 5 bps replacement cost.
2. Choose **Single ticker** or **Fund of funds**, with N from 2 to 12 for FoF.
   N stays fixed throughout a run. Initial capital is split equally; subsequent
   weights drift. Only breached holdings are reconsidered, with their proceeds
   remaining in the replacement sleeve.
3. Set the review cadence: monthly; quarterly in March, June, September and
   December; semiannually in June and December; or annually in December.
   Dates are the supplied holiday-adjusted roll dates, not month-end dates.
   Calendar anchors are shared by all launch cohorts, not measured from entry.
4. Rank by highest ECR, DBB Score, Buffer Integrity, or Cap Integrity. Ranking
   determines both initial purchases and subsequent replacement candidates.
   It is independent of the trigger. Ties retain incumbents, then break
   alphabetically. Unbreached holdings are excluded from replacement candidates.
5. Choose a trigger metric and a strictly-below threshold. ECR offers 1, 1.5,
   and 2. Component floors range from 0 to 1 in 0.05 input increments (typed
   decimals are also accepted); 0.80 means 80% of ideal fresh score 1.00.
6. Choose a date window and one or more launch months. Each launch month is
   a separate backtest, not another fund in the portfolio. Initial entry is
   the first observed session after that month's qualifying roll date. The
   actual entry date can therefore be later than the selected lower date bound.
7. Review the live summary and run. Execution, trading costs, minimum holding
   sessions, remaining outcome days and formula parameters are in expandable
   sections. Changing the preset restores those preset settings but retains
   the selected test window and launch months.

When a trigger fires, a trade still requires a different winning candidate.
A breached incumbent can remain if it still ranks highest. Holding all 12 funds
leaves no outside replacements. Candidates are not required to pass the trigger
floor before purchase; ranking and minimum remaining days govern eligibility.

The workbook multiplicative ECR formula is preserved. Choosing a component as
the selection score does not change the ECR calculation or its diagnostic
columns. Advanced formula edits can change its scale; a fixed 1.5 threshold may
then have different economic meaning. `Selection` and `Selection Score` in the
candidate audit, and `Selection Score at Signal` in diagnostics, record what
actually drove ranking. `Held Score` in the review log remains the held **ECR**;
`Best Score` is the chosen **selection** score. `Held Buffer Score` refers to the
chosen component trigger, including DBB or Cap Integrity.

## Results and exports

The results page belongs to the last completed run. Editing configuration does
not relabel or modify prior results. Rerun to apply edits; a failed run leaves the
previous results intact. Results are kept in the browser's Streamlit session;
download them before closing/restarting the session.

- **Excel:** Summary, Daily NAV, ECR Diagnostics, Review Decisions, Candidate Audit,
  Portfolio Holdings and Run Manifest. Entry/switch snapshots retain all ECR
  components. Turnover, downside deviation and Sortino are included in the summary.
- **Complete ZIP:** the Excel workbook, CSV tables, PNG performance/risk/trigger/
  allocation reports, interactive HTML charts, a shared local Plotly library,
  `config.json` and provenance manifest. Extract the whole archive for offline HTML.
- **Configuration JSON:** directly replay with
  `python main.py --config /path/to/config.json`. Save configuration is also
  available before running. Client presets in the repository are not overwritten.

The standard benchmarks are SPY, BUFR and Hold NAV. Single-ticker Hold NAV is the
launch month's original fund; FoF Hold NAV is the initial equal-weight basket.
Missing interior benchmark sessions appear as gaps and are disclosed in the
results. Invalid benchmark endpoints or missing held prices fail the run.

Costs are per replacement, applied once to the capital traded, with no initial
entry cost. Next-close execution retains the outgoing holding's return through
execution. A final-session signal cannot execute beyond the sample. The charts
show daily values; only scheduled review observations can trigger replacements.

## Further considerations

- **Robustness:** add saved scenario comparisons and a separate evaluation period
  before choosing a floor from the same sample used to measure performance.
- **Conservative objectives:** compare maximum drawdown, downside deviation,
  Sortino and turnover alongside returns. More holdings alone does not guarantee
  lower drawdown.
- **Eligibility:** the current universe is the 12 repository F-series funds.
  A client-specific eligible universe, liquidity screens and concentration
  limits would be useful next controls.
- **Portfolio policy:** equal initial capital and per-breach replacements are
  explicit assumptions. Weight caps, cash handling and forced full rebalancing
  would each require a distinct tested policy.
- **Reuse:** JSON exports are reproducible through the CLI. A saved-run library,
  JSON import and side-by-side UI comparison would make repeated client research
  easier. Current results are session-local rather than a shared run database.

## Validation

`python -m unittest discover -s tests` covers the engine's workbook regression,
selection changes for single and multi-fund runs, independent trigger thresholds,
calendar coverage, application preset changes/navigation, invalid configuration,
and ZIP/Excel contents. Install `requirements-app.txt` to include the application
and export tests; core tests remain usable with `requirements.txt` alone.
