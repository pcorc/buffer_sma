# Conservative ECR research and turnover controls

The annual workbook model and Super Buffer client remain the baseline. New
controls default off. All 35 named experiments keep **highest ECR** selection;
there is no additive 33/33/33 model. N denotes funds, not underlying stocks.

## Quarterly outcome structures

Outcome duration and review cadence are separate decisions. Quarterly reviews
of a 12-month fund do not create a quarterly-outcome strategy. The supplied
observations and engine universe remain the twelve annual F-series funds.
Real quarterly evaluation requires quarterly NAVs, returns since their own reset,
original/remaining cap and buffer, outcome days and the corresponding roll and
launch calendar, plus an adapter for that universe. Changing these parameters
on the current dataset is only a scoring sensitivity, not a quarterly backtest.

| Parameter | Annual baseline | Quarterly hypothesis |
| --- | --- | --- |
| Period-position center | 180 days | 45 days |
| Period-position width | 75 days | 18.75 days |
| Downside reset sigma | 0.07 | Compare 0.07 and 0.035 |
| Upside reset sigma | 0.05 | Compare 0.05 and 0.025 |
| DBB/reset coefficients | 1.5 / 1.2 | Initially unchanged |
| Buffer/cap gate exponents | 0.5 / 0.3 | Initially unchanged |
| Period-position bonus | 0.3 | Initially unchanged |

The period proposal preserves the curve in relative cycle time under a 360/90-day
convention: midpoint peak and equal proportional width. For actual varying term
lengths, a future adapter should calculate relative remaining time per fund.
Half-sized sigmas follow sqrt(90/360); they assume return dispersion scales with
square-root time. Keeping sigmas unchanged instead preserves the absolute-return
meaning of “near reset.” Compare these interpretations on actual quarterly data,
and consider scaling reset distance by the original cap/buffer width if those
structures have materially different economics. Do not select sigma solely by
whether the historical ranking looks attractive.

`backtesting.scenarios.quarterly_score_hypotheses()` exposes both parameter
proposals without changing the annual presets or running them as quarterly data.
The ECR formula stays multiplicative. The integrity ratios are dimensionless,
but equal ratios can still represent different absolute protection in shorter
structures. I would first measure those differences, then test DBB decay,
stronger buffer gating and lower cap emphasis separately. A monthly or more
frequent review may make more sense than waiting a whole quarter to observe
three-month funds; neither holding locks nor cooldowns should consume most of
the outcome period without deliberate testing.

## Experiments to run

Use the GUI's **Start from a client preset** dropdown; type `N=3`, `N=5`, `DBB`,
or `combined` to find a scenario. Their controls are editable in **Turnover &
weight controls**. Execution, fees and minimum holding sessions remain under
**Execution & eligibility**. A new run name identifies your custom variant.

The library includes one Super Buffer reference and, for each N in {3, 5}:

| Family | Settings |
| --- | --- |
| ECR reference | Sell below ECR 1.5 |
| Buffer floors | Sell below 0.70, 0.80 or 0.90 |
| DBB floors | Sell below 0.80, 0.90 or 0.95 |
| Improvement hurdle | Buffer 0.80; new ECR at least 0.15 higher |
| Confirmation | Buffer 0.80; two consecutive breached reviews |
| Entry/exit separation | Exit below Buffer 0.80; replacement must be at least 0.90 |
| Hold/cooldown | Buffer 0.80; hold 42 sessions, wait 21 after selling before repurchase |
| Replacement budget | Buffer 0.80; at most one replacement; turnover budget 40% for N=3, 25% for N=5 |
| Weight bands | Buffer 0.80; allow ±5 percentage points around 1/N |
| Combined controls | Buffer 0.80 or DBB 0.90; two confirmations, ECR gain ≥0.10, entry margin 0.05, hold/cooldown 21 sessions, one replacement, same turnover budget, ±5pp band |
| Defensive ECR | Buffer 0.80; DBB/reset 1.8/0.9, buffer/cap exponents 0.75/0.20, period bonus 0.30 |
| Slower reviews | Buffer 0.80; quarterly reviews |

The defensive coefficient variant retains the intended maximum 3.0. It tests
stronger buffer gating and less cap gating; it is a hypothesis, not a statement
that realized losses must be lower. DBB floors use the DBB **score**, not a raw
percentage loss. With DBB = exp(5 × input), scores 0.95/0.90/0.80 correspond to
inputs of approximately -1.03%/-2.11%/-4.46%. A higher floor responds earlier.

Run the entire suite with the same next-close execution and 5 bps fees:

```bash
python scripts/run_conservative_suite.py
```

For a smaller run or a separate evaluation period:

```bash
python scripts/run_conservative_suite.py --filter 'N=3' --cohorts JAN
python scripts/run_conservative_suite.py --start-date 2024-01-01 --end-date 2026-03-27
```

The new output folder contains `comparison.xlsx`, cohort and scenario CSVs,
interactive risk/return and cohort charts when Plotly is installed, source hashes,
and numbered JSON configs. JSON files retain their order in `experiments.json`;
replay any one with `python main.py --config output/.../configs/01.json` to export
a detailed single-run workbook and graphs. All scenarios use shared cohort
windows; summaries across cohorts are overlapping launch tests, not independent
statistical trials. The suite caches just two scoring panels.

## Exact control semantics

- **Minimum holding sessions:** starts at actual entry/replacement. It delays
  replacement eligibility, not weight-only resizing.
- **Confirmation reviews:** require consecutive raw breaches on scheduled
  reviews; recovery clears the count. At monthly cadence, two confirmations
  can delay an exit by a month. Actual replacement resets the count.
- **Minimum score improvement:** measured in the selected ranking's points;
  zero still preserves ties. For the library this is ECR points.
- **Entry margin:** positive values require replacement component ≥ exit floor
  + margin. Zero disables the filter, preserving the old behavior. It applies
  to replacements, not initial top-N selection. Candidates failing this screen
  cannot be rescued by a high ECR alone.
- **Repurchase cooldown:** measured in observed trading sessions from actual
  sale. It is ticker-specific, not a global pause. It expires at the configured
  session count; no forced exit is introduced.
- **Replacement count / turnover budget:** zero count means unlimited; budget
  is a fraction of whole portfolio NAV. Whole-sleeve replacement is never split
  to fit a budget. With replacement controls active, the worst triggered metric
  is considered first, then ticker for ties; already-held funds cannot be
  transferred between sleeves. Different choices can affect outcomes, not just
  cost. Review logs show blocked/rejected actions. Insufficient cash-free
  replacement opportunities leave the old holding in place.
- **Weight band:** optional symmetric absolute percentage-point band around
  equal weight 1/N. At a review, a strict band breach requests a full equal-weight
  reset, at the same/next close according to execution settings. Replacements
  execute first. Signal-time weight intent is frozen until execution, so a
  next-close reset can proceed even if weights have moved back inside the band.
  Weight-only trades do not reset entry metrics or holding clocks.
- **Weight-band limits:** these are review-date triggers, not hard concentration
  caps. Weights can exceed them between reviews or when a budget defers a reset.
  Tighter bands can **increase** turnover compared with allowing weights to
  drift forever; their benefit is fewer weight adjustments than unconditional
  equal-weight rebalancing. Component replacement controls address a separate
  source of turnover.
- **Accounting:** a replacement consumes sleeve capital / total pre-execution
  NAV. A weight reset consumes half the absolute pre-fee capital transfers /
  the same pre-execution portfolio NAV. The combined amount cannot exceed the
  per-review budget; replacements take priority. Fees equal turnover capital ×
  cost_bps / 10,000. Equal-weight targets divide remaining capital after fees.
  Initial allocations remain uncharged. Next-close prices are used to recheck
  the budget; a scheduled replacement/reset may be deferred if it no longer fits.
  No partial fills, automatic cash allocation or daily hard caps are implied.

Excel adds **Executed Trades** and **Weight Reviews**. The summary separates
replacement count from weight resets and reports total one-way turnover and
fees. The comparison adds annualized turnover. Review decisions retain raw
breaches, confirmation counts and control reasons; execution-budget deferrals
are in the trade log. No-op weight requests and unexecuted end-of-sample requests
remain auditable. Existing workbook snapshots still record entry/replacement
components; weight-only events are in their own ledger.

## How to choose a conservative variant

Compare worst drawdown, downside deviation, Sortino, turnover, cost sensitivity,
and periods spent below protection thresholds. Low turnover does not itself
mean conservative: confirmations, cooldowns, entry screens and tight budgets
can leave a deteriorated holding in place longer. Hold out a later period before
choosing parameters, and examine multiple market regimes rather than only a
full-sample ranking. Further experiments worth designing separately include
emergency protection exits that override ordinary limits, cap/buffer-width
normalization, overlapping exposure limits, and cash/short-duration alternatives
when no fund meets an entry floor. Each needs an explicit policy and suitable
price data; none is silently assumed by this implementation.
