# First ECR research comparison

These are exploratory historical comparisons using the supplied Website Metrics
observations through 2026-03-27. All twelve cohort windows, monthly reviews,
threshold <= 1.5, zero minimum hold, and other model parameters are held constant.
The buffer trigger is a 20% relative decline in Buffer Integrity from entry.

The original multiplicative formula is retained. There is no equal-weight linear
score. Coefficients and gate exponents have different mathematical roles.

| Experiment | Mean cumulative excess vs BUFR | Worst cohort drawdown | Mean switches over sample |
|---|---:|---:|---:|
| Workbook, cohort entry | +6.23 percentage points | -12.86% | 12.83 |
| Workbook, highest-score entry | +6.23 pp | -12.86% | 12.83 |
| DBB/reset coefficients 1.8/0.9, highest entry | -0.19 pp | -14.66% | 10.08 |
| Buffer gate exponent 1.0, highest entry | -3.02 pp | -16.11% | 14.83 |
| Cap gate exponent 0.8, highest entry | +4.29 pp | -12.86% | 16.42 |
| Buffer decline only, highest entry | +2.40 pp | -13.31% | 2.17 |
| ECR OR buffer decline, highest entry | +1.66 pp | -12.86% | 16.08 |
| ECR AND buffer decline, highest entry | +0.55 pp | -16.44% | 1.50 |

The standard same-close initial selections already match the highest-scoring
funds at these start dates; changing that rule alone has no effect in this sample.
Buffer-only substantially reduces turnover, with lower excess return. Stronger
buffer gating and DBB emphasis did not improve realized downside protection here.
A single common threshold can produce different trigger frequencies across scores.

With next-close execution and 5 bp per executed switch, the respective mean
cumulative excess returns were +6.50, +6.44, +0.69, -2.59, +4.40, +2.13, +2.01,
and -0.75 percentage points. This changes timing and costs together; it does not
isolate a cost effect. The twelve overlapping cohorts are not independent trials.
These are not annual alpha estimates or untouched out-of-sample results.

Reproduce both tables by running `notebooks/ecr_backtest.ipynb` top to bottom.
The session builds only four distinct scored panels for the sixteen runs and
reuses them across triggers and execution settings. Inputs are loaded once;
recreate the session after changing source files. No parallel workers are needed
for these small experiments.

Implemented views: linked NAV/drawdown/buffer charts with entry/switch markers,
cohort excess heatmap, return/drawdown/turnover scatter, and diagnostic/decision
and candidate tables. Next useful views are a holdings timeline, fund-rank
heatmap, gated component-contribution chart, and threshold/exponent sensitivity
surfaces evaluated on a separately reserved period.

Validation: 27 tests passed; the notebook executed fully under its registered
Python kernel. The standard configuration still reproduces all 166 workbook
events. Missing benchmark sessions remain explicitly missing in daily output.
