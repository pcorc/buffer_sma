# Conservative ECR research

The canonical model is now called **ECR**. The original workbook's historical
sheet names are used only when importing/verifying the supplied attachment.

## Method 1: component emphasis, ECR trigger and ECR selection

Use `ScoreConfig(component_weights=(0.75,0.125,0.125))` in DBB, buffer, cap order.
Weights must be finite/nonnegative and are normalized to sum to one. `(33,33,33)`
or `(1,1,1)` exactly reproduce standard ECR.

To retain the multiplicative formula, normalized weights `(d,b,c)` map to:

```
a = 3*d*w_DBB
beta = 3*b*gamma_Buffer
gamma = 3*c*gamma_Cap
normalizer = (w_DBB + w_Reset) / (a + w_Reset)
ECR = normalizer*(a*DBB + w_Reset*Reset)*Buffer^beta*max(0,Cap)^gamma
      + period_bonus*Period
```

This is an explicit parameter-emphasis mapping, not a linear weighted average or
portfolio allocation. The normalizer preserves the intended maximum of 3 under
the default parameters and usual component bounds. Equal weights recover the
workbook formula exactly. Higher buffer/cap emphasis strengthens that gate's
penalty; increasing DBB emphasis also relaxes the other gates. This is not a
promise of defensive behavior. Hold review schedule, execution timing and costs
constant when comparing configurations.

Trigger with `ECR < 1.5`, then select the highest ECR computed using the same
allocation. `config/ucit-conservative.json` provides a runnable example.

## Method 2: fresh-component floor, standard ECR selection

Set `trigger='metric_floor'`, `buffer_metric='dbb'`, `'buffer_integrity'`, or
`'cap_integrity'`, and `metric_floor=0.80`. The reference is the **ideal fresh
component score 1.0**, not the observed purchase/reset value. Net metrics may be
below one immediately after a real reset. A threshold of 0.80 means 80% remains
(20% below ideal fresh); enter fractions, not 80. Strict `<` or inclusive `<=`
uses `comparator`. Raw signed downside-before-buffer is not the score: the DBB
score is `exp(5 * downside_before_buffer_net)`.

Replacement selection always uses the highest ECR, not the trigger component.
The supplied comparison grid uses equal/default ECR weights for method 2 so the
experiment isolates the trigger change. Minimum holding time and review cadence
still apply; selecting the same fund creates no trade. Decision exports show the
metric value, fresh reference and floor alongside the ECR score. Existing
entry-relative deterioration modes remain available and distinctly documented.

```bash
python main.py --config config/ucit-conservative.json
python main.py --trigger metric_floor --buffer-metric dbb --metric-floor 0.8 --comparator '<' --initial-selection highest_score --execution-lag 1 --cost-bps 5
python main.py --trigger metric_floor --buffer-metric buffer_integrity --metric-floor 0.9
python main.py --trigger metric_floor --buffer-metric cap_integrity --metric-floor 0.7
```

`notebooks/ecr_backtest.ipynb` includes the 13-run controlled comparison: equal,
DBB-heavy, buffer-heavy and cap-heavy weights; three fresh floors (70/80/90%) for
each metric. Cohorts share history and are not independent trials. This evaluates
strategy hypotheses, not UCITS regulatory eligibility or client suitability.
