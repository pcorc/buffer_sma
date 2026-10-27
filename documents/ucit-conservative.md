# Conservative ECR research: buffer integrity trigger

Use the standard multiplicative **ECR** for initial and replacement selection.
The conservative experiment changes only the exit trigger: review the held fund's
buffer integrity and rebalance when it is below a configured fraction of its
**ideal fresh score of 1.00**.

The formula remains:

```
ECR = (1.5*DBB + 1.2*Reset) * max(0,Buffer)^0.5 * max(0,Cap)^0.3
      + 0.3*Period
```

The three-component allocation controls and their normalization have been
removed. There is no 33/33/33 or 75/12.5/12.5 score-weight experiment.

`metric_floor=0.80` with `comparator='<'` triggers below 80% of ideal fresh buffer
integrity. Equality does not trigger. This is not a decline from the purchase-date
value. For example, buying at 0.85 does not move the floor: it stays at 0.80.

At each permitted review, if the held fund breaches the floor, select the eligible
fund with the highest ECR. Do not select by buffer integrity alone. If that fund
is already held, no switch occurs. Minimum holding time, execution lag, costs,
and candidate eligibility retain their documented behavior.

```bash
python main.py --config config/ucit-conservative.json
python main.py --trigger metric_floor --buffer-metric buffer_integrity --metric-floor 0.9 --comparator '<' --initial-selection highest_score --execution-lag 1 --cost-bps 5
```

The preset uses monthly review, a 0.80 floor, highest-ECR entry, next-close
execution, zero trading cost and no minimum holding delay. These settings can be adjusted without
changing the ECR selection formula.

`notebooks/ecr_backtest.ipynb` compares four runs: the ECR-below-1.5 trigger as a
baseline, and buffer-integrity floors of 0.70, 0.80 and 0.90. All runs use the same
ECR parameters and cohort windows. It also repeats the comparison with next-close
execution and 5 bp costs. Decision records expose the metric value, fresh
reference and threshold. Daily plots use the actual constant floor for this mode.

The generic research engine retains older entry-relative modes for compatibility;
they are not part of this conservative comparison. Overlapping cohorts are not
independent trials. Lower turnover or a buffer-based exit does not guarantee lower
realized drawdown. This is strategy research, not UCITS eligibility analysis.

## Two clients on one trigger/selection chassis

| Preset | Number of funds | Trigger | Selection |
|---|---:|---|---|
| `config/super-buffer.json` | 1 | Held ECR **< 1.5** | Highest eligible ECR |
| `config/ucit-conservative.json` | Configurable N (example: 3) | Each held fund's buffer integrity **< floor** | Highest eligible ECR |

`config/ecr.json` remains the workbook-parity research configuration, including
its historical `<=` comparison. It is not the Super Buffer client preset. Both
clients expose trigger and selection independently and use the same scoring,
trigger evaluation and deterministic ranking helpers. The tested one-fund engine
remains the execution path for N=1; N>1 uses funded sleeves with the same rules.

### Portfolio mechanics

N is a positive integer from 1 to 12, fixed during one run. The engine fails if N
exceeds the eligible universe at inception; it never duplicates tickers to fill
positions. Multi-fund portfolios require highest-score initialization.

Initial capital is split equally among the top N eligible ECR funds. Thereafter,
weights drift with returns: there is no daily or whole-portfolio equal-weight
rebalance. On a review date, each position independently evaluates the configured
trigger and its own minimum holding age. Only triggered positions are reconsidered.
The others retain both their fund and their capital.

When several positions trigger together, candidates are ranked jointly, excluding
unaffected holdings, to prevent duplicate purchases. Triggered incumbents that
remain among the highest eligible ECR choices stay in their existing positions;
these are recorded as no-op reviews. If no distinct alternative is available,
no substitute or leverage is invented. Rank ties favor incumbents and then ticker
alphabetical order. Buffer integrity does not replace ECR as the ranking rule.

Each replacement inherits its outgoing position's proceeds less trading cost.
Costs are assessed on that traded capital only. With next-close execution, all
outgoing positions earn their returns through the execution close before the
pending replacements are applied. Signals, snapshots and decisions identify the
position so simultaneous trades remain distinguishable.

`holdings.csv` / the `Portfolio Holdings` Excel sheet records each day's actual
fund, position, capital, weight and score. Portfolio NAV equals the sum of position
capital. Daily aggregate score columns are capital-weighted displays, not trigger
inputs. The notebook charts per-position integrity and a stacked holdings-weight
timeline. For multi-fund runs, Hold NAV is the initial equal-weight basket held
without trading; the existing one-fund Hold NAV remains its named cohort fund.

`One-way Turnover` sums the fraction of total portfolio capital replaced at each
execution (a 25%-weight position is 0.25, not 1.0). Switch counts count individual
fund replacements. Compare turnover and downside deviation/Sortino alongside
returns and drawdowns when changing N.

```bash
python main.py --config config/super-buffer.json
python main.py --config config/ucit-conservative.json --portfolio-size 5 --metric-floor 0.85
```

The notebook's `client_experiments` grid compares the saved Super Buffer rule with
UCITS sizes 1/3/5 and floors 0.70/0.80/0.90, holding other run assumptions constant.
These are sensitivity examples, not recommended portfolio sizes or optimized
thresholds. Use a reserved evaluation period before choosing client settings.
