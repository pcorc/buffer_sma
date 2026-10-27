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
execution and 5 bp per executed switch. These settings can be adjusted without
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
