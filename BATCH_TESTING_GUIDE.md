# Batch Testing Guide

## Quick Start

1. **Open `run_batch_tests.py`**
2. **Set `BATCH_NUMBER = 1`** (line 25)
3. **Run**: `python run_batch_tests.py`
4. **Wait** ~15 minutes
5. **Review Excel output**
6. **Repeat** with BATCH_NUMBER = 2, 3, 4, 5, 6

## Batch Descriptions

### Batch 1: Time-Based Rebalancing (~48 sims, ~15 min)
**Strategy Type**: Neutral (systematic)
**Tests**:
- Monthly rebalancing (6 months × 4 selections = 24)
- Quarterly rebalancing (4 months × 4 selections = 16)
- Semi-annual (2 months × 4 selections = 8)
- **Total**: 48 simulations

**Goal**: Find best systematic rebalancing approach

---

### Batch 2: Cap Utilization Thresholds (~48 sims, ~15 min)
**Strategy Type**: Bearish/Defensive
**Tests**:
- Switch at 25%, 50%, 75%, 90% cap used
- 4 thresholds × 4 months × 4 selections = 64 simulations

**Goal**: Find optimal timing to preserve protection

---

### Batch 3: Remaining Cap Thresholds (~48 sims, ~15 min)
**Strategy Type**: Bearish/Defensive
**Tests**:
- Switch when remaining cap falls to 75%, 50%, 25%, 10%
- 4 thresholds × 4 months × 4 selections = 64 simulations

**Goal**: Identify when to rotate for fresh cap

---

### Batch 4: Reference Asset Thresholds (~60 sims, ~18 min)
**Strategy Type**: Directional (Bull + Bear)
**Tests**:
- Bearish: Switch when SPY down -5%, -10%
- Bullish: Switch when SPY up +5%, +10%, +15%
- 5 thresholds × 4 months × 3 selections = 60 simulations

**Goal**: Market-timing strategy validation

---

### Batch 5: Downside Before Buffer (~36 sims, ~12 min)
**Strategy Type**: Defensive
**Tests**:
- Switch when 5% from buffer, 2% from buffer, or in buffer
- 3 thresholds × 4 months × 3 selections = 36 simulations

**Goal**: Buffer breach protection strategies

---

### Batch 6: Best-of-Batch Cross-Validation (~48 sims, ~15 min)
**Strategy Type**: Mixed (top performers)
**Tests**:
- Top 2-3 combos from each previous batch
- Tested across ALL 12 launch months

**Goal**: Comprehensive validation of winners

**⚠️ NOTE**: Customize this batch after reviewing Batches 1-5 results!

---

## After Each Batch

### Review These Excel Tabs:
1. **Optimal-Bull (6M)** - Top 10 for bull markets
2. **Optimal-Bear (6M)** - Top 10 for bear markets  
3. **Optimal-Neutral (6M)** - Top 10 for neutral markets

### Look For:
- **High excess vs BUFR** (+2% or more is excellent)
- **Sufficient observations** (at least 5-10)
- **Consistency** with strategy intent

---

## Analysis Workflow

### After Batch 1:
✅ Identify best time-based rebalancing frequency
✅ Note which selection algo works best for systematic strategies

### After Batch 2:
✅ Identify optimal cap utilization threshold for defensive positioning
✅ Compare to Batch 1 performance in bear markets

### After Batch 3:
✅ Identify optimal remaining cap threshold
✅ Compare to Batch 2 (which defensive trigger is better?)

### After Batch 4:
✅ Validate if market-timing strategies actually work
✅ Identify best bullish and bearish directional combos

### After Batch 5:
✅ Validate buffer-based defensive strategies
✅ Compare all defensive approaches (Batches 2, 3, 5)

### After Batch 6:
✅ Final validation of top performers across all months
✅ Select final recommended strategies for each regime

---

## Final Output

After all 6 batches, you'll have identified:

### For BULL Markets:
- Top 3 trigger/selection combinations
- Best launch months for each
- Expected excess returns

### For BEAR Markets:
- Top 3 defensive combinations
- Best protective strategies
- Downside mitigation effectiveness

### For NEUTRAL Markets:
- Top 3 robust combinations
- Consistent performers
- All-weather strategies

---

## Total Time Investment

- 6 batches × ~15 minutes = ~1.5 hours
- Plus analysis time between batches = ~2-3 hours total
- Result: High-confidence strategy recommendations

---

## Tips

1. **Run batches sequentially** - Learn from each before moving to next
2. **Take notes** - Document which combos look promising
3. **Customize Batch 6** - Use learnings from Batches 1-5
4. **Compare across batches** - Find patterns in what works
5. **Check strategy intent** - Bearish strategies should protect in bear markets

---

## File Outputs

Each batch creates:
```
output/backtest_results/batch_N/
    batch_N_[description]_YYYYMMDD_HHMMSS.xlsx
```

Contains 15+ tabs with comprehensive analysis.