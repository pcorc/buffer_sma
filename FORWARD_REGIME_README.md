# Forward Regime Analysis & Train/Test Validation

## Overview

This framework now includes **forward-looking regime analysis** to identify which trigger/selection strategies perform best when entering different future market conditions (bull/bear/neutral).

## Key Concepts

### Forward Regime Classification
- **Traditional regime**: Uses backward-looking returns (T-6 to T) to classify current market
- **Forward regime**: Uses forward-looking returns (T to T+3, T to T+6) to classify upcoming market
- **Purpose**: Answer "If I choose strategy X today, and a bull market is coming, what's my expected return?"

### Dual Horizon Analysis
- **3-month forward**: Tactical horizon for shorter-term decisions
- **6-month forward**: Strategic horizon aligned with regime classification period

### Performance Attribution
- Measures strategy performance from entry date through the forward period
- Attributes returns to the future regime that actually materialized
- Enables identification of optimal strategies for each market outlook

## Running the Analysis

### Option 1: Basic Forward Regime Analysis

```bash
python main.py
```

This runs the full backtest suite with forward regime analysis and exports:
- Summary of all strategy combinations
- Future regime performance attribution
- Optimal strategies for bull/bear/neutral outlooks
- Strategy intent validation
- Robust strategy rankings

**Output**: `mainpy_consolidated_YYYYMMDD_HHMMSS.xlsx`

### Option 2: Train/Test Validation

```bash
python main_with_train_test.py
```

This extends the analysis with train/test split validation:

**Training Period**: 2020-01-01 through 2023-12-31 (4 years)
**Test Period**: 2024-01-01 through 2024-12-31+ (1+ year)

#### What It Does:
1. Runs backtests on full dataset
2. Identifies "optimal" strategies in training period
3. Validates whether those strategies actually perform well in test period
4. Reports stability metrics and recommendations

**Output**: `train_test_analysis_YYYYMMDD_HHMMSS.xlsx`

## Excel Output Structure

### From `main.py`:

**Tab 1: Summary** - All backtest iterations with comprehensive metrics

**Tab 2: Future Regime Analysis** - Full forward performance data
- Strategy entry dates
- Future regime classifications (3M and 6M)
- Forward returns over each horizon
- Excess returns vs SPY and BUFR

**Tab 3-5: Optimal Strategies by Future Regime**
- **Optimal for Bull (6M)**: Top 10 strategies for upcoming bull markets
- **Optimal for Bear (6M)**: Top 10 defensive strategies
- **Optimal for Neutral (6M)**: Top 10 for range-bound markets

**Tab 6: Intent vs Future** - Validates strategy design
- Do "bullish" strategies actually capture bull market upside?
- Do "bearish" strategies provide bear market protection?

**Tab 7: Robust Strategies** - Consistent performers across all regimes
- Strategies that work well in bull, bear, AND neutral markets
- Sorted by consistency score

**Tab 8: Ranked vs BUFR** - Full strategy rankings by excess returns

### From `main_with_train_test.py`:

**Additional Tabs:**

**Train-Optimal Bull/Bear/Neutral** - Strategies identified as optimal in training

**Test-Optimal Bull/Bear/Neutral** - Strategies that performed best in test

**Train vs Test Comparison** - Side-by-side performance
- Training rank vs test rank
- Performance delta
- Rank change

**Consistent Winners** - Strategies that were top performers in BOTH periods
- Highest validation confidence
- Recommended for actual implementation

## Understanding the Results

### Key Metrics

**Forward Return**: Strategy return from entry through T+3 or T+6

**Excess vs BUFR**: How much the strategy outperforms the benchmark fund-of-funds

**Rank**: Position within future regime (1 = best)

**Num Observations**: How many times this combination was tested

### Strategy Intent Validation

The framework tests whether strategies perform as designed:

- **Bullish strategies** (e.g., `ref_asset_return_threshold` with positive threshold)
  - *Expected*: High returns in future bull markets
  - *Check*: Does it actually capture upside?

- **Bearish strategies** (e.g., `cap_utilization_threshold`, `remaining_cap_threshold`)
  - *Expected*: Protection in future bear markets
  - *Check*: Does it actually limit downside?

- **Neutral strategies** (e.g., time-based rebalancing)
  - *Expected*: Consistent performance across regimes
  - *Check*: Minimal variance between bull/bear/neutral

### Train/Test Stability Metrics

**Stayed in Top N**: % of training top strategies that remained top performers in test

**Improved in Test**: % of strategies that performed better out-of-sample

**Avg Rank Change**: Average movement in ranking from train to test

**Stability Score**: Overall measure of consistency (higher = more reliable)

## Interpretation Guide

### High Confidence Signals

✅ **Strategy appears in top 10 for a regime in BOTH train and test**
- Example: "MAR + cap_utilization_threshold + select_most_recent_launch" ranks #3 in training bull and #2 in test bull
- **Action**: High confidence this combination captures bull market upside

✅ **Robust strategy with positive excess in all three regimes**
- Performs well regardless of market conditions
- **Action**: Consider for core allocation

✅ **Strategy intent matches actual performance**
- Bearish strategy shows high rank in bear market, low in bull
- **Action**: Use tactically based on market outlook

### Caution Flags

⚠️ **Strategy ranks high in training but drops significantly in test**
- Example: Training rank #2, test rank #15
- **Interpretation**: May have been overfit or regime-specific
- **Action**: Avoid or reduce allocation

⚠️ **Low stability score (<50%) for a regime**
- Top strategies in training don't persist in test
- **Interpretation**: Regime may be harder to capitalize on
- **Action**: More diversification needed

⚠️ **Strategy intent misalignment**
- "Bullish" strategy performs poorly in bull markets
- **Interpretation**: Design flaw or implementation issue
- **Action**: Re-examine strategy logic

## Practical Use Cases

### Use Case 1: Building a Portfolio
1. Run `main_with_train_test.py`
2. Open "Consistent Winners" tab
3. Select top 3-5 strategies with:
   - Top 10 rank in both train and test
   - Positive excess vs BUFR in both periods
   - Strategy intent matching your outlook
4. Allocate based on confidence levels

### Use Case 2: Tactical Regime Shifts
1. Identify current market regime (bull/bear/neutral)
2. Look up top strategies for that future regime
3. If confident in regime continuation, rotate into those combinations
4. Monitor stability score - if low, use more diversification

### Use Case 3: Risk Management
1. Review "Intent vs Future" tab
2. Identify bearish strategies with proven bear market protection
3. Hold as hedge allocation
4. Increase allocation if leading indicators suggest bear market ahead

### Use Case 4: Strategy Development
1. Examine which selection algorithms work best with which triggers
2. Look for patterns in optimal strategies
3. Test new combinations based on insights
4. Validate with train/test split before implementation

## Configuration

### Modifying Train/Test Split

Edit `main_with_train_test.py`:

```python
# Current: Train through 2023, test 2024+
TRAIN_END_DATE = '2023-12-31'
TEST_START_DATE = '2024-01-01'

# Example: Train through 2022, test 2023+
TRAIN_END_DATE = '2022-12-31'
TEST_START_DATE = '2023-01-01'
```

### Changing Forward Windows

Edit function calls in `main.py`:

```python
# Current: 63 and 126 trading days (3M and 6M)
df_forward_regimes = classify_forward_regimes(
    df_spy_for_regime,
    window_3m_days=63,   # Modify this
    window_6m_days=126,  # Modify this
    ...
)
```

### Adjusting Top N Strategies

```python
# Current: Top 10
optimal_6m = summarize_optimal_strategies(
    future_regime_df, 
    horizon='6M', 
    top_n=10  # Change to 5, 15, 20, etc.
)
```

## Technical Notes

### Data Requirements
- Minimum 2 years of data recommended for meaningful train/test split
- Forward regime classification requires at least 6 months of future data
- Strategies near end of dataset will have `unknown` future regime

### Performance Measurement
- All NAVs start at 100 on entry date
- Forward returns calculated as (NAV_at_T+N / NAV_at_entry) - 1
- Benchmarks (SPY, BUFR) measured over same forward period
- Excess returns = Strategy forward return - Benchmark forward return

### Known Limitations
- Last 6 months of data cannot have forward regime classification
- Strategies must have minimum observations in both train and test for comparison
- Short test periods (< 1 year) may not capture full regime cycles

## Next Steps

After reviewing results:

1. **Identify high-conviction strategies**: Top 5 from "Consistent Winners"

2. **Validate assumptions**: Check "Intent vs Future" matches expectations

3. **Build allocation**:
   - 40-50% to robust strategies (work in all regimes)
   - 30-40% to regime-specific strategies (rotate based on outlook)
   - 10-20% to experimental/test strategies

4. **Monitor and rebalance**:
   - Re-run analysis quarterly with new data
   - Update train/test split as dataset grows
   - Retire strategies that lose consistency

5. **Iterate**:
   - Test new trigger/selection combinations
   - Refine parameters based on findings
   - Document learnings

## Support

For questions or issues:
1. Check console output for detailed error messages
2. Verify input data format matches expected structure
3. Ensure all dependencies are installed
4. Review Excel output tabs systematically

---

**Remember**: Past performance does not guarantee future results. Use this analysis as one input in your decision-making process, combined with fundamental analysis, risk management, and current market conditions.