"""
Selection functions for determining WHAT fund to switch to.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from backtesting.regime_adaptive import select_ecr_v2_regime_adaptive_normalized
from config import settings


class RebalanceScoringTracker:
    """
    Tracks detailed scoring information at each rebalance event.
    """

    def __init__(self):
        self.rebalance_records = []

    def record_rebalance(self, date, month, weight_config, snapshot_scores, selected_fund):
        """
        Record scoring details for one rebalance event.

        Parameters:
            date: Rebalance date
            month: Launch month (e.g., 'JAN')
            weight_config: Tuple like ('111', 'Equal weights')
            snapshot_scores: List of dicts with fund scores
            selected_fund: The fund that was selected
        """
        weight_code, weight_desc = weight_config

        for fund_score in snapshot_scores:
            self.rebalance_records.append({
                'Date': date,
                'Month': month,
                'Weight_Config': f"{weight_code[0]},{weight_code[1]},{weight_code[2]}",
                'Weight_Description': weight_desc,
                'Fund': fund_score['fund'],
                'DBB_Score': fund_score.get('dbb_score', np.nan),
                'Buffer_Integrity': fund_score.get('buffer_integrity', np.nan),
                'Cap_Integrity': fund_score.get('cap_integrity', np.nan),
                'Time_Scaling': fund_score.get('time_scaling', np.nan),
                'Composite_Score': fund_score.get('composite_score', np.nan),
                'Selected': fund_score['fund'] == selected_fund,
                # Batch 11: Par Proximity fields (NaN for v1 runs, populated for par_prox runs)
                'Par_Proximity': fund_score.get('par_proximity', np.nan),
                'Time_Elapsed': fund_score.get('time_elapsed', np.nan),
                'Par_Bonus': fund_score.get('par_bonus', np.nan),
                # Raw metrics
                'Cap_Utilization': fund_score.get('cap_utilization', np.nan),
                'Remaining_Cap_Pct': fund_score.get('remaining_cap_pct', np.nan),
                'DBB_Pct': fund_score.get('dbb_pct', np.nan),
                'Remaining_Buffer': fund_score.get('remaining_buffer', np.nan),
                'Original_Buffer': fund_score.get('original_buffer', np.nan),
                'Remaining_Days': fund_score.get('remaining_days', np.nan),
                'Total_Days': fund_score.get('total_days', np.nan),
            })

    def export_to_excel(self, output_dir, batch_number):
        """Export all rebalance records to Excel with multiple views."""
        if not self.rebalance_records:
            print("⚠️  No rebalance records to export")
            return

        df = pd.DataFrame(self.rebalance_records)

        output_path = Path(output_dir) / f'batch_{batch_number}_rebalance_scoring.xlsx'

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: All Records (master data)
            df.to_excel(writer, sheet_name='All Rebalances', index=False)

            # Sheet 2: Selected Funds Only
            df_selected = df[df['Selected'] == True].copy()
            df_selected.to_excel(writer, sheet_name='Selected Funds', index=False)

            # Sheet 3: Pivot - Composite Scores by Date/Weight
            pivot_composite = df.pivot_table(
                index=['Date', 'Fund'],
                columns='Weight_Config',
                values='Composite_Score',
                aggfunc='first'
            )
            pivot_composite.to_excel(writer, sheet_name='Scores by Weight')

            # Sheet 4: Summary Stats by Weight Config
            summary = df.groupby(['Weight_Config', 'Weight_Description']).agg({
                'Composite_Score': ['mean', 'std', 'min', 'max'],
                'Date': 'count'
            }).round(4)
            summary.columns = ['Avg_Score', 'StdDev', 'Min_Score', 'Max_Score', 'N_Rebalances']
            summary.to_excel(writer, sheet_name='Summary by Weight')

            # Sheet 5: Component Score Correlations
            component_cols = ['DBB_Score', 'Buffer_Integrity', 'Cap_Integrity', 'Time_Scaling']
            if all(col in df.columns for col in component_cols):
                corr = df[component_cols + ['Composite_Score']].corr().round(3)
                corr.to_excel(writer, sheet_name='Component Correlations')


def calculate_ecr_scores_with_tracking(df_snapshot, current_date, series, w_dbb, w_buffer, w_cap):
    """
    Calculate ECR scores and return both the selected fund AND detailed scoring.

    This is a modified version of _select_highest_new_ecr_core that returns
    full scoring details for tracking.

    Returns:
        selected_fund: The ticker of the selected fund
        snapshot_scores: List of dicts with all fund scores
    """
    available = df_snapshot[
        (df_snapshot['Date'] == current_date) &
        (df_snapshot['Series'] == series)
        ].copy()

    if available.empty:
        return None, []

    snapshot_scores = []

    for _, row in available.iterrows():
        # Raw metrics
        cap_util = row.get('Cap_Utilization', np.nan)
        remaining_cap_pct = row.get('Remaining_Cap_Pct', np.nan)
        dbb_pct = row.get('Downside_Before_Buffer_Pct', np.nan)
        remaining_buffer = row.get('Remaining_Buffer', np.nan)
        original_buffer = row.get('Original_Buffer', np.nan)
        remaining_days = row.get('Remaining_Outcome_Days', np.nan)
        total_days = row.get('Total_Outcome_Days', 365)

        # Component 1: DBB Score (exponential penalty)
        if pd.notna(dbb_pct):
            dbb_score = np.exp(dbb_pct * 5)  # Exponential penalty for buffer proximity
        else:
            dbb_score = 0

        # Component 2: Buffer Integrity (incorporates distance to buffer)
        if pd.notna(remaining_buffer) and pd.notna(original_buffer) and pd.notna(dbb_pct):
            if original_buffer > 0:
                buffer_integrity = max(0, (remaining_buffer + dbb_pct)) / original_buffer
            else:
                buffer_integrity = 0
        else:
            buffer_integrity = 0

        # Component 3: Cap Integrity (capped at 1.0)
        if pd.notna(cap_util):
            cap_integrity = min(cap_util, 1.0)
        else:
            cap_integrity = 0

        # Component 4: Time Scaling (unchanged)
        if pd.notna(remaining_days) and remaining_days > 0:
            time_scaling = 1 - np.log(remaining_days / 365)
        else:
            time_scaling = 0

        # Composite Score
        composite = (
                w_dbb * dbb_score +
                w_buffer * buffer_integrity +
                w_cap * cap_integrity +
                time_scaling
        )

        snapshot_scores.append({
            'fund': row['Ticker'],
            'dbb_score': dbb_score,
            'buffer_integrity': buffer_integrity,
            'cap_integrity': cap_integrity,
            'time_scaling': time_scaling,
            'composite_score': composite,
            # Raw metrics for reference
            'cap_utilization': cap_util,
            'remaining_cap_pct': remaining_cap_pct,
            'dbb_pct': dbb_pct,
            'remaining_buffer': remaining_buffer,
            'original_buffer': original_buffer,
            'remaining_days': remaining_days,
            'total_days': total_days,
        })

    # Select fund with highest composite score
    if snapshot_scores:
        selected = max(snapshot_scores, key=lambda x: x['composite_score'])
        selected_fund = selected['fund']
    else:
        selected_fund = None

    return selected_fund, snapshot_scores


def select_remaining_cap(df_universe, current_date, series='F', df_regimes=None):
    """
    Legacy alias for select_remaining_cap_highest.
    Selects fund with highest remaining cap (most upside potential).
    """
    return select_remaining_cap_highest(df_universe, current_date, series)


def select_cap_utilization(df_universe, current_date, series='F', df_regimes=None):
    """
    Legacy alias for select_cap_utilization_lowest.
    Selects fund with lowest cap utilization (most cap remaining).
    """
    return select_cap_utilization_lowest(df_universe, current_date, series)



def select_most_recent_launch(df_universe, current_date, series='F', df_regimes=None, **kwargs):
    """
    Select the most recently launched fund.

    This is the baseline strategy for comparison.

    Parameters
    ----------
    df_universe : DataFrame
        Available funds at current date
    current_date : datetime
        Current backtest date
    series : str
        Fund series (default 'F')
    df_regimes : DataFrame, optional
        Regime data (not used, for API compatibility)
    **kwargs : dict
        Additional arguments (ignored, for API compatibility)

    Returns
    -------
    str or None
        Fund identifier with most recent roll date
    """
    if df_universe.empty:
        return None

    df_series = df_universe[df_universe['Fund'].str.startswith(series)].copy()

    if df_series.empty:
        return None

    # Get most recent launch
    if 'Roll_Date' in df_series.columns:
        most_recent = df_series.loc[df_series['Roll_Date'].idxmax()]
    else:
        most_recent = df_series.iloc[0]

    return most_recent['Fund']

def select_remaining_cap_highest(df_universe, current_date, series='F', df_regimes=None):
    """
    Select the fund with the HIGHEST remaining cap (bullish).

    Seeks maximum upside potential by choosing funds with the most cap remaining.
    If multiple funds have the same cap, selects the most recent launch.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker with highest remaining cap
    """
    if df_universe.empty:
        return None

    # Find maximum remaining cap
    max_cap = df_universe['Remaining Cap'].max()

    # Filter to funds with max cap
    max_cap_funds = df_universe[df_universe['Remaining Cap'] == max_cap].copy()

    # If tie, use most recent launch as tiebreaker
    if len(max_cap_funds) > 1:
        return select_most_recent_launch(max_cap_funds, current_date, series)

    return max_cap_funds.iloc[0]['Fund']


def select_remaining_cap_lowest(df_universe, current_date, series='F', df_regimes=None):
    """
    Select the fund with the LOWEST remaining cap (bearish/conservative).

    Chooses funds with caps nearly exhausted - more conservative positioning.
    If multiple funds have the same cap, selects the most recent launch.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker with lowest remaining cap
    """
    if df_universe.empty:
        return None

    # Find minimum remaining cap
    min_cap = df_universe['Remaining Cap'].min()

    # Filter to funds with min cap
    min_cap_funds = df_universe[df_universe['Remaining Cap'] == min_cap].copy()

    # If tie, use most recent launch as tiebreaker
    if len(min_cap_funds) > 1:
        return select_most_recent_launch(min_cap_funds, current_date, series)

    return min_cap_funds.iloc[0]['Fund']


def select_downside_buffer_highest(df_universe, current_date, series='F', df_regimes=None):
    """
    Select the fund with the HIGHEST downside before buffer (bullish).

    Chooses funds with the most cushion before hitting buffer zone.
    Higher downside % means more room to fall = more aggressive positioning.
    If multiple funds have the same downside, selects the most recent launch.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker with highest downside before buffer
    """
    if df_universe.empty:
        return None

    downside_col = 'Downside Before Buffer'

    if downside_col not in df_universe.columns:
        return df_universe.iloc[0]['Fund']

    # Filter out any null values
    valid_funds = df_universe[df_universe[downside_col].notna()].copy()

    if valid_funds.empty:
        return df_universe.iloc[0]['Fund']

    # Find maximum downside
    max_downside = valid_funds[downside_col].max()

    # Filter to funds with max downside
    max_downside_funds = valid_funds[valid_funds[downside_col] == max_downside].copy()

    # If tie, use most recent launch as tiebreaker
    if len(max_downside_funds) > 1:
        return select_most_recent_launch(max_downside_funds, current_date, series)

    return max_downside_funds.iloc[0]['Fund']


def select_downside_buffer_lowest(df_universe, current_date, series='F'):
    """
    Select the fund with the LOWEST downside before buffer (bearish/defensive).

    Chooses funds closest to or in the buffer zone - maximum downside protection.
    Lower downside % means closer to buffer = more defensive positioning.
    If multiple funds have the same downside, selects the most recent launch.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker with lowest downside before buffer
    """
    if df_universe.empty:
        return None

    downside_col = 'Downside Before Buffer'

    if downside_col not in df_universe.columns:
        return df_universe.iloc[0]['Fund']

    # Filter out any null values
    valid_funds = df_universe[df_universe[downside_col].notna()].copy()

    if valid_funds.empty:
        return df_universe.iloc[0]['Fund']

    # Find minimum downside
    valid_funds[downside_col] = abs(valid_funds[downside_col])
    min_downside = valid_funds[downside_col].min()

    # Filter to funds with min downside
    min_downside_funds = valid_funds[valid_funds[downside_col] == min_downside].copy()

    # If tie, use most recent launch as tiebreaker
    if len(min_downside_funds) > 1:
        return select_most_recent_launch(min_downside_funds, current_date, series)

    return min_downside_funds.iloc[0]['Fund']


def select_cap_utilization_lowest(df_universe, current_date, series='F', df_regimes=None):
    """
    Select the fund with the LOWEST cap utilization (bullish).

    Low cap utilization = most cap remaining = maximum upside potential.
    Example: 25% utilization means 75% of cap is still available.
    If multiple funds have the same utilization, selects the most recent launch.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker with lowest cap utilization
    """
    if df_universe.empty:
        return None

    if 'Cap_Utilization' not in df_universe.columns:
        return df_universe.iloc[0]['Fund']

    # Filter out any null values
    valid_funds = df_universe[df_universe['Cap_Utilization'].notna()].copy()

    if valid_funds.empty:
        return df_universe.iloc[0]['Fund']

    # Find minimum utilization
    min_util = valid_funds['Cap_Utilization'].min()

    # Filter to funds with min utilization
    min_util_funds = valid_funds[valid_funds['Cap_Utilization'] == min_util].copy()

    # If tie, use most recent launch as tiebreaker
    if len(min_util_funds) > 1:
        return select_most_recent_launch(min_util_funds, current_date, series)

    return min_util_funds.iloc[0]['Fund']


def select_cap_utilization_highest(df_universe, current_date, series='F', df_regimes=None):
    """
    Select the fund with the HIGHEST cap utilization (bearish/conservative).

    High cap utilization = little cap remaining = conservative positioning.
    Example: 85% utilization means only 15% of cap is available.
    If multiple funds have the same utilization, selects the most recent launch.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker with highest cap utilization
    """
    if df_universe.empty:
        return None

    if 'Cap_Utilization' not in df_universe.columns:
        return df_universe.iloc[0]['Fund']

    # Filter out any null values
    valid_funds = df_universe[df_universe['Cap_Utilization'].notna()].copy()

    if valid_funds.empty:
        return df_universe.iloc[0]['Fund']

    # Find maximum utilization
    max_util = valid_funds['Cap_Utilization'].max()

    # Filter to funds with max utilization
    max_util_funds = valid_funds[valid_funds['Cap_Utilization'] == max_util].copy()

    # If tie, use most recent launch as tiebreaker
    if len(max_util_funds) > 1:
        return select_most_recent_launch(max_util_funds, current_date, series)

    return max_util_funds.iloc[0]['Fund']


def select_highest_outcome_and_cap(df_universe, current_date, series='F', df_regimes=None):
    """
    Select fund with highest combined Remaining Outcome Days + Remaining Cap.

    If multiple funds have the same score, selects the most recent launch.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker
    """
    if df_universe.empty:
        return None

    df_universe = df_universe.copy()
    df_universe['Combined_Score'] = (
            df_universe['Remaining Outcome Days'] / 365 * 100 +
            df_universe['Remaining Cap']
    )

    # Find maximum score
    max_score = df_universe['Combined_Score'].max()

    # Filter to funds with max score
    max_score_funds = df_universe[df_universe['Combined_Score'] == max_score].copy()

    # If tie, use most recent launch as tiebreaker
    if len(max_score_funds) > 1:
        return select_most_recent_launch(max_score_funds, current_date, series)

    return max_score_funds.iloc[0]['Fund']


def select_cost_analysis(df_universe, current_date, series='F', df_regimes=None):
    """
    Select fund with lowest cost per day of protection.

    If multiple funds have the same cost, selects the most recent launch.

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker
    """
    if df_universe.empty:
        return None

    df_universe = df_universe.copy()
    downside_col = 'Downside Before Buffer'
    df_universe['Cost_Per_Day'] = np.inf

    valid_mask = (
            (df_universe['Remaining Outcome Days'] > 0) &
            (df_universe[downside_col].notna()) &
            (df_universe[downside_col] != 0)
    )

    df_universe.loc[valid_mask, 'Cost_Per_Day'] = (
            df_universe.loc[valid_mask, 'Fund Value (USD)'] /
            (abs(df_universe.loc[valid_mask, downside_col] / 100) *
             (df_universe.loc[valid_mask, 'Remaining Outcome Days'] / 365))
    )

    # Find minimum cost
    min_cost = df_universe['Cost_Per_Day'].min()

    # Filter to funds with min cost
    min_cost_funds = df_universe[df_universe['Cost_Per_Day'] == min_cost].copy()

    # If tie, use most recent launch as tiebreaker
    if len(min_cost_funds) > 1:
        return select_most_recent_launch(min_cost_funds, current_date, series)

    return min_cost_funds.iloc[0]['Fund']


def select_remaining_buffer_lowest(df_universe, current_date, series='F', df_regimes=None):
    """
    Select fund with LOWEST remaining buffer (bearish/defensive).

    Chooses funds where Remaining Buffer is most depleted, indicating
    the fund is closest to or in the buffer zone (defensive positioning).

    Logic:
    - Lower Remaining Buffer = closer to buffer activation
    - For F-series: Original Buffer = 10%
      - If Remaining Buffer = 8%, fund has 2% buffer cushion used
      - If Remaining Buffer = 5%, fund has 5% buffer cushion used (MORE defensive)

    This is bearish because:
    - Targets funds experiencing market stress
    - Positions in funds closest to full buffer protection
    - Defensive capital preservation focus

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker with lowest remaining buffer
    """
    if df_universe.empty:
        return None

    buffer_col = 'Remaining Buffer'

    if buffer_col not in df_universe.columns:
        return select_most_recent_launch(df_universe, current_date, series)

    # Filter out null values
    valid_funds = df_universe[df_universe[buffer_col].notna()].copy()

    if valid_funds.empty:
        return select_most_recent_launch(df_universe, current_date, series)

    # Find minimum remaining buffer (most depleted)
    min_buffer = valid_funds[buffer_col].min()

    # Filter to funds with min buffer
    min_buffer_funds = valid_funds[valid_funds[buffer_col] == min_buffer].copy()

    # If tie, use most recent launch as tiebreaker
    if len(min_buffer_funds) > 1:
        return select_most_recent_launch(min_buffer_funds, current_date, series)

    return min_buffer_funds.iloc[0]['Fund']


def select_enhanced_cost_ratio_bullish(df_universe, current_date, series='F', df_regimes=None):
    """
    Select fund with highest Enhanced Cost Ratio (Bullish weighting).

    Enhanced Cost Ratio is a composite score that normalizes Buffer ETFs across
    multiple dimensions for apples-to-apples comparison:

    Components:
    1. DBB Score (Downside Before Buffer): 1 + DBB_decimal [0.8-1.0 range]
    2. Buffer Integrity Score: (Remaining Buffer / Original Buffer) scaled to [0.5-1.0]
    3. Cap Integrity Score: Remaining Cap / Original Cap [0-1 range]
    4. Time Scaling Factor: 1 - ln(days_remaining / original_days)

    Bullish Weights: DBB=0.2, Buffer=0.1, Cap=0.7
    - Heavy emphasis on Cap Integrity (70%) - seeking maximum upside potential
    - Light weight on Buffer (10%) - less concerned with protection

    Formula:
    ECR = (0.2*DBB + 0.1*Buffer_Scaled + 0.7*Cap) / Time_Factor

    Higher ECR = Better fund for bullish positioning

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker with highest Enhanced Cost Ratio (bullish)
    """
    if df_universe.empty:
        return None

    # Required columns
    required_cols = ['Remaining Buffer', 'Original_Buffer', 'Remaining Cap',
                     'Original_Cap', 'Downside Before Buffer',
                     'Remaining Outcome Days', 'Total_Outcome_Days']

    if not all(col in df_universe.columns for col in required_cols):
        return select_most_recent_launch(df_universe, current_date, series)

    df_work = df_universe.copy()

    # Filter out funds with missing critical data
    valid_mask = (
        df_work['Remaining Buffer'].notna() &
        df_work['Original_Buffer'].notna() &
        df_work['Remaining Cap'].notna() &
        df_work['Original_Cap'].notna() &
        df_work['Downside Before Buffer'].notna() &
        df_work['Remaining Outcome Days'].notna() &
        df_work['Total_Outcome_Days'].notna() &
        (df_work['Original_Buffer'] > 0) &
        (df_work['Original_Cap'] > 0) &
        (df_work['Remaining Outcome Days'] > 0) &
        (df_work['Total_Outcome_Days'] > 0)
    )

    df_valid = df_work[valid_mask].copy()

    if df_valid.empty:
        return select_most_recent_launch(df_universe, current_date, series)

    # Component 1: DBB Score (no scaling)
    df_valid['DBB_Score'] = 1 + (df_valid['Downside Before Buffer'] / 100)

    # Component 2: Buffer Integrity Score (with scaling)
    df_valid['Buffer_Integrity'] = (df_valid['Remaining Buffer'] / 100) / df_valid['Original_Buffer']

    # Apply MinMaxScaler(0.5, 1.0) dynamically on current universe
    bi_min = df_valid['Buffer_Integrity'].min()
    bi_max = df_valid['Buffer_Integrity'].max()

    if bi_max > bi_min:
        # Standard scaling
        df_valid['Buffer_Score'] = ((df_valid['Buffer_Integrity'] - bi_min) / (bi_max - bi_min)) * 0.5 + 0.5
    else:
        # All values equal - assign midpoint
        df_valid['Buffer_Score'] = 0.75

    # Component 3: Cap Integrity Score (no scaling)
    df_valid['Cap_Score'] = (df_valid['Remaining Cap'] / 100) / df_valid['Original_Cap']

    # Component 4: Time Scaling Factor
    df_valid['Time_Ratio'] = df_valid['Remaining Outcome Days'] / df_valid['Total_Outcome_Days']
    df_valid['Time_Factor'] = 1 - np.log(df_valid['Time_Ratio'])

    # Bullish Weights: Cap-heavy (seeking upside)
    w_dbb = 0.2
    w_buffer = 0.1
    w_cap = 0.7

    # Calculate Enhanced Cost Ratio
    df_valid['Numerator'] = (w_dbb * df_valid['DBB_Score'] +
                              w_buffer * df_valid['Buffer_Score'] +
                              w_cap * df_valid['Cap_Score'])
    df_valid['ECR'] = df_valid['Numerator'] / df_valid['Time_Factor']

    # Select fund with highest ECR
    max_ecr = df_valid['ECR'].max()
    max_ecr_funds = df_valid[df_valid['ECR'] == max_ecr].copy()

    # Tiebreaker: most recent launch
    if len(max_ecr_funds) > 1:
        return select_most_recent_launch(max_ecr_funds, current_date, series)

    return max_ecr_funds.iloc[0]['Fund']


def select_enhanced_cost_ratio_bearish(df_universe, current_date, series='F', df_regimes=None):
    """
    Select fund with highest Enhanced Cost Ratio (Bearish weighting).

    Enhanced Cost Ratio is a composite score that normalizes Buffer ETFs across
    multiple dimensions for apples-to-apples comparison.

    Bearish Weights: DBB=0.4, Buffer=0.4, Cap=0.2
    - Heavy emphasis on DBB (40%) - seeking maximum downside protection
    - Heavy emphasis on Buffer (40%) - prioritizing safety cushion
    - Light weight on Cap (20%) - less concerned with upside

    Formula:
    ECR = (0.4*DBB + 0.4*Buffer_Scaled + 0.2*Cap) / Time_Factor

    Higher ECR = Better fund for bearish/defensive positioning

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker with highest Enhanced Cost Ratio (bearish)
    """
    if df_universe.empty:
        return None

    # Required columns
    required_cols = ['Remaining Buffer', 'Original_Buffer', 'Remaining Cap',
                     'Original_Cap', 'Downside Before Buffer',
                     'Remaining Outcome Days', 'Total_Outcome_Days']

    if not all(col in df_universe.columns for col in required_cols):
        return select_most_recent_launch(df_universe, current_date, series)

    df_work = df_universe.copy()

    # Filter out funds with missing critical data
    valid_mask = (
        df_work['Remaining Buffer'].notna() &
        df_work['Original_Buffer'].notna() &
        df_work['Remaining Cap'].notna() &
        df_work['Original_Cap'].notna() &
        df_work['Downside Before Buffer'].notna() &
        df_work['Remaining Outcome Days'].notna() &
        df_work['Total_Outcome_Days'].notna() &
        (df_work['Original_Buffer'] > 0) &
        (df_work['Original_Cap'] > 0) &
        (df_work['Remaining Outcome Days'] > 0) &
        (df_work['Total_Outcome_Days'] > 0)
    )

    df_valid = df_work[valid_mask].copy()

    if df_valid.empty:
        return select_most_recent_launch(df_universe, current_date, series)

    # Component 1: DBB Score (no scaling)
    df_valid['DBB_Score'] = 1 + (df_valid['Downside Before Buffer'] / 100)

    # Component 2: Buffer Integrity Score (with scaling)
    df_valid['Buffer_Integrity'] = (df_valid['Remaining Buffer'] / 100) / df_valid['Original_Buffer']

    # Apply MinMaxScaler(0.5, 1.0) dynamically on current universe
    bi_min = df_valid['Buffer_Integrity'].min()
    bi_max = df_valid['Buffer_Integrity'].max()

    if bi_max > bi_min:
        # Standard scaling
        df_valid['Buffer_Score'] = ((df_valid['Buffer_Integrity'] - bi_min) / (bi_max - bi_min)) * 0.5 + 0.5
    else:
        # All values equal - assign midpoint
        df_valid['Buffer_Score'] = 0.75

    # Component 3: Cap Integrity Score (no scaling)
    df_valid['Cap_Score'] = (df_valid['Remaining Cap'] / 100) / df_valid['Original_Cap']

    # Component 4: Time Scaling Factor
    df_valid['Time_Ratio'] = df_valid['Remaining Outcome Days'] / df_valid['Total_Outcome_Days']
    df_valid['Time_Factor'] = 1 - np.log(df_valid['Time_Ratio'])

    # Bearish Weights: DBB and Buffer heavy (seeking protection)
    w_dbb = 0.4
    w_buffer = 0.4
    w_cap = 0.2

    # Calculate Enhanced Cost Ratio
    df_valid['Numerator'] = (w_dbb * df_valid['DBB_Score'] +
                              w_buffer * df_valid['Buffer_Score'] +
                              w_cap * df_valid['Cap_Score'])
    df_valid['ECR'] = df_valid['Numerator'] / df_valid['Time_Factor']

    # Select fund with highest ECR
    max_ecr = df_valid['ECR'].max()
    max_ecr_funds = df_valid[df_valid['ECR'] == max_ecr].copy()

    # Tiebreaker: most recent launch
    if len(max_ecr_funds) > 1:
        return select_most_recent_launch(max_ecr_funds, current_date, series)

    return max_ecr_funds.iloc[0]['Fund']


def select_enhanced_cost_ratio_neutral(df_universe, current_date, series='F', df_regimes=None):
    """
    Select fund with highest Enhanced Cost Ratio (Neutral weighting).

    Enhanced Cost Ratio is a composite score that normalizes Buffer ETFs across
    multiple dimensions for apples-to-apples comparison.

    Neutral Weights: DBB=0.333, Buffer=0.333, Cap=0.333
    - Equal weighting across all three components
    - Balanced approach between upside and downside

    Formula:
    ECR = (0.333*DBB + 0.333*Buffer_Scaled + 0.333*Cap) / Time_Factor

    Higher ECR = Better fund for balanced positioning

    Parameters:
      df_universe: DataFrame with all funds on current date
      current_date: Current date
      series: Fund series

    Returns:
      String: Fund ticker with highest Enhanced Cost Ratio (neutral)
    """
    if df_universe.empty:
        return None

    # Required columns
    required_cols = ['Remaining Buffer', 'Original_Buffer', 'Remaining Cap',
                     'Original_Cap', 'Downside Before Buffer',
                     'Remaining Outcome Days', 'Total_Outcome_Days']

    if not all(col in df_universe.columns for col in required_cols):
        return select_most_recent_launch(df_universe, current_date, series)

    df_work = df_universe.copy()

    # Filter out funds with missing critical data
    valid_mask = (
        df_work['Remaining Buffer'].notna() &
        df_work['Original_Buffer'].notna() &
        df_work['Remaining Cap'].notna() &
        df_work['Original_Cap'].notna() &
        df_work['Downside Before Buffer'].notna() &
        df_work['Remaining Outcome Days'].notna() &
        df_work['Total_Outcome_Days'].notna() &
        (df_work['Original_Buffer'] > 0) &
        (df_work['Original_Cap'] > 0) &
        (df_work['Remaining Outcome Days'] > 0) &
        (df_work['Total_Outcome_Days'] > 0)
    )

    df_valid = df_work[valid_mask].copy()

    if df_valid.empty:
        return select_most_recent_launch(df_universe, current_date, series)

    # Component 1: DBB Score (no scaling)
    df_valid['DBB_Score'] = 1 + (df_valid['Downside Before Buffer'] / 100)

    # Component 2: Buffer Integrity Score (with scaling)
    df_valid['Buffer_Integrity'] = (df_valid['Remaining Buffer'] / 100) / df_valid['Original_Buffer']

    # Apply MinMaxScaler(0.5, 1.0) dynamically on current universe
    bi_min = df_valid['Buffer_Integrity'].min()
    bi_max = df_valid['Buffer_Integrity'].max()

    if bi_max > bi_min:
        # Standard scaling
        df_valid['Buffer_Score'] = ((df_valid['Buffer_Integrity'] - bi_min) / (bi_max - bi_min)) * 0.5 + 0.5
    else:
        # All values equal - assign midpoint
        df_valid['Buffer_Score'] = 0.75

    # Component 3: Cap Integrity Score (no scaling)
    df_valid['Cap_Score'] = (df_valid['Remaining Cap'] / 100) / df_valid['Original_Cap']

    # Component 4: Time Scaling Factor
    df_valid['Time_Ratio'] = df_valid['Remaining Outcome Days'] / df_valid['Total_Outcome_Days']
    df_valid['Time_Factor'] = 1 - np.log(df_valid['Time_Ratio'])

    # Neutral Weights: Equal balance
    w_dbb = 0.333
    w_buffer = 0.333
    w_cap = 0.333

    # Calculate Enhanced Cost Ratio
    df_valid['Numerator'] = (w_dbb * df_valid['DBB_Score'] +
                              w_buffer * df_valid['Buffer_Score'] +
                              w_cap * df_valid['Cap_Score'])
    df_valid['ECR'] = df_valid['Numerator'] / df_valid['Time_Factor']

    # Select fund with highest ECR
    max_ecr = df_valid['ECR'].max()
    max_ecr_funds = df_valid[df_valid['ECR'] == max_ecr].copy()

    # Tiebreaker: most recent launch
    if len(max_ecr_funds) > 1:
        return select_most_recent_launch(max_ecr_funds, current_date, series)

    return max_ecr_funds.iloc[0]['Fund']


# =============================================================================
# ADD THIS TO core/selections.py (after your existing ECR functions)
# =============================================================================

def _enhanced_cost_ratio_core(
        df_universe,
        current_date,
        series,
        w_dbb,
        w_buffer,
        w_cap
):
    """
    Core Enhanced Cost Ratio implementation with parameterized weights.

    This is the shared calculation logic - all extreme weight variants call this.

    Parameters:
        df_universe: DataFrame with all funds on current date
        current_date: Current date
        series: Fund series
        w_dbb: Weight for DBB component (0-1)
        w_buffer: Weight for Buffer Integrity component (0-1)
        w_cap: Weight for Cap Integrity component (0-1)

    Note: w_dbb + w_buffer + w_cap should equal 1.0

    Returns:
        String: Fund ticker with highest Enhanced Cost Ratio
    """
    if df_universe.empty:
        return None

    # Required columns
    required_cols = ['Remaining Buffer', 'Original_Buffer', 'Remaining Cap',
                     'Original_Cap', 'Downside Before Buffer',
                     'Remaining Outcome Days', 'Total_Outcome_Days']

    if not all(col in df_universe.columns for col in required_cols):
        return select_most_recent_launch(df_universe, current_date, series)

    df_work = df_universe.copy()

    # Filter out funds with missing critical data
    valid_mask = (
            df_work['Remaining Buffer'].notna() &
            df_work['Original_Buffer'].notna() &
            df_work['Remaining Cap'].notna() &
            df_work['Original_Cap'].notna() &
            df_work['Downside Before Buffer'].notna() &
            df_work['Remaining Outcome Days'].notna() &
            df_work['Total_Outcome_Days'].notna() &
            (df_work['Original_Buffer'] > 0) &
            (df_work['Original_Cap'] > 0) &
            (df_work['Remaining Outcome Days'] > 0) &
            (df_work['Total_Outcome_Days'] > 0)
    )

    df_valid = df_work[valid_mask].copy()

    if df_valid.empty:
        return select_most_recent_launch(df_universe, current_date, series)

    # Component 1: DBB Score (no scaling)
    df_valid['DBB_Score'] = 1 + (df_valid['Downside Before Buffer'] / 100)

    # Component 2: Buffer Integrity Score (with scaling)
    df_valid['Buffer_Integrity'] = (df_valid['Remaining Buffer'] / 100) / df_valid['Original_Buffer']

    # Apply MinMaxScaler(0.5, 1.0) dynamically on current universe
    bi_min = df_valid['Buffer_Integrity'].min()
    bi_max = df_valid['Buffer_Integrity'].max()

    if bi_max > bi_min:
        df_valid['Buffer_Score'] = ((df_valid['Buffer_Integrity'] - bi_min) / (bi_max - bi_min)) * 0.5 + 0.5
    else:
        df_valid['Buffer_Score'] = 0.75

    # Component 3: Cap Integrity Score (no scaling)
    df_valid['Cap_Score'] = (df_valid['Remaining Cap'] / 100) / df_valid['Original_Cap']

    # Component 4: Time Scaling Factor
    df_valid['Time_Ratio'] = df_valid['Remaining Outcome Days'] / df_valid['Total_Outcome_Days']
    df_valid['Time_Factor'] = 1 - np.log(df_valid['Time_Ratio'])

    # Calculate Enhanced Cost Ratio with CUSTOM WEIGHTS
    df_valid['Numerator'] = (w_dbb * df_valid['DBB_Score'] +
                             w_buffer * df_valid['Buffer_Score'] +
                             w_cap * df_valid['Cap_Score'])
    df_valid['ECR'] = df_valid['Numerator'] / df_valid['Time_Factor']

    # Select fund with highest ECR
    max_ecr = df_valid['ECR'].max()
    max_ecr_funds = df_valid[df_valid['ECR'] == max_ecr].copy()

    # Tiebreaker: most recent launch
    if len(max_ecr_funds) > 1:
        return select_most_recent_launch(max_ecr_funds, current_date, series)

    return max_ecr_funds.iloc[0]['Fund']


# =============================================================================
# PURE COMPONENT TESTS (100% focus on one component)
# =============================================================================

def select_ecr_pure_cap(df_universe, current_date, series='F', df_regimes=None):
    """100% Cap focus - ignore DBB and Buffer entirely."""
    return _enhanced_cost_ratio_core(df_universe, current_date, series,
                                     w_dbb=0.0, w_buffer=0.0, w_cap=1.0)


def select_ecr_pure_buffer(df_universe, current_date, series='F', df_regimes=None):
    """100% Buffer Integrity focus - ignore DBB and Cap entirely."""
    return _enhanced_cost_ratio_core(df_universe, current_date, series,
                                     w_dbb=0.0, w_buffer=1.0, w_cap=0.0)


def select_ecr_pure_dbb(df_universe, current_date, series='F', df_regimes=None):
    """100% DBB focus - ignore Buffer and Cap entirely."""
    return _enhanced_cost_ratio_core(df_universe, current_date, series,
                                     w_dbb=1.0, w_buffer=0.0, w_cap=0.0)


# =============================================================================
# EXTREME SKEWS (80-90% focus on one component)
# =============================================================================

def select_ecr_ultra_cap(df_universe, current_date, series='F', df_regimes=None):
    """Ultra Cap-Heavy: 90% cap, 5% each for DBB and Buffer."""
    return _enhanced_cost_ratio_core(df_universe, current_date, series,
                                     w_dbb=0.05, w_buffer=0.05, w_cap=0.90)


def select_ecr_ultra_protection(df_universe, current_date, series='F', df_regimes=None):
    """Ultra Protection: 45% DBB, 45% Buffer, 10% Cap."""
    return _enhanced_cost_ratio_core(df_universe, current_date, series,
                                     w_dbb=0.45, w_buffer=0.45, w_cap=0.10)


def select_ecr_dbb_dominant(df_universe, current_date, series='F', df_regimes=None):
    """DBB Dominant: 80% DBB, 10% each for Buffer and Cap."""
    return _enhanced_cost_ratio_core(df_universe, current_date, series,
                                     w_dbb=0.80, w_buffer=0.10, w_cap=0.10)


def select_ecr_buffer_dominant(df_universe, current_date, series='F', df_regimes=None):
    """Buffer Dominant: 80% Buffer, 10% each for DBB and Cap."""
    return _enhanced_cost_ratio_core(df_universe, current_date, series,
                                     w_dbb=0.10, w_buffer=0.80, w_cap=0.10)


# =============================================================================
# ABLATION TESTS (test by exclusion - remove one component)
# =============================================================================

def select_ecr_no_cap(df_universe, current_date, series='F', df_regimes=None):
    """No Cap Component: 50% DBB, 50% Buffer, 0% Cap."""
    return _enhanced_cost_ratio_core(df_universe, current_date, series,
                                     w_dbb=0.50, w_buffer=0.50, w_cap=0.0)


def select_ecr_no_buffer(df_universe, current_date, series='F', df_regimes=None):
    """No Buffer Component: 50% DBB, 0% Buffer, 50% Cap."""
    return _enhanced_cost_ratio_core(df_universe, current_date, series,
                                     w_dbb=0.50, w_buffer=0.0, w_cap=0.50)


# =============================================================================
# SHARED CORE FUNCTION WITH PARAMETERIZED MECHANICS
# =============================================================================

def _ecr_mechanics_core(
        df_universe,
        current_date,
        series,
        buffer_range=(0.5, 1.0),
        time_method='log',
        dbb_scaled=False
):
    """
    Core ECR implementation with parameterized calculation mechanics.

    Parameters:
        df_universe: DataFrame with all funds on current date
        current_date: Current date
        series: Fund series
        buffer_range: Tuple of (min, max) for Buffer Integrity scaling (default: (0.5, 1.0))
        time_method: Time scaling method - 'log', 'linear', 'sqrt', 'inverse', 'none'
        dbb_scaled: If True, scale DBB like Buffer Integrity; if False, use raw (1 + DBB_decimal)

    Returns:
        String: Fund ticker with highest ECR
    """
    if df_universe.empty:
        return None

    required_cols = ['Remaining Buffer', 'Original_Buffer', 'Remaining Cap',
                     'Original_Cap', 'Downside Before Buffer',
                     'Remaining Outcome Days', 'Total_Outcome_Days']

    if not all(col in df_universe.columns for col in required_cols):
        return select_most_recent_launch(df_universe, current_date, series)

    df_work = df_universe.copy()

    valid_mask = (
            df_work['Remaining Buffer'].notna() &
            df_work['Original_Buffer'].notna() &
            df_work['Remaining Cap'].notna() &
            df_work['Original_Cap'].notna() &
            df_work['Downside Before Buffer'].notna() &
            df_work['Remaining Outcome Days'].notna() &
            df_work['Total_Outcome_Days'].notna() &
            (df_work['Original_Buffer'] > 0) &
            (df_work['Original_Cap'] > 0) &
            (df_work['Remaining Outcome Days'] > 0) &
            (df_work['Total_Outcome_Days'] > 0)
    )

    df_valid = df_work[valid_mask].copy()

    if df_valid.empty:
        return select_most_recent_launch(df_universe, current_date, series)

    # =========================================================================
    # Component 1: DBB Score (with optional scaling)
    # =========================================================================

    if dbb_scaled:
        # Scale DBB like Buffer Integrity
        df_valid['DBB_Raw'] = 1 + (df_valid['Downside Before Buffer'] / 100)
        dbb_min = df_valid['DBB_Raw'].min()
        dbb_max = df_valid['DBB_Raw'].max()

        if dbb_max > dbb_min:
            df_valid['DBB_Score'] = ((df_valid['DBB_Raw'] - dbb_min) / (dbb_max - dbb_min)) * 0.5 + 0.5
        else:
            df_valid['DBB_Score'] = 0.75
    else:
        # Use raw DBB (current default)
        df_valid['DBB_Score'] = 1 + (df_valid['Downside Before Buffer'] / 100)

    # =========================================================================
    # Component 2: Buffer Integrity Score (with parameterized range)
    # =========================================================================

    df_valid['Buffer_Integrity'] = (df_valid['Remaining Buffer'] / 100) / df_valid['Original_Buffer']

    bi_min = df_valid['Buffer_Integrity'].min()
    bi_max = df_valid['Buffer_Integrity'].max()

    range_min, range_max = buffer_range

    if bi_max > bi_min:
        df_valid['Buffer_Score'] = ((df_valid['Buffer_Integrity'] - bi_min) / (bi_max - bi_min)) * (range_max - range_min) + range_min
    else:
        # All equal - assign midpoint
        df_valid['Buffer_Score'] = (range_min + range_max) / 2

    # =========================================================================
    # Component 3: Cap Integrity Score (no changes - always raw)
    # =========================================================================

    df_valid['Cap_Score'] = (df_valid['Remaining Cap'] / 100) / df_valid['Original_Cap']

    # =========================================================================
    # Component 4: Time Scaling Factor (with parameterized method)
    # =========================================================================

    df_valid['Time_Ratio'] = df_valid['Remaining Outcome Days'] / df_valid['Total_Outcome_Days']

    if time_method == 'log':
        # Current default: 1 - ln(days_remaining / original_days)
        df_valid['Time_Factor'] = 1 - np.log(df_valid['Time_Ratio'])

    elif time_method == 'linear':
        # Linear: days_remaining / original_days
        df_valid['Time_Factor'] = df_valid['Time_Ratio']

    elif time_method == 'sqrt':
        # Square root scaling
        df_valid['Time_Factor'] = np.sqrt(df_valid['Time_Ratio'])

    elif time_method == 'inverse':
        # Inverse linear: 1 - (days_remaining / original_days)
        df_valid['Time_Factor'] = 1 - df_valid['Time_Ratio']

    elif time_method == 'none':
        # No time scaling - constant
        df_valid['Time_Factor'] = 1.0

    else:
        # Fallback to log if unknown
        df_valid['Time_Factor'] = 1 - np.log(df_valid['Time_Ratio'])

    # Ensure Time_Factor is never zero or negative (for division)
    df_valid['Time_Factor'] = df_valid['Time_Factor'].clip(lower=0.01)

    # =========================================================================
    # Calculate ECR with neutral weights
    # =========================================================================

    w_dbb = 0.333
    w_buffer = 0.333
    w_cap = 0.333

    df_valid['Numerator'] = (w_dbb * df_valid['DBB_Score'] +
                             w_buffer * df_valid['Buffer_Score'] +
                             w_cap * df_valid['Cap_Score'])
    df_valid['ECR'] = df_valid['Numerator'] / df_valid['Time_Factor']

    # Select fund with highest ECR
    max_ecr = df_valid['ECR'].max()
    max_ecr_funds = df_valid[df_valid['ECR'] == max_ecr].copy()

    if len(max_ecr_funds) > 1:
        return select_most_recent_launch(max_ecr_funds, current_date, series)

    return max_ecr_funds.iloc[0]['Fund']


# =============================================================================
# GROUP 1: BUFFER INTEGRITY SCALING VARIANTS (4 functions)
# =============================================================================

def select_ecr_buffer_scale_full(df_universe, current_date, series='F', df_regimes=None):
    """
    ECR with Buffer Integrity scaled to [0.0, 1.0] (full range).

    Tests: Does giving full 0-100% range to buffer scores improve performance?
    Current default is [0.5, 1.0] which only penalizes down to 50%.
    """
    return _ecr_mechanics_core(df_universe, current_date, series,
                               buffer_range=(0.0, 1.0),
                               time_method='log',
                               dbb_scaled=False)


def select_ecr_buffer_scale_low_floor(df_universe, current_date, series='F', df_regimes=None):
    """
    ECR with Buffer Integrity scaled to [0.3, 1.0] (lower floor than default).

    Tests: Does allowing more penalty (down to 30%) improve differentiation?
    """
    return _ecr_mechanics_core(df_universe, current_date, series,
                               buffer_range=(0.3, 1.0),
                               time_method='log',
                               dbb_scaled=False)


def select_ecr_buffer_scale_default(df_universe, current_date, series='F', df_regimes=None):
    """
    ECR with Buffer Integrity scaled to [0.5, 1.0] (current default).

    Baseline for comparison - this is what Batch 6 and 6B used.
    """
    return _ecr_mechanics_core(df_universe, current_date, series,
                               buffer_range=(0.5, 1.0),
                               time_method='log',
                               dbb_scaled=False)


def select_ecr_buffer_scale_high_floor(df_universe, current_date, series='F', df_regimes=None):
    """
    ECR with Buffer Integrity scaled to [0.7, 1.0] (higher floor - less penalty).

    Tests: Does reducing buffer penalty (only down to 70%) improve performance?
    This makes buffer component less influential.
    """
    return _ecr_mechanics_core(df_universe, current_date, series,
                               buffer_range=(0.7, 1.0),
                               time_method='log',
                               dbb_scaled=False)


# =============================================================================
# GROUP 2: TIME SCALING METHOD VARIANTS (5 functions)
# =============================================================================

def select_ecr_time_log(df_universe, current_date, series='F', df_regimes=None):
    """
    ECR with Logarithmic time scaling: 1 - ln(days_remaining / original_days).

    Baseline - this is the current default.
    Heavily penalizes funds near expiration.
    """
    return _ecr_mechanics_core(df_universe, current_date, series,
                               buffer_range=(0.5, 1.0),
                               time_method='log',
                               dbb_scaled=False)


def select_ecr_time_linear(df_universe, current_date, series='F', df_regimes=None):
    """
    ECR with Linear time scaling: days_remaining / original_days.

    Tests: Does simple proportional time weighting work better than logarithmic?
    Example: Fund with 180/365 days gets Time_Factor = 0.493
    """
    return _ecr_mechanics_core(df_universe, current_date, series,
                               buffer_range=(0.5, 1.0),
                               time_method='linear',
                               dbb_scaled=False)


def select_ecr_time_sqrt(df_universe, current_date, series='F', df_regimes=None):
    """
    ECR with Square Root time scaling: sqrt(days_remaining / original_days).

    Tests: Does moderate non-linear scaling (between linear and log) work better?
    Less aggressive penalty than log, more aggressive than linear.
    """
    return _ecr_mechanics_core(df_universe, current_date, series,
                               buffer_range=(0.5, 1.0),
                               time_method='sqrt',
                               dbb_scaled=False)


def select_ecr_time_inverse(df_universe, current_date, series='F', df_regimes=None):
    """
    ECR with Inverse Linear time scaling: 1 - (days_remaining / original_days).

    Tests: Does inverting the time ratio improve performance?
    Example: Fund with 180/365 days gets Time_Factor = 0.507
    This gives slight preference to older funds.
    """
    return _ecr_mechanics_core(df_universe, current_date, series,
                               buffer_range=(0.5, 1.0),
                               time_method='inverse',
                               dbb_scaled=False)


def select_ecr_time_none(df_universe, current_date, series='F', df_regimes=None):
    """
    ECR with NO time scaling: Time_Factor = 1.0 (constant).

    Tests: Is time penalty even necessary? Maybe fund age doesn't matter.
    This removes time as a factor entirely.
    """
    return _ecr_mechanics_core(df_universe, current_date, series,
                               buffer_range=(0.5, 1.0),
                               time_method='none',
                               dbb_scaled=False)


# =============================================================================
# GROUP 3: DBB SCALING VARIANTS (2 functions)
# =============================================================================

def select_ecr_dbb_raw(df_universe, current_date, series='F', df_regimes=None):
    """
    ECR with Raw DBB Score: 1 + (DBB_decimal).

    Baseline - this is the current default.
    DBB is not scaled, just shifted to be positive.
    """
    return _ecr_mechanics_core(df_universe, current_date, series,
                               buffer_range=(0.5, 1.0),
                               time_method='log',
                               dbb_scaled=False)


def select_ecr_dbb_scaled(df_universe, current_date, series='F', df_regimes=None):
    """
    ECR with Scaled DBB Score: MinMaxScaler [0.5, 1.0] like Buffer Integrity.

    Tests: Does normalizing DBB across the universe improve performance?
    Makes DBB comparable in magnitude to Buffer score.
    """
    return _ecr_mechanics_core(df_universe, current_date, series,
                               buffer_range=(0.5, 1.0),
                               time_method='log',
                               dbb_scaled=True)


def select_enhanced_cost_ratio_neutral_v2(df_universe, current_date, series='F', df_regimes=None):
    """
    Enhanced Cost Ratio with updated buffer integrity logic (V2).

    Updates from V1:
    - New buffer scoring: Penalizes distance from ATM in rallies
    - In-buffer handling: Rewards remaining protection
    - Raw buffer scores (0.0-1.0) instead of MinMaxScaled [0.5-1.0]

    Components (all equal weighted at 0.333):
    1. DBB Score: 1 + (DBB% / 100) - unchanged
    2. Buffer Integrity: NEW distance-based scoring
    3. Cap Score: Remaining Cap / Original Cap - unchanged
    4. Time Factor: 1 - ln(Days Remaining / Total Days) - unchanged

    Formula:
    ECR = (0.333*DBB + 0.333*Buffer + 0.333*Cap) / Time_Factor

    Parameters:
        df_universe: DataFrame with all funds on current date
        current_date: Current date
        series: Fund series (default 'F')

    Returns:
        String: Fund ticker with highest ECR score
    """
    if df_universe.empty:
        return None

    # Required columns
    required_cols = ['Remaining Buffer', 'Original_Buffer', 'Remaining Cap',
                     'Original_Cap', 'Downside Before Buffer',
                     'Remaining Outcome Days', 'Total_Outcome_Days']

    if not all(col in df_universe.columns for col in required_cols):
        return select_most_recent_launch(df_universe, current_date, series)

    df_work = df_universe.copy()

    # Filter out funds with missing critical data
    valid_mask = (
            df_work['Remaining Buffer'].notna() &
            df_work['Original_Buffer'].notna() &
            df_work['Remaining Cap'].notna() &
            df_work['Original_Cap'].notna() &
            df_work['Downside Before Buffer'].notna() &
            df_work['Remaining Outcome Days'].notna() &
            df_work['Total_Outcome_Days'].notna() &
            (df_work['Original_Buffer'] > 0) &
            (df_work['Original_Cap'] > 0) &
            (df_work['Remaining Outcome Days'] > 0) &
            (df_work['Total_Outcome_Days'] > 0)
    )

    df_valid = df_work[valid_mask].copy()

    if df_valid.empty:
        return select_most_recent_launch(df_universe, current_date, series)

    # =========================================================================
    # Component 1: DBB Score (unchanged from V1)
    # =========================================================================

    df_valid['DBB_Score'] = 1 + (df_valid['Downside Before Buffer'] / 100)

    # =========================================================================
    # Component 2: Buffer Integrity Score (NEW LOGIC)
    # =========================================================================

    def calculate_buffer_score(row):
        """
        Calculate buffer score based on distance from ATM and in-buffer status.

        Logic:
        1. If DBB < 0: Market rallied from ATM
           - Penalize distance from ATM (larger abs(DBB) = lower score)
           - Max expected distance = Original_Buffer × 2

        2. If DBB == 0: Either at ATM or in buffer
           - If Remaining ≈ Original: At ATM → Score = 1.0
           - If Remaining < Original: In buffer → Score = Remaining / Original
        """
        dbb = row['Downside Before Buffer']
        remaining_buffer = row['Remaining Buffer'] / 100  # Convert to decimal
        original_buffer = row['Original_Buffer']

        if dbb < 0:
            # Market has rallied from ATM
            # Penalize distance from ideal starting point
            distance_from_atm = abs(dbb / 100)  # Convert to decimal
            max_expected_distance = original_buffer * 2  # e.g., 0.20 for 10% buffer

            if max_expected_distance > 0:
                buffer_score = 1.0 - (distance_from_atm / max_expected_distance)
                buffer_score = max(buffer_score, 0.0)  # Floor at 0
            else:
                buffer_score = 0.5

        elif dbb == 0:
            # Either at ATM (fresh) or in buffer zone
            # Distinguish by checking remaining buffer

            if remaining_buffer >= original_buffer * 0.95:
                # At ATM - fresh fund with full buffer
                buffer_score = 1.0
            else:
                # In buffer zone - score based on remaining protection
                if original_buffer > 0:
                    buffer_score = remaining_buffer / original_buffer
                    buffer_score = max(buffer_score, 0.0)
                else:
                    buffer_score = 0.0
        else:
            # DBB > 0 shouldn't happen (downside before buffer can't be positive)
            # Fallback to safe default
            buffer_score = 0.0

        return buffer_score

    df_valid['Buffer_Score'] = df_valid.apply(calculate_buffer_score, axis=1)

    # =========================================================================
    # Component 3: Cap Integrity Score (unchanged from V1)
    # =========================================================================

    df_valid['Cap_Score'] = (df_valid['Remaining Cap'] / 100) / df_valid['Original_Cap']

    # =========================================================================
    # Component 4: Time Scaling Factor (unchanged from V1)
    # =========================================================================

    df_valid['Time_Ratio'] = df_valid['Remaining Outcome Days'] / df_valid['Total_Outcome_Days']
    df_valid['Time_Factor'] = 1 - np.log(df_valid['Time_Ratio'])

    # Ensure Time_Factor is never zero (for division safety)
    df_valid['Time_Factor'] = df_valid['Time_Factor'].clip(lower=0.01)

    # =========================================================================
    # Calculate Enhanced Cost Ratio with Equal Weights
    # =========================================================================

    w_dbb = 0.15
    w_buffer = 0.35
    w_cap = 0.50

    df_valid['Numerator'] = (w_dbb * df_valid['DBB_Score'] +
                             w_buffer * df_valid['Buffer_Score'] +
                             w_cap * df_valid['Cap_Score'])
    df_valid['ECR'] = df_valid['Numerator'] / df_valid['Time_Factor']

    # =========================================================================
    # Select fund with highest ECR
    # =========================================================================

    max_ecr = df_valid['ECR'].max()
    max_ecr_funds = df_valid[df_valid['ECR'] == max_ecr].copy()

    # Tiebreaker: most recent launch
    if len(max_ecr_funds) > 1:
        return select_most_recent_launch(max_ecr_funds, current_date, series)

    return max_ecr_funds.iloc[0]['Fund']



def _select_ecr_v2_core(df_universe, current_date, series, w_dbb, w_buffer, w_cap):
    """
    Core ECR V2 implementation with parameterized weights.

    All the buffer integrity logic, time scaling, etc. is the same.
    Only difference is the weights can be adjusted.
    """

    if df_universe.empty:
        return None

    required_cols = ['Remaining Buffer', 'Original_Buffer', 'Remaining Cap',
                     'Original_Cap', 'Downside Before Buffer',
                     'Remaining Outcome Days', 'Total_Outcome_Days']

    if not all(col in df_universe.columns for col in required_cols):
        return select_most_recent_launch(df_universe, current_date, series)

    df_work = df_universe.copy()

    valid_mask = (
            df_work['Remaining Buffer'].notna() &
            df_work['Original_Buffer'].notna() &
            df_work['Remaining Cap'].notna() &
            df_work['Original_Cap'].notna() &
            df_work['Downside Before Buffer'].notna() &
            df_work['Remaining Outcome Days'].notna() &
            df_work['Total_Outcome_Days'].notna() &
            (df_work['Original_Buffer'] > 0) &
            (df_work['Original_Cap'] > 0) &
            (df_work['Remaining Outcome Days'] > 0) &
            (df_work['Total_Outcome_Days'] > 0)
    )

    df_valid = df_work[valid_mask].copy()

    if df_valid.empty:
        return select_most_recent_launch(df_universe, current_date, series)

    # Component 1: DBB Score
    df_valid['DBB_Score'] = 1 + (df_valid['Downside Before Buffer'] / 100)

    # Component 2: Buffer Integrity Score (V2 logic)
    def calculate_buffer_score(row):
        dbb = row['Downside Before Buffer']
        remaining_buffer = row['Remaining Buffer'] / 100
        original_buffer = row['Original_Buffer']

        if dbb < 0:
            distance_from_atm = abs(dbb / 100)
            max_expected_distance = original_buffer * 2

            if max_expected_distance > 0:
                buffer_score = 1.0 - (distance_from_atm / max_expected_distance)
                buffer_score = max(buffer_score, 0.0)
            else:
                buffer_score = 0.5

        elif dbb == 0:
            if remaining_buffer >= original_buffer * 0.95:
                buffer_score = 1.0
            else:
                if original_buffer > 0:
                    buffer_score = remaining_buffer / original_buffer
                    buffer_score = max(buffer_score, 0.0)
                else:
                    buffer_score = 0.0
        else:
            buffer_score = 0.0

        return buffer_score

    df_valid['Buffer_Score'] = df_valid.apply(calculate_buffer_score, axis=1)

    # Component 3: Cap Score
    df_valid['Cap_Score'] = (df_valid['Remaining Cap'] / 100) / df_valid['Original_Cap']

    # Component 4: Time Factor
    df_valid['Time_Ratio'] = df_valid['Remaining Outcome Days'] / df_valid['Total_Outcome_Days']
    df_valid['Time_Factor'] = 1 - np.log(df_valid['Time_Ratio'])
    df_valid['Time_Factor'] = df_valid['Time_Factor'].clip(lower=0.01)

    # Calculate ECR with PROVIDED weights
    df_valid['Numerator'] = (w_dbb * df_valid['DBB_Score'] +
                             w_buffer * df_valid['Buffer_Score'] +
                             w_cap * df_valid['Cap_Score'])
    df_valid['ECR'] = df_valid['Numerator'] / df_valid['Time_Factor']

    # Select fund with highest ECR
    max_ecr = df_valid['ECR'].max()
    max_ecr_funds = df_valid[df_valid['ECR'] == max_ecr].copy()

    if len(max_ecr_funds) > 1:
        return select_most_recent_launch(max_ecr_funds, current_date, series)

    return max_ecr_funds.iloc[0]['Fund']


# =============================================================================
# WRAPPER FUNCTIONS FOR EACH WEIGHT SCENARIO
# =============================================================================

def select_ecr_v2_equal(df_universe, current_date, series='F', df_regimes=None):
    """ECR V2: Equal weights (0.333, 0.333, 0.333)"""
    return _select_ecr_v2_core(df_universe, current_date, series,
                               w_dbb=0.333, w_buffer=0.333, w_cap=0.333)


def select_ecr_v2_cap_balanced(df_universe, current_date, series='F', df_regimes=None):
    """ECR V2: Cap-focused balanced (0.15, 0.35, 0.50)"""
    return _select_ecr_v2_core(df_universe, current_date, series,
                               w_dbb=0.15, w_buffer=0.35, w_cap=0.50)


def select_ecr_v2_cap_moderate(df_universe, current_date, series='F', df_regimes=None):
    """ECR V2: Cap-focused moderate (0.25, 0.25, 0.50)"""
    return _select_ecr_v2_core(df_universe, current_date, series,
                               w_dbb=0.25, w_buffer=0.25, w_cap=0.50)


def select_ecr_v2_cap_dominant(df_universe, current_date, series='F', df_regimes=None):
    """ECR V2: Cap-dominant (0.15, 0.15, 0.70)"""
    return _select_ecr_v2_core(df_universe, current_date, series,
                               w_dbb=0.15, w_buffer=0.15, w_cap=0.70)


def select_ecr_v2_protection(df_universe, current_date, series='F', df_regimes=None):
    """ECR V2: Protection-focused (0.40, 0.40, 0.20)"""
    return _select_ecr_v2_core(df_universe, current_date, series,
                               w_dbb=0.40, w_buffer=0.40, w_cap=0.20)



def _select_ecr_v2_normalized_core(df_universe, current_date, series, w_dbb, w_buffer, w_cap):
    """
    Core ECR V2 with MinMaxScaler normalization for fair weighting.

    Process:
    1. Calculate raw scores for all components
    2. Normalize each component to [0, 1] using MinMaxScaler
    3. Apply weights to normalized scores
    4. Divide by time factor
    """

    if df_universe.empty:
        return None

    required_cols = ['Remaining Buffer', 'Original_Buffer', 'Remaining Cap',
                     'Original_Cap', 'Downside Before Buffer',
                     'Remaining Outcome Days', 'Total_Outcome_Days']

    if not all(col in df_universe.columns for col in required_cols):
        return select_most_recent_launch(df_universe, current_date, series)

    df_work = df_universe.copy()

    valid_mask = (
            df_work['Remaining Buffer'].notna() &
            df_work['Original_Buffer'].notna() &
            df_work['Remaining Cap'].notna() &
            df_work['Original_Cap'].notna() &
            df_work['Downside Before Buffer'].notna() &
            df_work['Remaining Outcome Days'].notna() &
            df_work['Total_Outcome_Days'].notna() &
            (df_work['Original_Buffer'] > 0) &
            (df_work['Original_Cap'] > 0) &
            (df_work['Remaining Outcome Days'] > 0) &
            (df_work['Total_Outcome_Days'] > 0)
    )

    df_valid = df_work[valid_mask].copy()

    if df_valid.empty:
        return select_most_recent_launch(df_universe, current_date, series)

    # =========================================================================
    # STEP 1: Calculate Raw Scores
    # =========================================================================

    # Component 1: DBB Score (NO /100 - data already decimal!)
    df_valid['DBB_Score_Raw'] = 1 + df_valid['Downside Before Buffer']

    # Component 2: Buffer Integrity Score (V2 logic)
    def calculate_buffer_score(row):
        dbb = row['Downside Before Buffer']  # Already decimal
        remaining_buffer = row['Remaining Buffer']  # Already decimal
        original_buffer = row['Original_Buffer']  # Already decimal

        if dbb < 0:
            # Market rallied - penalize distance from ATM
            distance_from_atm = abs(dbb)  # NO /100 - already decimal!
            max_expected_distance = original_buffer * 2

            if max_expected_distance > 0:
                buffer_score = 1.0 - (distance_from_atm / max_expected_distance)
                buffer_score = max(buffer_score, 0.0)
            else:
                buffer_score = 0.5

        elif dbb == 0:
            # At ATM or in buffer
            if remaining_buffer >= original_buffer * 0.95:
                buffer_score = 1.0  # Fresh fund
            else:
                if original_buffer > 0:
                    buffer_score = remaining_buffer / original_buffer
                    buffer_score = max(buffer_score, 0.0)
                else:
                    buffer_score = 0.0
        else:
            buffer_score = 0.0

        return buffer_score

    df_valid['Buffer_Score_Raw'] = df_valid.apply(calculate_buffer_score, axis=1)

    # Component 3: Cap Score (NO /100 - data already decimal!)
    df_valid['Cap_Score_Raw'] = df_valid['Remaining Cap'] / df_valid['Original_Cap']

    # Component 4: Time Factor
    df_valid['Time_Ratio'] = df_valid['Remaining Outcome Days'] / df_valid['Total_Outcome_Days']
    df_valid['Time_Factor'] = 1 - np.log(df_valid['Time_Ratio'])
    df_valid['Time_Factor'] = df_valid['Time_Factor'].clip(lower=0.01)

    # =========================================================================
    # STEP 2: Normalize Each Component to [0, 1]
    # =========================================================================

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Normalize DBB Score
    dbb_values = df_valid[['DBB_Score_Raw']].values
    df_valid['DBB_Score_Normalized'] = scaler.fit_transform(dbb_values)

    # Normalize Buffer Score
    buffer_values = df_valid[['Buffer_Score_Raw']].values
    df_valid['Buffer_Score_Normalized'] = scaler.fit_transform(buffer_values)

    # Normalize Cap Score
    cap_values = df_valid[['Cap_Score_Raw']].values
    df_valid['Cap_Score_Normalized'] = scaler.fit_transform(cap_values)

    # =========================================================================
    # STEP 3: Apply Weights to NORMALIZED Scores
    # =========================================================================

    df_valid['Numerator'] = (w_dbb * df_valid['DBB_Score_Normalized'] +
                             w_buffer * df_valid['Buffer_Score_Normalized'] +
                             w_cap * df_valid['Cap_Score_Normalized'])

    df_valid['ECR'] = df_valid['Numerator'] / df_valid['Time_Factor']

    # =========================================================================
    # STEP 4: Select Fund with Highest ECR
    # =========================================================================

    max_ecr = df_valid['ECR'].max()
    max_ecr_funds = df_valid[df_valid['ECR'] == max_ecr].copy()

    if len(max_ecr_funds) > 1:
        return select_most_recent_launch(max_ecr_funds, current_date, series)

    return max_ecr_funds.iloc[0]['Fund']


def select_ecr_v2_equal_normalized(df_universe, current_date, series='F', df_regimes=None):
    """ECR V2 Normalized: Equal weights (0.333, 0.333, 0.333)"""
    return _select_ecr_v2_normalized_core(df_universe, current_date, series,
                                          w_dbb=0.333, w_buffer=0.333, w_cap=0.333)


def select_ecr_v2_cap_balanced_normalized(df_universe, current_date, series='F', df_regimes=None):
    """ECR V2 Normalized: Cap-focused balanced (0.15, 0.35, 0.50)"""
    return _select_ecr_v2_normalized_core(df_universe, current_date, series,
                                          w_dbb=0.15, w_buffer=0.35, w_cap=0.50)


def select_ecr_v2_cap_moderate_normalized(df_universe, current_date, series='F', df_regimes=None):
    """ECR V2 Normalized: Cap-focused moderate (0.25, 0.25, 0.50)"""
    return _select_ecr_v2_normalized_core(df_universe, current_date, series,
                                          w_dbb=0.25, w_buffer=0.25, w_cap=0.50)


def select_ecr_v2_cap_dominant_normalized(df_universe, current_date, series='F', df_regimes=None):
    """ECR V2 Normalized: Cap-dominant (0.15, 0.15, 0.70)"""
    return _select_ecr_v2_normalized_core(df_universe, current_date, series,
                                          w_dbb=0.15, w_buffer=0.15, w_cap=0.70)


def select_ecr_v2_protection_normalized(df_universe, current_date, series='F', df_regimes=None):
    """ECR V2 Normalized: Protection-focused (0.40, 0.40, 0.20)"""
    return _select_ecr_v2_normalized_core(df_universe, current_date, series,
                                          w_dbb=0.40, w_buffer=0.40, w_cap=0.20)


def select_highest_new_ecr_composite_111(df_snapshot, current_date, series, df_regimes=None, tracker=None, month=None, **kwargs):
    """Equal weights (1,1,1)"""
    return _select_highest_new_ecr_core(df_snapshot, current_date, series, 1, 1, 1, tracker=tracker, month=month)

def select_highest_new_ecr_composite_211(df_snapshot, current_date, series, df_regimes=None, tracker=None, month=None, **kwargs):
    """Emphasize DBB (2,1,1)"""
    return _select_highest_new_ecr_core(df_snapshot, current_date, series, 2, 1, 1, tracker=tracker, month=month)

def select_highest_new_ecr_composite_121(df_snapshot, current_date, series, df_regimes=None, tracker=None, month=None, **kwargs):
    """Emphasize Buffer Integrity (1,2,1)"""
    return _select_highest_new_ecr_core(df_snapshot, current_date, series, 1, 2, 1, tracker=tracker, month=month)

def select_highest_new_ecr_composite_112(df_snapshot, current_date, series, df_regimes=None, tracker=None, month=None, **kwargs):
    """Emphasize Cap Integrity (1,1,2)"""
    return _select_highest_new_ecr_core(df_snapshot, current_date, series, 1, 1, 2, tracker=tracker, month=month)

def select_highest_new_ecr_composite_101(df_snapshot, current_date, series, df_regimes=None, tracker=None, month=None, **kwargs):
    """DBB + Cap only (1,0,1)"""
    return _select_highest_new_ecr_core(df_snapshot, current_date, series, 1, 0, 1, tracker=tracker, month=month)

def select_highest_new_ecr_composite_221(df_snapshot, current_date, series, df_regimes=None, tracker=None, month=None, **kwargs):
    """Equal emphasis on buffer metrics (2,2,1)"""
    return _select_highest_new_ecr_core(df_snapshot, current_date, series, 2, 2, 1, tracker=tracker, month=month)

def select_highest_new_ecr_composite_110(df_snapshot, current_date, series, df_regimes=None, tracker=None, month=None, **kwargs):
    """Buffer-focused, ignore cap (1,1,0)"""
    return _select_highest_new_ecr_core(df_snapshot, current_date, series, 1, 1, 0, tracker=tracker, month=month)

def select_ecr_v2_regime_adaptive_new_methodology(df_universe, current_date, series='F', df_regimes=None):
    """
    Regime-adaptive wrapper for NEW ECR methodology.

    Switches weights based on current regime:
    - Bullish (1):  15/15/70 (cap-focused)
    - Neutral (0):  25/25/50 (balanced)
    - Bearish (-1): 40/40/20 (protection-focused)

    Uses the NEW ECR composite score calculation from _select_highest_new_ecr_core.

    Parameters
    ----------
    df_universe : DataFrame
        Available funds at current date
    current_date : datetime
        Current backtest date
    series : str
        Fund series (default 'F')
    df_regimes : DataFrame
        Regime classifications with Date and Regimes columns

    Returns
    -------
    str or None
        Selected fund identifier
    """
    if df_regimes is None:
        # Fallback to neutral weights if no regime data
        w_dbb, w_buffer, w_cap = 0.25, 0.25, 0.50
    else:
        # Get current regime
        regime_row = df_regimes[df_regimes['Date'] == current_date]

        if regime_row.empty:
            # No regime data for this date, use neutral
            w_dbb, w_buffer, w_cap = 0.25, 0.25, 0.50
        else:
            regime = regime_row.iloc[0]['Regimes']

            if regime == 1:  # Bullish
                w_dbb, w_buffer, w_cap = 0.15, 0.15, 0.70
            elif regime == -1:  # Bearish
                w_dbb, w_buffer, w_cap = 0.40, 0.40, 0.20
            else:  # Neutral (0)
                w_dbb, w_buffer, w_cap = 0.25, 0.25, 0.50

    # Call core function with unpacked weights as positional arguments
    return _select_highest_new_ecr_core(df_universe, current_date, series, w_dbb, w_buffer, w_cap)





def _select_highest_new_ecr_core(df_snapshot, current_date, series, w_dbb, w_buffer, w_cap, tracker=None, month=None):
    """Core ECR selection with optional tracking."""

    # Your existing debug...
    available = df_snapshot[df_snapshot['Fund'].str.startswith(series)].copy()

    if available.empty:
        return None

    # NEW: Debug the scoring loop
    fund_scores = []

    #print(f"      Starting scoring loop for {len(available)} funds...")

    for idx, (_, row) in enumerate(available.iterrows()):
        fund_name = row['Fund']
        #print(f"        Fund {idx + 1}/{len(available)}: {fund_name}")
        cap_util = row.get('Cap_Utilization', np.nan)
        cap_integrity_numerator = row.get('Remaining Cap Net', np.nan)
        cap_integrity_denominator = row.get('Original Cap Net', np.nan)
        dbb_pct = row.get('Downside Before Buffer Net', np.nan)
        remaining_buffer = row.get('Remaining Buffer Net', np.nan)
        original_buffer = row.get('Original Buffer Net', np.nan)
        remaining_days = row.get('Remaining Outcome Days', np.nan)

        #print(f"          cap_util={cap_util}, dbb_pct={dbb_pct}, rem_buf={remaining_buffer}, orig_buf={original_buffer}, days={remaining_days}")

        # Component scores
        dbb_score = np.exp(dbb_pct * 5) if pd.notna(dbb_pct) else 0

        if pd.notna(remaining_buffer) and pd.notna(original_buffer) and pd.notna(dbb_pct):
            buffer_integrity = max(0, (remaining_buffer + dbb_pct)) / original_buffer if original_buffer > 0 else 0
        else:
            buffer_integrity = 0

        cap_integrity = min(cap_integrity_numerator / cap_integrity_denominator, 1.0) if pd.notna(cap_integrity_numerator) else 0
        time_scaling = 1 - np.log(remaining_days / 365) if pd.notna(remaining_days) and remaining_days > 0 else 0
        composite = (w_dbb * dbb_score + w_buffer * buffer_integrity + w_cap * cap_integrity) / time_scaling

        #print(f"          Composite score: {composite}")

        fund_scores.append({
            'fund': fund_name,
            'composite_score': composite,
            'dbb_score': dbb_score,
            'buffer_integrity': buffer_integrity,
            'cap_integrity': cap_integrity,
            'time_scaling': time_scaling,
            'cap_utilization': cap_util,
            'dbb_pct': dbb_pct,
            'remaining_buffer': remaining_buffer,
            'original_buffer': original_buffer,
            'remaining_days': remaining_days,
            'total_days': 365,
        })

    #print(f"      Total scores calculated: {len(fund_scores)}")

    if not fund_scores:
        print(f"      ❌ No fund_scores - RETURNING NONE")
        return None

    selected = max(fund_scores, key=lambda x: x['composite_score'])
    selected_fund = selected['fund']

    #print(f"      ✅ Selected: {selected_fund} with score {selected['composite_score']}")

    # ADD THIS SECTION:
    if tracker is not None and month is not None:
        weight_code = (str(w_dbb), str(w_buffer), str(w_cap))
        weight_desc = f"Weights ({w_dbb},{w_buffer},{w_cap})"
        tracker.record_rebalance(current_date, month, (weight_code, weight_desc), fund_scores, selected_fund)

    return selected_fund


def _select_ecr_par_prox_core(df_snapshot, current_date, series,
                              w_dbb, w_buffer, w_cap, w_par, time_shape,
                              tracker=None, month=None):
    """
    ECR par-proximity selection core.

    Composite formula:
        composite = w_dbb * dbb_score
                  + w_buffer * buffer_integrity
                  + w_cap * cap_integrity
                  + w_par * par_proximity * time_elapsed

    NO time_scaling division — that responsibility moves to par_proximity × time_elapsed.

    Parameters:
        df_snapshot: DataFrame of available funds on current_date
        current_date: Current backtest date
        series: Fund series prefix ('F')
        w_dbb, w_buffer, w_cap: Weights for the three v1 components (locked at 1,1,1 for Batch 11)
        w_par: Weight on the par bonus
        time_shape: 'linear', 'sqrt', or 'sq'
        tracker: Optional RebalanceScoringTracker
        month: Optional launch month string for tracker

    Returns:
        Selected fund ticker, or None if no scorable funds
    """
    available = df_snapshot[df_snapshot['Fund'].str.startswith(series)].copy()

    if available.empty:
        return None

    fund_scores = []

    for _, row in available.iterrows():
        fund_name = row['Fund']

        dbb_net = row.get('Downside Before Buffer Net', np.nan)
        remaining_buffer = row.get('Remaining Buffer Net', np.nan)
        original_buffer = row.get('Original Buffer Net', np.nan)
        remaining_cap_net = row.get('Remaining Cap Net', np.nan)
        original_cap_net = row.get('Original Cap Net', np.nan)
        remaining_days = row.get('Remaining Outcome Days', np.nan)
        par_proximity = row.get('par_proximity', np.nan)

        # Component 1: DBB Score
        dbb_score = np.exp(dbb_net * 5) if pd.notna(dbb_net) else 0

        # Component 2: Buffer Integrity
        if pd.notna(remaining_buffer) and pd.notna(original_buffer) and pd.notna(dbb_net):
            buffer_integrity = max(0, (remaining_buffer + dbb_net)) / original_buffer \
                if original_buffer > 0 else 0
        else:
            buffer_integrity = 0

        # Component 3: Cap Integrity
        if pd.notna(remaining_cap_net) and pd.notna(original_cap_net) and original_cap_net > 0:
            cap_integrity = min(remaining_cap_net / original_cap_net, 1.0)
        else:
            cap_integrity = 0

        # Component 4: Par bonus
        time_elapsed = _compute_time_elapsed(remaining_days, time_shape)
        par_prox_val = par_proximity if pd.notna(par_proximity) else 0
        par_bonus = w_par * par_prox_val * time_elapsed

        # Composite — no time_scaling division
        composite = (w_dbb * dbb_score
                     + w_buffer * buffer_integrity
                     + w_cap * cap_integrity
                     + par_bonus)

        fund_scores.append({
            'fund': fund_name,
            'composite_score': composite,
            'dbb_score': dbb_score,
            'buffer_integrity': buffer_integrity,
            'cap_integrity': cap_integrity,
            'time_scaling': np.nan,  # not used in par_prox composite
            'par_proximity': par_prox_val,
            'time_elapsed': time_elapsed,
            'par_bonus': par_bonus,
            'cap_utilization': row.get('Cap_Utilization', np.nan),
            'dbb_pct': dbb_net,
            'remaining_buffer': remaining_buffer,
            'original_buffer': original_buffer,
            'remaining_days': remaining_days,
            'total_days': 365,
        })

    if not fund_scores:
        return None

    selected = max(fund_scores, key=lambda x: x['composite_score'])
    selected_fund = selected['fund']

    if tracker is not None and month is not None:
        weight_code = (str(w_dbb), str(w_buffer), str(w_cap))
        weight_desc = f"Weights ({w_dbb},{w_buffer},{w_cap}) par={w_par} shape={time_shape}"
        tracker.record_rebalance(current_date, month, (weight_code, weight_desc),
                                 fund_scores, selected_fund)

    return selected_fund


def compute_ecr_scores(df_universe, series, w_dbb, w_buffer, w_cap):
    """
    Compute ECR composite scores for all funds in universe.

    Used by triggers to evaluate the current fund's score and decide
    whether to fire. Pure weighted sum of the three v1 components —
    no time_scaling divisor (that role moves to par_proximity in selection).

    Returns:
        dict: {fund_name: composite_score}
    """
    available = df_universe[df_universe['Fund'].str.startswith(series)].copy()

    if available.empty:
        return {}

    scores = {}

    for _, row in available.iterrows():
        fund_name = row['Fund']

        dbb_net = row.get('Downside Before Buffer Net', np.nan)
        remaining_buffer = row.get('Remaining Buffer Net', np.nan)
        original_buffer = row.get('Original Buffer Net', np.nan)
        remaining_cap_net = row.get('Remaining Cap Net', np.nan)
        original_cap_net = row.get('Original Cap Net', np.nan)

        # DBB Score
        dbb_score = np.exp(dbb_net * 5) if pd.notna(dbb_net) else 0

        # Buffer Integrity
        if pd.notna(remaining_buffer) and pd.notna(original_buffer) and pd.notna(dbb_net):
            buffer_integrity = max(0, (remaining_buffer + dbb_net)) / original_buffer \
                if original_buffer > 0 else 0
        else:
            buffer_integrity = 0

        # Cap Integrity
        if pd.notna(remaining_cap_net) and pd.notna(original_cap_net) and original_cap_net > 0:
            cap_integrity = min(remaining_cap_net / original_cap_net, 1.0)
        else:
            cap_integrity = 0

        # Composite — pure weighted sum, no time scaling
        composite = w_dbb * dbb_score + w_buffer * buffer_integrity + w_cap * cap_integrity

        scores[fund_name] = composite

    return scores

def _compute_time_elapsed(remaining_days, shape):
    """
    Compute time-elapsed factor for the par_proximity bonus.

    All three shapes are 0 when remaining_days = 365 (fund just rolled)
    and 1 when remaining_days = 0 (fund at maturity).

    Parameters:
        remaining_days: Days remaining in outcome period (float or array)
        shape: 'linear', 'sqrt', or 'sq'

    Returns:
        Float in [0, 1]
    """
    if pd.isna(remaining_days) or remaining_days <= 0:
        return 1.0

    base = max(0.0, 1.0 - remaining_days / 365.0)

    if shape == 'linear':
        return base
    elif shape == 'sqrt':
        return np.sqrt(base)
    elif shape == 'sq':
        return base ** 2
    else:
        raise ValueError(f"Unknown time_shape: {shape}. Use 'linear', 'sqrt', or 'sq'.")


# =============================================================================
# Batch 11: Par Proximity Selection Wrappers
# =============================================================================
# Locked weights: w_dbb=1, w_buffer=1, w_cap=1
# Varied parameters: w_par × time_shape

def select_ecr_par_prox_w0p5_lin(df_snapshot, current_date, series='F',
                                  df_regimes=None, tracker=None, month=None, **kwargs):
    return _select_ecr_par_prox_core(df_snapshot, current_date, series,
                                     w_dbb=1, w_buffer=1, w_cap=1,
                                     w_par=0.5, time_shape='linear',
                                     tracker=tracker, month=month)

def select_ecr_par_prox_w0p5_sqrt(df_snapshot, current_date, series='F',
                                   df_regimes=None, tracker=None, month=None, **kwargs):
    return _select_ecr_par_prox_core(df_snapshot, current_date, series,
                                     w_dbb=1, w_buffer=1, w_cap=1,
                                     w_par=0.5, time_shape='sqrt',
                                     tracker=tracker, month=month)

def select_ecr_par_prox_w0p5_sq(df_snapshot, current_date, series='F',
                                 df_regimes=None, tracker=None, month=None, **kwargs):
    return _select_ecr_par_prox_core(df_snapshot, current_date, series,
                                     w_dbb=1, w_buffer=1, w_cap=1,
                                     w_par=0.5, time_shape='sq',
                                     tracker=tracker, month=month)

def select_ecr_par_prox_w1_lin(df_snapshot, current_date, series='F',
                                df_regimes=None, tracker=None, month=None, **kwargs):
    return _select_ecr_par_prox_core(df_snapshot, current_date, series,
                                     w_dbb=1, w_buffer=1, w_cap=1,
                                     w_par=1.0, time_shape='linear',
                                     tracker=tracker, month=month)

def select_ecr_par_prox_w1_sqrt(df_snapshot, current_date, series='F',
                                 df_regimes=None, tracker=None, month=None, **kwargs):
    return _select_ecr_par_prox_core(df_snapshot, current_date, series,
                                     w_dbb=1, w_buffer=1, w_cap=1,
                                     w_par=1.0, time_shape='sqrt',
                                     tracker=tracker, month=month)

def select_ecr_par_prox_w1_sq(df_snapshot, current_date, series='F',
                               df_regimes=None, tracker=None, month=None, **kwargs):
    return _select_ecr_par_prox_core(df_snapshot, current_date, series,
                                     w_dbb=1, w_buffer=1, w_cap=1,
                                     w_par=1.0, time_shape='sq',
                                     tracker=tracker, month=month)

def select_ecr_par_prox_w1p5_lin(df_snapshot, current_date, series='F',
                                  df_regimes=None, tracker=None, month=None, **kwargs):
    return _select_ecr_par_prox_core(df_snapshot, current_date, series,
                                     w_dbb=1, w_buffer=1, w_cap=1,
                                     w_par=1.5, time_shape='linear',
                                     tracker=tracker, month=month)

def select_ecr_par_prox_w1p5_sqrt(df_snapshot, current_date, series='F',
                                   df_regimes=None, tracker=None, month=None, **kwargs):
    return _select_ecr_par_prox_core(df_snapshot, current_date, series,
                                     w_dbb=1, w_buffer=1, w_cap=1,
                                     w_par=1.5, time_shape='sqrt',
                                     tracker=tracker, month=month)

def select_ecr_par_prox_w1p5_sq(df_snapshot, current_date, series='F',
                                 df_regimes=None, tracker=None, month=None, **kwargs):
    return _select_ecr_par_prox_core(df_snapshot, current_date, series,
                                     w_dbb=1, w_buffer=1, w_cap=1,
                                     w_par=1.5, time_shape='sq',
                                     tracker=tracker, month=month)

def select_ecr_par_prox_w2_lin(df_snapshot, current_date, series='F',
                                df_regimes=None, tracker=None, month=None, **kwargs):
    return _select_ecr_par_prox_core(df_snapshot, current_date, series,
                                     w_dbb=1, w_buffer=1, w_cap=1,
                                     w_par=2.0, time_shape='linear',
                                     tracker=tracker, month=month)

def select_ecr_par_prox_w2_sqrt(df_snapshot, current_date, series='F',
                                 df_regimes=None, tracker=None, month=None, **kwargs):
    return _select_ecr_par_prox_core(df_snapshot, current_date, series,
                                     w_dbb=1, w_buffer=1, w_cap=1,
                                     w_par=2.0, time_shape='sqrt',
                                     tracker=tracker, month=month)

def select_ecr_par_prox_w2_sq(df_snapshot, current_date, series='F',
                               df_regimes=None, tracker=None, month=None, **kwargs):
    return _select_ecr_par_prox_core(df_snapshot, current_date, series,
                                     w_dbb=1, w_buffer=1, w_cap=1,
                                     w_par=2.0, time_shape='sq',
                                     tracker=tracker, month=month)

# =============================================================================
# Batch 11: w_par = 2.5 variants
# =============================================================================

def select_ecr_par_prox_w2p5_lin(df_snapshot, current_date, series,
                                  df_regimes=None, tracker=None, month=None):
    """ECR par-proximity, w_par=2.5, time_shape=linear."""
    return _select_ecr_par_prox_core(
        df_snapshot, current_date, series,
        w_dbb=1, w_buffer=1, w_cap=1,
        w_par=2.5, time_shape='linear',
        tracker=tracker, month=month,
    )


def select_ecr_par_prox_w2p5_sqrt(df_snapshot, current_date, series,
                                   df_regimes=None, tracker=None, month=None):
    """ECR par-proximity, w_par=2.5, time_shape=sqrt."""
    return _select_ecr_par_prox_core(
        df_snapshot, current_date, series,
        w_dbb=1, w_buffer=1, w_cap=1,
        w_par=2.5, time_shape='sqrt',
        tracker=tracker, month=month,
    )


def select_ecr_par_prox_w2p5_sq(df_snapshot, current_date, series,
                                 df_regimes=None, tracker=None, month=None):
    """ECR par-proximity, w_par=2.5, time_shape=sq."""
    return _select_ecr_par_prox_core(
        df_snapshot, current_date, series,
        w_dbb=1, w_buffer=1, w_cap=1,
        w_par=2.5, time_shape='sq',
        tracker=tracker, month=month,
    )

# Selection registry for dynamic lookup
SELECTION_REGISTRY = {
    'select_most_recent_launch': select_most_recent_launch,
    'select_remaining_cap': select_remaining_cap,
    'select_cap_utilization': select_cap_utilization,
    'select_remaining_cap_highest': select_remaining_cap_highest,
    'select_remaining_cap_lowest': select_remaining_cap_lowest,
    'select_cap_utilization_lowest': select_cap_utilization_lowest,
    'select_cap_utilization_highest': select_cap_utilization_highest,
    'select_downside_buffer_highest': select_downside_buffer_highest,
    'select_downside_buffer_lowest': select_downside_buffer_lowest,
    'select_highest_outcome_and_cap': select_highest_outcome_and_cap,
    'select_cost_analysis': select_cost_analysis,
    'select_remaining_buffer_lowest': select_remaining_buffer_lowest,
    'select_enhanced_cost_ratio_bullish': select_enhanced_cost_ratio_bullish,
    'select_enhanced_cost_ratio_bearish': select_enhanced_cost_ratio_bearish,
    'select_enhanced_cost_ratio_neutral': select_enhanced_cost_ratio_neutral,

    # Batch 6B: Extreme Weight Variants
    'select_ecr_pure_cap': select_ecr_pure_cap,
    'select_ecr_pure_buffer': select_ecr_pure_buffer,
    'select_ecr_pure_dbb': select_ecr_pure_dbb,
    'select_ecr_ultra_cap': select_ecr_ultra_cap,
    'select_ecr_ultra_protection': select_ecr_ultra_protection,
    'select_ecr_dbb_dominant': select_ecr_dbb_dominant,
    'select_ecr_buffer_dominant': select_ecr_buffer_dominant,
    'select_ecr_no_cap': select_ecr_no_cap,
    'select_ecr_no_buffer': select_ecr_no_buffer,

    # Batch 6C: Component Mechanics
    'select_ecr_buffer_scale_full': select_ecr_buffer_scale_full,
    'select_ecr_buffer_scale_low_floor': select_ecr_buffer_scale_low_floor,
    'select_ecr_buffer_scale_default': select_ecr_buffer_scale_default,
    'select_ecr_buffer_scale_high_floor': select_ecr_buffer_scale_high_floor,
    'select_ecr_time_log': select_ecr_time_log,
    'select_ecr_time_linear': select_ecr_time_linear,
    'select_ecr_time_sqrt': select_ecr_time_sqrt,
    'select_ecr_time_inverse': select_ecr_time_inverse,
    'select_ecr_time_none': select_ecr_time_none,
    'select_ecr_dbb_raw': select_ecr_dbb_raw,
    'select_ecr_dbb_scaled': select_ecr_dbb_scaled,

    'select_enhanced_cost_ratio_neutral_v2': select_enhanced_cost_ratio_neutral_v2,

    'select_ecr_v2_equal': select_ecr_v2_equal,
    'select_ecr_v2_cap_balanced': select_ecr_v2_cap_balanced,
    'select_ecr_v2_cap_moderate': select_ecr_v2_cap_moderate,
    'select_ecr_v2_cap_dominant': select_ecr_v2_cap_dominant,
    'select_ecr_v2_protection': select_ecr_v2_protection,

    # Add these 5 new normalized ECR functions:
    'select_ecr_v2_equal_normalized': select_ecr_v2_equal_normalized,
    'select_ecr_v2_cap_balanced_normalized': select_ecr_v2_cap_balanced_normalized,
    'select_ecr_v2_cap_moderate_normalized': select_ecr_v2_cap_moderate_normalized,
    'select_ecr_v2_cap_dominant_normalized': select_ecr_v2_cap_dominant_normalized,
    'select_ecr_v2_protection_normalized': select_ecr_v2_protection_normalized,

    'select_ecr_v2_regime_adaptive_normalized': select_ecr_v2_regime_adaptive_normalized,

    'select_highest_new_ecr_composite_111': select_highest_new_ecr_composite_111,
    'select_highest_new_ecr_composite_211': select_highest_new_ecr_composite_211,
    'select_highest_new_ecr_composite_121': select_highest_new_ecr_composite_121,
    'select_highest_new_ecr_composite_112': select_highest_new_ecr_composite_112,
    'select_highest_new_ecr_composite_101': select_highest_new_ecr_composite_101,
    'select_highest_new_ecr_composite_221': select_highest_new_ecr_composite_221,
    'select_highest_new_ecr_composite_110': select_highest_new_ecr_composite_110,

    # Batch 11: Par Proximity Selection Functions
    'select_ecr_par_prox_w0p5_lin': select_ecr_par_prox_w0p5_lin,
    'select_ecr_par_prox_w0p5_sqrt': select_ecr_par_prox_w0p5_sqrt,
    'select_ecr_par_prox_w0p5_sq': select_ecr_par_prox_w0p5_sq,
    'select_ecr_par_prox_w1_lin': select_ecr_par_prox_w1_lin,
    'select_ecr_par_prox_w1_sqrt': select_ecr_par_prox_w1_sqrt,
    'select_ecr_par_prox_w1_sq': select_ecr_par_prox_w1_sq,
    'select_ecr_par_prox_w1p5_lin': select_ecr_par_prox_w1p5_lin,
    'select_ecr_par_prox_w1p5_sqrt': select_ecr_par_prox_w1p5_sqrt,
    'select_ecr_par_prox_w1p5_sq': select_ecr_par_prox_w1p5_sq,
    'select_ecr_par_prox_w2_lin': select_ecr_par_prox_w2_lin,
    'select_ecr_par_prox_w2_sqrt': select_ecr_par_prox_w2_sqrt,
    'select_ecr_par_prox_w2_sq': select_ecr_par_prox_w2_sq,
    'select_ecr_par_prox_w2p5_lin':  select_ecr_par_prox_w2p5_lin,
    'select_ecr_par_prox_w2p5_sqrt': select_ecr_par_prox_w2p5_sqrt,
    'select_ecr_par_prox_w2p5_sq':   select_ecr_par_prox_w2p5_sq,
}


def get_selection_function(selection_name):
    """
    Get selection function by name.

    Parameters:
      selection_name: String name of selection function

    Returns:
      Function reference
    """
    if selection_name not in SELECTION_REGISTRY:
        raise ValueError(f"Unknown selection function: {selection_name}")

    return SELECTION_REGISTRY[selection_name]