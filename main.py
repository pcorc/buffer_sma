"""
Batch Test Runner for Buffer ETF Rotation Strategies

Clean, modular implementation with batch configs and regime-adaptive logic
separated into dedicated modules.

Usage:
    1. Set BATCH_NUMBER below (0-7)
    2. Run: python run_batch_tests.py
    3. Review results in Excel output

Module structure:
    - config/batch_configs.py: All batch configuration functions
    - backtesting/regime_adaptive.py: Regime-adaptive ECR logic
    - run_batch_tests.py: Main execution logic only
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Core imports
from config import settings
from config.batch_configs import BATCH_CONFIGS, BATCH_DESCRIPTIONS
from backtesting.data_pipeline import load_and_preprocess_all_data
from backtesting.batch_runner import run_all_single_ticker_tests
from analysis.consolidator import consolidate_results
from utils.exporters import export_main_consolidated_workbook, extract_and_export_daily_nav
from utils.date_utils import get_first_trading_day_of_month
from visualization.performance_plots import generate_batch_visualizations

BATCH_NUMBER = '11w2p5'  # Change this to run different batches (0-7)


def main():
    """Run the selected batch."""

    if BATCH_NUMBER in BATCH_CONFIGS:
        configs = BATCH_CONFIGS[BATCH_NUMBER]()
        batch_desc = BATCH_DESCRIPTIONS[BATCH_NUMBER]
        df_regimes_loaded = None
    else:
        print(f"❌ Invalid BATCH_NUMBER: {BATCH_NUMBER}")
        return


    # Calculate simulations
    total_sims = sum(len(config['launch_months']) for config in configs)
    start_time = datetime.now()

    # Load data
    print("Loading data...")
    df_enriched, df_benchmarks, roll_dates_dict = load_and_preprocess_all_data(
        fund_file=settings.DATA_FILE,
        benchmark_file=settings.BENCHMARK_FILE,
        roll_dates_file=settings.ROLL_DATES_FILE,
        series=settings.SERIES,
        sigma=0.05,  # Par proximity Gaussian width
    )

    # # Apply start date filter if specified
    # if hasattr(settings, 'COMMON_START_DATE') and settings.COMMON_START_DATE:
    #     target_date = pd.to_datetime(settings.COMMON_START_DATE)
    #     actual_start_date = get_first_trading_day_of_month(
    #         df_enriched,
    #         target_date.year,
    #         target_date.month
    #     )
    #
    #     if actual_start_date:
    #         df_enriched = df_enriched[df_enriched['Date'] >= actual_start_date].copy()
    #         df_benchmarks = df_benchmarks[df_benchmarks['Date'] >= actual_start_date].copy()

    # Regime classification - disabled for Batch 10, re-enable if needed
    # df_spy_for_regime = df_benchmarks[['Date', 'SPY']].copy()
    # df_spy_for_regime.rename(columns={'SPY': 'Ref_Index'}, inplace=True)
    # df_regimes = classify_market_regimes(
    #     df_spy_for_regime,
    #     window_months=settings.REGIME_WINDOW_MONTHS,
    #     bull_threshold=settings.REGIME_BULL_THRESHOLD,
    #     bear_threshold=settings.REGIME_BEAR_THRESHOLD
    # )
    # df_forward_regimes = classify_forward_regimes(
    #     df_spy_for_regime,
    #     window_3m_days=63,
    #     window_6m_days=126,
    #     bull_threshold=settings.REGIME_BULL_THRESHOLD,
    #     bear_threshold=settings.REGIME_BEAR_THRESHOLD
    # )
    df_regimes = None
    df_forward_regimes = None

    # Run backtests
    print(f"\n{'=' * 80}")
    print(f"RUNNING BATCH {BATCH_NUMBER} BACKTESTS")
    print(f"{'=' * 80}\n")

    results_list = run_all_single_ticker_tests(
        df_enriched=df_enriched,
        df_benchmarks=df_benchmarks,
        roll_dates_dict=roll_dates_dict,
        trigger_selection_combos=configs,
        series=settings.SERIES
    )

    if not results_list:
        print("❌ No results generated")
        return

    # Analyze
    summary_df = consolidate_results(results_list)

    # future_regime_df = analyze_by_future_regime(
    #     results_list,
    #     df_forward_regimes,
    #     entry_frequency='quarterly'
    # )
    #
    # if not future_regime_df.empty:
    #     optimal_3m = summarize_optimal_strategies(future_regime_df, horizon='3M', top_n=10)
    #     optimal_6m = summarize_optimal_strategies(future_regime_df, horizon='6M', top_n=10)
    # else:
    #     optimal_3m = {}
    #     optimal_6m = {}

    future_regime_df = None
    optimal_3m = {}
    optimal_6m = {}

    # Create output directory
    output_dir = os.path.join(settings.RESULTS_DIR, f'batch_{BATCH_NUMBER}')
    os.makedirs(output_dir, exist_ok=True)

    # Extract daily time series
    extract_and_export_daily_nav(results_list, output_dir, BATCH_NUMBER, df_benchmarks)

    # Export to Excel
    workbook_path = export_main_consolidated_workbook(
        results_list=results_list,
        summary_df=summary_df,
        output_dir=output_dir,
        run_name=f'batch{BATCH_NUMBER}_{batch_desc.replace(" ", "_")}',
        df_forward_regimes=None if BATCH_NUMBER in [7, 9, 10] else df_forward_regimes,  # Skip for Batch 7
        future_regime_df=None if BATCH_NUMBER in [7, 9, 10] else future_regime_df,
        optimal_6m=None if BATCH_NUMBER in [7, 9, 10] else optimal_6m,
        optimal_3m=None if BATCH_NUMBER in [7, 9, 10] else optimal_3m,
        intent_vs_regime_6m=None,
        intent_vs_regime_3m=None,
        robust_strategies_6m=None,
        robust_strategies_3m=None,
        ranked_6m_vs_spy=None,
        ranked_6m_vs_bufr=None
    )

    # ============================================================================
    # EXPORT REBALANCE SCORING
    # ============================================================================

    all_records = []
    all_trade_scores = []

    for result in results_list:
        if 'scoring_tracker' in result and result['scoring_tracker']:
            all_records.extend(result['scoring_tracker'].rebalance_records)

        # Capture spread history per strategy
        if result.get('trigger_type') == 'ecr_percentile_trigger':
            trigger_params = result.get('trigger_params', {})
            weight_code = trigger_params.get('weight_code', '???')
            pct = trigger_params.get('percentile_threshold', 0)
            launch = result.get('launch_month', '???')

            # Trade history with ECR context
            trades = result.get('trade_history', pd.DataFrame())
            if not trades.empty:
                trades = trades.copy()
                trades['launch_month'] = launch
                trades['weight_code'] = weight_code
                trades['percentile_threshold'] = pct
                all_trade_scores.append(trades)

    if all_records:
        df_scoring = pd.DataFrame(all_records)
        scoring_path = Path(output_dir) / f'batch_{BATCH_NUMBER}_rebalance_scoring.xlsx'

        with pd.ExcelWriter(scoring_path, engine='openpyxl') as writer:
            df_scoring.to_excel(writer, sheet_name='All Rebalances', index=False)

            df_selected = df_scoring[df_scoring['Selected'] == True]
            df_selected.to_excel(writer, sheet_name='Selected Funds', index=False)

            if all_trade_scores:
                df_all_trades = pd.concat(all_trade_scores, ignore_index=True)
                df_all_trades.to_excel(writer, sheet_name='Trade Log ECR', index=False)

            pivot = df_scoring.pivot_table(
                index=['Date', 'Fund'],
                columns='Weight_Config',
                values='Composite_Score',
                aggfunc='first'
            )
            pivot.to_excel(writer, sheet_name='Scores by Weight')

    else:
        print("⚠️  No scoring records found")

    print("=" * 80 + "\n")
    # ============================================================================

    # Generate visualizations (DISABLED FOR BATCH 10)
    if BATCH_NUMBER not in [10, '10b', '10c', '10d', '11']:  # ← Add batch numbers to skip
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)

        try:
            generate_batch_visualizations(
                results_list=results_list,
                summary_df=summary_df,
                future_regime_df=future_regime_df if future_regime_df is not None else pd.DataFrame(),
                optimal_strategies=optimal_3m if optimal_3m else {},
                output_dir=output_dir,
                batch_number=BATCH_NUMBER
            )
        except KeyError as e:
            print(f"\n⚠️  Visualization error (non-critical): {e}")
            print("    Batch results are still saved in Excel workbook")
    else:
        print("\n⚠️  Batch visualizations skipped (use plot_batch_10_xxx.py instead)")

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Batch execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
