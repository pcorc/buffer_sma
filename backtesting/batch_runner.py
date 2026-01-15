"""
Batch execution of multiple backtest configurations.
"""

import pandas as pd
from backtesting.engine import run_single_ticker_backtest
from core.selections import get_selection_function


def run_all_single_ticker_tests(df_enriched, df_benchmarks, roll_dates_dict,
                                trigger_selection_combos, series='F'):
    """
    Run backtests for all combinations of launch months and trigger/selection configurations.

    This is the main orchestrator that:
    1. Loops through all trigger/selection combinations
    2. For each combo, loops through its specified launch months
    3. Calls run_single_ticker_backtest for each combination
    4. Collects all results into a list

    Parameters:
      df_enriched: Preprocessed fund data
      df_benchmarks: Benchmark data (SPY, BUFR) with daily returns
      roll_dates_dict: Dict of roll dates by frequency
      trigger_selection_combos: List of dicts with trigger/selection/launch_months configs
      series: Fund series (default 'F')

    Returns:
      List of result dicts from run_single_ticker_backtest
    """
    results_list = []

    # Calculate totals
    total_tests = sum(len(combo['launch_months']) for combo in trigger_selection_combos)
    current_test = 0
    failed_tests = []

    print(f"\n{'#' * 80}")
    print(f"STARTING BATCH BACKTEST EXECUTION")
    print(f"{'#' * 80}")
    print(f"Total tests to run: {total_tests}")
    print(f"  Combinations: {len(trigger_selection_combos)}")
    print(f"{'#' * 80}\n")

    for combo in trigger_selection_combos:
        # Extract launch months for this specific combo
        launch_months = combo['launch_months']

        print(f"\n{'=' * 80}")
        print(f"Combination: {combo['trigger_type']} + {combo['selection_func_name']}")
        print(f"Testing {len(launch_months)} months: {', '.join(launch_months)}")
        print(f"{'=' * 80}")

        for launch_month in launch_months:
            current_test += 1
            print(f"\n{'─' * 80}")
            print(f"Progress: {current_test}/{total_tests}")
            print(f"{'─' * 80}")

            try:
                # Get selection function from registry
                selection_func = get_selection_function(combo['selection_func_name'])

                result = run_single_ticker_backtest(
                    df_enriched=df_enriched,
                    df_benchmarks=df_benchmarks,
                    launch_month=launch_month,
                    trigger_config={
                        'type': combo['trigger_type'],
                        'params': combo['trigger_params']
                    },
                    selection_func=selection_func,
                    roll_dates_dict=roll_dates_dict,
                    series=series
                )

                if result:
                    results_list.append(result)
                else:
                    failed_tests.append({
                        'launch_month': launch_month,
                        'trigger_type': combo['trigger_type'],
                        'selection': combo['selection_func_name'],
                        'reason': 'No result returned'
                    })

            except Exception as e:
                print(f"\n❌ ERROR in backtest:")
                print(f"   Launch: {launch_month}")
                print(f"   Trigger: {combo['trigger_type']}")
                print(f"   Selection: {combo['selection_func_name']}")
                print(f"   Error: {str(e)}")

                failed_tests.append({
                    'launch_month': launch_month,
                    'trigger_type': combo['trigger_type'],
                    'selection': combo['selection_func_name'],
                    'reason': str(e)
                })

    print(f"\n{'#' * 80}")
    print(f"BATCH EXECUTION COMPLETE")
    print(f"{'#' * 80}")
    print(f"Successful tests: {len(results_list)}/{total_tests}")

    if failed_tests:
        print(f"\n⚠️  Failed tests: {len(failed_tests)}")
        for failed in failed_tests:
            print(f"  • {failed['launch_month']} | {failed['trigger_type']} | {failed['selection']}")
            print(f"    Reason: {failed['reason']}")
    else:
        print(f"✅ All tests completed successfully!")

    print(f"{'#' * 80}\n")

    return results_list


def run_subset_tests(df_enriched, df_benchmarks, roll_dates_dict,
                     trigger_selection_combos, series='F',
                     trigger_type_filter=None):
    """
    Run a filtered subset of backtests for testing/debugging.

    Note: Launch month filtering is now done via the 'launch_months' key
    in each combo config rather than as a separate parameter.

    Parameters:
      (same as run_all_single_ticker_tests)
      trigger_type_filter: List of specific trigger types to test (optional)

    Returns:
      List of result dicts
    """
    # Filter trigger combos
    if trigger_type_filter:
        trigger_selection_combos = [
            c for c in trigger_selection_combos
            if c['trigger_type'] in trigger_type_filter
        ]

    print(f"Running subset tests:")
    print(f"  Filtered to {len(trigger_selection_combos)} combinations")
    for combo in trigger_selection_combos:
        print(f"    {combo['trigger_type']} with {len(combo['launch_months'])} months")

    return run_all_single_ticker_tests(
        df_enriched, df_benchmarks, roll_dates_dict,
        trigger_selection_combos, series
    )