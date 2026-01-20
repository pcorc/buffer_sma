"""
Batch execution of multiple backtest configurations with parallel processing support.

This module supports both sequential and parallel execution of backtests.
Parallel processing can significantly reduce runtime for large-scale simulations.
"""

import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List, Dict, Any
import sys

from backtesting.engine import run_single_ticker_backtest
from core.selections import get_selection_function


# =============================================================================
# SINGLE BACKTEST WRAPPER FOR PARALLEL EXECUTION
# =============================================================================

def _run_single_backtest_wrapper(args):
    """
    Wrapper function for parallel execution of single backtest.

    This wrapper is necessary because multiprocessing.Pool.map() requires
    a function that takes a single argument.

    Parameters:
        args: Tuple of (backtest_params, shared_data)
            backtest_params: Dict with launch_month, combo
            shared_data: Dict with df_enriched, df_benchmarks, roll_dates_dict, series

    Returns:
        Result dict from run_single_ticker_backtest, or None if error
    """
    backtest_params, shared_data = args

    launch_month = backtest_params['launch_month']
    combo = backtest_params['combo']

    try:
        selection_func = get_selection_function(combo['selection_func_name'])

        result = run_single_ticker_backtest(
            df_enriched=shared_data['df_enriched'],
            df_benchmarks=shared_data['df_benchmarks'],
            launch_month=launch_month,
            trigger_config={
                'type': combo['trigger_type'],
                'params': combo['trigger_params']
            },
            selection_func=selection_func,
            roll_dates_dict=shared_data['roll_dates_dict'],
            series=shared_data['series']
        )

        return result

    except Exception as e:
        error_msg = (
            f"ERROR in backtest:\n"
            f"  Launch: {launch_month}\n"
            f"  Trigger: {combo['trigger_type']}\n"
            f"  Selection: {combo['selection_func_name']}\n"
            f"  Error: {str(e)}"
        )
        print(error_msg, file=sys.stderr)
        return None


# =============================================================================
# PARALLEL EXECUTION
# =============================================================================

def run_all_single_ticker_tests_parallel(
    df_enriched,
    df_benchmarks,
    roll_dates_dict,
    trigger_selection_combos,
    launch_months,
    series='F',
    n_processes=None,
    show_progress=True
):
    """
    Run backtests in parallel for all combinations of launch months and strategies.

    Uses multiprocessing.Pool to distribute backtests across CPU cores.
    Significantly faster than sequential execution for large batches.

    Parameters:
        df_enriched: Preprocessed fund data
        df_benchmarks: Benchmark data (SPY, BUFR) with daily returns
        roll_dates_dict: Dict of roll dates by frequency
        trigger_selection_combos: List of trigger/selection config dicts
        launch_months: List of launch month abbreviations
        series: Fund series (default 'F')
        n_processes: Number of processes to use (default: CPU count - 1)
        show_progress: Print progress updates (default: True)

    Returns:
        List of result dicts from run_single_ticker_backtest
    """
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)  # Leave one core free

    total_tests = len(launch_months) * len(trigger_selection_combos)

    print(f"\n{'#' * 80}")
    print(f"STARTING PARALLEL BACKTEST EXECUTION")
    print(f"{'#' * 80}")
    print(f"Total tests to run: {total_tests}")
    print(f"  Launch months: {len(launch_months)}")
    print(f"  Combinations per month: {len(trigger_selection_combos)}")
    print(f"  Parallel processes: {n_processes}")
    print(f"  Available CPUs: {cpu_count()}")
    print(f"{'#' * 80}\n")

    # Prepare shared data (will be passed to each worker)
    shared_data = {
        'df_enriched': df_enriched,
        'df_benchmarks': df_benchmarks,
        'roll_dates_dict': roll_dates_dict,
        'series': series
    }

    # Build list of all backtest parameter combinations
    backtest_tasks = []
    for launch_month in launch_months:
        for combo in trigger_selection_combos:
            backtest_params = {
                'launch_month': launch_month,
                'combo': combo
            }
            backtest_tasks.append((backtest_params, shared_data))

    # Execute in parallel
    start_time = datetime.now()
    results_list = []
    failed_tests = []

    if show_progress:
        print(f"Executing {total_tests} backtests across {n_processes} processes...")
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)

    with Pool(processes=n_processes) as pool:
        # Use imap_unordered for progress tracking
        completed = 0
        for result in pool.imap_unordered(_run_single_backtest_wrapper, backtest_tasks):
            completed += 1

            if result is not None:
                results_list.append(result)
            else:
                failed_tests.append(completed)

            # Progress update every 10% or every 50 tests
            if show_progress and (completed % max(1, total_tests // 10) == 0 or completed % 50 == 0):
                pct_complete = (completed / total_tests) * 100
                elapsed = datetime.now() - start_time
                print(f"  Progress: {completed}/{total_tests} ({pct_complete:.1f}%) - Elapsed: {elapsed}")

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print(f"PARALLEL EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Successful tests: {len(results_list)}/{total_tests}")
    print(f"Duration: {duration}")
    print(f"Average time per test: {duration / total_tests if total_tests > 0 else 0}")

    if failed_tests:
        print(f"\n⚠️  Failed tests: {len(failed_tests)}")
        print(f"Failed test indices: {failed_tests[:10]}{'...' if len(failed_tests) > 10 else ''}")
    else:
        print(f"\n✅ All tests completed successfully!")

    print("=" * 80 + "\n")

    return results_list


# =============================================================================
# SEQUENTIAL EXECUTION (Original Implementation)
# =============================================================================

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
        Each dict should have:
          - 'trigger_type': str
          - 'trigger_params': dict
          - 'selection_func_name': str
          - 'launch_months': list of str
          - 'strategy_intent': str (optional)
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
                        'params': combo['trigger_params'],
                        'strategy_intent': combo.get('strategy_intent', 'neutral')
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

# =============================================================================
# INTENT-GROUPED PARALLEL EXECUTION
# =============================================================================

def run_tests_by_intent_groups(
    df_enriched,
    df_benchmarks,
    roll_dates_dict,
    trigger_selection_combos,
    launch_months,
    series='F',
    n_processes=None
):
    """
    Run backtests grouped by strategy intent for better progress visibility.

    Executes each intent group (bullish/bearish/neutral/cost_optimized) sequentially,
    but parallelizes within each group. This provides clearer progress reporting.

    Parameters:
        (same as run_all_single_ticker_tests_parallel)

    Returns:
        List of result dicts from run_single_ticker_backtest
    """
    from config.strategy_intents import get_strategy_intent

    # Group combinations by intent
    intent_groups = {
        'bullish': [],
        'bearish': [],
        'neutral': [],
        'cost_optimized': []
    }

    for combo in trigger_selection_combos:
        try:
            intent = get_strategy_intent(
                combo['trigger_type'],
                combo['trigger_params'],
                combo['selection_func_name']
            )
            intent_groups[intent].append(combo)
        except KeyError:
            # If intent not found, add to neutral
            intent_groups['neutral'].append(combo)

    print(f"\n{'#' * 80}")
    print(f"INTENT-GROUPED PARALLEL EXECUTION")
    print(f"{'#' * 80}")
    print(f"Total combinations: {len(trigger_selection_combos)}")
    print(f"Launch months: {len(launch_months)}")
    for intent, combos in intent_groups.items():
        if combos:
            sims = len(combos) * len(launch_months)
            print(f"  {intent.upper():15s}: {len(combos):2d} combos × {len(launch_months):2d} months = {sims:3d} sims")
    print(f"{'#' * 80}\n")

    all_results = []
    overall_start = datetime.now()

    # Execute each intent group
    for intent in ['bullish', 'bearish', 'neutral', 'cost_optimized']:
        combos = intent_groups[intent]
        if not combos:
            continue

        print(f"\n{'=' * 80}")
        print(f"EXECUTING {intent.upper()} STRATEGIES")
        print(f"{'=' * 80}")
        print(f"Combinations: {len(combos)}")
        print(f"Total tests: {len(combos) * len(launch_months)}")
        print(f"{'=' * 80}\n")

        # Run this group in parallel
        group_results = run_all_single_ticker_tests_parallel(
            df_enriched=df_enriched,
            df_benchmarks=df_benchmarks,
            roll_dates_dict=roll_dates_dict,
            trigger_selection_combos=combos,
            launch_months=launch_months,
            series=series,
            n_processes=n_processes,
            show_progress=True
        )

        all_results.extend(group_results)

        print(f"✅ {intent.upper()} group complete: {len(group_results)} successful tests\n")

    overall_duration = datetime.now() - overall_start

    print(f"\n{'#' * 80}")
    print(f"ALL INTENT GROUPS COMPLETE")
    print(f"{'#' * 80}")
    print(f"Total successful tests: {len(all_results)}")
    print(f"Total duration: {overall_duration}")
    print(f"{'#' * 80}\n")

    return all_results

