"""
FINAL CORRECTED EXTRACT FUNCTION
=================================

This version properly identifies:
- ECR V2 (quarterly rebalance + enhanced_cost_ratio_neutral_v2)
- Existing 90% (cap_utilization_threshold + most_recent_launch)

Replace your extract_and_export_daily_nav() function in run_batch_tests.py
"""

def extract_and_export_daily_nav(results_list, output_dir, batch_number):
    """
    Extract daily NAV data from results_list and export to CSV.
    
    Properly identifies ECR V2 and Existing 90% strategies.
    """
    import pandas as pd
    from pathlib import Path
    
    print(f"\n{'='*80}")
    print(f"EXTRACTING DAILY TIME SERIES DATA")
    print(f"{'='*80}\n")
    
    if not results_list:
        print("❌ No results to extract")
        return
    
    # Collect all daily dataframes
    all_daily_data = {}
    
    for idx, result in enumerate(results_list, 1):
        # Get strategy identification
        launch = result.get('launch_month', 'UNK')
        trigger = result.get('trigger_type', 'unknown')
        selection = result.get('selection_algo', 'unknown')
        trigger_params = result.get('trigger_params', {})
        
        print(f"\nResult {idx}:")
        print(f"  Launch: {launch}")
        print(f"  Trigger: {trigger}")
        print(f"  Trigger Params: {trigger_params}")
        print(f"  Selection: {selection}")
        
        # GET THE DAILY DATA
        daily_df = result.get('daily_performance', None)
        
        if daily_df is None or daily_df.empty:
            print(f"  ⚠️  No daily_performance data")
            continue
        
        # IDENTIFY STRATEGY LABEL (include launch month prefix)
        label = None
        
        # ECR variants: rebalance_time_period + quarterly + various ECR functions
        if trigger == 'rebalance_time_period':
            if trigger_params.get('frequency') == 'quarterly':
                # Identify which ECR variant
                if 'select_ecr_v2_equal' in selection:
                    label = f'{launch}_ECR_V2_equal'
                    print(f"  ✅ Identified as: {launch} ECR Equal")
                elif 'select_ecr_v2_cap_balanced' in selection:
                    label = f'{launch}_ECR_V2_cap_balanced'
                    print(f"  ✅ Identified as: {launch} ECR Cap-Balanced")
                elif 'select_ecr_v2_cap_moderate' in selection:
                    label = f'{launch}_ECR_V2_cap_moderate'
                    print(f"  ✅ Identified as: {launch} ECR Cap-Moderate")
                elif 'select_ecr_v2_cap_dominant' in selection:
                    label = f'{launch}_ECR_V2_cap_dominant'
                    print(f"  ✅ Identified as: {launch} ECR Cap-Dominant")
                elif 'select_ecr_v2_protection' in selection:
                    label = f'{launch}_ECR_V2_protection'
                    print(f"  ✅ Identified as: {launch} ECR Protection")
                elif 'enhanced_cost_ratio_neutral_v2' in selection:
                    # Legacy/old naming
                    label = f'{launch}_ECR_V2'
                    print(f"  ✅ Identified as: {launch} ECR V2")
        
        # Existing 90%: cap_utilization_threshold + 0.9 + most_recent_launch
        if trigger == 'cap_utilization_threshold':
            if trigger_params.get('threshold') == 0.9 or trigger_params.get('threshold') == 0.90:
                if 'most_recent_launch' in selection:
                    label = f'{launch}_Existing_90'
                    print(f"  ✅ Identified as: {launch} Existing 90%")
        
        if label is None:
            # Fallback: use combination with month
            label = f"{launch}_{trigger[:10]}_{selection[:10]}"
            print(f"  ⚠️  Using fallback label: {label}")
        
        all_daily_data[label] = daily_df
        print(f"  ✅ Stored data: {label} ({len(daily_df)} rows)")
    
    if not all_daily_data:
        print("\n❌ No daily data found in any results")
        return
    
    print(f"\n{'='*80}")
    print(f"MERGING {len(all_daily_data)} STRATEGIES")
    print(f"{'='*80}")
    
    base_df = None
    benchmarks_added = False
    
    for label, daily_df in all_daily_data.items():
        print(f"\nProcessing: {label}")
        
        # Show available columns
        print(f"  Columns: {list(daily_df.columns)[:10]}...")
        
        # Required: Date column
        if 'Date' not in daily_df.columns:
            print(f"  ❌ No Date column")
            continue
        
        # Strategy NAV column
        strategy_nav_col = None
        for possible_col in ['Strategy_NAV', 'Strategy NAV', 'NAV', 'Strategy_Value']:
            if possible_col in daily_df.columns:
                strategy_nav_col = possible_col
                break
        
        if strategy_nav_col is None:
            print(f"  ❌ No Strategy NAV column found")
            print(f"     Available columns: {list(daily_df.columns)}")
            continue
        
        # Build dataframe for this strategy
        temp_df = pd.DataFrame()
        temp_df['Date'] = pd.to_datetime(daily_df['Date'])
        temp_df[f'{label}_NAV'] = daily_df[strategy_nav_col]
        
        print(f"  ✅ Using column '{strategy_nav_col}' as {label}_NAV")
        
        # Add benchmarks (only once)
        if not benchmarks_added:
            # Check for SPY
            spy_col = None
            for possible_col in ['SPY_NAV', 'SPY NAV', 'SPY', 'Benchmark_SPY']:
                if possible_col in daily_df.columns:
                    spy_col = possible_col
                    break
            
            if spy_col:
                temp_df['SPY_NAV'] = daily_df[spy_col]
                print(f"  ✅ Added SPY from column '{spy_col}'")
            
            # Check for BUFR
            bufr_col = None
            for possible_col in ['BUFR_NAV', 'BUFR NAV', 'BUFR', 'Benchmark_BUFR']:
                if possible_col in daily_df.columns:
                    bufr_col = possible_col
                    break
            
            if bufr_col:
                temp_df['BUFR_NAV'] = daily_df[bufr_col]
                print(f"  ✅ Added BUFR from column '{bufr_col}'")
            
            benchmarks_added = True
        
        # Merge or initialize
        if base_df is None:
            base_df = temp_df
        else:
            base_df = base_df.merge(
                temp_df[['Date', f'{label}_NAV']],
                on='Date',
                how='outer'
            )
    
    if base_df is None or base_df.empty:
        print("\n❌ Failed to merge data")
        return
    
    # Sort by date
    base_df.sort_values('Date', inplace=True)
    base_df.reset_index(drop=True, inplace=True)
    
    # Calculate returns
    print(f"\n{'='*80}")
    print(f"CALCULATING RETURNS")
    print(f"{'='*80}")
    
    for col in base_df.columns:
        if col.endswith('_NAV') and col != 'Date':
            return_col = col.replace('_NAV', '_Return')
            base_df[return_col] = base_df[col].pct_change()
            print(f"  ✅ {return_col}")
    
    # Reorder columns: Date, NAVs, then Returns
    nav_cols = sorted([c for c in base_df.columns if c.endswith('_NAV')])
    return_cols = sorted([c for c in base_df.columns if c.endswith('_Return')])
    base_df = base_df[['Date'] + nav_cols + return_cols]
    
    # Export
    output_path = Path(output_dir) / f'batch_{batch_number}_daily_time_series.csv'
    base_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"EXPORT COMPLETE")
    print(f"{'='*80}")
    print(f"✅ File: {output_path}")
    print(f"   Rows: {len(base_df):,}")
    print(f"   Columns: {len(base_df.columns)}")
    
    print(f"\nColumns included:")
    for i, col in enumerate(base_df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Show sample
    print(f"\n{'='*80}")
    print(f"SAMPLE DATA (first 5 rows)")
    print(f"{'='*80}")
    
    # Show just NAV columns for clarity
    sample_cols = ['Date'] + [c for c in base_df.columns if c.endswith('_NAV')]
    print(base_df[sample_cols].head().to_string(index=False))
    
    print(f"\n{'='*80}")
    
    return base_df
