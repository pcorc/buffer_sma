"""
COMPLETE BATCH 6 VISUALIZATION INTEGRATION
===========================================

STEP 1: Add these 3 functions to the END of visualization/performance_plots.py
        (Add them right before the last line of the file)

STEP 2: Add the Batch 6 conditional code to generate_batch_visualizations()
        (Find the section with "if batch_number == 9" and add after it)
"""

# =============================================================================
# STEP 1: ADD THESE THREE FUNCTIONS TO END OF performance_plots.py
# =============================================================================

def plot_ecr_regime_comparison(
        summary_df: pd.DataFrame,
        output_dir: str
) -> Optional[str]:
    """
    Compare performance of Enhanced Cost Ratio across three regime weightings.
    
    Creates bar chart showing:
    - Bullish (Cap-heavy) vs Bearish (Protection-heavy) vs Neutral (Balanced)
    - Average return, Sharpe ratio, max drawdown for each
    """
    print("\n  Generating: ECR Regime Weighting Comparison...")
    
    if summary_df.empty:
        print("    ⊘ Skipped: No data")
        return None
    
    # Filter to ECR strategies only
    ecr_mask = summary_df['selection_algo'].str.contains('enhanced_cost_ratio', na=False)
    ecr_df = summary_df[ecr_mask].copy()
    
    if ecr_df.empty:
        print("    ⊘ Skipped: No Enhanced Cost Ratio strategies found")
        return None
    
    # Extract regime type from selection function name
    def extract_regime(selection_algo):
        if 'bullish' in selection_algo:
            return 'Bullish'
        elif 'bearish' in selection_algo:
            return 'Bearish'
        elif 'neutral' in selection_algo:
            return 'Neutral'
        return 'Unknown'
    
    ecr_df['ecr_regime'] = ecr_df['selection_algo'].apply(extract_regime)
    
    # Calculate averages by regime
    regime_stats = ecr_df.groupby('ecr_regime').agg({
        'strategy_return': 'mean',
        'strategy_sharpe': 'mean',
        'strategy_max_dd': 'mean',
        'vs_bufr_excess': 'mean'
    }).reset_index()
    
    # Sort by Sharpe ratio
    regime_stats = regime_stats.sort_values('strategy_sharpe', ascending=False)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Enhanced Cost Ratio: Regime Weighting Performance Comparison', 
                 fontsize=16, fontweight='bold', pad=20)
    
    colors = {
        'Bullish': '#2E7D32',
        'Bearish': '#C62828',
        'Neutral': '#1565C0'
    }
    
    bar_colors = [colors.get(regime, '#757575') for regime in regime_stats['ecr_regime']]
    x_pos = range(len(regime_stats))
    labels = regime_stats['ecr_regime'].tolist()
    
    # Subplot 1: Average Return
    ax1 = axes[0, 0]
    returns_pct = regime_stats['strategy_return'].values * 100
    bars1 = ax1.bar(x_pos, returns_pct, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=2)
    ax1.set_title('Average Return', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Return (%)', fontsize=11)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, returns_pct):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')
    
    # Subplot 2: Sharpe Ratio
    ax2 = axes[0, 1]
    sharpes = regime_stats['strategy_sharpe'].values
    bars2 = ax2.bar(x_pos, sharpes, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=2)
    ax2.set_title('Average Sharpe Ratio', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Sharpe Ratio', fontsize=11)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, sharpes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height,
                f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')
    
    # Subplot 3: Max Drawdown
    ax3 = axes[1, 0]
    drawdowns_pct = regime_stats['strategy_max_dd'].values * 100
    bars3 = ax3.bar(x_pos, drawdowns_pct, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=2)
    ax3.set_title('Average Max Drawdown', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Max Drawdown (%)', fontsize=11)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars3, drawdowns_pct):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height,
                f'{val:.1f}%', ha='center', va='top' if height < 0 else 'bottom',
                fontsize=11, fontweight='bold')
    
    # Subplot 4: Excess vs BUFR
    ax4 = axes[1, 1]
    excess_pct = regime_stats['vs_bufr_excess'].values * 100
    bars4 = ax4.bar(x_pos, excess_pct, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=2)
    ax4.set_title('Average Excess Return vs BUFR', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Excess Return (%)', fontsize=11)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars4, excess_pct):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height,
                f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    filename = 'ecr_regime_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {filename}")
    
    # Print summary to console
    print("\n  ECR Regime Performance Summary:")
    print("  " + "=" * 70)
    for _, row in regime_stats.iterrows():
        print(f"  {row['ecr_regime']:10s} | "
              f"Return: {row['strategy_return']*100:+6.2f}% | "
              f"Sharpe: {row['strategy_sharpe']:5.2f} | "
              f"vs BUFR: {row['vs_bufr_excess']*100:+6.2f}%")
    print("  " + "=" * 70)
    
    return filepath


def plot_ecr_top_performers(
        results_list: List[Dict],
        summary_df: pd.DataFrame,
        output_dir: str,
        top_n: int = 5
) -> Optional[str]:
    """
    Plot normalized NAV of top N Enhanced Cost Ratio strategies.
    """
    print(f"\n  Generating: ECR Top {top_n} Performers...")
    
    if summary_df.empty or not results_list:
        print("    ⊘ Skipped: No data")
        return None
    
    # Filter to ECR strategies
    ecr_mask = summary_df['selection_algo'].str.contains('enhanced_cost_ratio', na=False)
    ecr_df = summary_df[ecr_mask].copy()
    
    if ecr_df.empty:
        print("    ⊘ Skipped: No Enhanced Cost Ratio strategies found")
        return None
    
    # Get top N by Sharpe ratio
    top_strategies = ecr_df.nlargest(top_n, 'strategy_sharpe')
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    colors = ['#2E7D32', '#1565C0', '#F57C00', '#7B1FA2', '#C62828']
    
    # Plot each top strategy
    for i, (idx, row) in enumerate(top_strategies.iterrows()):
        # Find matching result
        matching_result = None
        for result in results_list:
            if (result['launch_month'] == row['launch_month'] and
                result['trigger_type'] == row['trigger_type'] and
                result['selection_algo'] == row['selection_algo']):
                matching_result = result
                break
        
        if matching_result is None:
            continue
        
        daily = matching_result['daily_performance']
        
        # Extract regime type
        if 'bullish' in row['selection_algo']:
            regime = 'Bullish'
        elif 'bearish' in row['selection_algo']:
            regime = 'Bearish'
        else:
            regime = 'Neutral'
        
        # Create label
        trigger_short = row['trigger_type'].replace('_threshold', '').replace('_', ' ').title()
        label = f"#{i+1}: {regime} ECR - {trigger_short} ({row['launch_month']})"
        
        ax.plot(daily['Date'], daily['Strategy_NAV'],
                color=colors[i % len(colors)], linewidth=2.5, alpha=0.9,
                label=label, zorder=10 - i)
    
    # Plot benchmarks
    if results_list:
        benchmark = results_list[0]['daily_performance']
        ax.plot(benchmark['Date'], benchmark['SPY_NAV'],
                color='#757575', linewidth=2, linestyle='--',
                alpha=0.7, label='SPY', zorder=5)
        ax.plot(benchmark['Date'], benchmark['BUFR_NAV'],
                color='#9E9E9E', linewidth=2, linestyle=':',
                alpha=0.7, label='BUFR', zorder=5)
    
    # Formatting
    ax.set_title(f'Top {top_n} Enhanced Cost Ratio Strategies', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('NAV (Normalized to 100)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.axhline(y=100, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    filename = 'ecr_top_performers.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {filename}")
    return filepath


def plot_ecr_by_trigger_type(
        summary_df: pd.DataFrame,
        output_dir: str
) -> Optional[str]:
    """
    Compare ECR performance grouped by trigger type.
    """
    print("\n  Generating: ECR Performance by Trigger Type...")
    
    if summary_df.empty:
        print("    ⊘ Skipped: No data")
        return None
    
    # Filter to ECR strategies
    ecr_mask = summary_df['selection_algo'].str.contains('enhanced_cost_ratio', na=False)
    ecr_df = summary_df[ecr_mask].copy()
    
    if ecr_df.empty:
        print("    ⊘ Skipped: No Enhanced Cost Ratio strategies found")
        return None
    
    # Group by trigger type
    trigger_stats = ecr_df.groupby('trigger_type').agg({
        'strategy_sharpe': ['mean', 'std', 'count'],
        'vs_bufr_excess': 'mean'
    }).reset_index()
    
    # Flatten column names
    trigger_stats.columns = ['trigger_type', 'sharpe_mean', 'sharpe_std', 'count', 'excess_mean']
    
    # Sort by average Sharpe
    trigger_stats = trigger_stats.sort_values('sharpe_mean', ascending=False)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Enhanced Cost Ratio Performance by Trigger Type', 
                 fontsize=16, fontweight='bold', pad=20)
    
    x_pos = range(len(trigger_stats))
    trigger_labels = [t.replace('_', ' ').title() for t in trigger_stats['trigger_type']]
    
    # Sharpe ratio with error bars
    ax1.bar(x_pos, trigger_stats['sharpe_mean'], 
            yerr=trigger_stats['sharpe_std'],
            color='#1565C0', alpha=0.85, edgecolor='black', linewidth=2,
            capsize=5, error_kw={'linewidth': 2})
    
    ax1.set_title('Average Sharpe Ratio (with Std Dev)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Sharpe Ratio', fontsize=11)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(trigger_labels, rotation=45, ha='right', fontsize=10)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels and count
    for i, (bar, mean_val, count_val) in enumerate(zip(ax1.patches, 
                                                        trigger_stats['sharpe_mean'], 
                                                        trigger_stats['count'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{mean_val:.2f}\n(n={int(count_val)})',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    
    # Excess vs BUFR
    ax2.bar(x_pos, trigger_stats['excess_mean'] * 100,
            color='#2E7D32', alpha=0.85, edgecolor='black', linewidth=2)
    
    ax2.set_title('Average Excess Return vs BUFR', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Excess Return (%)', fontsize=11)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(trigger_labels, rotation=45, ha='right', fontsize=10)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(ax2.patches, trigger_stats['excess_mean'] * 100):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height,
                f'{val:+.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    filename = 'ecr_by_trigger_type.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {filename}")
    return filepath


# =============================================================================
# STEP 2: ADD THIS CODE TO generate_batch_visualizations() FUNCTION
# Find the section that has "if batch_number == 9:" and add this AFTER it
# =============================================================================

"""
    if batch_number == 9:
        filepath = plot_batch8_normalized_nav_ALIGNED_DETAILED_LEGEND(results_list, summary_df, output_dir)
        if filepath:
            generated_plots['normalized_nav_aligned'] = filepath

    # ===== ADD THIS BATCH 6 CODE HERE =====
    
    if batch_number == 6:
        # Enhanced Cost Ratio specialized plots
        filepath = plot_ecr_regime_comparison(summary_df, output_dir)
        if filepath:
            generated_plots['ecr_regime_comparison'] = filepath
        
        filepath = plot_ecr_top_performers(results_list, summary_df, output_dir, top_n=5)
        if filepath:
            generated_plots['ecr_top_performers'] = filepath
        
        filepath = plot_ecr_by_trigger_type(summary_df, output_dir)
        if filepath:
            generated_plots['ecr_by_trigger_type'] = filepath
    
    # ===== END BATCH 6 CODE =====

    # Regime performance (if we have regime data and optimal strategies)
    if not future_regime_df.empty and optimal_strategies:
        ...
"""
