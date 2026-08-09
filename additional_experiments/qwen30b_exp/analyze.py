"""
Medical Diagnosis Experiment Analysis - With 95% Confidence Intervals
======================================================================
- 95% CI for proportions (Accuracy, Convergence) using Wilson score interval
- 95% CI for continuous variables (Questions, Cost, Duration)
- Clean labels (no $ or % symbols on bars)
- Separate Pareto plot with distinct shapes and colors
- Condition-specific titles and filenames

Usage:
    python analyze.py results/results_merged.json --output-dir ./figures --condition prune_uniform
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys


# ============================================================================
# CONFIGURATION
# ============================================================================

Z = 1.96  # z-score for 95% CI

# Condition name mapping for titles
CONDITION_TITLES = {
    'notprune_notuniform': 'I-NP (Informative Prior, No Pruning)',
    'notprune_uniform': 'U-NP (Uniform Prior, No Pruning)',
    'prune_notuniform': 'I-P (Informative Prior, Pruning)',
    'prune_uniform': 'U-P (Uniform Prior, Pruning)',
}

# Colorblind-friendly palette - all 9 conditions
COLORS = {
    # Adaptive methods - blues/cyans
    'ADAPTIVE-EIG-COST-AWARE': '#33BBEE',        # cyan
    'ADAPTIVE-PD-COST-AWARE': '#0077BB',         # blue
    'ADAPTIVE-PD-EIG-COST-AWARE': '#004488',     # dark blue
    
    # Truncated methods - greens
    'TRUNCATED-EIG-COST-AWARE': '#009988',       # teal
    'TRUNCATED-PD-COST-AWARE': '#44AA99',        # green
    'TRUNCATED-PD-EIG-COST-AWARE': '#117733',    # dark green
    
    # Base methods - warm colors
    'EIG': '#EE3377',                            # magenta
    'EIG-COST-AWARE': '#EE7733',                 # orange
    'LLM-SELECT': '#CC3311',                     # red
}

# Markers for Pareto scatter plot - all 9 conditions with unique shapes
MARKERS = {
    # Adaptive methods - triangles
    'ADAPTIVE-EIG-COST-AWARE': '^',              # triangle up
    'ADAPTIVE-PD-COST-AWARE': 'v',               # triangle down
    'ADAPTIVE-PD-EIG-COST-AWARE': '<',           # triangle left
    
    # Truncated methods - squares/diamonds
    'TRUNCATED-EIG-COST-AWARE': 's',             # square
    'TRUNCATED-PD-COST-AWARE': 'D',              # diamond
    'TRUNCATED-PD-EIG-COST-AWARE': 'd',          # thin diamond
    
    # Base methods - other shapes
    'EIG': 'X',                                  # X
    'EIG-COST-AWARE': 'o',                       # circle
    'LLM-SELECT': 'P',                           # plus (filled)
}


# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def calc_ci_proportion_wilson(p, n):
    """
    Calculate 95% CI for a proportion using Wilson score interval.
    Returns (lower_bound, upper_bound) capped at [0, 100].
    
    Args:
        p: proportion as percentage (0-100)
        n: sample size
    
    Returns:
        Tuple of (lower_ci, upper_ci) in percentage points
    """
    if n <= 0:
        return 0, 0
    p_decimal = p / 100
    denominator = 1 + Z**2 / n
    center = (p_decimal + Z**2 / (2*n)) / denominator
    margin = Z * np.sqrt((p_decimal * (1 - p_decimal) + Z**2 / (4*n)) / n) / denominator
    
    lower = max(0, (center - margin)) * 100
    upper = min(100, (center + margin)) * 100
    
    return lower, upper


def calc_ci_proportion_wilson_halfwidth(p, n):
    """
    Calculate 95% CI half-width for a proportion, capped so bounds stay in [0, 100].
    For plotting error bars, we need asymmetric errors when near boundaries.
    
    Args:
        p: proportion as percentage (0-100)
        n: sample size
    
    Returns:
        Tuple of (lower_error, upper_error) for asymmetric error bars
    """
    # Handle edge cases
    if n <= 0:
        return 0, 0
    
    lower, upper = calc_ci_proportion_wilson(p, n)
    lower_err = max(0, p - lower)  # distance from p to lower bound, never negative
    upper_err = max(0, upper - p)  # distance from p to upper bound, never negative
    return lower_err, upper_err


def calc_ci_continuous(std, n):
    """
    Calculate 95% CI half-width for continuous variable.
    
    Formula: 1.96 * (std / sqrt(n))
    
    Args:
        std: standard deviation
        n: sample size
    
    Returns:
        CI half-width
    """
    if n <= 0:
        return 0
    return Z * (std / np.sqrt(n))


# ============================================================================
# DATA LOADING AND ANALYSIS
# ============================================================================

def load_and_analyze(json_file_path):
    """Load JSON data and analyze by condition."""
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    print(f"\nTotal records loaded: {len(df)}")
    print(f"\nConditions found: {df['condition'].unique()}")
    print(f"\nRecords per condition:")
    print(df['condition'].value_counts().sort_index())
    
    conditions = df['condition'].unique()
    all_metrics = []
    
    for cond in sorted(conditions):
        cond_df = df[df['condition'] == cond]
        
        # Overall metrics
        overall = compute_metrics(cond_df, cond, 'all')
        all_metrics.append(overall)
        
        # Correct only
        correct_df = cond_df[cond_df['correct'] == True]
        if len(correct_df) > 0:
            correct = compute_metrics(correct_df, cond, 'correct')
            all_metrics.append(correct)
        
        # Incorrect only
        incorrect_df = cond_df[cond_df['correct'] == False]
        if len(incorrect_df) > 0:
            incorrect = compute_metrics(incorrect_df, cond, 'incorrect')
            all_metrics.append(incorrect)
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # Add 95% CI columns
    # For proportions: asymmetric errors (lower, upper)
    metrics_df['ci_accuracy_lower'] = metrics_df.apply(
        lambda row: calc_ci_proportion_wilson_halfwidth(row['correct_rate'] * 100, row['n'])[0], axis=1)
    metrics_df['ci_accuracy_upper'] = metrics_df.apply(
        lambda row: calc_ci_proportion_wilson_halfwidth(row['correct_rate'] * 100, row['n'])[1], axis=1)
    metrics_df['ci_convergence_lower'] = metrics_df.apply(
        lambda row: calc_ci_proportion_wilson_halfwidth(row['correct_converge_rate'] * 100, row['n'])[0], axis=1)
    metrics_df['ci_convergence_upper'] = metrics_df.apply(
        lambda row: calc_ci_proportion_wilson_halfwidth(row['correct_converge_rate'] * 100, row['n'])[1], axis=1)
    
    # For continuous variables: symmetric errors
    metrics_df['ci_questions'] = metrics_df.apply(
        lambda row: calc_ci_continuous(row['std_questions'], row['n']), axis=1)
    metrics_df['ci_cost'] = metrics_df.apply(
        lambda row: calc_ci_continuous(row['std_cost'], row['n']), axis=1)
    metrics_df['ci_duration'] = metrics_df.apply(
        lambda row: calc_ci_continuous(row['std_duration'], row['n']), axis=1)
    
    return df, metrics_df


def compute_metrics(subset_df, condition, subset_type):
    """Compute metrics for a subset of data."""
    
    n = len(subset_df)
    
    metrics = {
        'condition': condition,
        'subset': subset_type,
        'n': n,
        'correct_rate': subset_df['correct'].mean() if n > 0 else 0,
        'correct_converge_rate': ((subset_df['converged'] == True) & (subset_df['correct'] == True)).mean() if n > 0 else 0,
        'avg_questions': subset_df['num_questions'].mean() if n > 0 else 0,
        'std_questions': subset_df['num_questions'].std() if n > 0 else 0,
        'avg_cost': subset_df['final_cost'].mean() if n > 0 else 0,
        'std_cost': subset_df['final_cost'].std() if n > 0 else 0,
        'avg_duration': subset_df['duration_seconds'].mean() if n > 0 else 0,
        'std_duration': subset_df['duration_seconds'].std() if n > 0 else 0,
        'avg_confidence': subset_df['final_confidence'].mean() if n > 0 else 0,
        'std_confidence': subset_df['final_confidence'].std() if n > 0 else 0,
    }
    
    return metrics


# ============================================================================
# PRINTING FUNCTIONS
# ============================================================================

def print_summary_table(metrics_df):
    """Print summary table for all conditions."""
    
    print("\n" + "="*110)
    print("SUMMARY: ALL CONDITIONS (All Samples) - with 95% CI")
    print("="*110)
    
    all_df = metrics_df[metrics_df['subset'] == 'all'].copy()
    
    print(f"\n{'Condition':<28} {'N':>5} {'Accuracy':>16} {'Correct&Conv':>16} {'Avg Q':>12} {'Avg Cost':>14}")
    print("-"*110)
    
    for _, row in all_df.iterrows():
        acc = row['correct_rate'] * 100
        conv = row['correct_converge_rate'] * 100
        # Get CI bounds
        acc_lo, acc_hi = calc_ci_proportion_wilson(acc, row['n'])
        conv_lo, conv_hi = calc_ci_proportion_wilson(conv, row['n'])
        
        print(f"{row['condition']:<28} {row['n']:>5} "
              f"{acc:>5.1f} [{acc_lo:.1f}-{acc_hi:.1f}] "
              f"{conv:>5.1f} [{conv_lo:.1f}-{conv_hi:.1f}] "
              f"{row['avg_questions']:>4.2f}±{row['ci_questions']:.2f} "
              f"${row['avg_cost']:>6.0f}±{row['ci_cost']:.0f}")
    
    print("="*110)


def print_detailed_breakdown(metrics_df, condition):
    """Print detailed breakdown for a specific condition."""
    
    cond_metrics = metrics_df[metrics_df['condition'] == condition]
    
    print(f"\n{'='*80}")
    print(f"{condition}")
    print(f"{'='*80}")
    
    for _, row in cond_metrics.iterrows():
        subset = row['subset'].upper()
        n = int(row['n'])
        
        # Get CI bounds for proportions
        acc = row['correct_rate'] * 100
        conv = row['correct_converge_rate'] * 100
        acc_lo, acc_hi = calc_ci_proportion_wilson(acc, n)
        conv_lo, conv_hi = calc_ci_proportion_wilson(conv, n)
        
        print(f"\n{subset} (n={n}):")
        print(f"  Accuracy:              {acc:>6.1f}% [{acc_lo:.1f}% - {acc_hi:.1f}%] (95% CI)")
        print(f"  Correct & Converge:    {conv:>6.1f}% [{conv_lo:.1f}% - {conv_hi:.1f}%] (95% CI)")
        print(f"  Avg Questions:         {row['avg_questions']:>6.2f} ± {row['ci_questions']:.2f} (95% CI)")
        print(f"  Avg Cost:              ${row['avg_cost']:>6.2f} ± ${row['ci_cost']:.2f} (95% CI)")
        print(f"  Avg Duration:          {row['avg_duration']:>6.1f}s ± {row['ci_duration']:.1f}s (95% CI)")
        print(f"  Avg Confidence:        {row['avg_confidence']:>6.1%} ± {row['std_confidence']:.1%} (SD)")


# ============================================================================
# FIGURE: COMPARISON (ALL 5 METRICS)
# ============================================================================

def create_comparison_figure(metrics_df, output_path='comparison_all_conditions.png', condition_key=None):
    """Create comprehensive comparison figure with 95% CI error bars."""
    
    # Filter for 'all' subset
    all_df = metrics_df[metrics_df['subset'] == 'all'].copy()
    num_conditions = len(all_df)
    methods = all_df['condition'].tolist()
    
    # Get title based on condition
    if condition_key and condition_key in CONDITION_TITLES:
        title = f'Detailed Results for {CONDITION_TITLES[condition_key]}'
    else:
        title = f'Comparison: All {num_conditions} Conditions'
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # ---------- 1. Accuracy (sorted descending) ----------
    ax = axes[0, 0]
    sorted_df = all_df.sort_values('correct_rate', ascending=False)
    conditions = sorted_df['condition'].tolist()
    values = sorted_df['correct_rate'].values * 100
    errors_lower = sorted_df['ci_accuracy_lower'].values
    errors_upper = sorted_df['ci_accuracy_upper'].values
    colors = [COLORS.get(c, '#999999') for c in conditions]
    
    bars = ax.bar(range(len(conditions)), values, color=colors, edgecolor='black', width=0.7,
                  yerr=[errors_lower, errors_upper], capsize=4, error_kw={'linewidth': 1.5, 'ecolor': '#333'})
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Accuracy', fontsize=12, fontweight='bold')
    max_val = max(values + errors_upper)
    ax.set_ylim(0, min(105, max_val * 1.1))  # Cap at 105 for visual space
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8, fontweight='bold')
    for i, (val, err_up) in enumerate(zip(values, errors_upper)):
        ax.text(i, val + err_up + 1.5, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # ---------- 2. Correct & Convergence Rate (sorted descending) ----------
    ax = axes[0, 1]
    sorted_df = all_df.sort_values('correct_converge_rate', ascending=False)
    conditions = sorted_df['condition'].tolist()
    values = sorted_df['correct_converge_rate'].values * 100
    errors_lower = sorted_df['ci_convergence_lower'].values
    errors_upper = sorted_df['ci_convergence_upper'].values
    colors = [COLORS.get(c, '#999999') for c in conditions]
    
    bars = ax.bar(range(len(conditions)), values, color=colors, edgecolor='black', width=0.7,
                  yerr=[errors_lower, errors_upper], capsize=4, error_kw={'linewidth': 1.5, 'ecolor': '#333'})
    ax.set_ylabel('Correct & Convergence Rate (%)', fontsize=11)
    ax.set_title('Correct & Convergence Rate', fontsize=12, fontweight='bold')
    max_val = max(values + errors_upper)
    ax.set_ylim(0, min(105, max_val * 1.1))  # Cap at 105 for visual space
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8, fontweight='bold')
    for i, (val, err_up) in enumerate(zip(values, errors_upper)):
        ax.text(i, val + err_up + 1.5, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # ---------- 3. Avg Questions (sorted ascending) ----------
    ax = axes[0, 2]
    sorted_df = all_df.sort_values('avg_questions', ascending=True)
    conditions = sorted_df['condition'].tolist()
    values = sorted_df['avg_questions'].values
    errors = sorted_df['ci_questions'].values
    colors = [COLORS.get(c, '#999999') for c in conditions]
    
    bars = ax.bar(range(len(conditions)), values, color=colors, edgecolor='black', width=0.7,
                  yerr=errors, capsize=4, error_kw={'linewidth': 1.5, 'ecolor': '#333'})
    ax.set_ylabel('Number of Questions', fontsize=11)
    ax.set_title('Average Questions', fontsize=12, fontweight='bold')
    max_val = max(values + errors)
    ax.set_ylim(0, max_val * 1.25)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8, fontweight='bold')
    for i, (val, err) in enumerate(zip(values, errors)):
        ax.text(i, val + err + max_val * 0.04, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # ---------- 4. Avg Cost (sorted ascending) ----------
    ax = axes[1, 0]
    sorted_df = all_df.sort_values('avg_cost', ascending=True)
    conditions = sorted_df['condition'].tolist()
    values = sorted_df['avg_cost'].values
    errors = sorted_df['ci_cost'].values
    colors = [COLORS.get(c, '#999999') for c in conditions]
    
    bars = ax.bar(range(len(conditions)), values, color=colors, edgecolor='black', width=0.7,
                  yerr=errors, capsize=4, error_kw={'linewidth': 1.5, 'ecolor': '#333'})
    ax.set_ylabel('Cost ($)', fontsize=11)
    ax.set_title('Average Cost', fontsize=12, fontweight='bold')
    max_val = max(values + errors)
    ax.set_ylim(0, max_val * 1.25)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8, fontweight='bold')
    for i, (val, err) in enumerate(zip(values, errors)):
        ax.text(i, val + err + max_val * 0.04, f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    # ---------- 5. Avg Duration (sorted ascending) ----------
    ax = axes[1, 1]
    sorted_df = all_df.sort_values('avg_duration', ascending=True)
    conditions = sorted_df['condition'].tolist()
    values = sorted_df['avg_duration'].values
    errors = sorted_df['ci_duration'].values
    colors = [COLORS.get(c, '#999999') for c in conditions]
    
    bars = ax.bar(range(len(conditions)), values, color=colors, edgecolor='black', width=0.7,
                  yerr=errors, capsize=4, error_kw={'linewidth': 1.5, 'ecolor': '#333'})
    ax.set_ylabel('Duration (seconds)', fontsize=11)
    ax.set_title('Average Duration', fontsize=12, fontweight='bold')
    max_val = max(values + errors)
    ax.set_ylim(0, max_val * 1.25)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8, fontweight='bold')
    for i, (val, err) in enumerate(zip(values, errors)):
        ax.text(i, val + err + max_val * 0.04, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # ---------- 6. Average Confidence (sorted descending) ----------
    ax = axes[1, 2]
    sorted_df = all_df.sort_values('avg_confidence', ascending=False)
    conditions = sorted_df['condition'].tolist()
    values = sorted_df['avg_confidence'].values * 100  # Convert to percentage
    # Use std_confidence for error bars (as CI would require different calculation)
    errors = sorted_df['std_confidence'].values * 100
    colors = [COLORS.get(c, '#999999') for c in conditions]
    
    bars = ax.bar(range(len(conditions)), values, color=colors, edgecolor='black', width=0.7,
                  yerr=errors, capsize=4, error_kw={'linewidth': 1.5, 'ecolor': '#333'})
    ax.set_ylabel('Confidence (%)', fontsize=11)
    ax.set_title('Average Final Confidence', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8, fontweight='bold')
    for i, (val, err) in enumerate(zip(values, errors)):
        ax.text(i, min(val + err + 2, 102), f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # REMOVED: Error bars note (will be in caption)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.02)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    # Also save as PDF
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Figure saved: {output_path}")
    print(f"Figure saved: {pdf_path}")


# ============================================================================
# FIGURE: PARETO PLOT (ACCURACY VS COST) - Optimized for single-column layout
# ============================================================================

def create_pareto_figure(metrics_df, output_path='pareto_accuracy_vs_cost.png', condition_key=None):
    """Create Pareto front scatter plot: Accuracy vs Cost with distinct shapes and colors.
    Includes 95% CI error bars for both axes.
    Features an inset showing full range including outliers (LLM-SELECT).
    Optimized for single-column layout with larger fonts."""
    
    all_df = metrics_df[metrics_df['subset'] == 'all'].copy()
    
    # Separate outliers (LLM-SELECT) from main methods
    outlier_methods = ['LLM-SELECT']
    main_df = all_df[~all_df['condition'].isin(outlier_methods)]
    outlier_df = all_df[all_df['condition'].isin(outlier_methods)]
    
    # Check if we have outliers to show in inset
    has_outliers = len(outlier_df) > 0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate axis limits for main plot (excluding outliers)
    main_costs = main_df['avg_cost'].values
    main_cost_cis = main_df['ci_cost'].values
    main_accs = main_df['correct_rate'].values * 100
    main_acc_upper = main_df['ci_accuracy_upper'].values
    
    # Main plot limits with padding
    main_x_min = 0
    main_x_max = max(main_costs + main_cost_cis) * 1.2
    main_y_min = min(main_accs - main_df['ci_accuracy_lower'].values) * 0.9
    main_y_max = min(105, max(main_accs + main_acc_upper) * 1.08)
    
    # Plot main methods with error bars
    for _, row in main_df.iterrows():
        method = row['condition']
        acc = row['correct_rate'] * 100
        cost = row['avg_cost']
        
        # Get CI values
        acc_err_lower = row['ci_accuracy_lower']
        acc_err_upper = row['ci_accuracy_upper']
        cost_err = row['ci_cost']
        
        # Plot error bars first (behind markers)
        ax.errorbar(cost, acc,
                    xerr=cost_err,
                    yerr=[[acc_err_lower], [acc_err_upper]],
                    fmt='none',
                    ecolor=COLORS.get(method, '#999999'),
                    elinewidth=1.5,
                    capsize=4,
                    capthick=1.5,
                    alpha=0.6,
                    zorder=2)
        
        # Plot scatter point on top
        ax.scatter(cost, acc, 
                   c=COLORS.get(method, '#999999'), 
                   marker=MARKERS.get(method, 'o'),
                   s=220,
                   edgecolors='white',
                   linewidths=1.5,
                   label=method,
                   zorder=3)
    
    # Set main plot limits
    ax.set_xlim(main_x_min, main_x_max)
    ax.set_ylim(main_y_min, main_y_max)
    
    # Larger fonts for single-column layout
    ax.set_xlabel('Average Cost ($)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    
    # Get title based on condition
    if condition_key and condition_key in CONDITION_TITLES:
        title = f'Accuracy vs. Cost: {CONDITION_TITLES[condition_key]}'
    else:
        title = 'Accuracy vs. Cost Tradeoff'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
    
    # Larger tick fonts
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    
    # Legend with larger font - position adjusted
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95, 
              edgecolor='#cccccc', fancybox=False)
    
    # Add "Better" arrow annotation
    ax.annotate('', xy=(main_x_min + main_x_max*0.02, main_y_max - 2),
                xytext=(main_x_max*0.15, main_y_max - 2),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(main_x_max*0.08, main_y_max - 0.5, 'Better', fontsize=11, 
            color='green', fontweight='bold', ha='center', va='bottom')
    
    # ==================== INSET PLOT ====================
    if has_outliers:
        # Create inset axes - smaller and positioned in top-right corner
        # [left, bottom, width, height] in figure coordinates
        ax_inset = fig.add_axes([0.62, 0.58, 0.28, 0.28])
        
        # Full range limits including outliers
        all_costs = all_df['avg_cost'].values
        all_cost_cis = all_df['ci_cost'].values
        all_accs = all_df['correct_rate'].values * 100
        
        full_x_max = max(all_costs + all_cost_cis) * 1.1
        full_y_min = 0
        full_y_max = 105
        
        # Plot ALL methods in inset (smaller markers)
        for _, row in all_df.iterrows():
            method = row['condition']
            acc = row['correct_rate'] * 100
            cost = row['avg_cost']
            
            # Get CI values
            acc_err_lower = row['ci_accuracy_lower']
            acc_err_upper = row['ci_accuracy_upper']
            cost_err = row['ci_cost']
            
            # Error bars (thinner for inset)
            ax_inset.errorbar(cost, acc,
                        xerr=cost_err,
                        yerr=[[acc_err_lower], [acc_err_upper]],
                        fmt='none',
                        ecolor=COLORS.get(method, '#999999'),
                        elinewidth=0.8,
                        capsize=1.5,
                        capthick=0.8,
                        alpha=0.5,
                        zorder=2)
            
            # Scatter points (smaller)
            ax_inset.scatter(cost, acc, 
                       c=COLORS.get(method, '#999999'), 
                       marker=MARKERS.get(method, 'o'),
                       s=40,
                       edgecolors='white',
                       linewidths=0.5,
                       zorder=3)
        
        # Set inset limits
        ax_inset.set_xlim(0, full_x_max)
        ax_inset.set_ylim(full_y_min, full_y_max)
        
        # Draw rectangle showing main plot region
        from matplotlib.patches import Rectangle
        rect = Rectangle((main_x_min, main_y_min), 
                         main_x_max - main_x_min, 
                         main_y_max - main_y_min,
                         fill=False, 
                         edgecolor='#E63946', 
                         linewidth=1.5,
                         linestyle='-',
                         zorder=5)
        ax_inset.add_patch(rect)
        
        # Style inset - minimal
        ax_inset.set_facecolor('#f9f9f9')
        ax_inset.grid(True, alpha=0.2, linestyle=':', zorder=0)
        ax_inset.tick_params(axis='both', which='major', labelsize=7)
        ax_inset.set_xlabel('Cost ($)', fontsize=7, fontweight='bold')
        ax_inset.set_ylabel('Acc (%)', fontsize=7, fontweight='bold')
        
        # Add border to inset
        for spine in ax_inset.spines.values():
            spine.set_edgecolor('#888888')
            spine.set_linewidth(1)
        
        # Label the outlier in inset - compact
        for _, row in outlier_df.iterrows():
            method = row['condition']
            acc = row['correct_rate'] * 100
            cost = row['avg_cost']
            ax_inset.annotate(method, 
                            xy=(cost, acc),
                            xytext=(cost - full_x_max*0.35, acc + 12),
                            fontsize=6,
                            fontweight='bold',
                            color=COLORS.get(method, '#999999'),
                            arrowprops=dict(arrowstyle='->', 
                                          color=COLORS.get(method, '#999999'),
                                          lw=0.8),
                            zorder=6)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    # Also save as PDF
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Figure saved: {output_path}")
    print(f"Figure saved: {pdf_path}")


# ============================================================================
# FIGURE: CORRECT VS INCORRECT COMPARISON
# ============================================================================

def create_correct_incorrect_comparison(df, metrics_df, output_path='correct_vs_incorrect.png', condition_key=None):
    """Create figure comparing correct vs incorrect predictions."""
    
    conditions = sorted(df['condition'].unique())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get title based on condition
    if condition_key and condition_key in CONDITION_TITLES:
        title = f'Correct vs Incorrect: {CONDITION_TITLES[condition_key]}'
    else:
        title = 'Correct vs Incorrect Predictions'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    metrics_to_plot = [
        ('avg_questions', 'ci_questions', 'Average Questions', 'Number of Questions'),
        ('avg_cost', 'ci_cost', 'Average Cost', 'Cost ($)'),
        ('avg_duration', 'ci_duration', 'Average Duration', 'Duration (s)')
    ]
    
    for idx, (metric, ci_col, title_subplot, ylabel) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        x = np.arange(len(conditions))
        width = 0.35
        
        correct_vals = []
        correct_cis = []
        incorrect_vals = []
        incorrect_cis = []
        
        for cond in conditions:
            correct_row = metrics_df[(metrics_df['condition'] == cond) & (metrics_df['subset'] == 'correct')]
            incorrect_row = metrics_df[(metrics_df['condition'] == cond) & (metrics_df['subset'] == 'incorrect')]
            
            correct_vals.append(correct_row[metric].values[0] if len(correct_row) > 0 else 0)
            correct_cis.append(correct_row[ci_col].values[0] if len(correct_row) > 0 else 0)
            incorrect_vals.append(incorrect_row[metric].values[0] if len(incorrect_row) > 0 else 0)
            incorrect_cis.append(incorrect_row[ci_col].values[0] if len(incorrect_row) > 0 else 0)
        
        bars1 = ax.bar(x - width/2, correct_vals, width, label='Correct', color='#2ecc71', 
                       edgecolor='black', yerr=correct_cis, capsize=3)
        bars2 = ax.bar(x + width/2, incorrect_vals, width, label='Incorrect', color='#e74c3c', 
                       edgecolor='black', yerr=incorrect_cis, capsize=3)
        
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title_subplot, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c[:12] for c in conditions], rotation=45, ha='right', fontsize=9, fontweight='bold')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # REMOVED: Error bars note (will be in caption)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    # Also save as PDF
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Figure saved: {output_path}")
    print(f"Figure saved: {pdf_path}")


# ============================================================================
# LATEX TABLE EXPORT
# ============================================================================

def export_to_latex(metrics_df, output_path='table.tex', condition_key=None):
    """Export metrics to a LaTeX table file using booktabs style with 95% CIs."""
    
    all_df = metrics_df[metrics_df['subset'] == 'all'].copy()
    all_df = all_df.sort_values('correct_rate', ascending=False)  # Sort by accuracy
    
    # Get condition title for caption
    if condition_key and condition_key in CONDITION_TITLES:
        condition_title = CONDITION_TITLES[condition_key]
    else:
        condition_title = "All Conditions"
    
    # Build LaTeX content
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Summary results for " + condition_title + r". Values show mean with 95\% CI in brackets.}",
        r"\label{tab:results-" + (condition_key if condition_key else "all") + r"}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Method & Acc (\%) & C\&C (\%) & Questions & Cost (\$) & Duration (s) \\",
        r"\midrule",
    ]
    
    for _, row in all_df.iterrows():
        method = row['condition']
        acc = row['correct_rate'] * 100
        cc = row['correct_converge_rate'] * 100
        q = row['avg_questions']
        cost = row['avg_cost']
        duration = row['avg_duration']
        
        # Get CI values for proportions (Wilson)
        acc_lo, acc_hi = calc_ci_proportion_wilson(acc, row['n'])
        cc_lo, cc_hi = calc_ci_proportion_wilson(cc, row['n'])
        
        # Get CI values for continuous variables
        q_ci = row['ci_questions']
        cost_ci = row['ci_cost']
        dur_ci = row['ci_duration']
        
        # Format with CI in brackets
        acc_str = f"{acc:.1f} [{acc_lo:.1f}--{acc_hi:.1f}]"
        cc_str = f"{cc:.1f} [{cc_lo:.1f}--{cc_hi:.1f}]"
        q_str = f"{q:.1f} $\\pm$ {q_ci:.1f}"
        cost_str = f"{cost:.0f} $\\pm$ {cost_ci:.0f}"
        dur_str = f"{duration:.1f} $\\pm$ {dur_ci:.1f}"
        
        latex_lines.append(
            f"{method} & {acc_str} & {cc_str} & {q_str} & {cost_str} & {dur_str} \\\\"
        )
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"LaTeX table exported to: {output_path}")


# ============================================================================
# FIGURE: DIAGNOSIS × METHOD HEATMAP
# ============================================================================

def create_diagnosis_heatmap(df, output_path='diagnosis_heatmap.png', condition_key=None, top_n=15):
    """Create heatmap showing accuracy for each diagnosis × method combination."""
    
    conditions = sorted(df['condition'].unique())
    n_conditions = len(conditions)
    
    # Get top N diagnoses by frequency
    top_diagnoses = df['true_diagnosis'].value_counts().head(top_n).index.tolist()
    
    # Build accuracy matrix
    acc_matrix = []
    patient_counts = []
    for diag in top_diagnoses:
        diag_df = df[df['true_diagnosis'] == diag]
        n_patients = len(diag_df) // n_conditions  # divide by number of conditions
        patient_counts.append(n_patients)
        row = []
        for cond in conditions:
            cond_diag_df = diag_df[diag_df['condition'] == cond]
            if len(cond_diag_df) > 0:
                row.append(cond_diag_df['correct'].mean() * 100)
            else:
                row.append(np.nan)
        acc_matrix.append(row)
    
    acc_matrix = np.array(acc_matrix)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot heatmap
    im = ax.imshow(acc_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    
    # Add text annotations
    for i in range(len(top_diagnoses)):
        for j in range(len(conditions)):
            val = acc_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 60 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', 
                        fontsize=9, color=color, fontweight='bold')
    
    # Set labels
    ax.set_xticks(range(len(conditions)))
    ax.set_yticks(range(len(top_diagnoses)))
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=10, fontweight='bold')
    
    # Add patient counts to y-axis labels
    y_labels = [f"{d[:35]} (n={patient_counts[i]})" for i, d in enumerate(top_diagnoses)]
    ax.set_yticklabels(y_labels, fontsize=10)
    
    # Get title based on condition
    if condition_key and condition_key in CONDITION_TITLES:
        title = f'Accuracy by Diagnosis: {CONDITION_TITLES[condition_key]}'
    else:
        title = 'Accuracy by Diagnosis × Method'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Diagnosis', fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Accuracy (%)', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    # Also save as PDF
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Figure saved: {output_path}")
    print(f"Figure saved: {pdf_path}")


# ============================================================================
# FIGURE: CONFIDENCE DISTRIBUTION HISTOGRAM
# ============================================================================

def create_confidence_histogram(df, output_path='confidence_histogram.png', condition_key=None):
    """Create histogram showing confidence distribution for correct vs incorrect predictions."""
    
    conditions = sorted(df['condition'].unique())
    n_conditions = len(conditions)
    
    # Determine grid size
    n_cols = 3
    n_rows = (n_conditions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_conditions > 1 else [axes]
    
    # Get title based on condition
    if condition_key and condition_key in CONDITION_TITLES:
        title = f'Confidence Distribution: {CONDITION_TITLES[condition_key]}'
    else:
        title = 'Confidence Distribution: Correct vs Incorrect'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    for idx, cond in enumerate(conditions):
        ax = axes[idx]
        cond_df = df[df['condition'] == cond]
        
        correct_conf = cond_df[cond_df['correct'] == True]['final_confidence']
        incorrect_conf = cond_df[cond_df['correct'] == False]['final_confidence']
        
        bins = np.linspace(0, 1, 21)
        
        # Plot histograms
        ax.hist(correct_conf, bins=bins, alpha=0.6, color='#2ecc71', 
                label=f'Correct (n={len(correct_conf)})', density=True, edgecolor='white')
        ax.hist(incorrect_conf, bins=bins, alpha=0.6, color='#e74c3c', 
                label=f'Incorrect (n={len(incorrect_conf)})', density=True, edgecolor='white')
        
        # Add vertical lines for means
        if len(correct_conf) > 0:
            ax.axvline(correct_conf.mean(), color='#27ae60', linestyle='--', 
                       linewidth=2, label=f'Mean correct: {correct_conf.mean():.2f}')
        if len(incorrect_conf) > 0:
            ax.axvline(incorrect_conf.mean(), color='#c0392b', linestyle='--', 
                       linewidth=2, label=f'Mean incorrect: {incorrect_conf.mean():.2f}')
        
        ax.set_xlim(0, 1)
        ax.set_xlabel('Confidence', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{cond}', fontsize=11, fontweight='bold')
        
        # Bold legend
        legend = ax.legend(loc='upper left', fontsize=8)
        for text in legend.get_texts():
            text.set_fontweight('bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide empty subplots
    for idx in range(len(conditions), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    # Also save as PDF
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Figure saved: {output_path}")
    print(f"Figure saved: {pdf_path}")


# ============================================================================
# TABLE: DISCRIMINABILITY (LATEX)
# ============================================================================

def compute_discriminability(df):
    """Compute discriminability (Cohen's d) for each condition."""
    
    conditions = sorted(df['condition'].unique())
    results = []
    
    for cond in conditions:
        cond_df = df[df['condition'] == cond]
        correct_conf = cond_df[cond_df['correct'] == True]['final_confidence']
        incorrect_conf = cond_df[cond_df['correct'] == False]['final_confidence']
        
        mean_correct = correct_conf.mean() if len(correct_conf) > 0 else 0
        mean_incorrect = incorrect_conf.mean() if len(incorrect_conf) > 0 else 0
        gap = mean_correct - mean_incorrect
        
        # Cohen's d
        if len(correct_conf) > 1 and len(incorrect_conf) > 1:
            pooled_std = np.sqrt(
                ((len(correct_conf) - 1) * correct_conf.std()**2 + 
                 (len(incorrect_conf) - 1) * incorrect_conf.std()**2) / 
                (len(correct_conf) + len(incorrect_conf) - 2)
            )
            cohens_d = gap / pooled_std if pooled_std > 0 else 0
        else:
            cohens_d = 0
        
        results.append({
            'condition': cond,
            'n_correct': len(correct_conf),
            'n_incorrect': len(incorrect_conf),
            'mean_correct': mean_correct,
            'mean_incorrect': mean_incorrect,
            'gap': gap,
            'cohens_d': cohens_d
        })
    
    return pd.DataFrame(results)


def export_discriminability_latex(df, output_path='table_discriminability.tex', condition_key=None):
    """Export discriminability table to LaTeX."""
    
    discrim_df = compute_discriminability(df)
    discrim_df = discrim_df.sort_values('cohens_d', ascending=False)
    
    # Get condition title for caption
    if condition_key and condition_key in CONDITION_TITLES:
        condition_title = CONDITION_TITLES[condition_key]
    else:
        condition_title = "All Conditions"
    
    # Build LaTeX content
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Discriminability analysis for " + condition_title + r". Cohen's $d$ measures how well confidence separates correct from incorrect predictions. Higher values indicate better self-awareness.}",
        r"\label{tab:discriminability-" + (condition_key if condition_key else "all") + r"}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Method & Conf (Correct) & Conf (Incorrect) & Gap & Cohen's $d$ & Interpretation \\",
        r"\midrule",
    ]
    
    for _, row in discrim_df.iterrows():
        method = row['condition']
        mean_c = row['mean_correct']
        mean_i = row['mean_incorrect']
        gap = row['gap']
        d = row['cohens_d']
        
        # Interpretation
        if d < 0:
            interp = "Inverted"
        elif d < 0.5:
            interp = "Small"
        elif d < 0.8:
            interp = "Medium"
        elif d < 1.2:
            interp = "Large"
        else:
            interp = "Very Large"
        
        latex_lines.append(
            f"{method} & {mean_c:.3f} & {mean_i:.3f} & {gap:+.3f} & {d:.2f} & {interp} \\\\"
        )
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"LaTeX table exported to: {output_path}")


# ============================================================================
# TABLE: CALIBRATION (LATEX)
# ============================================================================

def compute_calibration_summary(df):
    """Compute calibration summary for each condition (high-confidence region)."""
    
    conditions = sorted(df['condition'].unique())
    results = []
    
    for cond in conditions:
        cond_df = df[df['condition'] == cond]
        
        # High confidence region (>= 0.9) where most data lives
        high_conf_df = cond_df[cond_df['final_confidence'] >= 0.9]
        
        if len(high_conf_df) > 0:
            avg_conf = high_conf_df['final_confidence'].mean()
            accuracy = high_conf_df['correct'].mean()
            gap = avg_conf - accuracy
            n_high_conf = len(high_conf_df)
            pct_high_conf = len(high_conf_df) / len(cond_df) * 100
        else:
            avg_conf = 0
            accuracy = 0
            gap = 0
            n_high_conf = 0
            pct_high_conf = 0
        
        results.append({
            'condition': cond,
            'n_total': len(cond_df),
            'n_high_conf': n_high_conf,
            'pct_high_conf': pct_high_conf,
            'avg_confidence': avg_conf,
            'accuracy': accuracy,
            'calibration_gap': gap
        })
    
    return pd.DataFrame(results)


def export_calibration_latex(df, output_path='table_calibration.tex', condition_key=None):
    """Export calibration table to LaTeX."""
    
    calib_df = compute_calibration_summary(df)
    calib_df = calib_df.sort_values('calibration_gap', ascending=True)
    
    # Get condition title for caption
    if condition_key and condition_key in CONDITION_TITLES:
        condition_title = CONDITION_TITLES[condition_key]
    else:
        condition_title = "All Conditions"
    
    # Build LaTeX content
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Calibration analysis for " + condition_title + r". Analysis restricted to high-confidence predictions (confidence $\geq 0.9$), where the majority of predictions fall. ``\% High Conf'' shows the proportion of predictions in this region. ``Avg Conf'' and ``Accuracy'' are computed within this high-confidence subset. Gap = Avg Conf $-$ Accuracy; positive values indicate overconfidence.}",
        r"\label{tab:calibration-" + (condition_key if condition_key else "all") + r"}",
        r"\begin{tabular}{@{}lccccl@{}}",
        r"\toprule",
        r"Method & \% High Conf & Avg Conf & Accuracy & Gap & Status \\",
        r"\midrule",
    ]
    
    for _, row in calib_df.iterrows():
        method = row['condition']
        pct = row['pct_high_conf']
        conf = row['avg_confidence']
        acc = row['accuracy']
        gap = row['calibration_gap']
        
        # Status
        if abs(gap) < 0.05:
            status = "Well calibrated"
        elif gap > 0:
            status = "Overconfident"
        else:
            status = "Underconfident"
        
        latex_lines.append(
            f"{method} & {pct:.1f}\\% & {conf*100:.1f}\\% & {acc*100:.1f}\\% & {gap*100:+.1f}\\% & {status} \\\\"
        )
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"LaTeX table exported to: {output_path}")


# ============================================================================
# CSV EXPORT
# ============================================================================

def export_to_csv(metrics_df, output_path='metrics_summary.csv'):
    """Export metrics to CSV."""
    metrics_df.to_csv(output_path, index=False)
    print(f"Metrics exported to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main(json_file_path, output_dir='.', condition_key=None):
    """Main analysis function."""
    
    print("="*100)
    print("MEDICAL DIAGNOSIS EXPERIMENT ANALYSIS (with 95% CI)")
    print("="*100)
    print(f"\nInput file: {json_file_path}")
    if condition_key:
        print(f"Condition: {condition_key} -> {CONDITION_TITLES.get(condition_key, 'Unknown')}")
    
    # Load and analyze
    df, metrics_df = load_and_analyze(json_file_path)
    
    # Print summary
    print_summary_table(metrics_df)
    
    # Print detailed breakdown
    print("\n" + "="*100)
    print("DETAILED BREAKDOWN BY CONDITION")
    print("="*100)
    
    for condition in sorted(df['condition'].unique()):
        print_detailed_breakdown(metrics_df, condition)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figures with condition-specific filenames
    print("\n" + "="*100)
    print("GENERATING FIGURES")
    print("="*100)
    
    if condition_key:
        # Use condition-specific filenames
        all_filename = f'all_{condition_key}.png'
        pareto_filename = f'pareto_{condition_key}.png'
        correct_incorrect_filename = f'correct_vs_incorrect_{condition_key}.png'
        diagnosis_heatmap_filename = f'diagnosis_heatmap_{condition_key}.png'
        confidence_hist_filename = f'confidence_histogram_{condition_key}.png'
        csv_filename = f'metrics_summary_{condition_key}.csv'
        latex_filename = f'table_{condition_key}.tex'
        discriminability_latex_filename = f'table_discriminability_{condition_key}.tex'
        calibration_latex_filename = f'table_calibration_{condition_key}.tex'
    else:
        # Default filenames
        all_filename = 'comparison_all_conditions.png'
        pareto_filename = 'pareto_accuracy_vs_cost.png'
        correct_incorrect_filename = 'correct_vs_incorrect.png'
        diagnosis_heatmap_filename = 'diagnosis_heatmap.png'
        confidence_hist_filename = 'confidence_histogram.png'
        csv_filename = 'metrics_summary.csv'
        latex_filename = 'table.tex'
        discriminability_latex_filename = 'table_discriminability.tex'
        calibration_latex_filename = 'table_calibration.tex'
    
    create_comparison_figure(metrics_df, output_dir / all_filename, condition_key)
    create_pareto_figure(metrics_df, output_dir / pareto_filename, condition_key)
    # REMOVED: create_pareto_figure_wide (no longer needed)
    create_correct_incorrect_comparison(df, metrics_df, output_dir / correct_incorrect_filename, condition_key)
    create_diagnosis_heatmap(df, output_dir / diagnosis_heatmap_filename, condition_key)
    create_confidence_histogram(df, output_dir / confidence_hist_filename, condition_key)
    
    # Export CSV
    export_to_csv(metrics_df, output_dir / csv_filename)
    
    # Export LaTeX tables
    export_to_latex(metrics_df, output_dir / latex_filename, condition_key)
    export_discriminability_latex(df, output_dir / discriminability_latex_filename, condition_key)
    export_calibration_latex(df, output_dir / calibration_latex_filename, condition_key)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"All outputs saved to: {output_dir}")
    
    return df, metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze medical diagnosis experiment results (with 95% CI)")
    parser.add_argument("json_file", help="Path to JSON results file")
    parser.add_argument("--output-dir", default=".", help="Output directory for figures and CSV")
    parser.add_argument("--condition", default=None, 
                        choices=['notprune_notuniform', 'notprune_uniform', 'prune_notuniform', 'prune_uniform'],
                        help="Condition key for title and filename customization")
    
    args = parser.parse_args()
    
    if not Path(args.json_file).exists():
        print(f"Error: File not found: {args.json_file}")
        sys.exit(1)
    
    main(args.json_file, args.output_dir, args.condition)