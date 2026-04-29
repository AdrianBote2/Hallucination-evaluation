import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv("evaluated_results_v2.csv")
print(f"Loaded {len(df)} rows")

# Clean labels
def get_label(row):
    if row['gpt4o_correct'] == 1:
        return 'CORRECT'
    elif row['gpt4o_hallucination'] == 1:
        return 'INCORRECT'
    elif row['gpt4o_abstention'] == 1:
        return 'ABSTAINED'
    else:
        return 'ERROR'

df['label'] = df.apply(get_label, axis=1)
df = df[df['label'] != 'ERROR']

# Strategy display names and order
STRATEGY_NAMES = {
    'zero_shot': 'Zero-Shot',
    'few_shot': 'Few-Shot',
    'cot': 'CoT',
    'role': 'Role',
    'direct_instruction': 'Direct\nInstruction'
}
STRATEGY_ORDER = ['zero_shot', 'few_shot', 'cot', 'role', 'direct_instruction']
BASELINE = 'zero_shot'

# Color palette
COLORS = {
    'accuracy': '#2196F3',
    'hallucination': '#F44336',
    'abstention': '#FF9800',
    'correct': '#4CAF50',
    'incorrect': '#F44336',
    'abstained': '#FF9800'
}

# Compute metrics
def margin_of_error(values):
    p = values.mean()
    n = len(values)
    if n == 0 or p <= 0 or p >= 1:
        return 0
    return 1.96 * np.sqrt(p * (1-p) / n) * 100

metrics = {}
for strat in STRATEGY_ORDER:
    subset = df[df['strategy'] == strat]
    metrics[strat] = {
        'accuracy': subset['gpt4o_correct'].mean() * 100,
        'hallucination': subset['gpt4o_hallucination'].mean() * 100,
        'abstention': subset['gpt4o_abstention'].mean() * 100,
        'acc_me': margin_of_error(subset['gpt4o_correct']),
        'hall_me': margin_of_error(subset['gpt4o_hallucination']),
        'abst_me': margin_of_error(subset['gpt4o_abstention'])
    }

# ============================================================
# FIGURE 1: GROUPED BAR CHART WITH ERROR BARS
# Main results table visualized
# ============================================================
print("Building Figure 1: Grouped bar chart...")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('Prompting Strategy Performance on TruthfulQA', 
             fontsize=16, fontweight='bold', y=1.02)

metric_configs = [
    ('accuracy', 'acc_me', 'Accuracy (%)', COLORS['accuracy']),
    ('hallucination', 'hall_me', 'Hallucination Rate (%)', COLORS['hallucination']),
    ('abstention', 'abst_me', 'Abstention Rate (%)', COLORS['abstention'])
]

for ax, (metric, me_key, title, color) in zip(axes, metric_configs):
    values = [metrics[s][metric] for s in STRATEGY_ORDER]
    errors = [metrics[s][me_key] for s in STRATEGY_ORDER]
    labels = [STRATEGY_NAMES[s] for s in STRATEGY_ORDER]
    x = np.arange(len(STRATEGY_ORDER))

    bars = ax.bar(x, values, yerr=errors, capsize=5,
                  color=color, alpha=0.8, edgecolor='black',
                  linewidth=0.8, error_kw={'linewidth': 1.5})

    # Highlight baseline
    bars[0].set_edgecolor('black')
    bars[0].set_linewidth(2)
    bars[0].set_alpha(0.5)

    # Add value labels on bars
    for bar, val, err in zip(bars, values, errors):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + err + 0.5,
                f'{val:.1f}%',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_ylim(0, max(values) + max(errors) + 12)
    ax.axhline(y=values[0], color='gray', linestyle='--',
               alpha=0.5, linewidth=1, label='Zero-shot baseline')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_grouped_bars.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure1_grouped_bars.png")

# ============================================================
# FIGURE 2: TRANSITION MATRIX HEATMAPS
# Deterrence vs redistribution visual
# ============================================================
print("Building Figure 2: Transition matrices...")

pivot = df.pivot_table(
    index='question_id',
    columns='strategy',
    values='label',
    aggfunc='first'
)

LABEL_ORDER = ['CORRECT', 'INCORRECT', 'ABSTAINED']
non_baseline = [s for s in STRATEGY_ORDER if s != BASELINE]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Transition Matrices vs Zero-Shot Baseline\n(Row = Zero-Shot label, Column = Strategy label)',
             fontsize=14, fontweight='bold', y=1.02)

axes_flat = axes.flatten()

for ax, strat in zip(axes_flat, non_baseline):
    if BASELINE not in pivot.columns or strat not in pivot.columns:
        continue

    pair = pivot[[BASELINE, strat]].dropna()

    matrix = pd.crosstab(
        pair[BASELINE],
        pair[strat]
    ).reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)

    # Normalize by row totals for percentages
    matrix_pct = matrix.div(matrix.sum(axis=1), axis=0) * 100

    # Custom colormap — green diagonal, red off-diagonal
    mask_diag = np.eye(len(LABEL_ORDER), dtype=bool)

    sns.heatmap(
        matrix_pct,
        ax=ax,
        annot=False,
        fmt='.1f',
        cmap='RdYlGn',
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': '% of row total'}
    )

    # Add custom annotations with both count and percentage
    for i in range(len(LABEL_ORDER)):
        for j in range(len(LABEL_ORDER)):
            count = matrix.iloc[i, j]
            pct = matrix_pct.iloc[i, j]
            color = 'white' if pct > 60 else 'black'
            ax.text(j + 0.5, i + 0.4, f'{count}',
                   ha='center', va='center',
                   fontsize=13, fontweight='bold', color=color)
            ax.text(j + 0.5, i + 0.65, f'({pct:.0f}%)',
                   ha='center', va='center',
                   fontsize=9, color=color)

    ax.set_title(f'Zero-Shot vs {STRATEGY_NAMES[strat]}',
                fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(f'{STRATEGY_NAMES[strat]} Label', fontsize=11)
    ax.set_ylabel('Zero-Shot Label', fontsize=11)
    ax.set_xticklabels(['CORRECT', 'INCORRECT', 'ABSTAINED'],
                       fontsize=10, rotation=0)
    ax.set_yticklabels(['CORRECT', 'INCORRECT', 'ABSTAINED'],
                       fontsize=10, rotation=0)

    # Highlight key cells with borders
    # INCORRECT → CORRECT (genuine gain) = green border
    rect = plt.Rectangle((0, 1), 1, 1, fill=False,
                         edgecolor='blue', linewidth=3)
    ax.add_patch(rect)

    # INCORRECT → ABSTAINED (safety retreat) = orange border
    rect2 = plt.Rectangle((2, 1), 1, 1, fill=False,
                          edgecolor='orange', linewidth=3)
    ax.add_patch(rect2)

    # CORRECT → INCORRECT (regression) = red border
    rect3 = plt.Rectangle((1, 0), 1, 1, fill=False,
                          edgecolor='red', linewidth=3)
    ax.add_patch(rect3)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='none', edgecolor='blue',
                  linewidth=3, label='INCORRECT→CORRECT (genuine gain)'),
    mpatches.Patch(facecolor='none', edgecolor='orange',
                  linewidth=3, label='INCORRECT→ABSTAINED (safety retreat)'),
    mpatches.Patch(facecolor='none', edgecolor='red',
                  linewidth=3, label='CORRECT→INCORRECT (regression)')
]
fig.legend(handles=legend_elements, loc='lower center',
          ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig('figure2_transition_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure2_transition_matrices.png")

# ============================================================
# FIGURE 3: CATEGORY IMPROVEMENT CHART
# Top and bottom categories for role prompting
# ============================================================
print("Building Figure 3: Category breakdown...")

if 'category' in df.columns:
    cat_pivot = df.groupby(['category', 'strategy'])['gpt4o_correct'].mean().unstack()

    if BASELINE in cat_pivot.columns and 'role' in cat_pivot.columns:
        cat_pivot['role_improvement'] = (
            cat_pivot['role'] - cat_pivot[BASELINE]
        ) * 100
        cat_pivot = cat_pivot.sort_values('role_improvement', ascending=False)

        top10 = cat_pivot.head(10)
        bottom10 = cat_pivot.tail(10)
        combined = pd.concat([top10, bottom10])
        combined = combined.sort_values('role_improvement', ascending=True)

        fig, ax = plt.subplots(figsize=(12, 10))

        colors = ['#F44336' if v < 0 else '#4CAF50'
                 for v in combined['role_improvement']]

        bars = ax.barh(combined.index, combined['role_improvement'],
                      color=colors, alpha=0.85, edgecolor='black',
                      linewidth=0.8)

        # Add value labels
        for bar, val in zip(bars, combined['role_improvement']):
            x_pos = val + 0.5 if val >= 0 else val - 0.5
            ha = 'left' if val >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                   f'{val:+.1f}%', ha=ha, va='center',
                   fontsize=9, fontweight='bold')

        ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
        ax.set_xlabel('Accuracy Change vs Zero-Shot (%)',
                     fontsize=12, fontweight='bold')
        ax.set_title('Role Prompting: Category-Level Accuracy Change vs Zero-Shot\n(Top 10 Most Improved vs Bottom 10 Least Improved)',
                    fontsize=13, fontweight='bold', pad=15)

        legend_elements = [
            mpatches.Patch(facecolor='#4CAF50', alpha=0.85,
                          label='Improvement'),
            mpatches.Patch(facecolor='#F44336', alpha=0.85,
                          label='Decline')
        ]
        ax.legend(handles=legend_elements, fontsize=11,
                 loc='lower right')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig('figure3_category_breakdown.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved figure3_category_breakdown.png")
else:
    print("  No category column — skipping Figure 3")

# ============================================================
# FIGURE 4: INFORMATIVENESS-ADJUSTED ACCURACY
# Single score ranking all strategies
# ============================================================
print("Building Figure 4: Informativeness-adjusted accuracy...")

adj_scores = {}
for strat in STRATEGY_ORDER:
    s = df[df['strategy'] == strat]
    score = (s['gpt4o_correct'].mean() +
             0.5 * s['gpt4o_abstention'].mean()) * 100
    adj_scores[strat] = score

fig, ax = plt.subplots(figsize=(10, 6))

labels = [STRATEGY_NAMES[s] for s in STRATEGY_ORDER]
values = [adj_scores[s] for s in STRATEGY_ORDER]
colors = ['#90CAF9' if s == BASELINE else
          '#F44336' if s == 'cot' else
          '#4CAF50' for s in STRATEGY_ORDER]

bars = ax.bar(labels, values, color=colors, alpha=0.85,
             edgecolor='black', linewidth=0.8)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{val:.2f}%',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold')

ax.set_ylabel('Informativeness-Adjusted Accuracy (%)',
             fontsize=12, fontweight='bold')
ax.set_title('Informativeness-Adjusted Accuracy by Strategy\n(Score = Accuracy + 0.5 × Abstention Rate)',
            fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, max(values) + 5)

legend_elements = [
    mpatches.Patch(facecolor='#90CAF9', alpha=0.85,
                  label='Baseline (Zero-Shot)'),
    mpatches.Patch(facecolor='#4CAF50', alpha=0.85,
                  label='Improvement'),
    mpatches.Patch(facecolor='#F44336', alpha=0.85,
                  label='Net Harmful (CoT)')
]
ax.legend(handles=legend_elements, fontsize=10)
ax.axhline(y=adj_scores[BASELINE], color='gray',
          linestyle='--', alpha=0.5, linewidth=1.5,
          label='Baseline')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figure4_adjusted_accuracy.png',
           dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure4_adjusted_accuracy.png")

# ============================================================
# FIGURE 5: DETERRENCE vs REDISTRIBUTION SUMMARY
# Retreat-to-gain ratio for all strategies
# ============================================================
print("Building Figure 5: Deterrence vs redistribution summary...")

transition_data = {
    'few_shot':           {'genuine': 59, 'retreat': 7,  'regression': 55},
    'cot':                {'genuine': 48, 'retreat': 11, 'regression': 76},
    'role':               {'genuine': 81, 'retreat': 10, 'regression': 28},
    'direct_instruction': {'genuine': 67, 'retreat': 10, 'regression': 29}
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Hallucination Reduction Mechanism Analysis',
            fontsize=15, fontweight='bold')

# Left plot: Genuine gains vs safety retreats vs regressions
strats = list(transition_data.keys())
genuine = [transition_data[s]['genuine'] for s in strats]
retreat = [transition_data[s]['retreat'] for s in strats]
regression = [transition_data[s]['regression'] for s in strats]
labels = [STRATEGY_NAMES[s] for s in strats]

x = np.arange(len(strats))
width = 0.25

ax1 = axes[0]
b1 = ax1.bar(x - width, genuine, width, label='Genuine Gain\n(INCORRECT→CORRECT)',
             color='#4CAF50', alpha=0.85, edgecolor='black')
b2 = ax1.bar(x, retreat, width, label='Safety Retreat\n(INCORRECT→ABSTAINED)',
             color='#FF9800', alpha=0.85, edgecolor='black')
b3 = ax1.bar(x + width, regression, width, label='Regression\n(CORRECT→INCORRECT)',
             color='#F44336', alpha=0.85, edgecolor='black')

for bars in [b1, b2, b3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2,
                height + 0.5, f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

ax1.set_title('Question-Level Transitions\nvs Zero-Shot Baseline',
             fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11)
ax1.set_ylabel('Number of Questions', fontsize=11)
ax1.legend(fontsize=9, loc='upper right')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', alpha=0.3)

# Right plot: Retreat-to-gain ratio
ax2 = axes[1]
ratios = [transition_data[s]['retreat'] / transition_data[s]['genuine']
         for s in strats]
colors_ratio = ['#F44336' if r > 1 else
                '#FF9800' if r > 0.5 else
                '#4CAF50' for r in ratios]

bars = ax2.bar(labels, ratios, color=colors_ratio,
              alpha=0.85, edgecolor='black')

for bar, ratio in zip(bars, ratios):
    ax2.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f'{ratio:.2f}',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold')

ax2.axhline(y=1.0, color='red', linestyle='--',
           linewidth=2, label='Threshold: ratio=1\n(above = redistribution)')
ax2.axhline(y=0.5, color='orange', linestyle=':',
           linewidth=1.5, label='ratio=0.5')

ax2.set_title('Retreat-to-Gain Ratio\n(Lower = More Genuine Improvement)',
             fontsize=12, fontweight='bold')
ax2.set_ylabel('Ratio (Safety Retreats / Genuine Gains)', fontsize=11)
ax2.set_ylim(0, max(ratios) + 0.2)
ax2.legend(fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', alpha=0.3)

legend_elements = [
    mpatches.Patch(facecolor='#4CAF50', alpha=0.85,
                  label='Primarily Deterrent (ratio < 0.5)'),
    mpatches.Patch(facecolor='#FF9800', alpha=0.85,
                  label='Mixed (0.5 < ratio < 1.0)'),
    mpatches.Patch(facecolor='#F44336', alpha=0.85,
                  label='Primarily Redistribution (ratio > 1.0)')
]
fig.legend(handles=legend_elements, loc='lower center',
          ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
plt.savefig('figure5_deterrence_redistribution.png',
           dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure5_deterrence_redistribution.png")

# ============================================================
# DONE
# ============================================================
print("\n" + "="*60)
print("ALL FIGURES SAVED")
print("="*60)
print("figure1_grouped_bars.png       — Main results table")
print("figure2_transition_matrices.png — Deterrence vs redistribution")
print("figure3_category_breakdown.png  — Category analysis")
print("figure4_adjusted_accuracy.png   — Informativeness ranking")
print("figure5_deterrence_redistribution.png — Mechanism summary")
print("\nNext step: write your results section using these figures")