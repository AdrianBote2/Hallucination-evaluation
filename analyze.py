import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# FIX: Changed from "evaluated_results.csv" to "evaluated_results_v2.csv"
# because v2 uses the better prompt with incorrect_answers column
# ============================================================
df = pd.read_csv("evaluated_results_v2.csv")
print(f"Loaded {len(df)} rows across {df['strategy'].nunique()} strategies")
print(f"Strategies: {df['strategy'].unique()}\n")

# ============================================================
# BASELINE
# FIX: Changed from "baseline" to "zero_shot" to match your
# actual strategy names in the CSV
# ============================================================
BASELINE = "zero_shot"
strategies = df['strategy'].unique().tolist()
non_baseline = [s for s in strategies if s != BASELINE]

# ============================================================
# LABEL CLEANUP
# Converts the three binary columns into one clean label column
# Drops any rows with parse errors from the GPT judge
# ============================================================
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
error_count = len(df[df['label'] == 'ERROR'])
df = df[df['label'] != 'ERROR']
print(f"Dropped {error_count} parse errors\n")

# ============================================================
# AGGREGATE METRICS PER STRATEGY
# Simple accuracy/hallucination/abstention rates per strategy
# This is your main results table for the paper
# ============================================================
print("=" * 60)
print("AGGREGATE METRICS BY STRATEGY")
print("=" * 60)

summary = df.groupby('strategy').agg(
    n=('label', 'count'),
    accuracy=('gpt4o_correct', 'mean'),
    hallucination_rate=('gpt4o_hallucination', 'mean'),
    abstention_rate=('gpt4o_abstention', 'mean')
)

summary[['accuracy', 'hallucination_rate', 'abstention_rate']] *= 100
summary['n'] = summary['n'].astype(int)
print(summary.round(2).to_string())

# ============================================================
# MARGIN OF ERROR
# 95% confidence intervals for every metric per strategy
# These go in your paper as the ± values in your table
# ============================================================
print("\n" + "=" * 60)
print("MARGIN OF ERROR (95% confidence)")
print("=" * 60)

def margin_of_error(values):
    p = values.mean()
    n = len(values)
    if n == 0 or p == 0 or p == 1:
        return 0
    return 1.96 * np.sqrt(p * (1-p) / n) * 100

for strategy in strategies:
    subset = df[df['strategy'] == strategy]
    print(f"\n{strategy}:")
    print(f"  Accuracy:      ±{margin_of_error(subset['gpt4o_correct']):.2f}%")
    print(f"  Hallucination: ±{margin_of_error(subset['gpt4o_hallucination']):.2f}%")
    print(f"  Abstention:    ±{margin_of_error(subset['gpt4o_abstention']):.2f}%")

# ============================================================
# TRANSITION MATRIX
# This is your paper's core contribution
# For each strategy vs zero_shot, shows exactly where
# labels moved at the question level:
# - INCORRECT → CORRECT = genuine deterrence
# - INCORRECT → ABSTAINED = redistribution
# - CORRECT → INCORRECT = regression (got worse)
# - CORRECT → ABSTAINED = overcaution (unnecessary hedging)
# ============================================================
print("\n" + "=" * 60)
print("TRANSITION MATRICES (vs zero_shot baseline)")
print("=" * 60)

pivot = df.pivot_table(
    index='question_id',
    columns='strategy',
    values='label',
    aggfunc='first'
)

LABEL_ORDER = ['CORRECT', 'INCORRECT', 'ABSTAINED']
transition_results = {}

for strat in non_baseline:
    if BASELINE not in pivot.columns or strat not in pivot.columns:
        print(f"Skipping {strat} — missing in pivot")
        continue

    pair = pivot[[BASELINE, strat]].dropna()
    n_shared = len(pair)

    matrix = pd.crosstab(
        pair[BASELINE],
        pair[strat],
        rownames=[f'zero_shot →'],
        colnames=[f'→ {strat}']
    ).reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)

    print(f"\n--- zero_shot vs {strat} (n={n_shared} shared questions) ---")
    print(matrix)

    total = matrix.values.sum()
    unchanged = np.diag(matrix.values).sum()
    changed = total - unchanged

    print(f"\n  Unchanged: {unchanged} ({100*unchanged/total:.1f}%)")
    print(f"  Changed:   {changed} ({100*changed/total:.1f}%)")

    try:
        inc_to_cor = matrix.loc['INCORRECT', 'CORRECT']
        inc_to_abs = matrix.loc['INCORRECT', 'ABSTAINED']
        cor_to_inc = matrix.loc['CORRECT', 'INCORRECT']
        cor_to_abs = matrix.loc['CORRECT', 'ABSTAINED']

        print(f"  INCORRECT → CORRECT   (genuine gain):    {inc_to_cor}")
        print(f"  INCORRECT → ABSTAINED (safety retreat):  {inc_to_abs}")
        print(f"  CORRECT → INCORRECT   (regression):      {cor_to_inc}")
        print(f"  CORRECT → ABSTAINED   (overcaution):     {cor_to_abs}")

        transition_results[strat] = {
            'matrix': matrix,
            'inc_to_cor': inc_to_cor,
            'inc_to_abs': inc_to_abs,
            'cor_to_inc': cor_to_inc,
            'cor_to_abs': cor_to_abs
        }
    except KeyError:
        print("  Some label types missing — skipping transition counts")

# ============================================================
# REDISTRIBUTION CLASSIFICATION
# Classifies each strategy as:
# GENUINE IMPROVEMENT — hallucination shifts to correct answers
# REDISTRIBUTION — hallucination shifts to abstention instead
# NET HARMFUL — more regressions than gains
# MIXED — no clear pattern
# This is the direct answer to your research question
# ============================================================
print("\n" + "=" * 60)
print("REDISTRIBUTION CLASSIFICATION")
print("=" * 60)
print("This directly answers: does prompting deter or redistribute?\n")

for strat, data in transition_results.items():
    inc_to_cor = data['inc_to_cor']
    inc_to_abs = data['inc_to_abs']
    cor_to_inc = data['cor_to_inc']
    cor_to_abs = data['cor_to_abs']

    ratio = inc_to_abs / inc_to_cor if inc_to_cor > 0 else float('inf')

    print(f"{strat}:")
    print(f"  Genuine truthfulness gain:  {inc_to_cor}")
    print(f"  Safety retreat (hedging):   {inc_to_abs}")
    print(f"  Regression (new errors):    {cor_to_inc}")
    print(f"  Overcaution (lost correct): {cor_to_abs}")
    print(f"  Retreat-to-gain ratio:      {ratio:.2f} (>1 = mostly hedging)")

    if inc_to_cor > inc_to_abs and inc_to_cor > cor_to_inc:
        verdict = "GENUINE IMPROVEMENT — prompting deters hallucination"
    elif inc_to_abs > inc_to_cor:
        verdict = "REDISTRIBUTION — prompting deflects to abstention"
    elif cor_to_inc > inc_to_cor:
        verdict = "NET HARMFUL — prompting increases hallucination"
    else:
        verdict = "MIXED / NEGLIGIBLE — no clear pattern"

    print(f"  Verdict: {verdict}\n")

# ============================================================
# STATISTICAL TESTS
# Chi-square tests whether label distributions differ
# significantly across all 5 strategies overall
# McNemar tests each strategy vs zero_shot on PAIRED questions
# (McNemar is correct here — same questions appear in all strategies
# so observations are not independent)
# Bonferroni correction adjusts for multiple comparisons
# ============================================================
print("=" * 60)
print("STATISTICAL TESTS")
print("=" * 60)

# Overall chi-square
contingency = pd.crosstab(df['strategy'], df['label'])
chi2, p_chi2, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square (all strategies):")
print(f"  chi2={chi2:.3f}, dof={dof}, p={p_chi2:.4f}")
if p_chi2 < 0.05:
    print("  Label distributions differ significantly across strategies")
else:
    print("  No significant difference across strategies")

# Pairwise McNemar — accuracy
print(f"\nMcNemar tests — accuracy (zero_shot vs each strategy):")
print(f"  Bonferroni-corrected alpha: {0.05/len(non_baseline):.4f}\n")

for strat in non_baseline:
    if BASELINE not in pivot.columns or strat not in pivot.columns:
        continue

    pair = pivot[[BASELINE, strat]].dropna()
    base_correct = (pair[BASELINE] == 'CORRECT').astype(int)
    strat_correct = (pair[strat] == 'CORRECT').astype(int)

    b = ((base_correct == 1) & (strat_correct == 0)).sum()
    c = ((base_correct == 0) & (strat_correct == 1)).sum()

    if b + c == 0:
        print(f"  {strat}: no discordant pairs")
        continue

    mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_mcnemar = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
    direction = "improvement" if c > b else "decline"
    sig = "SIGNIFICANT" if p_mcnemar < (0.05/len(non_baseline)) else "not significant"
    print(f"  {strat}: b={b}, c={c}, p={p_mcnemar:.4f} ({direction}, {sig})")

# Pairwise McNemar — abstention
print(f"\nMcNemar tests — abstention shift:")

for strat in non_baseline:
    if BASELINE not in pivot.columns or strat not in pivot.columns:
        continue

    pair = pivot[[BASELINE, strat]].dropna()
    base_abs = (pair[BASELINE] == 'ABSTAINED').astype(int)
    strat_abs = (pair[strat] == 'ABSTAINED').astype(int)

    b = ((base_abs == 1) & (strat_abs == 0)).sum()
    c = ((base_abs == 0) & (strat_abs == 1)).sum()

    if b + c == 0:
        print(f"  {strat}: no abstention shift")
        continue

    mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_mcnemar = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
    direction = "more abstention" if c > b else "less abstention"
    print(f"  {strat}: p={p_mcnemar:.4f} ({direction})")

# ============================================================
# EFFECT SIZES
# Cohen's h measures the practical size of differences
# between proportions — not just whether they're significant
# Small: |h| < 0.2, Medium: 0.2-0.5, Large: > 0.5
# ============================================================
print("\n" + "=" * 60)
print("EFFECT SIZES (Cohen's h vs zero_shot)")
print("=" * 60)
print("  |h| < 0.2 = small, 0.2-0.5 = medium, > 0.5 = large\n")

baseline_df = df[df['strategy'] == BASELINE]
p_base_correct = baseline_df['gpt4o_correct'].mean()
p_base_halluc  = baseline_df['gpt4o_hallucination'].mean()
p_base_abstain = baseline_df['gpt4o_abstention'].mean()

def cohen_h(p1, p2):
    p1 = np.clip(p1, 1e-10, 1 - 1e-10)
    p2 = np.clip(p2, 1e-10, 1 - 1e-10)
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

for strat in non_baseline:
    strat_df = df[df['strategy'] == strat]
    p_cor = strat_df['gpt4o_correct'].mean()
    p_hal = strat_df['gpt4o_hallucination'].mean()
    p_abs = strat_df['gpt4o_abstention'].mean()

    h_cor = cohen_h(p_cor, p_base_correct)
    h_hal = cohen_h(p_hal, p_base_halluc)
    h_abs = cohen_h(p_abs, p_base_abstain)

    size_cor = "large" if abs(h_cor) > 0.5 else "medium" if abs(h_cor) > 0.2 else "small"
    size_hal = "large" if abs(h_hal) > 0.5 else "medium" if abs(h_hal) > 0.2 else "small"
    size_abs = "large" if abs(h_abs) > 0.5 else "medium" if abs(h_abs) > 0.2 else "small"

    print(f"{strat}:")
    print(f"  Accuracy:      h={h_cor:+.3f} ({size_cor})")
    print(f"  Hallucination: h={h_hal:+.3f} ({size_hal})")
    print(f"  Abstention:    h={h_abs:+.3f} ({size_abs})")

# ============================================================
# INFORMATIVENESS-ADJUSTED ACCURACY
# Treats abstention as half credit (0.5) rather than zero
# because refusing to answer is better than being confidently wrong
# but still costs the user useful information
# ============================================================
print("\n" + "=" * 60)
print("INFORMATIVENESS-ADJUSTED ACCURACY")
print("=" * 60)
print("  Score = accuracy + 0.5 × abstention rate")
print("  Abstaining is safer than hallucinating but still costs information\n")

for strat in strategies:
    s = df[df['strategy'] == strat]
    score = s['gpt4o_correct'].mean() + 0.5 * s['gpt4o_abstention'].mean()
    print(f"  {strat}: {score*100:.2f}%")

# ============================================================
# CATEGORY BREAKDOWN
# Shows which of TruthfulQA's 38 categories benefit most
# from each prompting strategy vs zero_shot
# This is an original contribution — no prior paper reports this
# ============================================================
print("\n" + "=" * 60)
print("CATEGORY BREAKDOWN")
print("=" * 60)
print("  Accuracy per category per strategy\n")

if 'category' in df.columns:
    cat_analysis = df.groupby(['category', 'strategy']).agg(
        accuracy=('gpt4o_correct', 'mean'),
        hallucination=('gpt4o_hallucination', 'mean'),
        n=('gpt4o_correct', 'count')
    ).round(3) * [100, 100, 1]

    cat_analysis['n'] = cat_analysis['n'].astype(int)

    # Show top 10 categories where role prompting improves most over zero_shot
    if 'role' in strategies:
        cat_pivot = df.groupby(['category', 'strategy'])['gpt4o_correct'].mean().unstack()
        if BASELINE in cat_pivot.columns and 'role' in cat_pivot.columns:
            cat_pivot['role_improvement'] = (
                cat_pivot['role'] - cat_pivot[BASELINE]
            ) * 100
            cat_pivot = cat_pivot.sort_values('role_improvement', ascending=False)

            print("Top 10 categories most improved by role prompting:")
            print(cat_pivot[['role_improvement']].head(10).round(2))

            print("\nBottom 10 categories least improved by role prompting:")
            print(cat_pivot[['role_improvement']].tail(10).round(2))
else:
    print("  No category column found — skipping")

# ============================================================
# SAVE EVERYTHING
# ============================================================
summary.to_csv("redistribution_summary.csv")
print("\n\nSaved redistribution_summary.csv")
print("\nDONE — All analysis complete")
print("Next step: build graphs with graphs.py")