import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import anton_util

from grn_code.pipeline_configuration import output_base_path

compiled_results_path = Path(f'{output_base_path}/simulated/compiled_results.pkl')
output_dir = Path(f'{output_base_path}/simulated/statistical_modelling')

df = anton_util.unpickle_object(str(compiled_results_path))
anton_util.log_timestamp(f'data loaded, shape: {df.shape}')
print(f'Columns available: {list(df.columns)}')

# OUTCOMES = ['AUPR ratio', 'AUPR', 'AUROC', 'top_k_accuracy']
OUTCOMES = ['AUROC']  # Debug

CATEGORICAL_PREDICTORS = {
    'method':                    'random',
    'pseudo_bulk':               'False',
    'cell normalised':           'False',
    'transform 1':               'none',
    'transform 2':               'none',
    'control_delta':             'False',
    'replicate':                 '0',
}
CONTINUOUS_PREDICTORS = ['snr', 'cell_count', 'n_TPs', 'ERMA',
                          '0_fraction__before_filtering__all']

# n_genes_after_harmonisation: tested, IQR=0 (median=50 for most datasets), not usable as continuous predictor

SIGNIFICANCE_THRESHOLD = 0.05

# Cast categorical predictors to string to avoid bool/int coercion issues
for col in CATEGORICAL_PREDICTORS:
    df[col] = df[col].astype(str)

def print_distributions(cols, header):
    print(f'\n{header} (min, Q25, median, Q75, max):')
    for col in cols:
        lo, q25, med, q75, hi = df[col].quantile([0, 0.25, 0.5, 0.75, 1.0])
        print(f'  {col}: {lo:.3g}, {q25:.3g}, {med:.3g}, {q75:.3g}, {hi:.3g}')

# Scale continuous predictors by 2 standard deviations (mean ± 1 std).
# Coefficients then represent the effect of going from mean-1sd to mean+1sd.
df_model_base = df.copy()

print('\nLevels of categorical predictors:')
for col, ref_level in CATEGORICAL_PREDICTORS.items():
    levels = sorted(df[col].unique())
    print(f'  {col}: {levels}  (reference: {ref_level})')
zero_fraction_cols = [c for c in df.columns if '0_fraction' in c]
print_distributions(zero_fraction_cols, 'Zero fraction variable distributions')

print_distributions(CONTINUOUS_PREDICTORS, 'Continuous predictor distributions')

print('\nContinuous predictor scaling (mean, sd, mean-1sd, mean+1sd):')
for col in CONTINUOUS_PREDICTORS:
    mean = df[col].mean()
    sd = df[col].std()
    if sd == 0:
        raise ValueError(f'SD is zero for {col!r} — cannot scale.')
    lo, hi = mean - sd, mean + sd
    print(f'  {col}: mean={mean:.3g}, sd={sd:.3g}, range=[{lo:.3g}, {hi:.3g}]')
    df_model_base[col] = (df[col] - mean) / (2 * sd)

print(f'\nContinuous predictors (±1sd-scaled): {CONTINUOUS_PREDICTORS}')

def needs_quoting(name):
    return ' ' in name or name[0].isdigit()

def ref(name):
    """Return a formula reference to a column, quoting if necessary."""
    return f'Q("{name}")' if needs_quoting(name) else name

def cat_term(name, ref_level):
    """Categorical term with explicit reference level."""
    return f'C({ref(name)}, Treatment("{ref_level}"))'

def cont_term(name):
    """Continuous term."""
    return ref(name)

def joint_f_pvalue(model, term_prefix):
    """F-test p-value for all levels of a categorical term."""
    matching = [t for t in model.params.index if t.startswith(term_prefix)]
    if not matching:
        return float('nan')
    return model.f_test([f'{t} = 0' for t in matching]).pvalue

for outcome in OUTCOMES:
    lhs = ref(outcome)

    # --- Screening: fit full model, drop insignificant predictors ---
    all_terms = (
        [cat_term(col, ref_level) for col, ref_level in CATEGORICAL_PREDICTORS.items()] +
        [cont_term(p) for p in CONTINUOUS_PREDICTORS]
    )
    screen_formula = lhs + ' ~ ' + ' + '.join(all_terms)
    cols_all = [outcome] + list(CATEGORICAL_PREDICTORS.keys()) + CONTINUOUS_PREDICTORS
    df_model = df_model_base[cols_all].dropna()
    screen_model = smf.ols(screen_formula, data=df_model).fit()

    print('\n' + '=' * 72)
    print(f'Screening stage for outcome: {outcome}')
    print(f'  {"Predictor":<45} {"p-value":>10}  {"included?":>10}')
    print(f'  {"-"*45} {"-"*10}  {"-"*10}')

    active_cat = {}
    for col, ref_level in CATEGORICAL_PREDICTORS.items():
        prefix = f'C({ref(col)}, Treatment("{ref_level}"))'
        pval = joint_f_pvalue(screen_model, prefix)
        keep = pval < SIGNIFICANCE_THRESHOLD
        if keep:
            active_cat[col] = ref_level
        print(f'  {col:<45} {pval:>10.4f}  {"yes" if keep else "no":>10}')

    active_cont = []
    for col in CONTINUOUS_PREDICTORS:
        term = cont_term(col)
        pval = screen_model.pvalues.get(term, float('nan'))
        keep = pval < SIGNIFICANCE_THRESHOLD
        if keep:
            active_cont.append(col)
        print(f'  {col:<45} {pval:>10.4f}  {"yes" if keep else "no":>10}')

    # --- Main model: refit with only significant predictors ---
    formula_terms_active = (
        [cat_term(col, ref_level) for col, ref_level in active_cat.items()] +
        [cont_term(p) for p in active_cont]
    )
    formula = lhs + ' ~ ' + ' + '.join(formula_terms_active)

    print('\n' + '=' * 72)
    print(f'Outcome : {outcome}')
    print(f'Formula : {formula}')
    print('=' * 72)

    model = smf.ols(formula, data=df_model).fit()
    print(model.summary())

    # --- Effect size summary ---
    # Each row is one coefficient (categorical levels individually, continuous as-is).
    # Categorical coefficients are relative to their reference level.
    # Continuous coefficients represent the effect of ±1sd around the mean.
    rows = []
    for col, ref_level in active_cat.items():
        prefix = f'C({ref(col)}, Treatment("{ref_level}"))'
        for term, coef in model.params.items():
            if not term.startswith(prefix):
                continue
            ci = model.conf_int().loc[term]
            # Extract the level name from the term string, e.g. "[T.True]" -> "True"
            level = term[len(prefix):]
            if level.startswith('[T.') and level.endswith(']'):
                level = level[3:-1]
            rows.append({
                'predictor': f'{col} = {level}',
                'abs_coef': abs(coef),
                'coef': coef,
                'ci_lo': ci[0],
                'ci_hi': ci[1],
            })
    for col in active_cont:
        term = cont_term(col)
        coef = model.params[term]
        ci = model.conf_int().loc[term]
        rows.append({
            'predictor': col,
            'abs_coef': abs(coef),
            'coef': coef,
            'ci_lo': ci[0],
            'ci_hi': ci[1],
        })

    rows.sort(key=lambda r: r['abs_coef'], reverse=True)

    print('\nEffect size ranking (categorical levels vs reference, continuous ±1sd):')
    print(f'  {"predictor":<55} {"coef":>8}  {"95% CI"}')
    print(f'  {"-"*55} {"-"*8}  {"-"*20}')
    for r in rows:
        print(f'  {r["predictor"]:<55} {r["coef"]:>8.4f}  [{r["ci_lo"]:.4f}, {r["ci_hi"]:.4f}]')

    # --- Coefficient forest plot ---
    # Plot all individual coefficients (excluding intercept), sorted by |coef|
    params = model.params.drop('Intercept')
    ci = model.conf_int().drop('Intercept')
    order = params.abs().sort_values(ascending=True).index
    params = params[order]
    ci = ci.loc[order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(params) * 0.3)))
    y = range(len(params))
    ax.barh(list(y), params.values, xerr=[params.values - ci[0].values, ci[1].values - params.values],
            align='center', height=0.6, color='steelblue', ecolor='black', capsize=3)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(params.index, fontsize=7)
    ax.set_xlabel('Coefficient (AUROC scale, continuous predictors ±1sd-scaled)')
    ax.set_title(f'Effect sizes: {outcome}')
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f'effect_sizes_{outcome}.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'\nCoefficient plot saved to {plot_path}')

    anton_util.log_timestamp(f'done with {outcome}')

anton_util.log_timestamp('done')
