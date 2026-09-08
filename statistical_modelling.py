import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
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

# All categorical predictors with their reference levels.
# All are screened; insignificant ones are dropped before the main model.
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
                          '0_fraction__before_filtering__all',
                          'n_genes_after_harmonisation']

SIGNIFICANCE_THRESHOLD = 0.05

# Cast categorical predictors to string to avoid bool/int coercion issues
for col in CATEGORICAL_PREDICTORS:
    df[col] = df[col].astype(str)

# Scale continuous predictors by IQR (Q25 to Q75).
# Coefficients then represent the effect of going from Q25 to Q75.
df_model_base = df.copy()
print('\nContinuous predictor distributions (min, Q25, median, Q75, max):')
for col in CONTINUOUS_PREDICTORS:
    lo, q25, med, q75, hi = df[col].quantile([0, 0.25, 0.5, 0.75, 1.0])
    print(f'  {col}: {lo:.3g}, {q25:.3g}, {med:.3g}, {q75:.3g}, {hi:.3g}')

zero_fraction_cols = [c for c in df.columns if '0_fraction' in c]
print('\nZero fraction variable distributions (min, Q25, median, Q75, max):')
for col in zero_fraction_cols:
    lo, q25, med, q75, hi = df[col].quantile([0, 0.25, 0.5, 0.75, 1.0])
    print(f'  {col}: {lo:.3g}, {q25:.3g}, {med:.3g}, {q75:.3g}, {hi:.3g}')

print('\nLevels of categorical predictors:')
for col, ref_level in CATEGORICAL_PREDICTORS.items():
    levels = sorted(df[col].unique())
    print(f'  {col}: {levels}  (reference: {ref_level})')

for col in CONTINUOUS_PREDICTORS:
    q25, q75 = df[col].quantile([0.25, 0.75])
    iqr = q75 - q25
    if iqr == 0:
        raise ValueError(f'IQR is zero for {col!r} — cannot scale. ')
    df_model_base[col] = (df[col] - q25) / iqr

print(f'\nContinuous predictors (IQR-scaled): {CONTINUOUS_PREDICTORS}')

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

    anton_util.log_timestamp(f'done with {outcome}')

anton_util.log_timestamp('done')
