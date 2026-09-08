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

CATEGORICAL_PREDICTORS = {
    'method':           'random',
    'pseudo_bulk':      'False',
    'cell normalised':  'False',
    'transform 1':      'none',
    'transform 2':      'none',
    'replicate':        '0',
    'control_delta':    'False',
}
CONTINUOUS_PREDICTORS = ['snr', 'cell_count', 'n_TPs',
                          'ERMA', '0_fraction__before_filtering__all']

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
    df_model_base[col] = (df[col] - q25) / (q75 - q25)

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

formula_terms = (
    [cat_term(col, ref_level) for col, ref_level in CATEGORICAL_PREDICTORS.items()] +
    [cont_term(p) for p in CONTINUOUS_PREDICTORS]
)

for outcome in OUTCOMES:
    lhs = ref(outcome)
    formula = lhs + ' ~ ' + ' + '.join(formula_terms)

    print('\n' + '=' * 72)
    print(f'Outcome : {outcome}')
    print(f'Formula : {formula}')
    print('=' * 72)

    cols = [outcome] + list(CATEGORICAL_PREDICTORS.keys()) + CONTINUOUS_PREDICTORS
    df_model = df_model_base[cols].dropna()

    model = smf.ols(formula, data=df_model).fit()
    print(model.summary())

    anton_util.log_timestamp(f'done with {outcome}')

anton_util.log_timestamp('done')
